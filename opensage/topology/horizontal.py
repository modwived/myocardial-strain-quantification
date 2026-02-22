"""
Horizontal Agent Topology for OpenSage.

From the paper: "horizontal agent topology, where multiple sub-agents
simultaneously execute the same task using distinct plans, with their
results integrated through an agent ensemble mechanism."

The horizontal topology implements the paper's agent ensemble, where:
1. Multiple agents tackle the same problem with different strategies
2. Each agent uses a distinct approach/plan
3. An integrator aggregates and selects the best solution
4. This improves robustness and solution quality

The paper notes this is used for "cost-efficiency" as well, since
multiple smaller/cheaper agents can collectively outperform one large run.
"""

import concurrent.futures
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from opensage.core.agent import SageAgent


@dataclass
class AgentApproach:
    """A distinct approach/strategy for an ensemble agent."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    strategy: str = ""
    system_prompt_suffix: str = ""


@dataclass
class EnsembleResult:
    """Aggregated result from the horizontal ensemble."""
    approach_results: List[Dict[str, Any]] = field(default_factory=list)
    best_result: str = ""
    consensus_summary: str = ""
    integration_method: str = "best_of_n"
    total_execution_time: float = 0.0


class HorizontalTopology:
    """
    Implements horizontal (ensemble) agent topology.

    Multiple SageAgent instances run concurrently on the same task
    with different strategies, and an integrator synthesizes the results.

    From the paper: "the agent ensemble mechanism" ensures that diverse
    approaches are explored and the best solution surfaces.
    """

    STRATEGIES = [
        AgentApproach(
            name="systematic",
            strategy="systematic_analysis",
            system_prompt_suffix=(
                "Approach this problem systematically: first analyze the problem space, "
                "then form a hypothesis, then test it step by step."
            ),
        ),
        AgentApproach(
            name="heuristic",
            strategy="heuristic_search",
            system_prompt_suffix=(
                "Approach this problem heuristically: use domain knowledge and intuition "
                "to quickly identify the most likely solution path, then validate it."
            ),
        ),
        AgentApproach(
            name="exhaustive",
            strategy="exhaustive_search",
            system_prompt_suffix=(
                "Approach this problem exhaustively: explore multiple possible solutions, "
                "enumerate edge cases, and verify correctness thoroughly."
            ),
        ),
    ]

    def __init__(self, parent_agent: "SageAgent", n_agents: int = 2):
        """
        Args:
            parent_agent: The parent agent providing resources (LLM, tools)
            n_agents: Number of ensemble agents to create (2-3 recommended)
        """
        self.parent = parent_agent
        self.n_agents = min(max(n_agents, 2), len(self.STRATEGIES))
        self._ensemble_agents: List["SageAgent"] = []

    def execute(self, task: str, custom_strategies: Optional[List[str]] = None) -> EnsembleResult:
        """
        Execute task using the horizontal ensemble.

        Creates N agents with different strategies, runs them concurrently,
        then integrates results into a consensus answer.

        Args:
            task: The task to solve
            custom_strategies: Optional custom strategy descriptions

        Returns:
            EnsembleResult with individual and integrated results
        """
        strategies = (
            [AgentApproach(name=f"agent_{i}", strategy=s, system_prompt_suffix=s)
             for i, s in enumerate(custom_strategies)]
            if custom_strategies
            else self.STRATEGIES[:self.n_agents]
        )

        start_time = time.time()
        approach_results = []

        # Run agents concurrently using thread pool
        # (In production, this would use async/await or process pool)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_agents) as executor:
            futures = {
                executor.submit(
                    self._run_ensemble_agent, task, approach
                ): approach
                for approach in strategies
            }

            for future in concurrent.futures.as_completed(futures):
                approach = futures[future]
                try:
                    result = future.result(timeout=300)
                    approach_results.append({
                        "agent_name": approach.name,
                        "strategy": approach.strategy,
                        "result": result,
                        "success": True,
                    })
                except Exception as e:
                    approach_results.append({
                        "agent_name": approach.name,
                        "strategy": approach.strategy,
                        "result": str(e),
                        "success": False,
                    })

        total_time = time.time() - start_time

        # Integrate results
        successful_results = [r for r in approach_results if r["success"]]

        if not successful_results:
            return EnsembleResult(
                approach_results=approach_results,
                best_result="All ensemble agents failed",
                consensus_summary="Ensemble failed",
                total_execution_time=total_time,
            )

        best_result, consensus = self._integrate_results(task, approach_results)

        # Store ensemble activity in parent memory
        self.parent.memory.observe(
            observation=(
                f"Horizontal ensemble: {len(approach_results)} agents ran, "
                f"{len(successful_results)} succeeded. "
                f"Strategies: {[r['strategy'] for r in approach_results]}"
            ),
            task_id=self.parent.memory._current_task_id,
            importance=0.8,
        )

        return EnsembleResult(
            approach_results=approach_results,
            best_result=best_result,
            consensus_summary=consensus,
            integration_method="llm_synthesis",
            total_execution_time=total_time,
        )

    def _run_ensemble_agent(self, task: str, approach: AgentApproach) -> str:
        """Create and run a single ensemble agent with a specific strategy."""
        from opensage.core.agent import SageAgent
        from opensage.memory.hierarchical import HierarchicalMemory
        from opensage.tools.manager import ToolManager

        # Each ensemble agent gets its own memory
        sub_memory = HierarchicalMemory(agent_id=f"ensemble_{approach.name}")

        # Shared tools (read-only access to the parent's tools)
        sub_tool_manager = ToolManager(executor=self.parent.executor)
        for name, tool in self.parent.tool_manager.tools.items():
            sub_tool_manager.register(tool)

        # Customize system prompt with strategy
        specialized_prompt = (
            f"{self.parent.system_prompt}\n\n"
            f"Your strategy: {approach.system_prompt_suffix}"
        )

        agent = SageAgent(
            name=f"ensemble_{approach.name}",
            system_prompt=specialized_prompt,
            llm=self.parent.llm,
            tool_manager=sub_tool_manager,
            memory=sub_memory,
            executor=self.parent.executor,
            max_iterations=10,
        )

        return agent.run(task)

    def _integrate_results(
        self, task: str, results: List[Dict[str, Any]]
    ) -> tuple[str, str]:
        """
        Use the parent LLM to synthesize ensemble results.

        The integrator selects the best solution or creates a synthesis
        from all approaches.
        """
        successful = [r for r in results if r["success"]]
        if len(successful) == 1:
            return successful[0]["result"], "Single agent succeeded"

        # Build integration prompt
        results_str = "\n\n".join(
            f"=== Agent: {r['agent_name']} (strategy: {r['strategy']}) ===\n{r['result']}"
            for r in successful
        )

        integration_prompt = f"""Multiple AI agents tackled the following task with different strategies.
Please synthesize the best solution from their approaches.

Task: {task}

Agent Results:
{results_str}

Provide:
1. The best/synthesized solution
2. A brief explanation of which approach worked best and why

Synthesized Solution:"""

        messages = [{"role": "user", "content": integration_prompt}]
        response = self.parent.llm.chat(
            messages=messages,
            system=(
                "You are an expert result integrator. Synthesize the best solution "
                "from multiple agent approaches, combining their insights."
            ),
            max_tokens=3000,
        )

        synthesis = response.content.strip()

        # Best individual result (longest successful one as heuristic)
        best_individual = max(successful, key=lambda r: len(r["result"]))["result"]

        return best_individual, synthesis

    def get_ensemble_summary(self, result: EnsembleResult) -> str:
        """Format the ensemble result for display."""
        lines = [
            f"\n{'='*60}",
            f"HORIZONTAL ENSEMBLE SUMMARY ({self.n_agents} agents)",
            f"Total time: {result.total_execution_time:.1f}s",
            "="*60,
        ]
        for r in result.approach_results:
            status = "✓" if r["success"] else "✗"
            lines.append(f"\n{status} Agent '{r['agent_name']}' [{r['strategy']}]:")
            preview = r["result"][:200] + "..." if len(r["result"]) > 200 else r["result"]
            lines.append(f"  {preview}")

        lines.extend([
            "\n--- INTEGRATED CONSENSUS ---",
            result.consensus_summary[:500],
        ])

        return "\n".join(lines)
