"""
OpenSage Engine: The top-level orchestrator.

The OpenSage engine is the ADK (Agent Development Kit) entry point.
It manages:
1. Creation of root-level SageAgents
2. Shared execution environment provisioning
3. Topology selection (vertical vs horizontal)
4. Session-level memory management
5. Benchmark task execution

From the paper: "We propose OpenSage, the first ADK that enables LLMs to
automatically create agents with self-generated topology and toolsets while
providing comprehensive and structured memory support."

Usage:
    engine = OpenSage(api_key="...", verbose=True)
    result = engine.solve("Fix the bug in fibonacci.py")
"""

import os
import time
from typing import Any, Dict, List, Optional

from opensage.core.agent import SageAgent
from opensage.llm.claude import ClaudeClient
from opensage.memory.hierarchical import HierarchicalMemory
from opensage.tools.executor import ExecutionEnvironment
from opensage.tools.manager import ToolManager
from opensage.tools.se_toolkit import get_se_toolkit


class OpenSage:
    """
    OpenSage Agent Development Kit.

    The top-level engine that creates and orchestrates SageAgents.
    Provides the infrastructure for self-programming agent generation.

    Key capabilities:
    - Automatic agent topology generation (vertical / horizontal)
    - Container-based tool execution
    - Hierarchical graph-based memory
    - AI-driven tool creation
    """

    DEFAULT_SYSTEM_PROMPT = """You are SageAgent, part of the OpenSage Agent Development Kit.

You are a highly capable AI agent that can:
1. Create specialized sub-agents for different aspects of a task (use create_sub_agent)
2. Write and register new tools at runtime (use create_tool)
3. Run multiple agents in parallel with different strategies (use use_ensemble)
4. Store and retrieve information from hierarchical memory (use store_memory / retrieve_memory)
5. Execute code, read/write files, and run shell commands

When facing a complex task:
- Decompose it into sub-tasks and create specialized sub-agents
- Generate custom tools when the built-in ones are insufficient
- Use ensemble mode for high-stakes decisions

Always:
- Check memory for relevant prior context before starting
- Store important findings in memory for future use
- Be systematic and thorough in your approach
- Report your findings and reasoning clearly
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5",
        work_dir: Optional[str] = None,
        verbose: bool = True,
        max_iterations: int = 15,
    ):
        """
        Initialize the OpenSage engine.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
            model: Claude model to use as backbone
            work_dir: Working directory for file operations
            verbose: Print agent reasoning steps
            max_iterations: Max iterations per agent run
        """
        self.model = model
        self.verbose = verbose
        self.max_iterations = max_iterations

        # Shared LLM client
        self.llm = ClaudeClient(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            model=model,
        )

        # Shared execution environment
        self.executor = ExecutionEnvironment(work_dir=work_dir)

        if verbose:
            print(f"[OpenSage] Initialized with model={model}")
            print(f"[OpenSage] Work directory: {self.executor.work_dir}")

    def create_agent(
        self,
        name: str = "SageAgent",
        system_prompt: Optional[str] = None,
        extra_tools: Optional[List] = None,
    ) -> SageAgent:
        """
        Create a fully-equipped SageAgent.

        The agent comes pre-loaded with the SE toolkit and all
        OpenSage meta-tools (create_sub_agent, create_tool, etc.).

        Args:
            name: Agent name
            system_prompt: Custom system prompt (uses default if None)
            extra_tools: Additional Tool objects to register

        Returns:
            A ready-to-use SageAgent
        """
        tool_manager = ToolManager(executor=self.executor)

        # Load SE toolkit
        se_tools = get_se_toolkit(self.executor)
        tool_manager.register_many(se_tools)

        # Register any extra tools
        if extra_tools:
            tool_manager.register_many(extra_tools)

        memory = HierarchicalMemory(agent_id=name)

        agent = SageAgent(
            name=name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            llm=self.llm,
            tool_manager=tool_manager,
            memory=memory,
            executor=self.executor,
            max_iterations=self.max_iterations,
            enable_topology=True,
            verbose=self.verbose,
        )

        return agent

    def solve(
        self,
        task: str,
        topology: str = "auto",
        agent_name: str = "SageAgent",
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Solve a task using OpenSage.

        The primary entry point for single-task execution.

        Args:
            task: The task to solve (natural language)
            topology: "auto" | "vertical" | "horizontal" | "single"
                - auto: Agent decides its own topology (recommended)
                - vertical: Force sequential sub-agent decomposition
                - horizontal: Force parallel ensemble
                - single: Single agent, no sub-agents
            agent_name: Name for the root agent
            system_prompt: Optional custom system prompt

        Returns:
            The final solution string
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[OpenSage] Task: {task}")
            print(f"[OpenSage] Topology: {topology}")
            print(f"{'='*60}")

        agent = self.create_agent(name=agent_name, system_prompt=system_prompt)

        start = time.time()

        if topology == "vertical":
            result = self._solve_vertical(agent, task)
        elif topology == "horizontal":
            result = self._solve_horizontal(agent, task)
        else:
            # "auto" or "single": let the agent decide its own topology
            result = agent.run(task)

        elapsed = time.time() - start

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[OpenSage] Completed in {elapsed:.1f}s")
            status = agent.get_status()
            print(f"[OpenSage] Sub-agents created: {status['sub_agents']}")
            print(f"[OpenSage] AI-generated tools: {status['ai_generated_tools']}")
            mem_summary = status["memory_summary"]
            print(f"[OpenSage] Memory nodes: {mem_summary['total_nodes']}")
            print(f"{'='*60}")

        return result

    def _solve_vertical(self, agent: SageAgent, task: str) -> str:
        """Execute with forced vertical topology."""
        from opensage.topology.vertical import VerticalTopology

        v_topo = VerticalTopology(parent_agent=agent)
        sub_task_specs = v_topo.decompose_via_llm(task)

        if self.verbose:
            print(f"[OpenSage] Vertical decomposition: {len(sub_task_specs)} sub-tasks")
            for i, spec in enumerate(sub_task_specs):
                print(f"  {i+1}. [{spec.agent_name}] {spec.description[:80]}")

        results = v_topo.execute(task, sub_task_specs)

        # Synthesize results
        successful = [r for r in results if r.success]
        all_outputs = "\n\n".join(
            f"=== {r.agent_name} ===\n{r.output}" for r in results
        )

        synthesis_prompt = (
            f"Multiple specialized agents completed sub-tasks for: {task}\n\n"
            f"Their results:\n{all_outputs}\n\n"
            f"Provide a final comprehensive answer synthesizing all findings:"
        )

        final = agent.llm.chat(
            messages=[{"role": "user", "content": synthesis_prompt}],
            system=agent.system_prompt,
            max_tokens=3000,
        )

        return final.content

    def _solve_horizontal(self, agent: SageAgent, task: str) -> str:
        """Execute with forced horizontal ensemble."""
        from opensage.topology.horizontal import HorizontalTopology

        ensemble = HorizontalTopology(parent_agent=agent, n_agents=2)
        result = ensemble.execute(task)

        if self.verbose:
            print(ensemble.get_ensemble_summary(result))

        return result.consensus_summary or result.best_result

    def run_benchmark(
        self,
        tasks: List[Dict[str, str]],
        topology: str = "auto",
    ) -> Dict[str, Any]:
        """
        Run OpenSage on a list of benchmark tasks.

        Args:
            tasks: List of {task_id, description} dicts
            topology: Topology mode for all tasks

        Returns:
            Dict with results, timing, and statistics
        """
        results = []
        total_start = time.time()

        for i, task_def in enumerate(tasks):
            task_id = task_def.get("task_id", str(i))
            description = task_def.get("description", "")

            print(f"\n[OpenSage] Task {i+1}/{len(tasks)}: {task_id}")

            task_start = time.time()
            try:
                result = self.solve(description, topology=topology)
                elapsed = time.time() - task_start
                results.append({
                    "task_id": task_id,
                    "success": True,
                    "result": result,
                    "time": elapsed,
                })
            except Exception as e:
                elapsed = time.time() - task_start
                results.append({
                    "task_id": task_id,
                    "success": False,
                    "error": str(e),
                    "time": elapsed,
                })

        total_elapsed = time.time() - total_start
        success_count = sum(1 for r in results if r.get("success"))

        return {
            "total_tasks": len(tasks),
            "successful": success_count,
            "failed": len(tasks) - success_count,
            "success_rate": success_count / len(tasks) if tasks else 0,
            "total_time": total_elapsed,
            "results": results,
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the OpenSage engine configuration."""
        return {
            "version": "0.1.0",
            "model": self.model,
            "work_dir": self.executor.work_dir,
            "features": {
                "vertical_topology": True,
                "horizontal_topology": True,
                "ai_tool_creation": True,
                "hierarchical_memory": True,
                "container_execution": self.executor.use_docker,
            },
        }
