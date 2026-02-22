"""
Vertical Agent Topology for OpenSage.

From the paper: "vertical agent topology, which decomposes complex tasks
into sequential sub-tasks to be completed by specialized sub-agents."

In vertical topology:
1. A parent agent receives a complex task
2. It decomposes the task into sequential sub-tasks
3. Creates specialized sub-agents for each sub-task
4. Each sub-agent has tailored system prompts and tools
5. Results flow back up the chain to the parent
6. Sub-agent states are preserved (not discarded, unlike static ADKs)

This is in contrast to existing ADKs where:
"the static agent structure lacks flexibility as a parent agent can only
assign tasks to pre-defined sub-agents, and the sub-agents' informative
states are discarded after execution."
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from opensage.core.agent import SageAgent


@dataclass
class SubTaskSpec:
    """Specification for a sub-task in the vertical topology."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    agent_name: str = ""
    agent_system_prompt: str = ""
    required_tools: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)  # task_ids this depends on
    context_from_parent: str = ""


@dataclass
class SubTaskResult:
    """Result from executing a sub-task."""
    task_id: str
    agent_name: str
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class VerticalTopology:
    """
    Implements vertical (hierarchical decomposition) agent topology.

    The vertical topology manager:
    1. Receives a decomposed task plan (list of sub-tasks)
    2. Creates specialized agents for each sub-task on demand
    3. Executes them sequentially, passing results forward
    4. Aggregates all results for the parent agent

    This is described in the paper as allowing "a parent agent to
    create sub-agent instances at runtime."
    """

    def __init__(self, parent_agent: "SageAgent"):
        self.parent = parent_agent
        self._sub_agents: Dict[str, "SageAgent"] = {}
        self._execution_log: List[SubTaskResult] = []

    def decompose_via_llm(self, task: str) -> List[SubTaskSpec]:
        """
        Use the parent agent's LLM to decompose a task into sub-tasks.

        The LLM is prompted to produce a structured decomposition with
        sub-task descriptions and agent specializations.
        """
        decomposition_prompt = f"""You are a task decomposition expert. Break down the following task into sequential sub-tasks.

Task: {task}

Respond with a JSON array of sub-task objects. Each object must have:
- "description": detailed description of the sub-task
- "agent_name": short name for the specialized agent (e.g. "code_analyzer", "test_runner")
- "agent_role": the role/expertise this agent should have (used in its system prompt)
- "required_tools": list of tool names this agent needs (from: read_file, write_file, run_command, run_python, search_code, list_files, run_tests, analyze_code)

Keep sub-tasks focused and sequential. Maximum 5 sub-tasks.

Respond with ONLY valid JSON, no other text:
```json
[...]
```"""

        messages = [{"role": "user", "content": decomposition_prompt}]
        response = self.parent.llm.chat(
            messages=messages,
            system="You are a task planning expert. Output only valid JSON arrays.",
            max_tokens=2000,
        )

        return self._parse_decomposition(response.content, task)

    def _parse_decomposition(self, llm_response: str, original_task: str) -> List[SubTaskSpec]:
        """Parse LLM's task decomposition response into SubTaskSpec objects."""
        import json
        import re

        # Extract JSON from code block if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", llm_response)
        json_str = json_match.group(1) if json_match else llm_response.strip()

        try:
            items = json.loads(json_str)
            specs = []
            for i, item in enumerate(items):
                spec = SubTaskSpec(
                    description=item.get("description", f"Sub-task {i+1}"),
                    agent_name=item.get("agent_name", f"agent_{i+1}"),
                    agent_system_prompt=self._build_agent_system_prompt(
                        item.get("agent_role", "specialist"),
                        item.get("description", ""),
                        original_task,
                    ),
                    required_tools=item.get("required_tools", []),
                )
                specs.append(spec)
            return specs
        except (json.JSONDecodeError, KeyError):
            # Fallback: single sub-task for the whole task
            return [
                SubTaskSpec(
                    description=original_task,
                    agent_name="general_agent",
                    agent_system_prompt=self._build_agent_system_prompt(
                        "general problem solver", original_task, original_task
                    ),
                    required_tools=list(self.parent.tool_manager.list_tool_names()),
                )
            ]

    def _build_agent_system_prompt(
        self, role: str, sub_task: str, parent_task: str
    ) -> str:
        return (
            f"You are a specialized {role} agent.\n"
            f"Parent task: {parent_task}\n"
            f"Your specific responsibility: {sub_task}\n"
            f"Complete your sub-task thoroughly and report your findings and results clearly."
        )

    def execute(
        self,
        task: str,
        sub_task_specs: Optional[List[SubTaskSpec]] = None,
    ) -> List[SubTaskResult]:
        """
        Execute a task using vertical topology.

        Creates specialized sub-agents dynamically and runs them in sequence,
        passing context from previous sub-tasks to subsequent ones.

        Args:
            task: The high-level task description
            sub_task_specs: Pre-defined sub-tasks (if None, uses LLM decomposition)

        Returns:
            List of SubTaskResults from all sub-agents
        """
        if sub_task_specs is None:
            sub_task_specs = self.decompose_via_llm(task)

        results: List[SubTaskResult] = []
        accumulated_context = f"Parent task: {task}\n\n"

        for spec in sub_task_specs:
            # Create a specialized sub-agent for this sub-task
            sub_agent = self._create_sub_agent(spec, accumulated_context)
            self._sub_agents[spec.agent_name] = sub_agent

            # Log agent creation to parent's memory
            self.parent.memory.record_agent_created(
                agent_name=spec.agent_name,
                agent_config={
                    "description": spec.description,
                    "tools": spec.required_tools,
                    "topology": "vertical",
                },
                task_id=self.parent.memory._current_task_id,
            )

            # Execute the sub-task
            start = time.time()
            try:
                task_for_agent = f"{spec.description}\n\nContext from previous steps:\n{accumulated_context}"
                result_text = sub_agent.run(task_for_agent)
                elapsed = time.time() - start

                result = SubTaskResult(
                    task_id=spec.task_id,
                    agent_name=spec.agent_name,
                    success=True,
                    output=result_text,
                    execution_time=elapsed,
                )

                # Accumulate context for the next sub-agent
                accumulated_context += (
                    f"\n--- {spec.agent_name} completed ---\n{result_text}\n"
                )

            except Exception as e:
                elapsed = time.time() - start
                result = SubTaskResult(
                    task_id=spec.task_id,
                    agent_name=spec.agent_name,
                    success=False,
                    output="",
                    error=str(e),
                    execution_time=elapsed,
                )

            results.append(result)
            self._execution_log.append(result)

            # Store result in parent's memory
            self.parent.memory.observe(
                observation=f"Sub-agent {spec.agent_name}: {'SUCCESS' if result.success else 'FAILED'}\n{result.output}",
                task_id=self.parent.memory._current_task_id,
                importance=0.9,
            )

        return results

    def _create_sub_agent(
        self, spec: SubTaskSpec, context: str
    ) -> "SageAgent":
        """
        Dynamically instantiate a specialized sub-agent.

        The sub-agent inherits the parent's LLM and executor but gets
        its own memory and a filtered set of tools.
        """
        from opensage.core.agent import SageAgent
        from opensage.memory.hierarchical import HierarchicalMemory
        from opensage.tools.manager import ToolManager

        # Give the sub-agent only the tools it needs
        sub_tool_manager = ToolManager(executor=self.parent.executor)
        for tool_name in spec.required_tools:
            tool = self.parent.tool_manager.get(tool_name)
            if tool:
                sub_tool_manager.register(tool)

        # If no specific tools requested, give all tools
        if not spec.required_tools:
            for name, tool in self.parent.tool_manager.tools.items():
                sub_tool_manager.register(tool)

        sub_memory = HierarchicalMemory(agent_id=spec.agent_name)

        # Inject parent context as initial memory
        sub_memory.observe(
            f"Context from parent agent:\n{context}",
            importance=0.9,
        )

        sub_agent = SageAgent(
            name=spec.agent_name,
            system_prompt=spec.agent_system_prompt,
            llm=self.parent.llm,
            tool_manager=sub_tool_manager,
            memory=sub_memory,
            executor=self.parent.executor,
            max_iterations=8,
        )

        return sub_agent

    def get_execution_summary(self) -> str:
        """Get a summary of all executed sub-tasks."""
        lines = [f"Vertical Topology Execution ({len(self._execution_log)} sub-tasks):"]
        for result in self._execution_log:
            status = "✓" if result.success else "✗"
            lines.append(
                f"  {status} [{result.agent_name}] ({result.execution_time:.1f}s): "
                f"{result.output[:100]}..."
                if len(result.output) > 100
                else f"  {status} [{result.agent_name}] ({result.execution_time:.1f}s): {result.output}"
            )
        return "\n".join(lines)
