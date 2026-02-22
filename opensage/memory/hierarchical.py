"""
Hierarchical memory system for OpenSage.

Wraps the GraphMemory with a higher-level API that implements
the hierarchical structure described in the paper:

  Task → [SubTask, SubTask, ...] → [Observation, Plan, Result, ...]

The hierarchy allows efficient context-length management and
prevents redundant querying by organizing memories into levels:
  - Level 0: Session / global context
  - Level 1: Task-level summaries
  - Level 2: Sub-task execution traces
  - Level 3: Fine-grained tool outputs / observations
"""

from typing import Any, Dict, List, Optional

from opensage.memory.graph import GraphMemory, MemoryNode, MemoryType


class HierarchicalMemory:
    """
    Hierarchical, graph-based memory system.

    Provides a structured API over GraphMemory that enforces the
    task hierarchy and exposes context-efficient retrieval methods.
    """

    def __init__(self, agent_id: str = "root"):
        self.graph = GraphMemory()
        self.agent_id = agent_id

        # Root session node acts as the top-level anchor
        self._session_node = self.graph.add_node(
            content=f"Session memory for agent: {agent_id}",
            memory_type=MemoryType.TASK,
            agent_id=agent_id,
            importance=1.0,
        )
        self._current_task_id: Optional[str] = None

    @property
    def session_id(self) -> str:
        return self._session_node.id

    # ------------------------------------------------------------------ #
    # Task management                                                      #
    # ------------------------------------------------------------------ #

    def start_task(self, task_description: str, metadata: Optional[Dict] = None) -> str:
        """
        Register the start of a new top-level task.

        Returns:
            task_id for subsequent memory operations
        """
        node = self.graph.add_node(
            content=task_description,
            memory_type=MemoryType.TASK,
            agent_id=self.agent_id,
            parent_id=self._session_node.id,
            relation="child_of",
            metadata=metadata or {},
            importance=1.0,
        )
        self._current_task_id = node.id
        return node.id

    def start_subtask(self, subtask_description: str, parent_task_id: str) -> str:
        """Register a sub-task under a parent task."""
        node = self.graph.add_node(
            content=subtask_description,
            memory_type=MemoryType.TASK,
            agent_id=self.agent_id,
            parent_id=parent_task_id,
            relation="child_of",
            importance=0.9,
        )
        return node.id

    # ------------------------------------------------------------------ #
    # Writing memories                                                     #
    # ------------------------------------------------------------------ #

    def observe(
        self,
        observation: str,
        task_id: Optional[str] = None,
        importance: float = 0.7,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Store an observation (tool output, environment state, etc.)."""
        parent_id = task_id or self._current_task_id or self._session_node.id
        node = self.graph.add_node(
            content=observation,
            memory_type=MemoryType.OBSERVATION,
            agent_id=self.agent_id,
            parent_id=parent_id,
            relation="child_of",
            importance=importance,
            metadata=metadata or {},
        )
        return node.id

    def plan(self, plan_text: str, task_id: Optional[str] = None) -> str:
        """Store a plan or strategy."""
        parent_id = task_id or self._current_task_id or self._session_node.id
        node = self.graph.add_node(
            content=plan_text,
            memory_type=MemoryType.PLAN,
            agent_id=self.agent_id,
            parent_id=parent_id,
            relation="child_of",
            importance=0.9,
        )
        return node.id

    def record_result(
        self,
        result: str,
        task_id: Optional[str] = None,
        importance: float = 1.0,
    ) -> str:
        """Store the result of a task or sub-task."""
        parent_id = task_id or self._current_task_id or self._session_node.id
        node = self.graph.add_node(
            content=result,
            memory_type=MemoryType.RESULT,
            agent_id=self.agent_id,
            parent_id=parent_id,
            relation="child_of",
            importance=importance,
        )
        return node.id

    def record_fact(self, fact: str, task_id: Optional[str] = None) -> str:
        """Store a domain knowledge fact."""
        parent_id = task_id or self._current_task_id or self._session_node.id
        node = self.graph.add_node(
            content=fact,
            memory_type=MemoryType.FACT,
            agent_id=self.agent_id,
            parent_id=parent_id,
            relation="child_of",
            importance=0.8,
        )
        return node.id

    def record_code(
        self,
        code: str,
        language: str = "python",
        description: str = "",
        task_id: Optional[str] = None,
    ) -> str:
        """Store a code snippet with language metadata."""
        parent_id = task_id or self._current_task_id or self._session_node.id
        node = self.graph.add_node(
            content=code,
            memory_type=MemoryType.CODE,
            agent_id=self.agent_id,
            parent_id=parent_id,
            relation="child_of",
            importance=0.85,
            metadata={"language": language, "description": description},
        )
        return node.id

    def record_tool_created(
        self, tool_name: str, tool_spec: Dict, task_id: Optional[str] = None
    ) -> str:
        """Record an AI-generated tool."""
        import json

        parent_id = task_id or self._current_task_id or self._session_node.id
        node = self.graph.add_node(
            content=f"Tool created: {tool_name}\n{json.dumps(tool_spec, indent=2)}",
            memory_type=MemoryType.TOOL,
            agent_id=self.agent_id,
            parent_id=parent_id,
            relation="child_of",
            importance=0.95,
            metadata={"tool_name": tool_name, "spec": tool_spec},
        )
        return node.id

    def record_agent_created(
        self, agent_name: str, agent_config: Dict, task_id: Optional[str] = None
    ) -> str:
        """Record a dynamically created sub-agent."""
        import json

        parent_id = task_id or self._current_task_id or self._session_node.id
        node = self.graph.add_node(
            content=f"Sub-agent created: {agent_name}\n{json.dumps(agent_config, indent=2)}",
            memory_type=MemoryType.AGENT,
            agent_id=self.agent_id,
            parent_id=parent_id,
            relation="child_of",
            importance=1.0,
            metadata={"agent_name": agent_name, "config": agent_config},
        )
        return node.id

    def record_error(
        self, error: str, task_id: Optional[str] = None, importance: float = 0.6
    ) -> str:
        """Record an error for future avoidance."""
        parent_id = task_id or self._current_task_id or self._session_node.id
        node = self.graph.add_node(
            content=error,
            memory_type=MemoryType.ERROR,
            agent_id=self.agent_id,
            parent_id=parent_id,
            relation="child_of",
            importance=importance,
        )
        return node.id

    # ------------------------------------------------------------------ #
    # Reading / retrieval                                                  #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, max_results: int = 8) -> List[MemoryNode]:
        """Retrieve relevant memories using keyword + importance search."""
        return self.graph.search(query, max_results=max_results)

    def get_task_context(self, task_id: str, max_tokens: int = 3000) -> str:
        """Get context-efficient string for a specific task."""
        return self.graph.get_context_window(
            focal_node_id=task_id,
            max_tokens_approx=max_tokens,
            use_summaries=True,
        )

    def get_recent_context(self, max_tokens: int = 2000) -> str:
        """Get the most recent memory context."""
        task_id = self._current_task_id or self._session_node.id
        return self.graph.get_context_window(
            focal_node_id=task_id,
            max_tokens_approx=max_tokens,
            use_summaries=True,
        )

    def summarize_node(self, node_id: str, summary: str) -> None:
        """Attach a compressed summary to a node (used by MemoryAgent)."""
        self.graph.update_node(node_id, summary=summary)

    def get_all_tools(self) -> List[MemoryNode]:
        """Get all recorded AI-generated tools."""
        return self.graph.get_all_by_type(MemoryType.TOOL)

    def get_all_agents(self) -> List[MemoryNode]:
        """Get all recorded sub-agents."""
        return self.graph.get_all_by_type(MemoryType.AGENT)

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the current memory graph state."""
        return self.graph.to_summary_dict()
