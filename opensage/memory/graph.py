"""
Graph-based memory system for OpenSage.

Implements the hierarchical, graph-based memory described in the OpenSage paper.
Memory is organized as a directed graph where:
- Nodes represent memory entries (facts, observations, results, plans)
- Edges represent relationships (parent-child, causal, temporal)
- The hierarchy allows efficient context-length management
- Graph traversal enables relevant memory retrieval
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Set

import networkx as nx


class MemoryType(str, Enum):
    """Categories of memory in the OpenSage hierarchy."""
    TASK = "task"           # High-level task definitions
    OBSERVATION = "observation"  # Agent observations / tool outputs
    PLAN = "plan"           # Agent plans and strategies
    RESULT = "result"       # Final or intermediate results
    TOOL = "tool"           # Created tools and their specs
    AGENT = "agent"         # Created sub-agents and their configs
    FACT = "fact"           # Domain knowledge / facts discovered
    ERROR = "error"         # Errors encountered
    CODE = "code"           # Code snippets produced or analyzed


@dataclass
class MemoryNode:
    """
    A node in the OpenSage memory graph.

    Each node stores a piece of information with metadata
    for efficient retrieval and context management.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.OBSERVATION
    agent_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None  # Compressed summary for context efficiency
    importance: float = 1.0  # 0-1 relevance score for retrieval prioritization

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "summary": self.summary,
            "importance": self.importance,
        }

    def to_context_str(self, use_summary: bool = False) -> str:
        """Render the node as a context string for LLM consumption."""
        text = self.summary if (use_summary and self.summary) else self.content
        return f"[{self.memory_type.value.upper()}] {text}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        node = cls()
        node.id = data["id"]
        node.content = data["content"]
        node.memory_type = MemoryType(data["memory_type"])
        node.agent_id = data.get("agent_id", "")
        node.timestamp = data.get("timestamp", time.time())
        node.metadata = data.get("metadata", {})
        node.summary = data.get("summary")
        node.importance = data.get("importance", 1.0)
        return node


class GraphMemory:
    """
    Directed graph memory store as described in OpenSage.

    The graph structure enables:
    1. Hierarchical organization (task → subtask → observation)
    2. Causal chains (observation → plan → result)
    3. Cross-agent memory sharing (sub-agents can reference parent memories)
    4. Efficient pruning for context length management

    Node relationships (edges) represent:
    - "child_of": Parent-child task hierarchy
    - "caused_by": Causal / temporal sequence
    - "used_by": Tool or agent usage
    - "informs": Knowledge flow between memories
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, MemoryNode] = {}

    def add_node(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.OBSERVATION,
        agent_id: str = "",
        parent_id: Optional[str] = None,
        relation: str = "child_of",
        metadata: Optional[Dict] = None,
        importance: float = 1.0,
    ) -> MemoryNode:
        """
        Add a new memory node to the graph.

        Args:
            content: The memory content
            memory_type: Category of this memory
            agent_id: Which agent created this memory
            parent_id: ID of the parent node (creates an edge)
            relation: Edge type between parent and this node
            metadata: Additional metadata
            importance: Relevance score (0-1)

        Returns:
            The created MemoryNode
        """
        node = MemoryNode(
            content=content,
            memory_type=memory_type,
            agent_id=agent_id,
            metadata=metadata or {},
            importance=importance,
        )

        self.graph.add_node(node.id, data=node)
        self._nodes[node.id] = node

        if parent_id and parent_id in self._nodes:
            self.graph.add_edge(parent_id, node.id, relation=relation)

        return node

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a memory node by ID."""
        return self._nodes.get(node_id)

    def update_node(self, node_id: str, **kwargs) -> bool:
        """Update fields of an existing memory node."""
        if node_id not in self._nodes:
            return False
        node = self._nodes[node_id]
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
        return True

    def link_nodes(self, from_id: str, to_id: str, relation: str = "informs") -> bool:
        """Create a directed edge between two existing nodes."""
        if from_id not in self._nodes or to_id not in self._nodes:
            return False
        self.graph.add_edge(from_id, to_id, relation=relation)
        return True

    def get_children(self, node_id: str) -> List[MemoryNode]:
        """Get all direct children of a node."""
        children = []
        for successor in self.graph.successors(node_id):
            node = self._nodes.get(successor)
            if node:
                children.append(node)
        return children

    def get_ancestors(self, node_id: str, max_depth: int = 5) -> List[MemoryNode]:
        """Walk up the graph to collect ancestor context."""
        ancestors = []
        visited: Set[str] = set()
        queue = [(node_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth or current_id in visited:
                continue
            visited.add(current_id)

            for pred_id in self.graph.predecessors(current_id):
                node = self._nodes.get(pred_id)
                if node and pred_id not in visited:
                    ancestors.append(node)
                    queue.append((pred_id, depth + 1))

        return ancestors

    def get_subgraph(self, root_id: str) -> List[MemoryNode]:
        """
        Get all descendants of a root node (BFS traversal).
        Used to retrieve all memories related to a task.
        """
        if root_id not in self._nodes:
            return []

        result = []
        visited: Set[str] = {root_id}
        queue = [root_id]

        while queue:
            current = queue.pop(0)
            node = self._nodes.get(current)
            if node:
                result.append(node)
            for successor in self.graph.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)

        return result

    def search(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        agent_id: Optional[str] = None,
        max_results: int = 10,
        min_importance: float = 0.0,
    ) -> List[MemoryNode]:
        """
        Text-based memory search using keyword matching and importance scoring.

        In a production system this would use vector embeddings for semantic
        similarity. Here we use keyword matching + importance weighting.

        Args:
            query: Search query string
            memory_types: Filter by memory type
            agent_id: Filter by agent
            max_results: Maximum results to return
            min_importance: Minimum importance threshold

        Returns:
            Ranked list of matching MemoryNodes
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored: List[tuple[float, MemoryNode]] = []

        for node in self._nodes.values():
            # Apply filters
            if memory_types and node.memory_type not in memory_types:
                continue
            if agent_id and node.agent_id != agent_id:
                continue
            if node.importance < min_importance:
                continue

            # Score: keyword overlap + importance
            content_lower = node.content.lower()
            content_terms = set(content_lower.split())
            overlap = len(query_terms & content_terms)
            if overlap == 0 and query_lower not in content_lower:
                continue

            # TF-like score + importance weighting
            score = (overlap / max(len(query_terms), 1)) * node.importance
            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:max_results]]

    def get_context_window(
        self,
        focal_node_id: str,
        max_tokens_approx: int = 4000,
        use_summaries: bool = True,
    ) -> str:
        """
        Build a context-efficient memory summary around a focal node.

        Collects ancestors + siblings + immediate children, then formats
        them as a context string for the LLM. Uses summaries when available
        to maximize the information density within the token budget.

        This implements the paper's claim that the hierarchical memory
        "significantly optimize[s] context length while preventing redundant
        and repeated queries."
        """
        nodes = []

        # Ancestors give high-level context
        ancestors = self.get_ancestors(focal_node_id, max_depth=3)
        nodes.extend(ancestors)

        # Focal node itself
        focal = self._nodes.get(focal_node_id)
        if focal:
            nodes.append(focal)

        # Immediate children give detail
        children = self.get_children(focal_node_id)
        nodes.extend(children[:10])  # cap to avoid overflow

        # Deduplicate preserving order
        seen: Set[str] = set()
        unique_nodes: List[MemoryNode] = []
        for n in nodes:
            if n.id not in seen:
                seen.add(n.id)
                unique_nodes.append(n)

        # Sort by timestamp ascending
        unique_nodes.sort(key=lambda x: x.timestamp)

        # Build context string respecting token budget (rough 4-char-per-token)
        char_budget = max_tokens_approx * 4
        parts = []
        used_chars = 0

        for node in unique_nodes:
            text = node.to_context_str(use_summary=use_summaries)
            if used_chars + len(text) <= char_budget:
                parts.append(text)
                used_chars += len(text)
            else:
                # Try summary if full content too long
                summary_text = node.to_context_str(use_summary=True)
                if used_chars + len(summary_text) <= char_budget:
                    parts.append(summary_text)
                    used_chars += len(summary_text)

        return "\n".join(parts)

    def prune_old_observations(self, max_age_seconds: float = 3600) -> int:
        """
        Remove low-importance, old observation nodes to manage graph size.
        Returns number of nodes pruned.
        """
        now = time.time()
        to_remove = []

        for node_id, node in self._nodes.items():
            age = now - node.timestamp
            if (
                node.memory_type == MemoryType.OBSERVATION
                and age > max_age_seconds
                and node.importance < 0.3
            ):
                to_remove.append(node_id)

        for node_id in to_remove:
            self.graph.remove_node(node_id)
            del self._nodes[node_id]

        return len(to_remove)

    def get_all_by_type(self, memory_type: MemoryType) -> List[MemoryNode]:
        """Get all nodes of a specific type."""
        return [n for n in self._nodes.values() if n.memory_type == memory_type]

    def get_root_nodes(self) -> List[MemoryNode]:
        """Get nodes with no parents (entry points of the graph)."""
        roots = []
        for node_id in self.graph.nodes():
            if self.graph.in_degree(node_id) == 0:
                node = self._nodes.get(node_id)
                if node:
                    roots.append(node)
        return roots

    def to_summary_dict(self) -> Dict[str, Any]:
        """Summarize the current state of the memory graph."""
        type_counts: Dict[str, int] = {}
        for node in self._nodes.values():
            key = node.memory_type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": type_counts,
            "root_count": len(self.get_root_nodes()),
        }

    def export_json(self) -> str:
        """Export the full memory graph to JSON."""
        nodes = [n.to_dict() for n in self._nodes.values()]
        edges = [
            {"from": u, "to": v, "relation": d.get("relation", "")}
            for u, v, d in self.graph.edges(data=True)
        ]
        return json.dumps({"nodes": nodes, "edges": edges}, indent=2)
