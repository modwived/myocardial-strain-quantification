"""
Memory Agent for OpenSage.

The MemoryAgent actively manages the memory graph, performing:
1. Summarization of lengthy observation chains
2. Deduplication of redundant memories
3. Importance scoring updates based on access patterns
4. Context window optimization

From the paper: "The hierarchical memory system and dedicated memory agent
significantly optimize context length while preventing redundant and repeated queries."
"""

from typing import TYPE_CHECKING, List, Optional

from opensage.memory.graph import MemoryNode, MemoryType
from opensage.memory.hierarchical import HierarchicalMemory

if TYPE_CHECKING:
    from opensage.llm.base import LLMClient


class MemoryAgent:
    """
    A specialized agent for managing the hierarchical memory graph.

    The MemoryAgent is always present alongside a SageAgent and
    performs background memory maintenance to keep context efficient.
    """

    SUMMARIZE_THRESHOLD = 500  # Characters above which we attempt summarization

    def __init__(self, memory: HierarchicalMemory, llm_client: "LLMClient"):
        self.memory = memory
        self.llm_client = llm_client
        self._access_counts: dict = {}

    def record_access(self, node_id: str) -> None:
        """Track how often a node is accessed."""
        self._access_counts[node_id] = self._access_counts.get(node_id, 0) + 1

    def update_importance_from_access(self) -> None:
        """Boost importance of frequently accessed nodes."""
        for node_id, count in self._access_counts.items():
            node = self.memory.graph.get_node(node_id)
            if node:
                boost = min(count * 0.05, 0.3)
                new_importance = min(node.importance + boost, 1.0)
                self.memory.graph.update_node(node_id, importance=new_importance)

    def summarize_long_nodes(self) -> int:
        """
        Summarize observations that are too long for efficient context use.
        Returns the number of nodes summarized.
        """
        summarized = 0
        for node in self.memory.graph._nodes.values():
            if (
                node.memory_type == MemoryType.OBSERVATION
                and node.summary is None
                and len(node.content) > self.SUMMARIZE_THRESHOLD
            ):
                summary = self._generate_summary(node)
                if summary:
                    self.memory.summarize_node(node.id, summary)
                    summarized += 1
        return summarized

    def _generate_summary(self, node: MemoryNode) -> Optional[str]:
        """Use the LLM to compress a memory node."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following in 1-2 sentences, preserving key details:\n\n"
                        f"{node.content[:2000]}"
                    ),
                }
            ]
            response = self.llm_client.chat(
                messages=messages,
                system="You are a memory compression assistant. Be concise and preserve essential information.",
                max_tokens=150,
            )
            return response.content.strip()
        except Exception:
            return None

    def deduplicate(self) -> int:
        """
        Mark highly similar observation nodes as lower importance
        to reduce redundancy.
        Returns number of nodes deduplicated.
        """
        observations = self.memory.graph.get_all_by_type(MemoryType.OBSERVATION)
        deduped = 0

        for i, node_a in enumerate(observations):
            if node_a.importance < 0.2:
                continue
            for node_b in observations[i + 1 :]:
                if self._similarity(node_a.content, node_b.content) > 0.8:
                    # Keep the more recent / important one
                    if node_a.timestamp >= node_b.timestamp:
                        self.memory.graph.update_node(node_b.id, importance=0.1)
                    else:
                        self.memory.graph.update_node(node_a.id, importance=0.1)
                    deduped += 1
                    break

        return deduped

    def _similarity(self, a: str, b: str) -> float:
        """Simple token overlap similarity."""
        a_terms = set(a.lower().split())
        b_terms = set(b.lower().split())
        if not a_terms or not b_terms:
            return 0.0
        intersection = len(a_terms & b_terms)
        union = len(a_terms | b_terms)
        return intersection / union

    def prune(self, max_age_seconds: float = 7200) -> int:
        """Remove old, low-importance observations."""
        return self.memory.graph.prune_old_observations(max_age_seconds)

    def run_maintenance(self) -> dict:
        """
        Run a full maintenance cycle: summarize, deduplicate, update importance.
        Returns a summary of actions taken.
        """
        self.update_importance_from_access()
        summarized = self.summarize_long_nodes()
        deduped = self.deduplicate()
        pruned = self.prune()

        return {
            "nodes_summarized": summarized,
            "nodes_deduplicated": deduped,
            "nodes_pruned": pruned,
        }

    def get_focused_context(self, task_id: str, query: str, max_tokens: int = 3000) -> str:
        """
        Build a focused context for a task, incorporating:
        1. Task hierarchy context
        2. Relevant retrieved memories
        3. Recent observations

        This implements the context-length optimization from the paper.
        """
        # Task hierarchy context
        task_context = self.memory.get_task_context(task_id, max_tokens=max_tokens // 2)

        # Retrieve relevant memories
        relevant = self.memory.retrieve(query, max_results=5)
        for node in relevant:
            self.record_access(node.id)

        relevant_str = "\n".join(
            node.to_context_str(use_summary=True) for node in relevant
        )

        sections = []
        if task_context:
            sections.append(f"=== Task Context ===\n{task_context}")
        if relevant_str:
            sections.append(f"=== Relevant Memories ===\n{relevant_str}")

        return "\n\n".join(sections)
