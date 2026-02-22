"""
Tests for OpenSage hierarchical graph-based memory system.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensage.memory.graph import GraphMemory, MemoryNode, MemoryType
from opensage.memory.hierarchical import HierarchicalMemory


class TestGraphMemory:
    """Test the core graph memory."""

    def test_add_node(self):
        mem = GraphMemory()
        node = mem.add_node("test content", MemoryType.OBSERVATION)
        assert node.content == "test content"
        assert node.memory_type == MemoryType.OBSERVATION
        assert node.id in mem._nodes

    def test_parent_child_relationship(self):
        mem = GraphMemory()
        parent = mem.add_node("parent task", MemoryType.TASK)
        child = mem.add_node("child obs", MemoryType.OBSERVATION, parent_id=parent.id)
        assert mem.graph.has_edge(parent.id, child.id)

    def test_get_children(self):
        mem = GraphMemory()
        parent = mem.add_node("parent", MemoryType.TASK)
        child1 = mem.add_node("child1", MemoryType.OBSERVATION, parent_id=parent.id)
        child2 = mem.add_node("child2", MemoryType.OBSERVATION, parent_id=parent.id)
        children = mem.get_children(parent.id)
        assert len(children) == 2

    def test_get_ancestors(self):
        mem = GraphMemory()
        root = mem.add_node("root", MemoryType.TASK)
        child = mem.add_node("child", MemoryType.TASK, parent_id=root.id)
        grandchild = mem.add_node("grandchild", MemoryType.OBSERVATION, parent_id=child.id)
        ancestors = mem.get_ancestors(grandchild.id, max_depth=3)
        ancestor_ids = [a.id for a in ancestors]
        assert child.id in ancestor_ids or root.id in ancestor_ids

    def test_search_basic(self):
        mem = GraphMemory()
        mem.add_node("python bug median calculation error", MemoryType.OBSERVATION, importance=0.9)
        mem.add_node("unrelated content xyz", MemoryType.OBSERVATION, importance=0.5)
        results = mem.search("median calculation")
        assert len(results) >= 1
        assert results[0].content.startswith("python bug")

    def test_search_with_type_filter(self):
        mem = GraphMemory()
        mem.add_node("observation node", MemoryType.OBSERVATION)
        mem.add_node("fact node with same terms", MemoryType.FACT)
        results = mem.search("node", memory_types=[MemoryType.OBSERVATION])
        for r in results:
            assert r.memory_type == MemoryType.OBSERVATION

    def test_get_subgraph(self):
        mem = GraphMemory()
        root = mem.add_node("root task", MemoryType.TASK)
        child1 = mem.add_node("subtask 1", MemoryType.TASK, parent_id=root.id)
        child2 = mem.add_node("subtask 2", MemoryType.TASK, parent_id=root.id)
        grandchild = mem.add_node("obs", MemoryType.OBSERVATION, parent_id=child1.id)
        subgraph = mem.get_subgraph(root.id)
        subgraph_ids = {n.id for n in subgraph}
        assert root.id in subgraph_ids
        assert child1.id in subgraph_ids
        assert child2.id in subgraph_ids
        assert grandchild.id in subgraph_ids

    def test_get_root_nodes(self):
        mem = GraphMemory()
        root1 = mem.add_node("root1", MemoryType.TASK)
        root2 = mem.add_node("root2", MemoryType.TASK)
        child = mem.add_node("child", MemoryType.OBSERVATION, parent_id=root1.id)
        roots = mem.get_root_nodes()
        root_ids = {r.id for r in roots}
        assert root1.id in root_ids
        assert root2.id in root_ids
        assert child.id not in root_ids

    def test_context_window(self):
        mem = GraphMemory()
        root = mem.add_node("root task description", MemoryType.TASK)
        for i in range(5):
            mem.add_node(f"observation {i}", MemoryType.OBSERVATION, parent_id=root.id)
        ctx = mem.get_context_window(root.id, max_tokens_approx=1000)
        assert len(ctx) > 0
        assert len(ctx) <= 4000  # 1000 tokens * 4 chars

    def test_to_summary_dict(self):
        mem = GraphMemory()
        mem.add_node("task", MemoryType.TASK)
        mem.add_node("obs", MemoryType.OBSERVATION)
        summary = mem.to_summary_dict()
        assert summary["total_nodes"] == 2
        assert "task" in summary["nodes_by_type"]
        assert "observation" in summary["nodes_by_type"]

    def test_prune_old_observations(self):
        import time
        mem = GraphMemory()
        node = mem.add_node("old obs", MemoryType.OBSERVATION, importance=0.1)
        # Manually backdate the timestamp
        mem._nodes[node.id].timestamp = time.time() - 8000
        pruned = mem.prune_old_observations(max_age_seconds=7200)
        assert pruned == 1
        assert node.id not in mem._nodes

    def test_export_json(self):
        import json
        mem = GraphMemory()
        parent = mem.add_node("parent", MemoryType.TASK)
        child = mem.add_node("child", MemoryType.OBSERVATION, parent_id=parent.id)
        exported = json.loads(mem.export_json())
        assert len(exported["nodes"]) == 2
        assert len(exported["edges"]) == 1


class TestHierarchicalMemory:
    """Test the hierarchical memory API."""

    def test_start_task(self):
        mem = HierarchicalMemory(agent_id="test")
        task_id = mem.start_task("Fix the bug")
        assert task_id is not None
        assert mem._current_task_id == task_id

    def test_observe(self):
        mem = HierarchicalMemory(agent_id="test")
        task_id = mem.start_task("test task")
        obs_id = mem.observe("found a bug", task_id=task_id)
        assert obs_id is not None
        node = mem.graph.get_node(obs_id)
        assert node.content == "found a bug"

    def test_record_result(self):
        mem = HierarchicalMemory(agent_id="test")
        task_id = mem.start_task("test task")
        result_id = mem.record_result("task complete", task_id=task_id)
        node = mem.graph.get_node(result_id)
        assert node.memory_type == MemoryType.RESULT

    def test_retrieve(self):
        mem = HierarchicalMemory(agent_id="test")
        mem.observe("python bug in median computation")
        mem.observe("unrelated content about databases")
        results = mem.retrieve("median computation")
        assert len(results) >= 1

    def test_get_all_tools(self):
        mem = HierarchicalMemory(agent_id="test")
        mem.record_tool_created("my_tool", {"description": "a tool"})
        tools = mem.get_all_tools()
        assert len(tools) == 1

    def test_get_all_agents(self):
        mem = HierarchicalMemory(agent_id="test")
        mem.record_agent_created("my_agent", {"tools": ["read_file"]})
        agents = mem.get_all_agents()
        assert len(agents) == 1

    def test_start_subtask(self):
        mem = HierarchicalMemory(agent_id="test")
        task_id = mem.start_task("main task")
        subtask_id = mem.start_subtask("sub task", parent_task_id=task_id)
        assert subtask_id != task_id
        # Subtask should be a child of main task
        children = mem.graph.get_children(task_id)
        child_ids = {c.id for c in children}
        assert subtask_id in child_ids

    def test_graph_summary(self):
        mem = HierarchicalMemory(agent_id="test")
        task_id = mem.start_task("task")
        mem.observe("obs1")
        mem.plan("plan")
        summary = mem.get_graph_summary()
        assert summary["total_nodes"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
