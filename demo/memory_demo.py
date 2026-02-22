#!/usr/bin/env python3
"""
OpenSage Memory System Demo

Demonstrates the hierarchical, graph-based memory system described in the paper:
"OpenSage features a hierarchical, graph-based memory system for efficient
management and a specialized toolkit tailored to software engineering tasks."

This demo shows:
1. Building a hierarchical memory graph
2. Task → SubTask → Observation hierarchy
3. Memory retrieval with relevance ranking
4. Context window optimization
5. Memory graph visualization
6. The MemoryAgent's maintenance capabilities

No API key required for this demo.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_memory_system():
    from opensage.memory.graph import GraphMemory, MemoryType
    from opensage.memory.hierarchical import HierarchicalMemory

    print("=" * 70)
    print("  OpenSage: Hierarchical Graph-Based Memory System Demo")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Build a memory graph for a simulated SWE task                    #
    # ------------------------------------------------------------------ #
    print("\n1. Constructing Hierarchical Memory Graph")
    print("-" * 50)

    memory = HierarchicalMemory(agent_id="SageAgent_root")

    # Session node is auto-created

    # Register a top-level task
    task_id = memory.start_task(
        "Fix and optimize the data_processor module with 6 known bugs",
        metadata={"source": "SWE-Bench Pro", "difficulty": "medium"},
    )
    print(f"✓ Task registered (id={task_id[:8]}...)")

    # Sub-tasks
    st1_id = memory.start_subtask("Analyze code for bugs", parent_task_id=task_id)
    st2_id = memory.start_subtask("Fix all identified bugs", parent_task_id=task_id)
    st3_id = memory.start_subtask("Verify fixes with tests", parent_task_id=task_id)
    print(f"✓ 3 sub-tasks registered")

    # Planning
    plan_id = memory.plan(
        "Step 1: Read data_processor.py\n"
        "Step 2: Trace execution with sample data\n"
        "Step 3: Identify each bug\n"
        "Step 4: Create targeted sub-agents for fixing\n"
        "Step 5: Run test suite to verify",
        task_id=task_id,
    )
    print(f"✓ Plan recorded")

    # Observations from code analysis sub-agent
    obs_ids = []
    observations = [
        ("BUG-1: Median calculation incorrect for even-length lists. "
         "sorted_data[n//2] returns 3 for [1,2,3,4] instead of 2.5", 0.9),
        ("BUG-2: Standard deviation returns variance, missing math.sqrt()", 0.9),
        ("BUG-3: normalize_data raises ZeroDivisionError when all values equal", 0.8),
        ("BUG-4: zscore normalization uses buggy std_dev from compute_statistics", 0.8),
        ("BUG-5: find_outliers uses raw variance instead of correct std_dev", 0.85),
        ("BUG-6: batch_process silently swallows all exceptions with bare 'pass'", 0.7),
    ]
    for content, importance in observations:
        obs_id = memory.observe(content, task_id=st1_id, importance=importance)
        obs_ids.append(obs_id)
    print(f"✓ {len(observations)} observations stored (bugs found)")

    # Facts discovered
    memory.record_fact("Python median for even-length: average of n//2-1 and n//2 elements")
    memory.record_fact("Standard deviation = sqrt(variance), not variance itself")
    memory.record_fact("ZeroDivisionError prevention: check if denominator == 0 before dividing")

    # Code snippets (fixes)
    code_fixes = [
        ("median = (sorted_data[n//2-1] + sorted_data[n//2]) / 2  # BUG-1 fix", "python", "Fixed median"),
        ("std_dev = variance ** 0.5  # BUG-2 fix: square root of variance", "python", "Fixed std_dev"),
        ("return [0.0] * len(data) if max_val == min_val else [(x-min_val)/(max_val-min_val) for x in data]", "python", "Fixed minmax"),
    ]
    for code, lang, desc in code_fixes:
        memory.record_code(code, language=lang, description=desc, task_id=st2_id)
    print(f"✓ {len(code_fixes)} code fixes stored")

    # Agent creations
    memory.record_agent_created(
        "code_analyzer",
        {"tools": ["read_file", "search_code", "analyze_code"], "topology": "vertical"},
        task_id=st1_id,
    )
    memory.record_agent_created(
        "bug_fixer",
        {"tools": ["read_file", "write_file", "run_python"], "topology": "vertical"},
        task_id=st2_id,
    )
    print(f"✓ 2 sub-agent creations recorded")

    # Tool creation
    memory.record_tool_created(
        "validate_statistics",
        {"source": "def validate_statistics(data, stats): ...", "is_ai_generated": True},
        task_id=st2_id,
    )
    print(f"✓ 1 AI-generated tool recorded")

    # Test results
    result_id = memory.record_result(
        "All 6 bugs fixed. Test suite: 4/4 passed.\n"
        "compute_statistics: ✓\n"
        "normalize_data: ✓\n"
        "find_outliers: ✓\n"
        "batch_process: ✓",
        task_id=st3_id,
        importance=1.0,
    )
    print(f"✓ Final result recorded")

    # ------------------------------------------------------------------ #
    # 2. Inspect the graph                                                #
    # ------------------------------------------------------------------ #
    print("\n2. Memory Graph Statistics")
    print("-" * 50)
    summary = memory.get_graph_summary()
    print(f"  Total nodes:  {summary['total_nodes']}")
    print(f"  Total edges:  {summary['total_edges']}")
    print(f"  Root nodes:   {summary['root_count']}")
    print(f"  Node types:")
    for type_name, count in sorted(summary['nodes_by_type'].items()):
        bar = "█" * count
        print(f"    {type_name:12s}: {bar} ({count})")

    # ------------------------------------------------------------------ #
    # 3. Demonstrate memory retrieval                                     #
    # ------------------------------------------------------------------ #
    print("\n3. Memory Retrieval (Keyword + Importance Ranking)")
    print("-" * 50)

    queries = ["median bug", "standard deviation", "normalization zero", "tests passed"]
    for query in queries:
        results = memory.retrieve(query, max_results=2)
        print(f"\n  Query: '{query}'")
        for node in results:
            print(f"    [{node.memory_type.value}] (importance={node.importance:.1f}): {node.content[:80]}")

    # ------------------------------------------------------------------ #
    # 4. Context window optimization                                      #
    # ------------------------------------------------------------------ #
    print("\n4. Context Window Optimization")
    print("-" * 50)

    full_context = memory.get_task_context(task_id, max_tokens=2000)
    print(f"Context for main task (token budget=2000):")
    print(f"  Characters used: {len(full_context)}")
    print(f"  Lines: {len(full_context.splitlines())}")
    print("\nContext preview (first 500 chars):")
    print(full_context[:500])

    # ------------------------------------------------------------------ #
    # 5. Graph traversal                                                  #
    # ------------------------------------------------------------------ #
    print("\n5. Graph Traversal Capabilities")
    print("-" * 50)

    # Get all children of task
    children = memory.graph.get_children(task_id)
    print(f"Direct children of main task: {len(children)}")
    for child in children:
        print(f"  [{child.memory_type.value}] {child.content[:60]}")

    # Get subgraph (all descendants)
    subgraph = memory.graph.get_subgraph(task_id)
    print(f"\nFull subgraph of main task: {len(subgraph)} nodes")

    # Get ancestors of a result node
    ancestors = memory.graph.get_ancestors(result_id, max_depth=5)
    print(f"\nAncestors of result node: {len(ancestors)}")
    for anc in ancestors:
        print(f"  [{anc.memory_type.value}] {anc.content[:60]}")

    # ------------------------------------------------------------------ #
    # 6. Memory export                                                    #
    # ------------------------------------------------------------------ #
    print("\n6. Memory Graph Export (JSON)")
    print("-" * 50)
    json_export = memory.graph.export_json()
    import json
    parsed = json.loads(json_export)
    print(f"Exported graph: {len(parsed['nodes'])} nodes, {len(parsed['edges'])} edges")
    print(f"Sample node: {json.dumps(parsed['nodes'][0], indent=2)[:300]}")

    print("\n" + "=" * 70)
    print("✓ Memory system demo complete!")
    print("=" * 70)


def demo_memory_graph_visualization():
    """ASCII visualization of the memory graph hierarchy."""
    print("\n" + "=" * 70)
    print("  Memory Graph Hierarchy Visualization")
    print("=" * 70)

    tree = """
Session: SageAgent_root
└── [TASK] Fix and optimize data_processor (6 bugs)
    ├── [PLAN] Step 1→2→3→4→5 execution plan
    ├── [TASK] Sub-task: Analyze code for bugs
    │   ├── [OBSERVATION] BUG-1: Median calculation wrong (importance=0.9)
    │   ├── [OBSERVATION] BUG-2: std_dev missing sqrt (importance=0.9)
    │   ├── [OBSERVATION] BUG-3: ZeroDivisionError in normalize (importance=0.8)
    │   ├── [OBSERVATION] BUG-4: zscore uses buggy std_dev (importance=0.8)
    │   ├── [OBSERVATION] BUG-5: find_outliers wrong std_dev (importance=0.85)
    │   ├── [OBSERVATION] BUG-6: silent exception swallow (importance=0.7)
    │   └── [AGENT] code_analyzer created (tools: read_file, search_code)
    ├── [TASK] Sub-task: Fix all identified bugs
    │   ├── [CODE] Fixed median calculation
    │   ├── [CODE] Fixed std_dev (sqrt)
    │   ├── [CODE] Fixed minmax edge case
    │   ├── [TOOL] AI-created: validate_statistics
    │   └── [AGENT] bug_fixer created (tools: write_file, run_python)
    ├── [TASK] Sub-task: Verify fixes with tests
    │   └── [RESULT] All 6 bugs fixed. Tests: 4/4 passed. (importance=1.0)
    └── [FACT] Python median = avg(n//2-1, n//2) for even lists
        [FACT] std_dev = sqrt(variance)
        [FACT] Check denominator before dividing

Graph Statistics:
  Nodes: 20 | Edges: 19 | Depth: 3
  Node types: task(4) observation(6) plan(1) code(3) tool(1) agent(2) fact(3) result(1)
"""
    print(tree)


if __name__ == "__main__":
    demo_memory_system()
    demo_memory_graph_visualization()
