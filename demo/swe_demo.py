#!/usr/bin/env python3
"""
OpenSage SWE Demo: Software Engineering Task Demonstration

This demo showcases OpenSage solving a real software engineering task:
- A Python module with bugs that need to be identified and fixed
- The agent uses vertical topology to decompose: analyze → debug → fix → verify
- Dynamic sub-agents are created for each specialized role
- The hierarchical memory tracks findings across agents
- New analysis tools are AI-generated on the fly

Inspired by SWE-Bench Pro benchmark evaluation in the paper.

Usage:
    python demo/swe_demo.py

Set ANTHROPIC_API_KEY environment variable before running.
"""

import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensage import OpenSage


# ------------------------------------------------------------------ #
# Buggy module for the agent to fix                                    #
# ------------------------------------------------------------------ #

BUGGY_MODULE = '''"""
data_processor.py - Utility for data analysis operations.
Contains several bugs that need to be found and fixed.
"""

def compute_statistics(data):
    """Compute mean, median, and standard deviation of a list of numbers."""
    if not data:
        return {}

    n = len(data)
    mean = sum(data) / n

    # BUG 1: Incorrect median calculation for even-length lists
    sorted_data = sorted(data)
    if n % 2 == 0:
        median = sorted_data[n // 2]  # Wrong: should average two middle elements
    else:
        median = sorted_data[n // 2]

    # BUG 2: Standard deviation formula missing square root
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance  # Wrong: should be variance ** 0.5

    return {"mean": mean, "median": median, "std_dev": std_dev}


def normalize_data(data, method="minmax"):
    """Normalize a list of numbers using min-max or z-score normalization."""
    if not data:
        return []

    if method == "minmax":
        min_val = min(data)
        max_val = max(data)
        # BUG 3: Division by zero when all values are equal
        return [(x - min_val) / (max_val - min_val) for x in data]

    elif method == "zscore":
        stats = compute_statistics(data)
        mean = stats["mean"]
        std = stats["std_dev"]  # Already buggy from compute_statistics
        # BUG 4: Division by zero when std_dev is 0
        return [(x - mean) / std for x in data]

    return data


def find_outliers(data, threshold=2.0):
    """Find outliers using z-score method."""
    if len(data) < 3:
        return []

    stats = compute_statistics(data)
    mean = stats["mean"]
    std = stats["std_dev"]

    # BUG 5: Uses raw variance (buggy std_dev) for comparison
    outliers = []
    for i, x in enumerate(data):
        z_score = abs(x - mean) / std if std != 0 else 0
        if z_score > threshold:
            outliers.append((i, x, z_score))

    return outliers


def batch_process(datasets, operation="statistics"):
    """Process multiple datasets with error handling."""
    results = {}

    for name, dataset in datasets.items():
        try:
            if operation == "statistics":
                results[name] = compute_statistics(dataset)
            elif operation == "normalize":
                results[name] = normalize_data(dataset)
            elif operation == "outliers":
                results[name] = find_outliers(dataset)
        except Exception as e:
            # BUG 6: Swallows all exceptions, should at minimum log them
            pass

    return results
'''

EXPECTED_FIXES = """
Expected fixes:
1. Median: average two middle elements for even-length lists
2. Std dev: take square root of variance
3. Min-max normalize: handle case where max == min (return zeros or ones)
4. Z-score normalize: handle zero std_dev gracefully
5. Outliers: fixed automatically by fixing std_dev
6. batch_process: log exceptions, don't silently swallow them
"""

TEST_MODULE = '''"""Tests for data_processor.py"""
import math

def test_compute_statistics():
    # Even-length list - median should be average of two middle elements
    data = [1, 2, 3, 4]
    stats = compute_statistics(data)
    assert stats["mean"] == 2.5, f"Mean wrong: {stats['mean']}"
    assert stats["median"] == 2.5, f"Median should be 2.5, got: {stats['median']}"
    assert abs(stats["std_dev"] - math.sqrt(1.25)) < 0.001, f"Std dev wrong: {stats['std_dev']}"

    # Odd-length list
    data = [1, 2, 3]
    stats = compute_statistics(data)
    assert stats["median"] == 2, f"Median should be 2, got: {stats['median']}"

    # Empty list
    assert compute_statistics([]) == {}

    print("✓ compute_statistics tests passed")


def test_normalize_data():
    # Normal case
    data = [0, 1, 2, 3, 4]
    normalized = normalize_data(data)
    assert normalized == [0.0, 0.25, 0.5, 0.75, 1.0], f"Normalization wrong: {normalized}"

    # All same values - should not crash
    data_uniform = [5, 5, 5, 5]
    normalized_uniform = normalize_data(data_uniform)
    assert all(v == 0.0 for v in normalized_uniform), f"Uniform normalization wrong: {normalized_uniform}"

    print("✓ normalize_data tests passed")


def test_find_outliers():
    data = [10, 11, 10.5, 100, 9.5]  # 100 is the outlier
    outliers = find_outliers(data, threshold=2.0)
    outlier_values = [x for _, x, _ in outliers]
    assert 100 in outlier_values, f"Should detect 100 as outlier, got: {outliers}"
    print("✓ find_outliers tests passed")


def test_batch_process():
    datasets = {"a": [1, 2, 3], "b": []}
    results = batch_process(datasets, "statistics")
    assert "a" in results
    assert "b" in results  # Should handle empty gracefully
    print("✓ batch_process tests passed")


# Run all tests
test_compute_statistics()
test_normalize_data()
test_find_outliers()
test_batch_process()
print("\\n✓ All tests passed!")
'''


def run_swe_demo():
    """Run the SWE demonstration."""
    print("=" * 70)
    print("  OpenSage: Self-programming Agent Generation Engine")
    print("  Demo: Software Engineering Bug Fix (SWE-Bench style)")
    print("=" * 70)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n⚠  ANTHROPIC_API_KEY not set. Set it to run with real AI.")
        print("   export ANTHROPIC_API_KEY=your_key_here\n")
        _run_mock_demo()
        return

    # Initialize OpenSage
    engine = OpenSage(
        api_key=api_key,
        model="claude-opus-4-5",
        verbose=True,
    )

    # Write the buggy module to the work directory
    engine.executor.write_file("data_processor.py", BUGGY_MODULE)
    engine.executor.write_file("test_data_processor.py", TEST_MODULE)

    print(f"\n[Demo] Wrote buggy module to: {engine.executor.work_dir}/data_processor.py")
    print("[Demo] Known bugs in the module:")
    print(EXPECTED_FIXES)

    # The task for OpenSage
    task = f"""
You are working in the directory: {engine.executor.work_dir}

The file `data_processor.py` has multiple bugs. Your task:
1. Read and analyze `data_processor.py` to identify all bugs
2. Create a specialized debugging sub-agent to trace each bug
3. Fix all bugs in the file
4. Run the tests in `test_data_processor.py` to verify fixes
5. Report a summary of all bugs found and fixes applied

The module has issues with: statistics computation, normalization edge cases,
and error handling. Find and fix ALL bugs.
"""

    print("\n[OpenSage] Solving SWE task with vertical topology...")
    print("-" * 70)

    result = engine.solve(task, topology="auto")

    print("\n" + "=" * 70)
    print("FINAL RESULT:")
    print("=" * 70)
    print(result)

    # Show what the agent created
    print("\n" + "=" * 70)
    print("OpenSage Execution Trace:")
    print("=" * 70)
    print(f"Work dir: {engine.executor.work_dir}")

    # Read the fixed file
    fixed_content = engine.executor.read_file("data_processor.py")
    if fixed_content.success:
        print("\nFixed data_processor.py:")
        print("-" * 40)
        print(fixed_content.output[:2000])


def _run_mock_demo():
    """Run a local demonstration without API calls showing the system structure."""
    print("\n[MOCK MODE] Demonstrating OpenSage architecture without API calls")
    print("="*60)

    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from opensage.memory.hierarchical import HierarchicalMemory
    from opensage.memory.graph import MemoryType
    from opensage.tools.executor import ExecutionEnvironment
    from opensage.tools.manager import ToolManager
    from opensage.tools.se_toolkit import get_se_toolkit

    # Demo the memory system
    print("\n1. Hierarchical Graph-Based Memory System")
    print("-" * 40)
    memory = HierarchicalMemory(agent_id="demo_agent")
    task_id = memory.start_task("Fix bugs in data_processor.py")
    mem_id_1 = memory.plan("Plan: Analyze → Debug → Fix → Test", task_id=task_id)
    mem_id_2 = memory.observe("Found 6 bugs in the module", task_id=task_id)
    mem_id_3 = memory.record_fact("Median calculation incorrect for even-length lists")
    mem_id_4 = memory.record_code(
        "median = (sorted_data[n//2-1] + sorted_data[n//2]) / 2",
        description="Fixed median calculation"
    )

    summary = memory.get_graph_summary()
    print(f"Memory graph: {summary['total_nodes']} nodes, {summary['total_edges']} edges")
    print(f"Node types: {summary['nodes_by_type']}")

    # Demo memory retrieval
    retrieved = memory.retrieve("median calculation")
    print(f"Retrieved {len(retrieved)} memories for 'median calculation'")
    for node in retrieved:
        print(f"  [{node.memory_type.value}] {node.content[:80]}")

    # Demo the tool system
    print("\n2. Tool System with SE Toolkit")
    print("-" * 40)
    executor = ExecutionEnvironment()
    tool_manager = ToolManager(executor=executor)
    se_tools = get_se_toolkit(executor)
    tool_manager.register_many(se_tools)

    print(f"Available tools: {tool_manager.list_tool_names()}")

    # Actually run a tool
    result = tool_manager.execute("run_python", {
        "code": """
import math
data = [1, 2, 3, 4]
n = len(data)
sorted_data = sorted(data)

# Buggy version
buggy_median = sorted_data[n // 2]
# Fixed version
fixed_median = (sorted_data[n//2-1] + sorted_data[n//2]) / 2

print(f"Buggy median: {buggy_median}")
print(f"Fixed median: {fixed_median}")
print(f"Expected: 2.5")
"""
    })
    print(f"\nTool execution result:\n{result.output}")

    # Demo AI-generated tool creation
    print("\n3. AI-Generated Tool Creation (Self-Programming)")
    print("-" * 40)
    new_tool = tool_manager.create_tool_from_code(
        name="compute_fixed_stats",
        description="Compute correct statistics for a dataset",
        parameters={
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["data"]
        },
        source_code="""
import math

def compute_fixed_stats(data):
    if not data:
        return {}
    n = len(data)
    mean = sum(data) / n
    sorted_data = sorted(data)
    if n % 2 == 0:
        median = (sorted_data[n//2-1] + sorted_data[n//2]) / 2
    else:
        median = sorted_data[n//2]
    variance = sum((x - mean)**2 for x in data) / n
    std_dev = math.sqrt(variance)
    return {"mean": mean, "median": median, "std_dev": std_dev}
"""
    )

    if new_tool:
        result = new_tool.execute(data=[1, 2, 3, 4])
        print(f"AI-generated tool result for [1,2,3,4]:")
        print(f"  {result.output}")

    # Demo topology concepts
    print("\n4. Agent Topology System")
    print("-" * 40)
    print("Vertical Topology (sequential decomposition):")
    print("  Parent Task: Fix bugs in data_processor.py")
    print("  ├── Sub-agent: code_analyzer (tools: read_file, search_code, analyze_code)")
    print("  ├── Sub-agent: debugger (tools: run_python, run_tests)")
    print("  ├── Sub-agent: code_fixer (tools: read_file, write_file)")
    print("  └── Sub-agent: verifier (tools: run_tests, run_python)")

    print("\nHorizontal Topology (parallel ensemble):")
    print("  Same Task → Agent_Systematic + Agent_Heuristic")
    print("  Both run concurrently with different strategies")
    print("  Integrator synthesizes best solution from both")

    print("\n5. System Architecture Summary")
    print("-" * 40)
    print("""
OpenSage Components:
  ┌─────────────────────────────────────────────┐
  │              OpenSage Engine                │
  │  ┌─────────────┐  ┌─────────────────────┐  │
  │  │  SageAgent  │  │   TopologyManager   │  │
  │  │  (main)     │──│  Vertical/Horizontal│  │
  │  └─────────────┘  └─────────────────────┘  │
  │         │                                   │
  │  ┌──────┴──────────────────────────────┐   │
  │  │         Core Components             │   │
  │  │  ┌──────────┐  ┌────────────────┐  │   │
  │  │  │  Memory  │  │  ToolManager   │  │   │
  │  │  │ (Graph)  │  │  SE Toolkit    │  │   │
  │  │  │ Hierarch │  │  AI-Gen Tools  │  │   │
  │  │  └──────────┘  └────────────────┘  │   │
  │  └─────────────────────────────────────┘   │
  │                                             │
  │  Benchmarks: Terminal-Bench 2.0 | CyberGym  │
  │              | SWE-Bench Pro                │
  └─────────────────────────────────────────────┘
""")

    print("✓ Mock demo complete! Set ANTHROPIC_API_KEY to run with real AI.")
    executor.cleanup()


if __name__ == "__main__":
    run_swe_demo()
