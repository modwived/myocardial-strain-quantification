#!/usr/bin/env python3
"""
OpenSage Tool Creation Demo

Demonstrates the AI-driven dynamic tool creation capability:
"OpenSage empowers AI to construct its own tools for targeting tasks
and provides tool management, including overall tool orchestration,
execution isolation, and state management."

This demo shows:
1. Pre-built SE toolkit tools
2. AI generating new tools at runtime from Python code
3. Container-based execution isolation
4. Tool caching and state management

No API key required for the tool creation demos.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_tool_creation():
    from opensage.tools.executor import ExecutionEnvironment
    from opensage.tools.manager import ToolManager
    from opensage.tools.se_toolkit import get_se_toolkit

    print("=" * 70)
    print("  OpenSage: Dynamic Tool Creation Demo")
    print("=" * 70)

    executor = ExecutionEnvironment()
    tool_manager = ToolManager(executor=executor)

    # ------------------------------------------------------------------ #
    # 1. Pre-built SE Toolkit                                             #
    # ------------------------------------------------------------------ #
    print("\n1. Pre-built SE Toolkit")
    print("-" * 50)

    se_tools = get_se_toolkit(executor)
    tool_manager.register_many(se_tools)
    print(f"Registered {len(se_tools)} SE tools: {tool_manager.list_tool_names()}")

    # Use the run_python tool
    result = tool_manager.execute("run_python", {
        "code": """
import sys
print(f"Python {sys.version}")
print("OpenSage tool execution works!")
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
print(f"Sorted: {sorted(data)}")
"""
    })
    print(f"\nrun_python output:\n{result.output}")

    # Use the analyze_code tool
    code_to_analyze = """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib
"""
    result = tool_manager.execute("analyze_code", {"code": code_to_analyze})
    print(f"\nanalyze_code output:\n{result.output}")

    # ------------------------------------------------------------------ #
    # 2. AI-Generated Tool Creation                                       #
    # ------------------------------------------------------------------ #
    print("\n2. AI-Generated Tool Creation (Self-Programming)")
    print("-" * 50)
    print("The agent writes new Python tools at runtime for task-specific needs.")
    print()

    # Simulate what an AI agent would do: write a task-specific tool

    # Tool 1: Vulnerability analyzer
    vuln_tool = tool_manager.create_tool_from_code(
        name="detect_sql_injection",
        description="Detect potential SQL injection vulnerabilities in Python code",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to analyze"},
            },
            "required": ["code"],
        },
        source_code="""
import re

def detect_sql_injection(code):
    '''AI-generated tool to detect SQL injection patterns.'''
    patterns = [
        (r'["\']\\s*\\+\\s*\\w+\\s*\\+\\s*["\'].*(?:SELECT|INSERT|UPDATE|DELETE)',
         'String concatenation in SQL query'),
        (r'execute\\s*\\(\\s*["\'].*%s', 'Using %s format in SQL'),
        (r'execute\\s*\\(\\s*f["\'].*\\{', 'Using f-string in SQL execute'),
        (r'format\\(.*\\).*(?:SELECT|INSERT|UPDATE|DELETE)',
         'Using .format() in SQL'),
    ]

    issues = []
    lines = code.split('\\n')
    for i, line in enumerate(lines, 1):
        for pattern, description in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append(f"Line {i}: {description}\\n  Code: {line.strip()}")

    if not issues:
        return "No SQL injection patterns detected"

    return f"Found {len(issues)} potential SQL injection issue(s):\\n" + "\\n".join(issues)
""",
    )

    if vuln_tool:
        test_code = '''
user_input = request.GET['username']
# Vulnerable!
cursor.execute("SELECT * FROM users WHERE name = '" + user_input + "'")

# Also vulnerable (f-string)
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
'''
        result = vuln_tool.execute(code=test_code)
        print(f"AI-generated tool 'detect_sql_injection':")
        print(f"  {result.output}")
        print(f"  [Tool is AI-generated: {vuln_tool.is_ai_generated}]")

    # Tool 2: Statistical analyzer
    stats_tool = tool_manager.create_tool_from_code(
        name="statistical_report",
        description="Generate a comprehensive statistical report for a dataset",
        parameters={
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {"type": "number"}},
                "label": {"type": "string", "description": "Dataset label"},
            },
            "required": ["data"],
        },
        source_code="""
import math
from collections import Counter

def statistical_report(data, label="Dataset"):
    if not data:
        return f"{label}: empty"

    n = len(data)
    mean = sum(data) / n
    sorted_d = sorted(data)

    # Median
    if n % 2 == 0:
        median = (sorted_d[n//2-1] + sorted_d[n//2]) / 2
    else:
        median = sorted_d[n//2]

    # Std dev
    variance = sum((x - mean)**2 for x in data) / n
    std_dev = math.sqrt(variance)

    # Mode
    counts = Counter(data)
    mode = counts.most_common(1)[0][0]

    # Percentiles
    def percentile(p):
        idx = int(n * p / 100)
        return sorted_d[min(idx, n-1)]

    lines = [
        f"=== {label} (n={n}) ===",
        f"  Range:    [{min(data):.2f}, {max(data):.2f}]",
        f"  Mean:     {mean:.4f}",
        f"  Median:   {median:.4f}",
        f"  Mode:     {mode}",
        f"  Std Dev:  {std_dev:.4f}",
        f"  P25:      {percentile(25):.4f}",
        f"  P75:      {percentile(75):.4f}",
        f"  IQR:      {percentile(75)-percentile(25):.4f}",
    ]
    return "\\n".join(lines)
""",
    )

    if stats_tool:
        result = stats_tool.execute(
            data=[23, 45, 12, 67, 34, 89, 23, 56, 78, 12, 45, 90, 34, 23],
            label="Test Dataset"
        )
        print(f"\nAI-generated tool 'statistical_report':")
        print(f"  {result.output}")

    # Tool 3: Pattern-based code fixer
    fixer_tool = tool_manager.create_tool_from_code(
        name="fix_common_python_bugs",
        description="Automatically fix common Python anti-patterns and bugs",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
            },
            "required": ["code"],
        },
        source_code="""
import re

def fix_common_python_bugs(code):
    '''AI-generated tool that applies automated fixes for common Python bugs.'''
    fixes_applied = []
    fixed = code

    # Fix 1: Mutable default arguments
    pattern = r'def (\\w+)\\(([^)]*=\\s*(?:\\[|\\{)[^)]*)?\\):'
    if re.search(r'=\\s*\\[\\]', fixed) or re.search(r'=\\s*\\{\\}', fixed):
        fixes_applied.append("WARNING: Mutable default argument detected (use None instead)")

    # Fix 2: Bare except
    fixed, count = re.subn(r'except:\\s*\\n(\\s+)pass', r'except Exception:\\n\\1pass  # Fixed: bare except', fixed)
    if count:
        fixes_applied.append(f"Fixed {count} bare except clause(s)")

    # Fix 3: == None comparisons
    fixed, count = re.subn(r'(\\w+)\\s*==\\s*None', r'\\1 is None', fixed)
    if count:
        fixes_applied.append(f"Fixed {count} '== None' to 'is None'")

    # Fix 4: != None comparisons
    fixed, count = re.subn(r'(\\w+)\\s*!=\\s*None', r'\\1 is not None', fixed)
    if count:
        fixes_applied.append(f"Fixed {count} '!= None' to 'is not None'")

    report = f"Fixes applied ({len(fixes_applied)}):\\n"
    report += "\\n".join(f"  - {f}" for f in fixes_applied) if fixes_applied else "  No fixes needed"
    report += f"\\n\\nFixed code:\\n{fixed}"
    return report
""",
    )

    if fixer_tool:
        test_code = """
def process(data={}):
    if data == None:
        return None
    try:
        result = data['key']
    except:
        pass
    if result != None:
        return result
"""
        result = fixer_tool.execute(code=test_code)
        print(f"\nAI-generated tool 'fix_common_python_bugs':")
        print(result.output[:500])

    # ------------------------------------------------------------------ #
    # 3. Tool stats                                                       #
    # ------------------------------------------------------------------ #
    print("\n3. Tool Registry Status")
    print("-" * 50)
    all_tools = tool_manager.list_tool_names()
    ai_tools = [t for t, tool in tool_manager.tools.items() if tool.is_ai_generated]
    builtin_tools = [t for t in all_tools if t not in ai_tools]

    print(f"Total tools: {len(all_tools)}")
    print(f"Built-in SE tools ({len(builtin_tools)}): {builtin_tools}")
    print(f"AI-generated tools ({len(ai_tools)}): {ai_tools}")

    # ------------------------------------------------------------------ #
    # 4. Execution isolation                                              #
    # ------------------------------------------------------------------ #
    print("\n4. Execution Isolation (Container-based)")
    print("-" * 50)
    print("From the paper: 'Each tool set specifies its environment requirements")
    print("in metadata, and OpenSage automatically provisions an isolated Docker")
    print("container with the appropriate configuration.'")
    print()
    print("Current mode: subprocess isolation (Docker fallback)")
    print(f"  Docker available: {executor.use_docker}")
    print(f"  Work directory: {executor.work_dir}")
    print(f"  Installed packages: {list(executor._installed_packages) or 'none yet'}")

    # Show that execution is isolated from main process
    result = executor.run_python_code("""
import os
import sys
print(f"Isolated subprocess PID: {os.getpid()}")
print(f"Python: {sys.executable}")
# This variable doesn't exist in main process
isolated_var = "only_in_subprocess"
print(f"Isolated var: {isolated_var}")
""")
    print(f"\nIsolated execution result:\n{result.output}")

    executor.cleanup()
    print("\n✓ Tool creation demo complete!")


if __name__ == "__main__":
    demo_tool_creation()
