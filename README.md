# OpenSage: Self-programming Agent Generation Engine

Implementation of the research paper:
> **OpenSage: Self-programming Agent Generation Engine**
> Hongwei Li, Zhun Wang et al. — arXiv:2602.16891 (February 2026)

## Overview

OpenSage is the **first ADK (Agent Development Kit) that enables LLMs to automatically create agents with self-generated topology and toolsets** while providing comprehensive and structured memory support.

Existing ADKs (OpenHands, Google ADK, LangChain) require humans to manually design agent topology, tools, and memory — creating a "human-centered paradigm" similar to early ML with handcrafted features. OpenSage shifts this to an **AI-centered paradigm** where the LLM itself programs the agent system.

## Key Innovations

### 1. Self-Generated Agent Topology

Two topology modes, both created by the LLM at runtime:

**Vertical Topology** — Sequential task decomposition:
```
Parent Task
├── Sub-agent: code_analyzer  (tools: read_file, search_code)
├── Sub-agent: bug_fixer      (tools: write_file, run_python)
└── Sub-agent: test_runner    (tools: run_tests, analyze_code)
```

**Horizontal Topology** — Parallel ensemble:
```
Same Task → Agent_Systematic + Agent_Heuristic + Agent_Exhaustive
                              ↓
                    Integrator synthesizes best solution
```

### 2. Dynamic Tool Creation

Agents write their own Python tools at runtime using `create_tool`:
```python
agent.run("I need a tool to detect SQL injection - let me write one")
# → Agent writes, compiles, and registers a new tool
# → Immediately available for use in the same session
```

### 3. Hierarchical Graph-Based Memory

```
Session
└── Task: Fix bugs in data_processor.py
    ├── Plan: Analyze → Fix → Test
    ├── SubTask: Code Analysis
    │   ├── Observation: BUG-1 median wrong (importance=0.9)
    │   ├── Observation: BUG-2 missing sqrt (importance=0.9)
    │   └── Agent: code_analyzer created
    ├── SubTask: Bug Fixing
    │   ├── Code: fixed_median = (a + b) / 2
    │   └── Tool: validate_statistics (AI-generated)
    └── Result: All 6 bugs fixed, tests: 4/4 ✓
```

The MemoryAgent optimizes context length, deduplicates redundant memories, and summarizes long observations — preventing token overflow on complex tasks.

## Architecture

```
opensage/
├── core/
│   ├── agent.py          # SageAgent: main agent class with full agentic loop
│   └── engine.py         # OpenSage: top-level ADK orchestrator
├── llm/
│   ├── base.py           # Abstract LLM interface
│   └── claude.py         # Anthropic Claude backend
├── memory/
│   ├── graph.py          # GraphMemory: directed graph with BFS/DFS retrieval
│   ├── hierarchical.py   # HierarchicalMemory: task-level API
│   └── memory_agent.py   # MemoryAgent: context optimization & maintenance
├── tools/
│   ├── base.py           # Tool & ToolResult base classes
│   ├── manager.py        # ToolManager: registration + AI tool creation
│   ├── executor.py       # ExecutionEnvironment: subprocess/container isolation
│   └── se_toolkit/       # Software Engineering domain tools
│       └── core.py       # read_file, write_file, run_python, search_code, ...
└── topology/
    ├── vertical.py       # VerticalTopology: sequential sub-task decomposition
    └── horizontal.py     # HorizontalTopology: parallel ensemble + integration
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Quick Start

```python
from opensage import OpenSage

engine = OpenSage(verbose=True)

# Auto topology: agent decides its own structure
result = engine.solve("Fix the bug in fibonacci.py and write tests")
print(result)
```

### Forced Vertical Topology

```python
result = engine.solve(
    "Analyze, fix, and verify the data_processor module",
    topology="vertical"
)
```

### Forced Horizontal Ensemble

```python
result = engine.solve(
    "Implement a high-performance sorting algorithm",
    topology="horizontal"
)
```

### Direct Agent Control

```python
agent = engine.create_agent(name="SecurityAgent")

# Agent can create sub-agents at runtime
result = agent.run("""
    Analyze auth.py for security vulnerabilities.
    Create a specialized scanner sub-agent and a patch writer sub-agent.
    Generate and apply fixes for all issues found.
""")
```

## Running Demos

```bash
# Memory system demo (no API key needed)
python demo/memory_demo.py

# Tool creation demo (no API key needed)
python demo/tool_creation_demo.py

# Topology visualization (no API key needed)
python demo/topology_demo.py

# Full SWE demo (requires ANTHROPIC_API_KEY)
python demo/swe_demo.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Benchmarks

The paper evaluates OpenSage on:

| Benchmark | Description | OpenSage Result |
|-----------|-------------|-----------------|
| **CyberGym** | 1,507 real C/C++ vulnerabilities | 🥇 #1 on leaderboard (>20% over OpenHands) |
| **Terminal-Bench 2.0** | 89 expert terminal tasks | 🥇 #1 on leaderboard |
| **SWE-Bench Pro** | 1,865 enterprise SE problems | Outperforms SWE-agent baseline |

### Ablation Study

| Config | Impact |
|--------|--------|
| Without horizontal topology | -8% to -15% |
| Without vertical topology | -12% to -23% |
| Without all features | -31% to -41% |

## Paper Citation

```bibtex
@article{opensage2026,
  title={OpenSage: Self-programming Agent Generation Engine},
  author={Hongwei Li and Zhun Wang and others},
  journal={arXiv preprint arXiv:2602.16891},
  year={2026}
}
```
