# Agent-Based Parallel Development Strategy

A battle-tested framework for building complex software projects using parallel AI agents. Decompose, delegate, integrate.

## What This Is

A repeatable process for turning a project plan into thousands of lines of production code in a single session by running multiple AI agents in parallel — each owning a distinct module with clear interfaces.

**Proven results:** 6,400+ lines across 32 files, 224 passing tests, built in one session using 6 parallel agents.

## The 6-Step Process

```
Plan → Scaffold → Instruct → Launch → Integrate → Iterate
```

| Step | What You Do | Output |
|------|-------------|--------|
| **Plan** | Decompose project into independent phases | Dependency graph |
| **Scaffold** | Create skeleton files and interface contracts | Importable project structure |
| **Instruct** | Write detailed agent instruction files | `agents/phaseN_*.md` files |
| **Launch** | Run all agents in parallel | Code written across all modules |
| **Integrate** | Commit, test, fix interface bugs | Green test suite |
| **Iterate** | End-to-end validation, refinement | Working system |

## Quick Start

1. Read the [full strategy guide](STRATEGY.md)
2. Copy the [agent instruction template](templates/agent_instruction_template.md) for each phase
3. Copy the [launch prompt template](templates/launch_prompt_template.md) for each agent
4. Follow the [integration checklist](templates/integration_checklist.md) after agents complete

## Repository Contents

```
├── STRATEGY.md                              # Full strategy guide (start here)
├── templates/
│   ├── agent_instruction_template.md        # Template for agent instruction files
│   ├── launch_prompt_template.md            # Template for agent launch prompts
│   └── integration_checklist.md             # Post-agent integration checklist
└── examples/
    └── medical-ai-pipeline/                 # Real-world example with 6 phases
        ├── phase1_data_pipeline.md
        ├── phase2_segmentation.md
        ├── phase3_motion_estimation.md
        ├── phase4_strain_computation.md
        ├── phase5_risk_stratification.md
        └── phase6_api_deployment.md
```

## Core Principles

1. **One owner per file** — no two agents touch the same file
2. **Interfaces before implementation** — define contracts upfront
3. **Scaffold before launch** — create skeleton so agents see the full layout
4. **Launch in parallel** — all agents in a single message
5. **Test at the seams** — bugs live where modules connect
6. **Instruction files are documentation** — they outlive the session

## When to Use This

- Projects with 3+ distinct modules
- Any system that can be decomposed into a DAG
- When wall-clock time matters more than token cost
- Team projects where multiple people build different parts

## License

MIT
