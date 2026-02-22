# Agent-Based Parallel Development Strategy

A practical guide for using AI agents to implement complex software projects by decomposing work into independent phases, assigning each to a parallel agent, and integrating the results.

---

## Overview

This strategy breaks a project into independent modules, assigns each to a dedicated agent with detailed instructions, runs all agents in parallel, then integrates, tests, and fixes. It works for any multi-component system — not just ML pipelines.

**What this achieves:**
- Thousands of lines of production code in a single session
- All modules developed in parallel (wall-clock time of one, not six)
- Clear ownership boundaries prevent conflicts
- Agent instruction files serve as living documentation

---

## The Process

```
Step 1: Plan          →  Decompose into independent phases
Step 2: Scaffold      →  Create skeleton files and interfaces
Step 3: Instruct      →  Write detailed agent instruction files
Step 4: Launch        →  Run all agents in parallel
Step 5: Integrate     →  Commit, test, fix issues
Step 6: Iterate       →  Train, validate, refine
```

---

## Step 1: Plan — Decompose Into Independent Phases

Break your project into modules that can be developed **independently** with well-defined interfaces between them.

### Rules for good decomposition:
- Each phase should own a distinct directory/package
- Phases communicate through **documented interfaces** (function signatures, data formats)
- Minimize circular dependencies — use a DAG structure
- Phases that depend on each other's output should still be developable in isolation (mock inputs)

### Example decompositions:

**ML Pipeline:**
```
Phase 1: Data Pipeline        (no dependencies)
Phase 2: Model A              (depends on Phase 1 interfaces)
Phase 3: Model B              (depends on Phase 1 interfaces)
Phase 4: Core Logic           (depends on Phase 2 + 3 outputs)
Phase 5: Business Logic       (depends on Phase 4 outputs)
Phase 6: API & Integration    (depends on all, but skeleton-first)
```

**Web Application:**
```
Phase 1: Database / ORM       (no dependencies)
Phase 2: Auth Module          (depends on Phase 1)
Phase 3: Core Business Logic  (depends on Phase 1)
Phase 4: REST API             (depends on Phases 2 + 3)
Phase 5: Frontend Components  (depends on Phase 4 API contracts)
Phase 6: DevOps / CI/CD       (depends on all, but config-first)
```

**CLI Tool:**
```
Phase 1: Config & Parsing     (no dependencies)
Phase 2: Core Engine          (depends on Phase 1)
Phase 3: Output Formatters    (depends on Phase 2 output types)
Phase 4: Plugin System        (depends on Phase 2 interfaces)
Phase 5: Tests & Docs         (depends on all)
```

Phases at the same level can always run in parallel. Even phases with dependencies can be developed simultaneously if interfaces are defined upfront.

---

## Step 2: Scaffold — Create Skeleton Files and Interfaces

Before launching agents, create:

1. **Directory structure** — all packages and `__init__.py` files
2. **Skeleton source files** — function signatures, docstrings, placeholder implementations
3. **Interface contracts** — what each module produces and consumes
4. **Config files** — shared configuration (YAML, env vars)
5. **Dependency files** — requirements.txt, package.json, Cargo.toml, etc.

### Why scaffold first?
- Agents see the full project layout before writing code
- Interfaces are locked so agents don't make incompatible assumptions
- Reduces merge conflicts between parallel agents
- The project is importable from the start

### Example skeleton:
```python
# module_a/processor.py (skeleton)
def process(data: InputData) -> OutputData:
    """Process input data and return results.

    Args:
        data: Validated input from the data pipeline.

    Returns:
        Processed output for downstream consumers.
    """
    raise NotImplementedError
```

This skeleton tells the agent: the function exists, here's the signature, now implement it.

---

## Step 3: Instruct — Write Agent Instruction Files

Create one detailed instruction file per phase in an `agents/` directory. Use the [agent instruction template](templates/agent_instruction_template.md).

### Key principles for good instructions:
- **Be explicit** — don't assume the agent knows your conventions
- **Include interface contracts** — the most important section
- **List concrete test cases** — agents write better code when they know what will be tested
- **Add "If You Get Stuck"** — saves agent cycles on known gotchas
- **Mark existing vs create** — indicate which files exist (to edit) vs need creating
- **Include function signatures** — with types, docstrings, and expected behavior

---

## Step 4: Launch — Run All Agents in Parallel

Launch all agents simultaneously. Each agent reads its instruction file and the existing skeleton code, then implements everything. Use the [launch prompt template](templates/launch_prompt_template.md).

### Launch checklist:
- [ ] All agents launched in a single message (true parallelism)
- [ ] Each agent has a clear, non-overlapping file ownership
- [ ] No two agents edit the same file (prevents conflicts)
- [ ] Each agent gets the full instruction file path
- [ ] Agents are told to read existing files before modifying

---

## Step 5: Integrate — Commit, Test, Fix

After all agents complete, follow the [integration checklist](templates/integration_checklist.md).

### Common issues after parallel agent development:
- **Import errors** — missing `__init__.py` or circular imports
- **Interface mismatches** — one agent returns a dict, another expects a tuple
- **Shape/type bugs** — tensor dimensions, schema differences between modules
- **Missing dependencies** — one agent added a package but didn't update dependency files
- **Gitignore collisions** — broad patterns matching source files (e.g., `data/` matching `src/data/`)

### The seams are where bugs live
The code *within* each module is usually correct. The bugs appear at **module boundaries** — where one agent's output becomes another's input. Focus testing here.

---

## Step 6: Iterate — Validate and Refine

After integration:

1. **End-to-end test** — run real (or synthetic) data through the full pipeline
2. **Generate test data** — if real data isn't available, create synthetic data that exercises all code paths
3. **Profile** — identify bottlenecks
4. **Benchmark** — compare against baselines
5. **Refine** — create new agent instruction files for improvements, launch another round

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Do This Instead |
|---|---|---|
| No skeleton before agents | Agents make incompatible assumptions | Scaffold interfaces first |
| Two agents own the same file | Merge conflicts, overwritten work | One owner per file, always |
| Vague instructions | Agent guesses wrong, wastes cycles | Be explicit with signatures and contracts |
| No interface contracts | Module A outputs X, Module B expects Y | Define input/output formats upfront |
| Launching sequentially | Loses the parallelism benefit | Launch all in one message |
| Skipping tests after integration | Silent bugs between modules | Always run full suite after merge |
| Giant monolithic phases | Agent runs out of context | Break into smaller, focused phases |
| Not reading existing code | Agent rewrites instead of extending | Always instruct "read first, then modify" |

---

## Scaling Tips

### For larger projects (10+ phases):
- Group phases into tiers: foundation → core → application
- Run tier 1 agents first, commit, then run tier 2
- Use a shared `types.py` or `interfaces.py` that all agents import

### For teams:
- Store agent instruction files in `agents/` — they double as onboarding docs
- Each team member can "resume" an agent if work is incomplete
- The instruction files capture requirements more precisely than tickets
- Review instruction files in PR reviews, not just the generated code

### For iteration:
- After the first pass, create new instruction files for refinements
- Reference the existing code: "Read X, then improve Y"
- Keep instruction files updated as the codebase evolves
- Version your instruction files — they're as valuable as the code

---

## Quick Reference

```bash
# 1. Plan and scaffold
mkdir -p module_a module_b module_c agents tests

# 2. Write instruction files
# agents/phase1_module_a.md
# agents/phase2_module_b.md
# agents/phase3_module_c.md

# 3. Launch all agents in parallel
# (In Claude Code, use Task tool with run_in_background=true for each)

# 4. After agents complete — integrate
git status --short              # Check all changes
git diff --stat                 # See scope of changes
python -m pytest tests/ -v      # Run tests
git add -A && git commit        # Commit

# 5. Validate end-to-end
python -c "from module_a import ...; from module_b import ..."
```
