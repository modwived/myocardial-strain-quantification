# Agent-Based Parallel Development Strategy

A practical guide for using AI agents to implement complex software projects by decomposing work into independent phases, assigning each to a parallel agent, and integrating the results.

---

## Overview

This strategy breaks a project into independent modules, assigns each to a dedicated agent with detailed instructions, runs all agents in parallel, then integrates, tests, and fixes. It works for any multi-component system — not just ML pipelines.

**What this achieves:**
- 6,000+ lines of production code in a single session
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

### Example decomposition:

```
Phase 1: Data Pipeline        (no dependencies)
Phase 2: Model A              (depends on Phase 1 interfaces)
Phase 3: Model B              (depends on Phase 1 interfaces)
Phase 4: Core Logic           (depends on Phase 2 + 3 outputs)
Phase 5: Business Logic       (depends on Phase 4 outputs)
Phase 6: API & Integration    (depends on all, but skeleton-first)
```

Phases 2 and 3 can always run in parallel. Even phases with dependencies can be developed simultaneously if interfaces are defined upfront.

---

## Step 2: Scaffold — Create Skeleton Files and Interfaces

Before launching agents, create:

1. **Directory structure** — all packages and `__init__.py` files
2. **Skeleton source files** — function signatures, docstrings, placeholder implementations
3. **Interface contracts** — what each module produces and consumes
4. **Config files** — shared configuration (YAML, env vars)
5. **Dependency files** — requirements.txt, Dockerfile, etc.

### Why scaffold first?
- Agents see the full project layout before writing code
- Interfaces are locked so agents don't make incompatible assumptions
- Reduces merge conflicts between parallel agents
- The project is importable from the start

---

## Step 3: Instruct — Write Agent Instruction Files

Create one detailed instruction file per phase in an `agents/` directory. Each file should contain:

### Template: `agents/phaseN_module_name.md`

```markdown
# Agent: Phase N — Module Name

## Mission
One paragraph describing what this agent builds and why.

## Status: IN PROGRESS

## Files You Own
- `path/to/file.py` — description (EXISTS / CREATE)

## Detailed Requirements
For each file:
- [ ] Checkboxed list of what to implement
- [ ] Include function signatures with types
- [ ] Include expected behavior and edge cases

## Interface Contract
What this module receives from other phases:
```python
input_format = {"key": Type, ...}
```
What this module outputs to other phases:
```python
output_format = {"key": Type, ...}
```

## Configuration
Inline YAML or reference to config file.

## Tests to Write
- List of specific test cases with expected outcomes.

## If You Get Stuck
- Links to relevant docs
- Common pitfalls and solutions
- Debugging hints specific to this module
```

### Key principles for good instructions:
- **Be explicit** — don't assume the agent knows your conventions
- **Include interface contracts** — the most important section
- **List concrete test cases** — agents write better code when they know what will be tested
- **Add "If You Get Stuck"** — saves agent cycles on known gotchas
- **Check existing vs create** — mark which files exist (to edit) vs need creating

---

## Step 4: Launch — Run All Agents in Parallel

Launch all agents simultaneously. Each agent reads its instruction file and the existing skeleton code, then implements everything.

### Prompt template for each agent:

```
You are implementing Phase N (Module Name) of [project description].

Read the agent instructions at agents/phaseN_module_name.md for full details.

Your job: WRITE CODE to implement all the TODO items. The existing skeleton files are at:
- path/to/existing/file1.py
- path/to/existing/file2.py

You need to:
1. Read each existing file first
2. [List specific tasks]
3. Create [new files]

Work in /path/to/project/. Edit existing files, create new ones.
Write production-quality code.
```

### Launch checklist:
- [ ] All agents launched in a single message (true parallelism)
- [ ] Each agent has a clear, non-overlapping file ownership
- [ ] No two agents edit the same file (prevents conflicts)
- [ ] Each agent gets the full instruction file path

---

## Step 5: Integrate — Commit, Test, Fix

After all agents complete:

### 5a. Check for conflicts
```bash
git status --short       # See all changes
git diff --stat          # Summary of modifications
```

If two agents modified the same file (shouldn't happen with good decomposition), resolve manually.

### 5b. Fix gitignore / build issues
Agents may create files that hit gitignore patterns or have import issues. Fix these first.

### 5c. Run the full test suite
```bash
python -m pytest tests/ -v --tb=short
```

### 5d. Fix failing tests
Common issues after parallel agent development:
- **Import errors** — missing `__init__.py` or circular imports
- **Interface mismatches** — one agent returns a dict, another expects a tuple
- **Channel/shape bugs** — in ML projects, tensor dimensions between modules
- **Missing dependencies** — one agent added a package but didn't update requirements.txt

### 5e. Commit in logical chunks
```bash
git add <phase-specific-files>
git commit -m "Implement Phase N: description"
```

Or commit all at once if everything works:
```bash
git add -A
git commit -m "Implement all N phases via parallel agents"
```

---

## Step 6: Iterate — Validate and Refine

After integration:

1. **End-to-end test** — run data through the full pipeline
2. **Generate test data** — if real data isn't available, create synthetic data that exercises all code paths
3. **Profile** — identify bottlenecks
4. **Train** — for ML projects, train on synthetic then real data
5. **Benchmark** — compare against baselines

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

### For iteration:
- After the first pass, create new instruction files for refinements
- Reference the existing code: "Read X, then improve Y"
- Keep instruction files updated as the codebase evolves

---

## Quick Reference

```bash
# 1. Plan and scaffold
mkdir -p module_a module_b module_c agents

# 2. Write instruction files
# agents/phase1_module_a.md
# agents/phase2_module_b.md
# agents/phase3_module_c.md

# 3. Launch all agents in parallel (in Claude Code)
# Use Task tool with run_in_background=true for each phase

# 4. After agents complete
git status --short              # Check all changes
python -m pytest tests/ -v      # Run tests
git add -A && git commit        # Commit

# 5. Validate end-to-end
python -c "from module_a import ...; from module_b import ..."
```
