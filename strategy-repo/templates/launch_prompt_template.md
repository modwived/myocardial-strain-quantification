# Agent Launch Prompt Template

Use this template when launching each agent. Customize the bracketed sections.

---

## The Prompt

```
You are implementing Phase [N] ([Module Name]) of [project description].

Read the agent instructions at [path/to/agents/phaseN_module_name.md] for full details.

Your job: WRITE CODE to implement all the TODO items in the instruction file.

The existing skeleton files are at:
- [path/to/existing/file1.py]
- [path/to/existing/file2.py]

You need to:
1. Read each existing file first
2. Enhance [file1.py]: [list specific additions]
3. Enhance [file2.py]: [list specific additions]
4. Create [path/to/new_file.py]: [describe]
5. Create [tests/test_module.py] with unit tests

Work in [/absolute/path/to/project/]. Edit existing files, create new ones.
Write production-quality code.
```

---

## Launching Multiple Agents in Parallel

All agents must be launched in a **single message** to run in parallel. Example structure:

```
[Agent 1 - Phase 1: Data Pipeline]
Prompt: "You are implementing Phase 1..."
Files: data/loader.py, data/preprocessing.py
Background: true

[Agent 2 - Phase 2: Model Training]
Prompt: "You are implementing Phase 2..."
Files: models/network.py, models/losses.py
Background: true

[Agent 3 - Phase 3: Evaluation]
Prompt: "You are implementing Phase 3..."
Files: eval/metrics.py, eval/benchmark.py
Background: true
```

---

## Tips for Effective Prompts

1. **Always reference the instruction file** — the prompt is a pointer, the instruction file has the details
2. **List existing files explicitly** — so the agent reads them before modifying
3. **Be specific about what to create vs enhance** — "enhance X" vs "create Y"
4. **Include the working directory** — absolute path prevents confusion
5. **Say "production-quality"** — signals the agent to add types, docstrings, error handling
