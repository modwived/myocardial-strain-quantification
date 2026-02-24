"""
opensage - Enterprise AI developer tool powered by OpenSage.

Usage:
    opensage [GLOBAL OPTIONS] COMMAND [OPTIONS] [ARGS]

Global Options:
    --json          Output results as JSON (for scripting / CI pipelines)
    --model MODEL   Claude model to use (overrides config)
    --work-dir DIR  Project root (default: current directory)
    -v, --verbose   Show full agent reasoning traces
    --no-log        Skip writing an audit log entry for this run

Commands:
    config          Manage opensage settings
    fix             Fix a bug or issue in your codebase
    implement       Implement a new feature
    analyze         Analyse the codebase for issues
    ask             Ask a question about your codebase
    review          Review code changes (staged, last commit, or diff file)
    test            Generate, fix, or run tests
    run             Execute any arbitrary engineering task
    info            Show system / project information

Examples:
    opensage config set api_key sk-ant-...
    opensage fix "login fails when password contains special chars" --file auth.py
    opensage implement "add rate limiting to all REST endpoints"
    opensage analyze --type security
    opensage ask "how does the payment reconciliation flow work?"
    opensage review --staged
    opensage test --generate src/payments/processor.py
    opensage test --fix
    opensage run "migrate the ORM from SQLAlchemy 1.x to 2.x"
    opensage info
"""

import argparse
import datetime
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from opensage.config import Config, ConfigError, VALID_MODELS, VALID_KEYS
from opensage.project import detect_project, format_project_summary, build_agent_context


# ---------------------------------------------------------------------------
# Terminal colours (disabled automatically when not a TTY or NO_COLOR is set)
# ---------------------------------------------------------------------------

class _C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    CYAN    = "\033[36m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    WHITE   = "\033[37m"

def _disable_colours() -> None:
    for attr in vars(_C):
        if not attr.startswith("_"):
            setattr(_C, attr, "")

if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    _disable_colours()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _header(text: str) -> None:
    print(f"\n{_C.BOLD}{_C.CYAN}» {text}{_C.RESET}")

def _success(text: str) -> None:
    print(f"{_C.GREEN}✓ {text}{_C.RESET}")

def _warn(text: str) -> None:
    print(f"{_C.YELLOW}⚠ {text}{_C.RESET}")

def _err(text: str) -> None:
    print(f"{_C.RED}✗ {text}{_C.RESET}", file=sys.stderr)

def _dim(text: str) -> None:
    print(f"{_C.DIM}  {text}{_C.RESET}")

def _print_result(text: str, as_json: bool, json_payload: Dict) -> None:
    if as_json:
        print(json.dumps(json_payload, indent=2))
    else:
        print(f"\n{_C.WHITE}{text}{_C.RESET}")


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def _audit_log(command: str, task: str, result: str, elapsed: float, cfg: Config) -> None:
    """Append one JSONL line to ~/.opensage/logs/YYYY-MM-DD.jsonl."""
    if not cfg.get("log_sessions", True):
        return
    log_dir = Path.home() / ".opensage" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts":      datetime.datetime.now().isoformat(),
        "cmd":     command,
        "task":    task[:400],
        "result_len": len(result),
        "elapsed": round(elapsed, 2),
        "cwd":     os.getcwd(),
        "model":   cfg.get("model"),
    }
    log_file = log_dir / f"{datetime.date.today().isoformat()}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _make_engine(cfg: Config, work_dir: str, verbose: bool, model: Optional[str]):
    """Build and return an OpenSage engine, raising ConfigError on bad config."""
    from opensage import OpenSage

    api_key = cfg.require_api_key()
    use_model = model or cfg.get("model")
    max_iter = int(cfg.get("max_iterations", 15))

    return OpenSage(
        api_key=api_key,
        model=use_model,
        work_dir=work_dir,
        verbose=verbose,
        max_iterations=max_iter,
    )


# ---------------------------------------------------------------------------
# Shared argparse parent (global flags that every subcommand inherits)
# ---------------------------------------------------------------------------

def _global_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--json",     action="store_true", help="Output results as JSON")
    p.add_argument("--model",    metavar="MODEL",     help="Claude model (overrides config)")
    p.add_argument("--work-dir", metavar="DIR",       help="Project root (default: cwd)")
    p.add_argument("-v", "--verbose", action="store_true", help="Show agent reasoning traces")
    p.add_argument("--no-log",   action="store_true", help="Skip audit log for this run")
    return p


# ---------------------------------------------------------------------------
# Command: config
# ---------------------------------------------------------------------------

def cmd_config(args: argparse.Namespace, cfg: Config) -> int:
    action = args.config_action

    if action == "set":
        key, value = args.key, args.value
        if key not in VALID_KEYS:
            _err(f"Unknown key: '{key}'")
            _dim(f"Valid keys: {', '.join(sorted(VALID_KEYS))}")
            return 1
        if key == "model" and value not in VALID_MODELS:
            _warn(f"Model '{value}' is not in the known list; proceeding anyway.")
        try:
            cfg.set(key, value)
        except ConfigError as e:
            _err(str(e))
            return 1
        if key == "api_key":
            masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            _success(f"api_key = {masked}")
        else:
            _success(f"{key} = {value}")
        return 0

    if action == "get":
        val = cfg.get(args.key)
        if val is None:
            _err(f"Key not set: {args.key}")
            return 1
        if args.key == "api_key":
            val = f"{val[:8]}...{val[-4:]}" if len(val) > 12 else "***"
        print(val)
        return 0

    if action == "unset":
        removed = cfg.unset(args.key)
        if removed:
            _success(f"Unset {args.key}")
        else:
            _warn(f"Key '{args.key}' was not set")
        return 0

    if action == "show":
        settings = cfg.all_settings()
        if getattr(args, "json", False):
            print(json.dumps(settings, indent=2))
        else:
            _header("OpenSage Configuration")
            for k, v in sorted(settings.items()):
                print(f"  {_C.CYAN}{k:<20}{_C.RESET} {_C.WHITE}{v}{_C.RESET}")
            print(f"\n  {_C.DIM}Config file: {Path.home() / '.opensage' / 'config.json'}{_C.RESET}")
            print(f"  {_C.DIM}Logs dir:    {Path.home() / '.opensage' / 'logs'}{_C.RESET}")
        return 0

    return 1


# ---------------------------------------------------------------------------
# Command: fix
# ---------------------------------------------------------------------------

def cmd_fix(args: argparse.Namespace, cfg: Config) -> int:
    """Fix a bug or issue described in natural language."""
    work_dir = args.work_dir or os.getcwd()
    description = " ".join(args.description)

    if not description.strip():
        _err("No description provided.")
        _dim('Example: opensage fix "authentication fails for users with + in their email"')
        return 1

    _header(f"Fix: {description}")
    project_info = detect_project(work_dir)
    _dim(format_project_summary(project_info))

    task_parts: List[str] = [
        f"Fix the following issue in this codebase:\n{description}",
    ]
    if getattr(args, "file", None):
        task_parts.append(f"\nThe issue is in: {args.file}")

    task_parts.append(build_agent_context(project_info))
    task_parts.append(
        "After fixing:\n"
        "1. Verify the fix addresses the root cause, not just the symptom\n"
        "2. Run existing tests if a test suite is present and report results\n"
        "3. Summarise: what file(s) changed, what the root cause was, and how you fixed it\n"
        "4. List any related issues you noticed (do NOT fix them unless asked)"
    )

    if getattr(args, "dry_run", False):
        task_parts.append(
            "\nDRY RUN MODE: Describe exactly what you would change and why, "
            "but do NOT write or modify any files."
        )

    task = "\n\n".join(filter(None, task_parts))

    try:
        engine = _make_engine(cfg, work_dir, args.verbose, args.model)
    except ConfigError as e:
        _err(str(e))
        return 1

    topology = getattr(args, "topology", None) or cfg.get("topology", "auto")

    if not args.verbose:
        _dim("Running agent…")

    start = time.time()
    try:
        result = engine.solve(task, topology=topology)
    except Exception as e:
        _err(f"Agent error: {e}")
        return 1
    elapsed = time.time() - start

    _print_result(result, args.json, {
        "command": "fix",
        "description": description,
        "dry_run": getattr(args, "dry_run", False),
        "result": result,
        "elapsed_seconds": round(elapsed, 2),
    })

    if not getattr(args, "no_log", False):
        _audit_log("fix", description, result, elapsed, cfg)
    _success(f"Done in {elapsed:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# Command: implement
# ---------------------------------------------------------------------------

def cmd_implement(args: argparse.Namespace, cfg: Config) -> int:
    """Implement a new feature described in natural language."""
    work_dir = args.work_dir or os.getcwd()
    description = " ".join(args.description)

    if not description.strip():
        _err("No description provided.")
        _dim('Example: opensage implement "add JWT refresh token rotation"')
        return 1

    _header(f"Implement: {description}")
    project_info = detect_project(work_dir)
    _dim(format_project_summary(project_info))

    task_parts: List[str] = [
        f"Implement the following feature in this codebase:\n{description}",
    ]
    if getattr(args, "target_file", None):
        task_parts.append(f"\nPrimary target file: {args.target_file}")

    task_parts.append(build_agent_context(project_info))
    task_parts.append(
        "Implementation requirements:\n"
        "1. Study the existing code style and architecture before writing anything\n"
        "2. Make minimal, focused changes — do not refactor unrelated code\n"
        "3. Do not break existing functionality\n"
        "4. Add or update tests if the project has a test directory\n"
        "5. Summarise every file you created or modified and the key design decisions"
    )

    if getattr(args, "dry_run", False):
        task_parts.append(
            "\nDRY RUN MODE: Produce an implementation plan with file-by-file changes, "
            "but do NOT write or modify any files."
        )

    task = "\n\n".join(filter(None, task_parts))

    try:
        engine = _make_engine(cfg, work_dir, args.verbose, args.model)
    except ConfigError as e:
        _err(str(e))
        return 1

    topology = getattr(args, "topology", None) or cfg.get("topology", "auto")

    if not args.verbose:
        _dim("Running agent…")

    start = time.time()
    try:
        result = engine.solve(task, topology=topology)
    except Exception as e:
        _err(f"Agent error: {e}")
        return 1
    elapsed = time.time() - start

    _print_result(result, args.json, {
        "command": "implement",
        "description": description,
        "dry_run": getattr(args, "dry_run", False),
        "result": result,
        "elapsed_seconds": round(elapsed, 2),
    })

    if not getattr(args, "no_log", False):
        _audit_log("implement", description, result, elapsed, cfg)
    _success(f"Done in {elapsed:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# Command: analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace, cfg: Config) -> int:
    """Analyse the codebase for security, performance, or quality issues."""
    work_dir = args.work_dir or os.getcwd()
    target = getattr(args, "path", None) or work_dir

    analysis_type = getattr(args, "type", "all") or "all"
    type_focus = {
        "security":    "security vulnerabilities (OWASP Top 10), injection risks, hardcoded secrets, insecure deserialization, missing auth/authz checks",
        "performance": "performance bottlenecks, N+1 query problems, synchronous I/O in async paths, large allocations, inefficient algorithms",
        "quality":     "code quality issues: dead code, duplicated logic, missing error handling, poor naming, overly complex methods, lack of tests",
        "all":         "security vulnerabilities, performance bottlenecks, and code quality issues",
    }

    _header(f"Analyse ({analysis_type}): {target}")
    project_info = detect_project(work_dir)
    _dim(format_project_summary(project_info))

    task = (
        f"Perform a thorough {analysis_type} analysis of: {target}\n\n"
        f"Focus areas: {type_focus.get(analysis_type, type_focus['all'])}\n\n"
        f"{build_agent_context(project_info)}\n\n"
        "Deliverables:\n"
        "1. Scan all relevant source files systematically\n"
        "2. For each finding: severity (CRITICAL/HIGH/MEDIUM/LOW), file path, "
        "   line number(s), description, and a concrete suggested fix\n"
        "3. Executive summary: overall risk/quality assessment\n"
        "4. Prioritised top-5 action items\n"
        "5. Do NOT modify any files — analysis only"
    )

    try:
        engine = _make_engine(cfg, work_dir, args.verbose, args.model)
    except ConfigError as e:
        _err(str(e))
        return 1

    if not args.verbose:
        _dim("Scanning codebase…")

    start = time.time()
    try:
        # Vertical topology works well here: decompose into sub-tasks per file/module
        result = engine.solve(task, topology="vertical")
    except Exception as e:
        _err(f"Agent error: {e}")
        return 1
    elapsed = time.time() - start

    output_file = getattr(args, "output", None)
    if output_file:
        Path(output_file).write_text(result)
        _success(f"Report written to {output_file}")

    _print_result(result, args.json, {
        "command": "analyze",
        "path": target,
        "analysis_type": analysis_type,
        "result": result,
        "elapsed_seconds": round(elapsed, 2),
    })

    if not getattr(args, "no_log", False):
        _audit_log("analyze", f"{analysis_type}:{target}", result, elapsed, cfg)
    _success(f"Done in {elapsed:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# Command: ask
# ---------------------------------------------------------------------------

def cmd_ask(args: argparse.Namespace, cfg: Config) -> int:
    """Answer a question about the codebase."""
    work_dir = args.work_dir or os.getcwd()
    question = " ".join(args.question)

    if not question.strip():
        _err("No question provided.")
        _dim('Example: opensage ask "where is the database connection pool configured?"')
        return 1

    _header(f"Ask: {question}")
    project_info = detect_project(work_dir)

    task = (
        f"Answer this question about the codebase:\n{question}\n\n"
        f"{build_agent_context(project_info)}\n\n"
        "Requirements:\n"
        "1. Read relevant source files before answering\n"
        "2. Give a precise, accurate answer with specific file:line references\n"
        "3. Include relevant code snippets where they aid understanding\n"
        "4. Do NOT modify any files"
    )

    try:
        engine = _make_engine(cfg, work_dir, args.verbose, args.model)
    except ConfigError as e:
        _err(str(e))
        return 1

    if not args.verbose:
        _dim("Researching codebase…")

    start = time.time()
    try:
        result = engine.solve(task, topology="single")
    except Exception as e:
        _err(f"Agent error: {e}")
        return 1
    elapsed = time.time() - start

    _print_result(result, args.json, {
        "command": "ask",
        "question": question,
        "result": result,
        "elapsed_seconds": round(elapsed, 2),
    })

    if not getattr(args, "no_log", False):
        _audit_log("ask", question, result, elapsed, cfg)
    return 0


# ---------------------------------------------------------------------------
# Command: review
# ---------------------------------------------------------------------------

def cmd_review(args: argparse.Namespace, cfg: Config) -> int:
    """Review code changes — staged, last commit, or a diff file."""
    work_dir = args.work_dir or os.getcwd()

    if getattr(args, "staged", False):
        scope = "staged git changes"
        fetch_instruction = (
            "Retrieve the staged diff with: git diff --cached\n"
            "If there are no staged changes, report that clearly."
        )
    elif getattr(args, "last_commit", False):
        scope = "most recent git commit"
        fetch_instruction = "Retrieve the commit with: git show HEAD"
    elif getattr(args, "diff_file", None):
        scope = f"diff file: {args.diff_file}"
        fetch_instruction = f"Read the diff from the file: {args.diff_file}"
    else:
        scope = "all uncommitted changes"
        fetch_instruction = (
            "Retrieve all uncommitted changes with: git diff HEAD\n"
            "If the working tree is clean, report that."
        )

    _header(f"Review: {scope}")
    project_info = detect_project(work_dir)
    _dim(format_project_summary(project_info))

    task = (
        f"Perform a thorough code review of: {scope}\n\n"
        f"{fetch_instruction}\n\n"
        f"{build_agent_context(project_info)}\n\n"
        "Review criteria:\n"
        "- Correctness: does the logic do what it claims?\n"
        "- Security: are there new attack surfaces, secrets in code, injection risks?\n"
        "- Performance: are there obvious inefficiencies introduced?\n"
        "- Style consistency: does it match the surrounding codebase conventions?\n"
        "- Test coverage: are new/changed code paths adequately tested?\n\n"
        "Output format:\n"
        "## Summary\n"
        "One-paragraph overall assessment and verdict: APPROVE / REQUEST CHANGES\n\n"
        "## Issues\n"
        "Numbered list, each with: severity, file:line, description, suggested fix\n\n"
        "## Suggestions (optional)\n"
        "Non-blocking improvement ideas\n\n"
        "Do NOT modify any files."
    )

    try:
        engine = _make_engine(cfg, work_dir, args.verbose, args.model)
    except ConfigError as e:
        _err(str(e))
        return 1

    if not args.verbose:
        _dim("Reviewing changes…")

    start = time.time()
    try:
        result = engine.solve(task, topology="single")
    except Exception as e:
        _err(f"Agent error: {e}")
        return 1
    elapsed = time.time() - start

    _print_result(result, args.json, {
        "command": "review",
        "scope": scope,
        "result": result,
        "elapsed_seconds": round(elapsed, 2),
    })

    if not getattr(args, "no_log", False):
        _audit_log("review", scope, result, elapsed, cfg)
    _success(f"Review completed in {elapsed:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# Command: test
# ---------------------------------------------------------------------------

def cmd_test(args: argparse.Namespace, cfg: Config) -> int:
    """Generate, fix, or run tests."""
    work_dir = args.work_dir or os.getcwd()
    target = getattr(args, "target", None) or "."
    project_info = detect_project(work_dir)
    ctx = build_agent_context(project_info)

    if getattr(args, "generate", False):
        _header(f"Generate tests: {target}")
        _dim(format_project_summary(project_info))
        task = (
            f"Generate comprehensive tests for: {target}\n\n"
            f"{ctx}\n\n"
            "Requirements:\n"
            "1. Study the source code to understand its behaviour and contracts\n"
            "2. Identify the existing test framework and file naming conventions\n"
            "3. Generate tests covering: happy paths, edge cases, boundary conditions, "
            "   error cases, and any observable side effects\n"
            "4. Place test files in the correct test directory with correct naming\n"
            "5. Ensure every generated test can actually run and pass with the current code\n"
            "6. Report: which files were created and what each test verifies"
        )
        topology = "vertical"
        log_key = "test:generate"

    elif getattr(args, "fix", False):
        _header("Fix failing tests")
        _dim(format_project_summary(project_info))
        task = (
            f"Find and fix all failing tests in this codebase.\n\n"
            f"{ctx}\n\n"
            "Procedure:\n"
            "1. Run the full test suite to identify all failures\n"
            "2. For each failure: read the test, read the relevant source code, "
            "   understand whether the test or the implementation is wrong\n"
            "3. Fix the root cause (either update incorrect test expectations OR "
            "   fix the implementation bug — not both unless both are wrong)\n"
            "4. Re-run the test suite to confirm all fixes\n"
            "5. Report: test name, root cause, what was changed to fix it"
        )
        topology = "vertical"
        log_key = "test:fix"

    elif getattr(args, "run", False):
        _header(f"Run tests: {target}")
        task = (
            f"Run the test suite at: {target}\n\n"
            f"{ctx}\n\n"
            "1. Detect the test framework in use\n"
            "2. Run all tests\n"
            "3. Report: total tests, passed, failed, skipped; full failure output"
        )
        topology = "single"
        log_key = "test:run"

    else:
        _err("Specify an action: --generate, --fix, or --run")
        return 1

    try:
        engine = _make_engine(cfg, work_dir, args.verbose, args.model)
    except ConfigError as e:
        _err(str(e))
        return 1

    if not args.verbose:
        _dim("Running…")

    start = time.time()
    try:
        result = engine.solve(task, topology=topology)
    except Exception as e:
        _err(f"Agent error: {e}")
        return 1
    elapsed = time.time() - start

    _print_result(result, args.json, {
        "command": log_key,
        "target": target,
        "result": result,
        "elapsed_seconds": round(elapsed, 2),
    })

    if not getattr(args, "no_log", False):
        _audit_log(log_key, target, result, elapsed, cfg)
    _success(f"Done in {elapsed:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# Command: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace, cfg: Config) -> int:
    """Execute any arbitrary engineering task."""
    work_dir = args.work_dir or os.getcwd()
    task_desc = " ".join(args.task)

    if not task_desc.strip():
        _err("No task provided.")
        _dim('Example: opensage run "migrate SQLAlchemy 1.x declarative models to 2.x style"')
        return 1

    _header(f"Task: {task_desc}")
    project_info = detect_project(work_dir)
    _dim(format_project_summary(project_info))

    ctx = build_agent_context(project_info)
    full_task = f"{task_desc}\n\n{ctx}" if ctx else task_desc

    try:
        engine = _make_engine(cfg, work_dir, args.verbose, args.model)
    except ConfigError as e:
        _err(str(e))
        return 1

    topology = getattr(args, "topology", None) or cfg.get("topology", "auto")

    if not args.verbose:
        _dim("Running agent…")

    start = time.time()
    try:
        result = engine.solve(full_task, topology=topology)
    except Exception as e:
        _err(f"Agent error: {e}")
        return 1
    elapsed = time.time() - start

    _print_result(result, args.json, {
        "command": "run",
        "task": task_desc,
        "result": result,
        "elapsed_seconds": round(elapsed, 2),
    })

    if not getattr(args, "no_log", False):
        _audit_log("run", task_desc, result, elapsed, cfg)
    _success(f"Done in {elapsed:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# Command: info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace, cfg: Config) -> int:
    """Display system, configuration, and project information."""
    work_dir = args.work_dir or os.getcwd()

    try:
        cfg.require_api_key()
        api_status = "configured"
    except ConfigError:
        api_status = "NOT CONFIGURED"

    project_info = detect_project(work_dir)

    payload: Dict[str, Any] = {
        "opensage_version": "1.0.0",
        "model":            cfg.get("model"),
        "topology":         cfg.get("topology"),
        "max_iterations":   cfg.get("max_iterations"),
        "api_key_status":   api_status,
        "config_file":      str(Path.home() / ".opensage" / "config.json"),
        "log_dir":          str(Path.home() / ".opensage" / "logs"),
        "work_dir":         work_dir,
        "project": {
            "primary_language": project_info.get("primary_language"),
            "languages":        project_info.get("languages"),
            "frameworks":       project_info.get("frameworks"),
            "has_git":          project_info.get("has_git"),
            "git_branch":       project_info.get("git_branch"),
            "has_ci":           project_info.get("has_ci"),
            "ci_system":        project_info.get("ci_system"),
            "total_files":      project_info.get("total_files"),
            "approximate_loc":  project_info.get("approximate_loc"),
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _header("OpenSage Information")
        print(f"\n  {_C.BOLD}Tool{_C.RESET}")
        print(f"  {'Version':<18} 1.0.0")
        print(f"  {'Model':<18} {payload['model']}")
        print(f"  {'Topology':<18} {payload['topology']}")
        print(f"  {'Max iterations':<18} {payload['max_iterations']}")
        print(f"  {'API key':<18} {api_status}")
        print(f"  {'Config file':<18} {payload['config_file']}")
        print(f"  {'Audit logs':<18} {payload['log_dir']}")

        print(f"\n  {_C.BOLD}Project{_C.RESET}  {_C.DIM}({work_dir}){_C.RESET}")
        for line in format_project_summary(project_info).splitlines():
            print(f"  {line}")

        if api_status == "NOT CONFIGURED":
            print(f"\n  {_C.YELLOW}⚠  API key not set. Configure it with:{_C.RESET}")
            print(f"     opensage config set api_key YOUR_ANTHROPIC_API_KEY")
            print(f"  {_C.DIM}Or: export ANTHROPIC_API_KEY=YOUR_KEY{_C.RESET}")

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    gp = _global_parent()

    parser = argparse.ArgumentParser(
        prog="opensage",
        description=(
            "OpenSage — AI-powered developer tool for enterprise software engineering.\n"
            "Fix bugs, implement features, analyse code, review PRs, and more.\n"
            "Powered by the OpenSage Self-programming Agent framework (arXiv:2602.16891)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  opensage config set api_key sk-ant-...\n"
            "  opensage info\n\n"
            "Examples:\n"
            "  opensage fix 'login fails when password has special chars'\n"
            "  opensage implement 'add pagination to GET /users'\n"
            "  opensage analyze --type security\n"
            "  opensage ask 'how does user session invalidation work?'\n"
            "  opensage review --staged\n"
            "  opensage test --generate src/payments/\n"
            "  opensage test --fix\n"
            "  opensage run 'migrate from SQLAlchemy 1.x to 2.x'\n"
        ),
        parents=[gp],
    )
    parser.add_argument(
        "--version", action="version", version="opensage 1.0.0"
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- config --------------------------------------------------------
    cfg_p = sub.add_parser(
        "config", parents=[gp],
        help="Manage opensage settings",
        description="Read and write persistent opensage configuration.",
    )
    cfg_sub = cfg_p.add_subparsers(dest="config_action", metavar="ACTION")
    cfg_sub.required = True

    s = cfg_sub.add_parser("set",   parents=[gp], help="Set a value")
    s.add_argument("key",   help="Config key (api_key, model, topology, …)")
    s.add_argument("value", help="Value to set")

    g = cfg_sub.add_parser("get",   parents=[gp], help="Get a value")
    g.add_argument("key",   help="Config key")

    cfg_sub.add_parser("show",  parents=[gp], help="Show all settings")

    u = cfg_sub.add_parser("unset", parents=[gp], help="Remove a key from config")
    u.add_argument("key",   help="Config key to remove")

    # ---- fix -----------------------------------------------------------
    fix_p = sub.add_parser(
        "fix", parents=[gp],
        help="Fix a bug or issue in the codebase",
        description=(
            "Describe a bug in plain English. OpenSage will locate the root cause, "
            "apply a fix, run existing tests, and summarise the change."
        ),
    )
    fix_p.add_argument("description", nargs="+",
                       help="Natural language description of the bug")
    fix_p.add_argument("--file",      metavar="FILE",
                       help="Specific file containing the bug (narrows the search)")
    fix_p.add_argument("--topology",  choices=["auto", "vertical", "horizontal", "single"],
                       help="Execution topology (default: auto)")
    fix_p.add_argument("--dry-run",   action="store_true",
                       help="Describe what would be fixed without modifying files")

    # ---- implement -----------------------------------------------------
    impl_p = sub.add_parser(
        "implement", parents=[gp],
        help="Implement a new feature",
        description=(
            "Describe a feature in plain English. OpenSage will study the existing "
            "codebase, implement the feature following existing conventions, and "
            "add tests if a test directory is present."
        ),
    )
    impl_p.add_argument("description", nargs="+",
                        help="Natural language feature description")
    impl_p.add_argument("--in",   dest="target_file", metavar="FILE",
                        help="Primary file to extend or create")
    impl_p.add_argument("--topology", choices=["auto", "vertical", "horizontal", "single"],
                        help="Execution topology (default: auto)")
    impl_p.add_argument("--dry-run",  action="store_true",
                        help="Produce an implementation plan without writing files")

    # ---- analyze -------------------------------------------------------
    an_p = sub.add_parser(
        "analyze", parents=[gp],
        help="Analyse the codebase for issues",
        description=(
            "Scan the codebase for security vulnerabilities, performance bottlenecks, "
            "or code quality problems. Outputs a prioritised issue list."
        ),
    )
    an_p.add_argument("path",   nargs="?",
                      help="Path to analyse (default: current directory)")
    an_p.add_argument("--type", choices=["security", "performance", "quality", "all"],
                      default="all",
                      help="Analysis focus (default: all)")
    an_p.add_argument("--output", metavar="FILE",
                      help="Write the report to a file as well")

    # ---- ask -----------------------------------------------------------
    ask_p = sub.add_parser(
        "ask", parents=[gp],
        help="Ask a question about the codebase",
        description=(
            "Ask any question about the codebase in plain English. "
            "The agent will read source files and provide a precise answer with "
            "file:line references and code snippets."
        ),
    )
    ask_p.add_argument("question", nargs="+",
                       help="Question to answer")

    # ---- review --------------------------------------------------------
    rev_p = sub.add_parser(
        "review", parents=[gp],
        help="Review code changes",
        description=(
            "Perform an AI code review on staged changes, the last commit, "
            "a specific diff file, or all uncommitted changes."
        ),
    )
    rev_exc = rev_p.add_mutually_exclusive_group()
    rev_exc.add_argument("--staged",      action="store_true",
                         help="Review staged git changes (git diff --cached)")
    rev_exc.add_argument("--last-commit", action="store_true",
                         help="Review the most recent git commit")
    rev_exc.add_argument("--diff-file",   metavar="FILE",
                         help="Review a specific diff file")

    # ---- test ----------------------------------------------------------
    tst_p = sub.add_parser(
        "test", parents=[gp],
        help="Generate, fix, or run tests",
        description=(
            "Work with the test suite: generate missing tests for a file or module, "
            "fix all failing tests, or run the full test suite."
        ),
    )
    tst_p.add_argument("target", nargs="?",
                       help="Target file or directory (used with --generate and --run)")
    tst_act = tst_p.add_mutually_exclusive_group(required=True)
    tst_act.add_argument("--generate", action="store_true",
                         help="Generate missing tests for TARGET")
    tst_act.add_argument("--fix",      action="store_true",
                         help="Find and fix all failing tests")
    tst_act.add_argument("--run",      action="store_true",
                         help="Run the test suite")

    # ---- run -----------------------------------------------------------
    run_p = sub.add_parser(
        "run", parents=[gp],
        help="Execute any arbitrary engineering task",
        description=(
            "General-purpose command: describe any engineering task in natural language "
            "and let the OpenSage agent execute it."
        ),
    )
    run_p.add_argument("task", nargs="+",
                       help="Task description")
    run_p.add_argument("--topology", choices=["auto", "vertical", "horizontal", "single"],
                       help="Execution topology (default: from config)")

    # ---- info ----------------------------------------------------------
    sub.add_parser(
        "info", parents=[gp],
        help="Show system / project information",
        description="Display opensage configuration and project detection results.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _handle_interrupt(sig: int, frame: object) -> None:
    print(f"\n{_C.YELLOW}Interrupted{_C.RESET}")
    sys.exit(130)

signal.signal(signal.SIGINT, _handle_interrupt)


COMMAND_MAP = {
    "config":    cmd_config,
    "fix":       cmd_fix,
    "implement": cmd_implement,
    "analyze":   cmd_analyze,
    "ask":       cmd_ask,
    "review":    cmd_review,
    "test":      cmd_test,
    "run":       cmd_run,
    "info":      cmd_info,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = Config()

    # Temporarily disable audit logging if --no-log was passed
    if getattr(args, "no_log", False):
        cfg._data["log_sessions"] = False  # in-memory only, not persisted

    handler = COMMAND_MAP.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        exit_code = handler(args, cfg)
    except KeyboardInterrupt:
        print(f"\n{_C.YELLOW}Interrupted{_C.RESET}")
        exit_code = 130

    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
