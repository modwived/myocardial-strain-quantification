"""
Core SE toolkit tools for OpenSage SageAgents.

These tools are always available to agents working on software engineering
tasks (SWE-Bench, bug fixing, code generation, etc.).
"""

import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

from opensage.tools.base import Tool, ToolEnvironment, ToolResult
from opensage.tools.executor import ExecutionEnvironment


def get_se_toolkit(executor: Optional[ExecutionEnvironment] = None) -> List[Tool]:
    """
    Build and return the standard SE toolkit tool list.

    Args:
        executor: Shared ExecutionEnvironment for command-based tools

    Returns:
        List of Tool objects ready for registration
    """
    env = executor or ExecutionEnvironment()
    return [
        _make_read_file_tool(env),
        _make_write_file_tool(env),
        _make_run_command_tool(env),
        _make_run_python_tool(env),
        _make_search_code_tool(),
        _make_list_files_tool(env),
        _make_apply_patch_tool(env),
        _make_run_tests_tool(env),
        _make_analyze_code_tool(env),
    ]


# ------------------------------------------------------------------ #
# Tool implementations                                                 #
# ------------------------------------------------------------------ #

def _make_read_file_tool(env: ExecutionEnvironment) -> Tool:
    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        result = env.read_file(path)
        if not result.success:
            return result.error or f"Could not read {path}"
        lines = result.output.splitlines()
        start = max(0, start_line - 1)
        end = end_line if end_line > 0 else len(lines)
        snippet = lines[start:end]
        numbered = [f"{start + i + 1}: {line}" for i, line in enumerate(snippet)]
        return "\n".join(numbered)

    return Tool(
        name="read_file",
        description="Read a source file, optionally a specific line range. Returns numbered lines.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "start_line": {
                    "type": "integer",
                    "description": "Start line (1-indexed)",
                    "default": 1,
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line (0 = read all)",
                    "default": 0,
                },
            },
            "required": ["path"],
        },
        func=read_file,
    )


def _make_write_file_tool(env: ExecutionEnvironment) -> Tool:
    def write_file(path: str, content: str) -> str:
        # Allow absolute paths too
        if os.path.isabs(path):
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)
                return f"Wrote {len(content)} bytes to {path}"
            except Exception as e:
                return f"Error: {e}"
        result = env.write_file(path, content)
        return result.output if result.success else f"Error: {result.error}"

    return Tool(
        name="write_file",
        description="Write content to a file (creates or overwrites).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write to"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
        func=write_file,
    )


def _make_run_command_tool(env: ExecutionEnvironment) -> Tool:
    def run_command(command: str, timeout: int = 30) -> str:
        result = env.run_shell_command(command, timeout=timeout)
        if result.success:
            return result.output or "(no output)"
        return f"Exit code != 0\n{result.output}\n{result.error}"

    return Tool(
        name="run_command",
        description="Run a shell command and return its output.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
        func=run_command,
    )


def _make_run_python_tool(env: ExecutionEnvironment) -> Tool:
    def run_python(code: str, timeout: int = 30) -> str:
        result = env.run_python_code(code, timeout=timeout)
        return result.output if result.success else f"Error:\n{result.error}\n{result.output}"

    return Tool(
        name="run_python",
        description="Execute Python code and return the output. Use print() for output.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 30,
                },
            },
            "required": ["code"],
        },
        func=run_python,
    )


def _make_search_code_tool() -> Tool:
    def search_code(pattern: str, path: str = ".", file_extensions: str = "") -> str:
        """Search for a pattern in source files."""
        try:
            extensions = (
                [f".{e.strip()}" for e in file_extensions.split(",") if e.strip()]
                if file_extensions
                else []
            )

            matches = []
            search_root = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
            compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

            for root, dirs, files in os.walk(search_root):
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".") and d not in ("__pycache__", "node_modules")
                ]
                for fname in files:
                    if extensions and not any(fname.endswith(e) for e in extensions):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", errors="replace") as f:
                            for i, line in enumerate(f, 1):
                                if compiled.search(line):
                                    rel = os.path.relpath(fpath, search_root)
                                    matches.append(f"{rel}:{i}: {line.rstrip()}")
                    except (PermissionError, UnicodeDecodeError):
                        pass

            if not matches:
                return f"No matches for '{pattern}'"
            return "\n".join(matches[:50])  # cap at 50 matches
        except Exception as e:
            return f"Search error: {e}"

    return Tool(
        name="search_code",
        description="Search for a regex pattern in source files. Returns matching lines with file:line.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search"},
                "path": {
                    "type": "string",
                    "description": "Directory to search in",
                    "default": ".",
                },
                "file_extensions": {
                    "type": "string",
                    "description": "Comma-separated file extensions (e.g. 'py,js'). Empty = all files.",
                    "default": "",
                },
            },
            "required": ["pattern"],
        },
        func=search_code,
    )


def _make_list_files_tool(env: ExecutionEnvironment) -> Tool:
    def list_files(directory: str = ".") -> str:
        result = env.list_files(directory)
        return result.output if result.success else f"Error: {result.error}"

    return Tool(
        name="list_files",
        description="List files in a directory recursively.",
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to list",
                    "default": ".",
                }
            },
            "required": [],
        },
        func=list_files,
    )


def _make_apply_patch_tool(env: ExecutionEnvironment) -> Tool:
    def apply_patch(patch: str, target_file: Optional[str] = None) -> str:
        """Apply a unified diff patch to a file or write a new file."""
        if target_file and not patch.startswith("---"):
            # Direct content replacement
            result = env.write_file(target_file, patch)
            return result.output if result.success else f"Error: {result.error}"

        # Apply unified diff
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch)
            patch_path = f.name

        result = env.run_shell_command(f"patch -p1 < {patch_path}", timeout=30)
        os.unlink(patch_path)
        return result.output if result.success else f"Patch failed:\n{result.error}\n{result.output}"

    return Tool(
        name="apply_patch",
        description="Apply a unified diff patch or write a new file. Use for code changes.",
        parameters={
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Patch content or new file content"},
                "target_file": {
                    "type": "string",
                    "description": "Target file path (for direct replacement)",
                },
            },
            "required": ["patch"],
        },
        func=apply_patch,
    )


def _make_run_tests_tool(env: ExecutionEnvironment) -> Tool:
    def run_tests(
        test_path: str = ".",
        framework: str = "pytest",
        verbose: bool = False,
    ) -> str:
        flags = "-v" if verbose else ""
        if framework == "pytest":
            cmd = f"{sys.executable} -m pytest {test_path} {flags} --tb=short 2>&1"
        elif framework == "unittest":
            cmd = f"{sys.executable} -m unittest discover {test_path} 2>&1"
        else:
            cmd = f"{sys.executable} {test_path}"

        result = env.run_shell_command(cmd, timeout=120)
        return result.output or result.error or "(no output)"

    return Tool(
        name="run_tests",
        description="Run tests using pytest or unittest. Returns test results.",
        parameters={
            "type": "object",
            "properties": {
                "test_path": {
                    "type": "string",
                    "description": "Path to tests",
                    "default": ".",
                },
                "framework": {
                    "type": "string",
                    "description": "Test framework: pytest or unittest",
                    "enum": ["pytest", "unittest", "direct"],
                    "default": "pytest",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Show verbose output",
                    "default": False,
                },
            },
            "required": [],
        },
        func=run_tests,
    )


def _make_analyze_code_tool(env: ExecutionEnvironment) -> Tool:
    def analyze_code(code: str, check_type: str = "all") -> str:
        """Analyze Python code for issues."""
        analysis = []

        # Syntax check
        try:
            import ast
            tree = ast.parse(code)
            if check_type in ("syntax", "all"):
                analysis.append("✓ Syntax: Valid Python")

            if check_type in ("structure", "all"):
                # Count functions, classes
                funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                analysis.append(f"Structure: {len(funcs)} functions, {len(classes)} classes")

        except SyntaxError as e:
            analysis.append(f"✗ Syntax Error: {e}")
            return "\n".join(analysis)

        # Write to temp file and run pylint/flake8 if available
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp = f.name

        if check_type in ("style", "all"):
            result = env.run_shell_command(
                f"{sys.executable} -m py_compile {tmp} 2>&1", timeout=10
            )
            if result.success:
                analysis.append("✓ Compilation: OK")
            else:
                analysis.append(f"✗ Compilation error:\n{result.output}")

        os.unlink(tmp)
        return "\n".join(analysis)

    return Tool(
        name="analyze_code",
        description="Analyze Python code for syntax errors, style issues, and structure.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to analyze"},
                "check_type": {
                    "type": "string",
                    "enum": ["syntax", "structure", "style", "all"],
                    "description": "Type of analysis",
                    "default": "all",
                },
            },
            "required": ["code"],
        },
        func=analyze_code,
    )
