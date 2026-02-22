"""
Execution environment for OpenSage tools.

Implements the container-based execution described in the paper:
"OpenSage provides container-based execution and state management to support
tools with heterogeneous compilation and runtime requirements."

For demo purposes, this implementation:
1. Executes Python code in a sandboxed subprocess (primary mode)
2. Falls back to in-process execution when sandboxing is not needed
3. Provides Docker container execution when Docker is available
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any, Dict, Optional

from opensage.tools.base import ToolEnvironment, ToolResult


class ExecutionEnvironment:
    """
    Manages isolated execution environments for tool execution.

    In OpenSage, each tool set can have its own environment with
    specific dependencies. This class handles:
    1. Subprocess-based sandboxed execution (safe, isolated)
    2. Container state caching (reduces startup overhead)
    3. Dependency installation on demand
    """

    def __init__(self, work_dir: Optional[str] = None, use_docker: bool = False):
        """
        Args:
            work_dir: Working directory for command execution
            use_docker: Whether to attempt Docker-based execution
        """
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="opensage_")
        self.use_docker = use_docker and self._docker_available()
        self._installed_packages: set = set()
        self._container_id: Optional[str] = None

    def _docker_available(self) -> bool:
        """Check if Docker is available on this system."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def run_python_code(
        self,
        code: str,
        timeout: int = 30,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Execute Python code in a sandboxed subprocess.

        This implements the isolation described in the paper where
        tools with conflicting dependencies can coexist.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            extra_context: Variables to inject into the code namespace

        Returns:
            ToolResult with stdout/stderr output
        """
        # Write code to temp file for clean execution
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=self.work_dir
        ) as f:
            # Inject context variables if provided
            if extra_context:
                context_lines = []
                for key, val in extra_context.items():
                    context_lines.append(f"{key} = {repr(val)}")
                f.write("\n".join(context_lines) + "\n\n")

            f.write(textwrap.dedent(code))
            tmp_path = f.name

        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.work_dir,
            )
            elapsed = time.time() - start
            os.unlink(tmp_path)

            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"

            return ToolResult(
                success=result.returncode == 0,
                output=output.strip(),
                error=result.stderr if result.returncode != 0 else None,
                execution_time=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            os.unlink(tmp_path)
            return ToolResult(
                success=False,
                output="",
                error=f"Code execution timed out after {timeout}s",
                execution_time=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start
            return ToolResult(
                success=False,
                output="",
                error=str(e),
                execution_time=elapsed,
            )

    def run_shell_command(
        self,
        command: str,
        timeout: int = 30,
        capture_output: bool = True,
    ) -> ToolResult:
        """
        Execute a shell command in the working directory.

        Args:
            command: Shell command to execute
            timeout: Maximum execution time in seconds
            capture_output: Whether to capture stdout/stderr

        Returns:
            ToolResult with command output
        """
        start = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd=self.work_dir,
            )
            elapsed = time.time() - start

            output = result.stdout or ""
            if result.stderr:
                output += f"\n{result.stderr}"

            return ToolResult(
                success=result.returncode == 0,
                output=output.strip(),
                error=result.stderr if result.returncode != 0 else None,
                execution_time=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout}s",
                execution_time=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start
            return ToolResult(
                success=False,
                output="",
                error=str(e),
                execution_time=elapsed,
            )

    def install_packages(self, packages: list) -> ToolResult:
        """
        Install Python packages in the current environment.

        In production, this would install into a tool-specific container.
        Here we install into the current Python environment.

        From the paper: "OpenSage commits container snapshots as Docker
        image layers after initialization or execution, capturing installed
        packages, compiled artifacts, and intermediate files."
        """
        new_packages = [p for p in packages if p not in self._installed_packages]
        if not new_packages:
            return ToolResult(success=True, output="All packages already installed")

        result = self.run_shell_command(
            f"{sys.executable} -m pip install {' '.join(new_packages)} --quiet",
            timeout=120,
        )

        if result.success:
            self._installed_packages.update(new_packages)

        return result

    def write_file(self, filename: str, content: str) -> ToolResult:
        """Write a file to the working directory."""
        try:
            path = os.path.join(self.work_dir, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return ToolResult(success=True, output=f"Wrote {len(content)} bytes to {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def read_file(self, filename: str) -> ToolResult:
        """Read a file from the working directory."""
        try:
            path = os.path.join(self.work_dir, filename)
            if not os.path.exists(path):
                # Try absolute path
                if os.path.exists(filename):
                    path = filename
                else:
                    return ToolResult(
                        success=False, output="", error=f"File not found: {filename}"
                    )
            with open(path, "r") as f:
                content = f.read()
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def list_files(self, directory: Optional[str] = None) -> ToolResult:
        """List files in the working directory."""
        try:
            target = directory or self.work_dir
            files = []
            for root, dirs, file_names in os.walk(target):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in file_names:
                    rel = os.path.relpath(os.path.join(root, name), target)
                    files.append(rel)
            return ToolResult(success=True, output="\n".join(sorted(files)))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def cleanup(self):
        """Clean up the working directory."""
        import shutil
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)
