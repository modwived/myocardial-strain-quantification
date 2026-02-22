"""
Base classes for OpenSage tools.

Tools in OpenSage can be:
1. Pre-built (provided by the ADK, e.g., SE toolkit)
2. AI-generated (created by agents at runtime using code generation)

Each tool specifies its environment requirements in metadata,
allowing OpenSage to provision the correct execution environment.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_str(self) -> str:
        if self.success:
            return self.output
        return f"ERROR: {self.error}\n{self.output}"


@dataclass
class ToolEnvironment:
    """
    Environment specification for a tool.

    From the paper: "Each tool set specifies its environment requirements
    in metadata, and OpenSage automatically provisions an isolated
    Docker container with the appropriate configuration."
    """
    python_packages: List[str] = field(default_factory=list)
    system_packages: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    docker_image: Optional[str] = None   # e.g. "python:3.11-slim"
    use_container: bool = False


class Tool:
    """
    A tool that an agent can use.

    Tools are the action interface between agents and the environment.
    They can be pre-built or AI-generated at runtime.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
        environment: Optional[ToolEnvironment] = None,
        is_ai_generated: bool = False,
        source_code: Optional[str] = None,
    ):
        """
        Initialize a tool.

        Args:
            name: Tool name (used in LLM function calls)
            description: What this tool does
            parameters: JSON Schema-style parameter definition
            func: The actual Python callable implementing the tool
            environment: Execution environment requirements
            is_ai_generated: Whether this tool was created by an agent
            source_code: Original source code (for AI-generated tools)
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        self.environment = environment or ToolEnvironment()
        self.is_ai_generated = is_ai_generated
        self.source_code = source_code
        self._call_count = 0
        self._total_time = 0.0

    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given arguments."""
        start = time.time()
        try:
            output = self.func(**kwargs)
            elapsed = time.time() - start
            self._call_count += 1
            self._total_time += elapsed
            return ToolResult(
                success=True,
                output=str(output) if output is not None else "",
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

    def to_schema(self) -> Dict[str, Any]:
        """
        Convert to LLM-ready tool schema (generic format).
        Will be converted to provider-specific format by LLMClient.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        avg_time = self._total_time / self._call_count if self._call_count > 0 else 0
        return {
            "name": self.name,
            "call_count": self._call_count,
            "avg_execution_time": avg_time,
            "is_ai_generated": self.is_ai_generated,
        }
