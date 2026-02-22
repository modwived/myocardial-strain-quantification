"""
Tool Manager for OpenSage.

Manages the lifecycle of tools:
1. Registration (pre-built tools)
2. AI-generated tool creation
3. Execution routing (local or containerized)
4. Tool discovery for agents

From the paper: OpenSage empowers AI to construct its own tools for
targeting tasks and provides tool management including overall tool
orchestration, execution isolation, and state management.
"""

import ast
import textwrap
import types
from typing import Any, Dict, List, Optional

from opensage.tools.base import Tool, ToolEnvironment, ToolResult
from opensage.tools.executor import ExecutionEnvironment


class ToolManager:
    """
    Manages a collection of tools for a SageAgent.

    Supports both pre-built tools and AI-generated tools,
    with proper isolation and state management.
    """

    def __init__(self, executor: Optional[ExecutionEnvironment] = None):
        self.tools: Dict[str, Tool] = {}
        self.executor = executor or ExecutionEnvironment()

    def register(self, tool: Tool) -> None:
        """Register a pre-built tool."""
        self.tools[tool.name] = tool

    def register_many(self, tools: List[Tool]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Get tool schemas for all registered tools."""
        return [tool.to_schema() for tool in self.tools.values()]

    def list_tool_names(self) -> List[str]:
        """Get names of all registered tools."""
        return list(self.tools.keys())

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a named tool with arguments.

        Routes to the appropriate execution environment based on
        the tool's environment requirements.
        """
        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' not found. Available: {self.list_tool_names()}",
            )

        # Install any required packages before first use
        if tool.environment.python_packages:
            install_result = self.executor.install_packages(
                tool.environment.python_packages
            )
            if not install_result.success:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Failed to install dependencies: {install_result.error}",
                )

        return tool.execute(**arguments)

    def create_tool_from_code(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        source_code: str,
        function_name: Optional[str] = None,
        required_packages: Optional[List[str]] = None,
    ) -> Optional[Tool]:
        """
        Create a new tool from AI-generated Python source code.

        This is the core of OpenSage's dynamic tool creation capability.
        From the paper: "OpenSage empowers AI to construct its own tools
        for targeting tasks."

        Args:
            name: Tool name
            description: Tool description
            parameters: JSON Schema parameter definition
            source_code: Python source code defining the tool function
            function_name: Name of the function in source_code to use
            required_packages: pip packages needed

        Returns:
            The created Tool, or None if compilation failed
        """
        fn_name = function_name or name.replace("-", "_").replace(" ", "_")

        # Validate that the source code is syntactically correct
        try:
            ast.parse(source_code)
        except SyntaxError as e:
            return None

        # Install required packages before compiling
        if required_packages:
            install_result = self.executor.install_packages(required_packages)
            if not install_result.success:
                return None

        # Compile and extract the function in a safe namespace
        try:
            namespace: Dict[str, Any] = {}
            exec(compile(source_code, "<ai_generated>", "exec"), namespace)

            if fn_name not in namespace:
                # Try to find any callable in the namespace
                callables = [
                    k for k, v in namespace.items()
                    if callable(v) and not k.startswith("_")
                ]
                if not callables:
                    return None
                fn_name = callables[0]

            func = namespace[fn_name]
        except Exception:
            return None

        environment = ToolEnvironment(
            python_packages=required_packages or [],
        )

        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
            environment=environment,
            is_ai_generated=True,
            source_code=source_code,
        )

        self.tools[name] = tool
        return tool

    def get_tool_for_llm(self) -> List[Dict[str, Any]]:
        """Get all tool schemas formatted for LLM consumption."""
        return self.list_tools()

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the manager."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def get_stats(self) -> List[Dict[str, Any]]:
        """Get execution statistics for all tools."""
        return [tool.stats for tool in self.tools.values()]
