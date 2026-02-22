"""
Tests for OpenSage tool system.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensage.tools.base import Tool, ToolEnvironment, ToolResult
from opensage.tools.manager import ToolManager
from opensage.tools.executor import ExecutionEnvironment
from opensage.tools.se_toolkit import get_se_toolkit


class TestTool:
    def test_basic_execution(self):
        tool = Tool(
            name="add",
            description="Add two numbers",
            parameters={"type": "object", "properties": {
                "a": {"type": "number"}, "b": {"type": "number"}
            }, "required": ["a", "b"]},
            func=lambda a, b: a + b,
        )
        result = tool.execute(a=1, b=2)
        assert result.success
        assert result.output == "3"

    def test_error_handling(self):
        tool = Tool(
            name="divider",
            description="Divide a by b",
            parameters={"type": "object", "properties": {
                "a": {"type": "number"}, "b": {"type": "number"}
            }, "required": ["a", "b"]},
            func=lambda a, b: a / b,
        )
        result = tool.execute(a=1, b=0)
        assert not result.success
        assert result.error is not None

    def test_to_schema(self):
        tool = Tool(
            name="my_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            func=lambda: None,
        )
        schema = tool.to_schema()
        assert schema["name"] == "my_tool"
        assert "description" in schema
        assert "parameters" in schema

    def test_stats(self):
        tool = Tool(
            name="counter",
            description="Returns count",
            parameters={"type": "object", "properties": {}},
            func=lambda: 42,
        )
        tool.execute()
        tool.execute()
        stats = tool.stats
        assert stats["call_count"] == 2
        assert stats["avg_execution_time"] >= 0


class TestToolManager:
    def setup_method(self):
        self.executor = ExecutionEnvironment()
        self.manager = ToolManager(executor=self.executor)

    def teardown_method(self):
        self.executor.cleanup()

    def test_register_and_get(self):
        tool = Tool(
            name="my_tool",
            description="test",
            parameters={"type": "object", "properties": {}},
            func=lambda: "ok",
        )
        self.manager.register(tool)
        retrieved = self.manager.get("my_tool")
        assert retrieved is not None
        assert retrieved.name == "my_tool"

    def test_list_tools(self):
        tool = Tool(
            name="tool_a",
            description="tool a",
            parameters={"type": "object", "properties": {}},
            func=lambda: None,
        )
        self.manager.register(tool)
        names = self.manager.list_tool_names()
        assert "tool_a" in names

    def test_execute_known_tool(self):
        tool = Tool(
            name="greet",
            description="greeting",
            parameters={"type": "object", "properties": {
                "name": {"type": "string"}
            }, "required": ["name"]},
            func=lambda name: f"Hello, {name}!",
        )
        self.manager.register(tool)
        result = self.manager.execute("greet", {"name": "OpenSage"})
        assert result.success
        assert "OpenSage" in result.output

    def test_execute_unknown_tool(self):
        result = self.manager.execute("nonexistent", {})
        assert not result.success
        assert "not found" in result.error.lower()

    def test_create_tool_from_code(self):
        source = """
def double(n):
    return n * 2
"""
        tool = self.manager.create_tool_from_code(
            name="double",
            description="Double a number",
            parameters={"type": "object", "properties": {
                "n": {"type": "number"}
            }, "required": ["n"]},
            source_code=source,
        )
        assert tool is not None
        assert tool.is_ai_generated
        result = tool.execute(n=5)
        assert result.success
        assert result.output == "10"

    def test_create_tool_invalid_syntax(self):
        tool = self.manager.create_tool_from_code(
            name="broken",
            description="broken tool",
            parameters={"type": "object", "properties": {}},
            source_code="def broken(: invalid syntax",
        )
        assert tool is None

    def test_remove_tool(self):
        tool = Tool(
            name="removable",
            description="test",
            parameters={"type": "object", "properties": {}},
            func=lambda: None,
        )
        self.manager.register(tool)
        assert "removable" in self.manager.list_tool_names()
        result = self.manager.remove_tool("removable")
        assert result
        assert "removable" not in self.manager.list_tool_names()


class TestExecutionEnvironment:
    def setup_method(self):
        self.env = ExecutionEnvironment()

    def teardown_method(self):
        self.env.cleanup()

    def test_run_python_simple(self):
        result = self.env.run_python_code("print('hello opensage')")
        assert result.success
        assert "hello opensage" in result.output

    def test_run_python_error(self):
        result = self.env.run_python_code("raise ValueError('test error')")
        assert not result.success

    def test_run_python_timeout(self):
        result = self.env.run_python_code("import time; time.sleep(100)", timeout=1)
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_run_shell_command(self):
        result = self.env.run_shell_command("echo 'test_output'")
        assert result.success
        assert "test_output" in result.output

    def test_write_and_read_file(self):
        content = "hello from opensage"
        write_result = self.env.write_file("test.txt", content)
        assert write_result.success

        read_result = self.env.read_file("test.txt")
        assert read_result.success
        assert content in read_result.output

    def test_read_nonexistent_file(self):
        result = self.env.read_file("definitely_does_not_exist.txt")
        assert not result.success

    def test_list_files(self):
        self.env.write_file("file1.py", "# file1")
        self.env.write_file("file2.py", "# file2")
        result = self.env.list_files()
        assert result.success
        assert "file1.py" in result.output
        assert "file2.py" in result.output


class TestSEToolkit:
    def setup_method(self):
        self.executor = ExecutionEnvironment()
        self.manager = ToolManager(executor=self.executor)
        tools = get_se_toolkit(self.executor)
        self.manager.register_many(tools)

    def teardown_method(self):
        self.executor.cleanup()

    def test_toolkit_registers_all_tools(self):
        names = self.manager.list_tool_names()
        expected = ["read_file", "write_file", "run_command", "run_python",
                    "search_code", "list_files", "apply_patch", "run_tests", "analyze_code"]
        for tool_name in expected:
            assert tool_name in names, f"Missing tool: {tool_name}"

    def test_write_and_read_file(self):
        write_result = self.manager.execute("write_file", {
            "path": "test_file.txt",
            "content": "line1\nline2\nline3"
        })
        assert write_result.success

        read_result = self.manager.execute("read_file", {
            "path": os.path.join(self.executor.work_dir, "test_file.txt"),
        })
        assert read_result.success
        assert "line1" in read_result.output

    def test_run_python(self):
        result = self.manager.execute("run_python", {
            "code": "print(1 + 1)"
        })
        assert result.success
        assert "2" in result.output

    def test_analyze_code_valid(self):
        result = self.manager.execute("analyze_code", {
            "code": "def hello():\n    return 'world'\n",
            "check_type": "syntax"
        })
        assert result.success
        assert "Valid Python" in result.output or "Syntax" in result.output

    def test_analyze_code_invalid(self):
        result = self.manager.execute("analyze_code", {
            "code": "def broken(: invalid",
            "check_type": "syntax"
        })
        assert result.success  # Tool itself succeeded
        assert "Error" in result.output or "error" in result.output.lower()

    def test_search_code(self):
        # Write a file to search in
        self.executor.write_file("searchable.py", "def compute_median(data):\n    pass\n")
        result = self.manager.execute("search_code", {
            "pattern": "compute_median",
            "path": self.executor.work_dir,
        })
        assert result.success
        assert "compute_median" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
