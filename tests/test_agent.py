"""
Tests for SageAgent - focuses on components that don't require API calls.
Tests that require LLM use mock objects.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensage.core.agent import SageAgent
from opensage.tools.manager import ToolManager
from opensage.tools.executor import ExecutionEnvironment
from opensage.tools.se_toolkit import get_se_toolkit
from opensage.memory.hierarchical import HierarchicalMemory
from opensage.llm.base import LLMResponse, ToolCall


def make_mock_llm(responses=None):
    """Create a mock LLM client for testing."""
    mock = MagicMock()
    responses = responses or []
    call_count = [0]

    def mock_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        if responses:
            return responses[idx]
        return LLMResponse(content="Task complete.", tool_calls=[])

    mock.chat.side_effect = mock_chat
    return mock


class TestSageAgentBasics:
    def setup_method(self):
        self.executor = ExecutionEnvironment()
        self.tool_manager = ToolManager(executor=self.executor)
        tools = get_se_toolkit(self.executor)
        self.tool_manager.register_many(tools)
        self.memory = HierarchicalMemory(agent_id="test_agent")

    def teardown_method(self):
        self.executor.cleanup()

    def test_agent_creation(self):
        mock_llm = make_mock_llm()
        agent = SageAgent(
            name="test_agent",
            system_prompt="You are a test agent.",
            llm=mock_llm,
            tool_manager=self.tool_manager,
            memory=self.memory,
            executor=self.executor,
        )
        assert agent.name == "test_agent"

    def test_meta_tools_registered(self):
        """Verify OpenSage meta-tools are registered when topology is enabled."""
        mock_llm = make_mock_llm()
        agent = SageAgent(
            name="test_agent",
            system_prompt="test",
            llm=mock_llm,
            tool_manager=self.tool_manager,
            memory=self.memory,
            executor=self.executor,
            enable_topology=True,
        )
        tool_names = agent.tool_manager.list_tool_names()
        assert "create_sub_agent" in tool_names
        assert "invoke_sub_agent" in tool_names
        assert "create_tool" in tool_names
        assert "store_memory" in tool_names
        assert "retrieve_memory" in tool_names
        assert "use_ensemble" in tool_names

    def test_no_meta_tools_when_disabled(self):
        """Meta-tools should NOT be registered when topology is disabled."""
        mock_llm = make_mock_llm()
        agent = SageAgent(
            name="test_agent",
            system_prompt="test",
            llm=mock_llm,
            tool_manager=self.tool_manager,
            memory=self.memory,
            executor=self.executor,
            enable_topology=False,
        )
        tool_names = agent.tool_manager.list_tool_names()
        assert "create_sub_agent" not in tool_names

    def test_run_simple_task(self):
        """Test a task that completes without tool calls."""
        mock_llm = make_mock_llm([
            LLMResponse(content="The answer is 42.", tool_calls=[])
        ])
        agent = SageAgent(
            name="test",
            system_prompt="You are a helpful agent.",
            llm=mock_llm,
            tool_manager=self.tool_manager,
            memory=self.memory,
            executor=self.executor,
            enable_topology=False,
        )
        result = agent.run("What is the answer?")
        assert "42" in result

    def test_run_with_tool_call(self):
        """Test a task that uses a tool call."""
        from opensage.tools.base import Tool

        results_tracker = []

        def my_tool(value: str):
            results_tracker.append(value)
            return f"processed: {value}"

        self.tool_manager.register(Tool(
            name="process",
            description="Process a value",
            parameters={"type": "object", "properties": {
                "value": {"type": "string"}
            }, "required": ["value"]},
            func=my_tool,
        ))

        mock_llm = make_mock_llm([
            # First call: use a tool
            LLMResponse(
                content="I'll process this.",
                tool_calls=[ToolCall(id="call1", name="process", arguments={"value": "hello"})],
            ),
            # Second call: no more tool calls
            LLMResponse(content="Done. Result: processed: hello", tool_calls=[]),
        ])

        agent = SageAgent(
            name="test",
            system_prompt="You are a helpful agent.",
            llm=mock_llm,
            tool_manager=self.tool_manager,
            memory=self.memory,
            executor=self.executor,
            enable_topology=False,
        )

        result = agent.run("Process the value 'hello'")
        assert "hello" in results_tracker
        assert "processed" in result or "Done" in result

    def test_create_sub_agent(self):
        """Test dynamic sub-agent creation."""
        mock_llm = make_mock_llm()
        agent = SageAgent(
            name="parent",
            system_prompt="test",
            llm=mock_llm,
            tool_manager=self.tool_manager,
            memory=self.memory,
            executor=self.executor,
            enable_topology=True,
        )
        result = agent._create_sub_agent(
            name="child_agent",
            description="A child agent for testing",
            system_prompt="You are a child agent.",
            tool_names=["run_python"],
        )
        assert "child_agent" in result
        assert "child_agent" in agent._sub_agents

    def test_create_tool_dynamically(self):
        """Test AI-generated tool creation."""
        mock_llm = make_mock_llm()
        agent = SageAgent(
            name="test",
            system_prompt="test",
            llm=mock_llm,
            tool_manager=self.tool_manager,
            memory=self.memory,
            executor=self.executor,
        )
        import json
        params = json.dumps({
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": ["x"]
        })
        result = agent._create_tool(
            name="square",
            description="Compute square of a number",
            parameters_json=params,
            source_code="def square(x): return x * x",
        )
        assert "created" in result.lower() or "success" in result.lower()
        assert "square" in agent.tool_manager.list_tool_names()

    def test_memory_integration(self):
        """Test that agent stores memories during task execution."""
        mock_llm = make_mock_llm([
            LLMResponse(content="Task done.", tool_calls=[])
        ])
        agent = SageAgent(
            name="test",
            system_prompt="test",
            llm=mock_llm,
            tool_manager=self.tool_manager,
            memory=self.memory,
            executor=self.executor,
        )
        agent.run("Remember this task")
        summary = agent.memory.get_graph_summary()
        # Memory should have grown
        assert summary["total_nodes"] >= 2

    def test_get_status(self):
        """Test agent status reporting."""
        mock_llm = make_mock_llm()
        agent = SageAgent(
            name="test",
            system_prompt="test",
            llm=mock_llm,
            tool_manager=self.tool_manager,
            memory=self.memory,
            executor=self.executor,
        )
        status = agent.get_status()
        assert status["name"] == "test"
        assert "tools" in status
        assert "sub_agents" in status
        assert "memory_summary" in status


class TestVerticalTopology:
    def setup_method(self):
        self.executor = ExecutionEnvironment()
        self.tool_manager = ToolManager(executor=self.executor)
        tools = get_se_toolkit(self.executor)
        self.tool_manager.register_many(tools)
        self.memory = HierarchicalMemory(agent_id="test")

    def teardown_method(self):
        self.executor.cleanup()

    def test_parse_decomposition_valid_json(self):
        from opensage.topology.vertical import VerticalTopology, SubTaskSpec

        mock_llm = make_mock_llm()
        agent = SageAgent(
            name="parent",
            system_prompt="test",
            llm=mock_llm,
            tool_manager=self.tool_manager,
            memory=self.memory,
            executor=self.executor,
        )
        v_topo = VerticalTopology(parent_agent=agent)

        # Test parsing a valid decomposition
        test_json = """```json
[
  {"description": "Analyze code", "agent_name": "analyzer", "agent_role": "code analyst", "required_tools": ["read_file"]},
  {"description": "Fix bugs", "agent_name": "fixer", "agent_role": "developer", "required_tools": ["write_file"]}
]
```"""
        specs = v_topo._parse_decomposition(test_json, "Fix bugs in code")
        assert len(specs) == 2
        assert specs[0].agent_name == "analyzer"
        assert specs[1].agent_name == "fixer"

    def test_parse_decomposition_invalid_json_fallback(self):
        from opensage.topology.vertical import VerticalTopology

        mock_llm = make_mock_llm()
        agent = SageAgent(
            name="parent",
            system_prompt="test",
            llm=mock_llm,
            tool_manager=self.tool_manager,
            memory=self.memory,
            executor=self.executor,
        )
        v_topo = VerticalTopology(parent_agent=agent)

        # Invalid JSON should produce a single fallback task
        specs = v_topo._parse_decomposition("not valid json at all", "Fix bugs")
        assert len(specs) == 1
        assert specs[0].description == "Fix bugs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
