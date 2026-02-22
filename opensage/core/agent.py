"""
SageAgent: The core agent class for OpenSage.

SageAgent implements the full agentic loop described in the paper:
1. Receives a task
2. Uses LLM with tools to reason and act
3. Can create sub-agents dynamically (vertical topology)
4. Uses hierarchical graph-based memory throughout
5. Generates tools on the fly as needed

The agent's built-in tools include:
- All SE toolkit tools (read/write/execute/search)
- create_sub_agent: Create a new specialized sub-agent at runtime
- create_tool: Write and register a new tool from AI-generated code
- store_memory / retrieve_memory: Explicit memory management
- invoke_sub_agent: Run a previously created sub-agent
- use_ensemble: Trigger horizontal ensemble mode

This implements the paper's "self-programming" aspect: the agent actively
programs its own toolset and agent topology to fit the task.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from opensage.llm.base import LLMClient, LLMResponse, ToolCall
from opensage.memory.hierarchical import HierarchicalMemory
from opensage.memory.memory_agent import MemoryAgent
from opensage.tools.base import Tool, ToolResult
from opensage.tools.executor import ExecutionEnvironment
from opensage.tools.manager import ToolManager


class SageAgent:
    """
    The primary agent class in OpenSage.

    SageAgent drives the agent execution loop, managing:
    - LLM-based reasoning with tool calling
    - Dynamic sub-agent creation and management
    - Hierarchical memory read/write
    - Dynamic tool creation (self-programming)
    - Both vertical and horizontal topology execution
    """

    # Maximum agentic loop iterations before forcing a conclusion
    DEFAULT_MAX_ITERATIONS = 15

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: LLMClient,
        tool_manager: ToolManager,
        memory: Optional[HierarchicalMemory] = None,
        executor: Optional[ExecutionEnvironment] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        enable_topology: bool = True,
        verbose: bool = False,
    ):
        """
        Create a SageAgent.

        Args:
            name: Agent identifier
            system_prompt: The agent's role and instructions
            llm: LLM client for reasoning
            tool_manager: Manages available tools
            memory: Hierarchical graph memory (created if None)
            executor: Execution environment (created if None)
            max_iterations: Max agent loop steps
            enable_topology: Allow creating sub-agents (vertical/horizontal)
            verbose: Print agent reasoning steps
        """
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tool_manager = tool_manager
        self.memory = memory or HierarchicalMemory(agent_id=name)
        self.executor = executor or ExecutionEnvironment()
        self.max_iterations = max_iterations
        self.enable_topology = enable_topology
        self.verbose = verbose

        # Sub-agents created by this agent at runtime
        self._sub_agents: Dict[str, "SageAgent"] = {}

        # Memory agent for background maintenance
        self._memory_agent = MemoryAgent(self.memory, self.llm)

        # Register OpenSage meta-tools (create_agent, create_tool, etc.)
        if enable_topology:
            self._register_meta_tools()

    # ------------------------------------------------------------------ #
    # Meta-tools: OpenSage's "self-programming" capabilities              #
    # ------------------------------------------------------------------ #

    def _register_meta_tools(self) -> None:
        """Register the OpenSage-specific meta-tools."""

        # create_sub_agent: Runtime agent creation (vertical topology)
        def create_sub_agent(
            name: str,
            description: str,
            system_prompt: str,
            tools: Optional[List[str]] = None,
        ) -> str:
            return self._create_sub_agent(name, description, system_prompt, tools or [])

        self.tool_manager.register(
            Tool(
                name="create_sub_agent",
                description=(
                    "Dynamically create a specialized sub-agent to handle a specific sub-task. "
                    "The sub-agent will have its own memory and tailored tools. "
                    "Use when you need specialized expertise for a portion of the task."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Unique name for the sub-agent (e.g. 'code_analyzer')",
                        },
                        "description": {
                            "type": "string",
                            "description": "What this sub-agent specializes in",
                        },
                        "system_prompt": {
                            "type": "string",
                            "description": "Detailed role and instructions for the sub-agent",
                        },
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of tool names for this agent. "
                                f"Available: {list(self.tool_manager.tools.keys())}"
                            ),
                        },
                    },
                    "required": ["name", "description", "system_prompt"],
                },
                func=create_sub_agent,
            )
        )

        # invoke_sub_agent: Run a previously created sub-agent
        def invoke_sub_agent(agent_name: str, task: str) -> str:
            return self._invoke_sub_agent(agent_name, task)

        self.tool_manager.register(
            Tool(
                name="invoke_sub_agent",
                description=(
                    "Invoke a previously created sub-agent to solve a task. "
                    "The sub-agent will use its specialized tools and return its result."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Name of the sub-agent to invoke",
                        },
                        "task": {
                            "type": "string",
                            "description": "Task description for the sub-agent",
                        },
                    },
                    "required": ["agent_name", "task"],
                },
                func=invoke_sub_agent,
            )
        )

        # create_tool: AI writes a new tool at runtime
        def create_tool(
            name: str,
            description: str,
            parameters_json: str,
            source_code: str,
            function_name: Optional[str] = None,
        ) -> str:
            return self._create_tool(
                name, description, parameters_json, source_code, function_name
            )

        self.tool_manager.register(
            Tool(
                name="create_tool",
                description=(
                    "Write and register a new Python tool at runtime. "
                    "The tool will be immediately available for use. "
                    "Use this when you need a capability not in your current toolset."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Tool name (snake_case, e.g. 'parse_json')",
                        },
                        "description": {
                            "type": "string",
                            "description": "What the tool does",
                        },
                        "parameters_json": {
                            "type": "string",
                            "description": "JSON Schema for tool parameters as a JSON string",
                        },
                        "source_code": {
                            "type": "string",
                            "description": "Python function definition(s) implementing the tool",
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Name of the main function (defaults to tool name)",
                        },
                    },
                    "required": ["name", "description", "parameters_json", "source_code"],
                },
                func=create_tool,
            )
        )

        # store_memory: Explicit memory storage
        def store_memory(content: str, memory_type: str = "observation") -> str:
            mem_map = {
                "observation": self.memory.observe,
                "fact": self.memory.record_fact,
                "plan": self.memory.plan,
                "result": self.memory.record_result,
                "code": lambda c: self.memory.record_code(c, description="AI-stored code"),
            }
            store_fn = mem_map.get(memory_type, self.memory.observe)
            node_id = store_fn(content)
            return f"Stored in memory (id={node_id})"

        self.tool_manager.register(
            Tool(
                name="store_memory",
                description="Explicitly store information in the hierarchical memory for later retrieval.",
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Information to store",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["observation", "fact", "plan", "result", "code"],
                            "description": "Category of memory",
                            "default": "observation",
                        },
                    },
                    "required": ["content"],
                },
                func=store_memory,
            )
        )

        # retrieve_memory: Query the memory graph
        def retrieve_memory(query: str, max_results: int = 5) -> str:
            nodes = self.memory.retrieve(query, max_results=max_results)
            if not nodes:
                return "No relevant memories found"
            lines = [f"Found {len(nodes)} relevant memories:"]
            for node in nodes:
                lines.append(f"  [{node.memory_type.value}] {node.content[:200]}")
            return "\n".join(lines)

        self.tool_manager.register(
            Tool(
                name="retrieve_memory",
                description="Retrieve relevant information from the hierarchical memory graph.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                func=retrieve_memory,
            )
        )

        # use_ensemble: Trigger horizontal topology
        def use_ensemble(task: str, n_agents: int = 2) -> str:
            return self._use_ensemble(task, n_agents)

        self.tool_manager.register(
            Tool(
                name="use_ensemble",
                description=(
                    "Run multiple specialized agents in parallel on the same task using "
                    "different strategies, then integrate the best solution. "
                    "Use for high-stakes tasks where multiple perspectives improve quality."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Task to solve with the ensemble",
                        },
                        "n_agents": {
                            "type": "integer",
                            "description": "Number of ensemble agents (2-3)",
                            "default": 2,
                        },
                    },
                    "required": ["task"],
                },
                func=use_ensemble,
            )
        )

    # ------------------------------------------------------------------ #
    # Meta-tool implementations                                            #
    # ------------------------------------------------------------------ #

    def _create_sub_agent(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tool_names: List[str],
    ) -> str:
        """Create and register a specialized sub-agent."""
        from opensage.memory.hierarchical import HierarchicalMemory
        from opensage.tools.manager import ToolManager

        # Build tool manager with requested tools
        sub_tools = ToolManager(executor=self.executor)
        for tool_name in tool_names:
            tool = self.tool_manager.get(tool_name)
            if tool:
                sub_tools.register(tool)

        # If no specific tools, inherit all parent tools except meta-tools
        if not tool_names:
            meta_tool_names = {
                "create_sub_agent", "invoke_sub_agent", "create_tool",
                "store_memory", "retrieve_memory", "use_ensemble"
            }
            for tn, tool in self.tool_manager.tools.items():
                if tn not in meta_tool_names:
                    sub_tools.register(tool)

        sub_memory = HierarchicalMemory(agent_id=name)

        sub_agent = SageAgent(
            name=name,
            system_prompt=f"{system_prompt}\n\nYou are a specialized sub-agent. Complete your assigned task thoroughly.",
            llm=self.llm,
            tool_manager=sub_tools,
            memory=sub_memory,
            executor=self.executor,
            max_iterations=8,
            enable_topology=False,  # Sub-agents don't recurse by default
            verbose=self.verbose,
        )

        self._sub_agents[name] = sub_agent

        # Record in memory
        self.memory.record_agent_created(
            agent_name=name,
            agent_config={
                "description": description,
                "tools": tool_names,
                "system_prompt": system_prompt[:200],
            },
            task_id=self.memory._current_task_id,
        )

        if self.verbose:
            print(f"  [OpenSage] Created sub-agent: {name} with tools: {tool_names or 'all'}")

        return f"Sub-agent '{name}' created successfully. Tools: {tool_names or list(sub_tools.list_tool_names())}"

    def _invoke_sub_agent(self, agent_name: str, task: str) -> str:
        """Invoke a registered sub-agent."""
        agent = self._sub_agents.get(agent_name)
        if agent is None:
            return f"Sub-agent '{agent_name}' not found. Created agents: {list(self._sub_agents.keys())}"

        if self.verbose:
            print(f"  [OpenSage] Invoking sub-agent: {agent_name}")

        # Give sub-agent access to parent's current task context
        recent_ctx = self.memory.get_recent_context(max_tokens=1000)
        enriched_task = f"{task}\n\nParent memory context:\n{recent_ctx}"

        result = agent.run(enriched_task)

        # Import result into parent memory
        self.memory.observe(
            f"Sub-agent '{agent_name}' result:\n{result}",
            task_id=self.memory._current_task_id,
            importance=0.9,
        )

        return result

    def _create_tool(
        self,
        name: str,
        description: str,
        parameters_json: str,
        source_code: str,
        function_name: Optional[str] = None,
    ) -> str:
        """Create a new AI-generated tool and register it."""
        try:
            parameters = json.loads(parameters_json)
        except json.JSONDecodeError:
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            }

        tool = self.tool_manager.create_tool_from_code(
            name=name,
            description=description,
            parameters=parameters,
            source_code=source_code,
            function_name=function_name,
        )

        if tool is None:
            return f"Failed to create tool '{name}'. Check source code syntax."

        # Record in memory
        self.memory.record_tool_created(
            tool_name=name,
            tool_spec={"description": description, "source": source_code[:500]},
            task_id=self.memory._current_task_id,
        )

        if self.verbose:
            print(f"  [OpenSage] AI created new tool: {name}")

        return f"Tool '{name}' created and registered successfully. You can now call it directly."

    def _use_ensemble(self, task: str, n_agents: int = 2) -> str:
        """Trigger horizontal ensemble for a task."""
        from opensage.topology.horizontal import HorizontalTopology

        if self.verbose:
            print(f"  [OpenSage] Launching horizontal ensemble ({n_agents} agents)...")

        ensemble = HorizontalTopology(parent_agent=self, n_agents=n_agents)
        result = ensemble.execute(task)

        return (
            f"Ensemble complete ({len(result.approach_results)} agents).\n"
            f"Best solution:\n{result.best_result}\n\n"
            f"Synthesis:\n{result.consensus_summary}"
        )

    # ------------------------------------------------------------------ #
    # Main agent execution loop                                            #
    # ------------------------------------------------------------------ #

    def run(self, task: str) -> str:
        """
        Execute a task using the agentic reasoning loop.

        This is the main entry point. The agent:
        1. Registers the task in memory
        2. Runs the LLM → tool use → LLM loop
        3. Each iteration may call tools (including meta-tools)
        4. Loop terminates when the LLM produces a final answer
           (no tool calls) or max_iterations is reached

        Args:
            task: Natural language task description

        Returns:
            The agent's final response
        """
        task_id = self.memory.start_task(task)
        conversation: List[Dict[str, Any]] = [{"role": "user", "content": task}]

        if self.verbose:
            print(f"\n[{self.name}] Starting task: {task[:100]}...")

        for iteration in range(self.max_iterations):
            # Build system prompt with current memory context
            memory_context = self.memory.get_recent_context(max_tokens=1500)
            full_system = self._build_system_prompt(memory_context)

            # Get all tool schemas
            tools = self.tool_manager.get_tool_for_llm()

            # Call LLM
            try:
                response = self.llm.chat(
                    messages=conversation,
                    system=full_system,
                    tools=tools if tools else None,
                    max_tokens=4096,
                )
            except Exception as e:
                return f"LLM error: {e}"

            if self.verbose and response.content:
                print(f"  [{self.name}] Iteration {iteration+1}: {response.content[:150]}...")

            # If no tool calls, the agent is done
            if not response.tool_calls:
                final_answer = response.content
                self.memory.record_result(final_answer, task_id=task_id)

                # Run memory maintenance in background
                try:
                    self._memory_agent.summarize_long_nodes()
                except Exception:
                    pass

                return final_answer

            # Add assistant message to conversation
            assistant_content = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            conversation.append({"role": "assistant", "content": assistant_content})

            # Execute all tool calls and collect results
            tool_results_content = []
            for tool_call in response.tool_calls:
                if self.verbose:
                    print(f"  [{self.name}] → {tool_call.name}({list(tool_call.arguments.keys())})")

                # Execute the tool
                tool_result = self.tool_manager.execute(tool_call.name, tool_call.arguments)
                result_str = tool_result.to_str()

                if self.verbose:
                    print(f"  [{self.name}] ← {result_str[:100]}...")

                # Store tool result in memory
                self.memory.observe(
                    f"Tool '{tool_call.name}': {result_str[:500]}",
                    task_id=task_id,
                    importance=0.7,
                )

                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result_str,
                })

            # Add tool results as user message
            conversation.append({
                "role": "user",
                "content": tool_results_content,
            })

        # Max iterations reached — force a conclusion
        if self.verbose:
            print(f"  [{self.name}] Max iterations reached, generating conclusion...")

        conversation.append({
            "role": "user",
            "content": "You have reached the iteration limit. Please provide your best answer based on what you've found so far.",
        })

        try:
            final_response = self.llm.chat(
                messages=conversation,
                system=self._build_system_prompt(""),
                max_tokens=2048,
            )
            final = final_response.content
        except Exception as e:
            final = f"Agent reached max iterations. Last error: {e}"

        self.memory.record_result(final, task_id=task_id)
        return final

    def _build_system_prompt(self, memory_context: str) -> str:
        """Build the full system prompt including memory context."""
        parts = [self.system_prompt]

        if memory_context:
            parts.append(
                "\n=== Memory Context ===\n"
                + memory_context
                + "\n=== End Memory Context ==="
            )

        parts.append(
            "\nYou have access to tools. Use them to complete your task. "
            "When you have gathered enough information and can provide a final answer, "
            "respond without calling any tools."
        )

        available_agents = list(self._sub_agents.keys())
        if available_agents:
            parts.append(f"\nRegistered sub-agents: {available_agents}")

        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Status / inspection                                                  #
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of this agent."""
        return {
            "name": self.name,
            "sub_agents": list(self._sub_agents.keys()),
            "tools": self.tool_manager.list_tool_names(),
            "memory_summary": self.memory.get_graph_summary(),
            "ai_generated_tools": [
                name for name, tool in self.tool_manager.tools.items()
                if tool.is_ai_generated
            ],
        }

    def __repr__(self) -> str:
        return (
            f"SageAgent(name={self.name!r}, "
            f"tools={self.tool_manager.list_tool_names()}, "
            f"sub_agents={list(self._sub_agents.keys())})"
        )
