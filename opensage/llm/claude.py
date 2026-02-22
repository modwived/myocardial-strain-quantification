"""
Claude/Anthropic LLM client implementation for OpenSage.

Uses the Anthropic SDK to interact with Claude models.
Supports tool use (function calling) for agent actions.
"""

import json
import os
from typing import Any, Dict, List, Optional

import anthropic

from opensage.llm.base import LLMClient, LLMResponse, ToolCall


class ClaudeClient(LLMClient):
    """
    Anthropic Claude LLM client.

    Implements the LLMClient interface using the Anthropic API
    with support for tool use, conversation history, and streaming.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5",
    ):
        """
        Initialize the Claude client.

        Args:
            api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        system: str = "",
        tools: Optional[List[Dict]] = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Chat with Claude using the Anthropic Messages API.

        Handles both simple text responses and tool_use responses,
        accumulating multiple text blocks if present.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [self.get_tool_schema(t) for t in tools]

        response = self.client.messages.create(**kwargs)

        # Extract text content
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw=response,
        )

    def get_tool_schema(self, tool_def: Dict) -> Dict:
        """
        Convert generic tool definition to Anthropic's expected format.

        Anthropic expects:
          {name, description, input_schema: {type, properties, required}}
        """
        schema = {
            "name": tool_def["name"],
            "description": tool_def.get("description", ""),
            "input_schema": {
                "type": "object",
                "properties": tool_def.get("parameters", {}).get("properties", {}),
                "required": tool_def.get("parameters", {}).get("required", []),
            },
        }
        return schema

    def format_tool_result(self, tool_call_id: str, result: Any) -> Dict:
        """
        Format a tool result for inclusion in the message history.

        Returns a user message with tool_result content block.
        """
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": str(result) if not isinstance(result, str) else result,
                }
            ],
        }
