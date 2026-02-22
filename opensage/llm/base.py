"""
Abstract LLM client interface for OpenSage.
Allows swapping out different LLM backends (Claude, GPT, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """Represents an LLM-initiated tool call."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Structured response from an LLM."""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        system: str = "",
        tools: Optional[List[Dict]] = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send messages to the LLM and get a response.

        Args:
            messages: Conversation history in {role, content} format
            system: System prompt for the LLM
            tools: Tool definitions in the LLM's expected format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse with content and optional tool calls
        """
        ...

    @abstractmethod
    def get_tool_schema(self, tool_def: Dict) -> Dict:
        """
        Convert a generic tool definition to the LLM's expected schema.

        Args:
            tool_def: Generic tool definition with name, description, parameters

        Returns:
            Tool schema in the LLM's expected format
        """
        ...
