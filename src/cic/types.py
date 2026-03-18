"""CiC data types.

All public types returned by CiCClient methods are defined here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Estimated token usage for a single chat call.

    Because CiC routes through the claude CLI rather than a REST API, token
    counts are estimated from character counts (chars / 4 ≈ tokens). They are
    useful for rough budgeting and logging but should not be treated as exact.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class ToolCall:
    """A single tool call decision made by the model.

    The model populates ``name`` and ``arguments``; the caller is responsible
    for executing the tool and returning the result in the next chat turn.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def arguments_json(self) -> str:
        """Return arguments serialised as a JSON string."""
        return json.dumps(self.arguments)


@dataclass
class ChatResult:
    """Result of a single chat call.

    Exactly one of ``content`` or ``tool_calls`` will be non-empty on a
    well-formed response. If the model returns a plain text answer, ``content``
    holds it and ``tool_calls`` is empty. If the model decides to use tools,
    ``tool_calls`` is populated and ``content`` is ``None``.
    """

    content: str | None
    """Plain-text response from the model, or None if tool calls were made."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    """Tool call decisions. Non-empty only when the model chose to use tools."""

    model: str = ""
    """Identifier of the model that produced this response, e.g. ``cic/sonnet``."""

    usage: TokenUsage = field(default_factory=TokenUsage)
    """Estimated token usage for this call."""

    raw: dict[str, Any] = field(default_factory=dict)
    """Raw OpenAI-format response dict for inspection or compatibility."""

    @property
    def has_tool_calls(self) -> bool:
        """True if the model made at least one tool call."""
        return bool(self.tool_calls)

    def to_openai_dict(self) -> dict[str, Any]:
        """Return this result in OpenAI chat completion wire format.

        Useful when CiC is used as a drop-in replacement for openai.ChatCompletion.
        """
        return self.raw


# Type alias for the OpenAI-format messages list.
# Each message is a dict with at minimum a "role" key and a "content" or "tool_calls" key.
Messages = list[dict[str, Any]]

# Type alias for an OpenAI-format tool definition.
Tool = dict[str, Any]
