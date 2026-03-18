"""CiC — Context in Claude Code.

Build AI agents using your Claude Pro/Max subscription.
Zero per-token costs. No API keys required.

Quick start::

    from cic import CiCClient

    client = CiCClient(model="sonnet")
    result = client.chat([{"role": "user", "content": "Hello!"}])
    print(result.content)

With tools (agent loop)::

    from cic import CiCClient

    client = CiCClient(model="sonnet")
    tools = [
        {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]

    messages = [{"role": "user", "content": "Read /etc/hostname and tell me its contents."}]
    result = client.chat(messages, tools=tools)

    while result.has_tool_calls:
        for tc in result.tool_calls:
            output = your_tool_executor(tc.name, tc.arguments)
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments_json()},
                    }
                ],
            })
            messages.append({"role": "tool", "name": tc.name, "content": output})
        result = client.chat(messages, tools=tools)

    print(result.content)

With smart routing::

    from cic import CiCClient

    client = CiCClient(routing={"simple": "haiku", "moderate": "sonnet", "complex": "opus"})
    client.set_complexity("complex")
    result = client.chat(messages)  # uses Opus automatically
"""

from .client import CiCClient
from .exceptions import (
    CiCError,
    ClaudeNotFoundError,
    ClaudeSubprocessError,
    ClaudeTimeoutError,
    ResponseParseError,
)
from .routing import CiCRouter, DEFAULT_ROUTING
from .types import ChatResult, Messages, Tool, ToolCall, TokenUsage

__all__ = [
    # Main client
    "CiCClient",
    # Routing
    "CiCRouter",
    "DEFAULT_ROUTING",
    # Types
    "ChatResult",
    "ToolCall",
    "TokenUsage",
    "Messages",
    "Tool",
    # Exceptions
    "CiCError",
    "ClaudeNotFoundError",
    "ClaudeSubprocessError",
    "ClaudeTimeoutError",
    "ResponseParseError",
]

__version__ = "0.1.0"
