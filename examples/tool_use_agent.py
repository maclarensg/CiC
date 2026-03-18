"""Tool use agent example.

Demonstrates building a simple agent loop with CiC. The agent can read
files and list directories. It runs until it produces a final answer.

Run with:
    python examples/tool_use_agent.py
"""

from __future__ import annotations

import os
from typing import Any

from cic import CiCClient

# Tool definitions in OpenAI format
TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": "List the files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"},
            },
            "required": ["path"],
        },
    },
]


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool and return its output as a string."""
    if name == "read_file":
        path = arguments.get("path", "")
        try:
            with open(path) as f:
                return f.read()
        except OSError as e:
            return f"Error reading {path}: {e}"

    if name == "list_directory":
        path = arguments.get("path", ".")
        try:
            entries = os.listdir(path)
            return "\n".join(sorted(entries))
        except OSError as e:
            return f"Error listing {path}: {e}"

    return f"Unknown tool: {name}"


def run_agent(question: str, max_turns: int = 10) -> str:
    """Run an agent loop until it produces a final answer.

    Args:
        question: The user's question or task.
        max_turns: Maximum number of tool-call turns before giving up.

    Returns:
        The agent's final text answer.
    """
    client = CiCClient(model="sonnet")
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    for turn in range(max_turns):
        result = client.chat(messages, tools=TOOLS)

        if not result.has_tool_calls:
            return result.content or "[no response]"

        print(f"  Turn {turn + 1}: {len(result.tool_calls)} tool call(s)")

        # Add the assistant's tool call decisions to history
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_json()},
                }
                for tc in result.tool_calls
            ],
        })

        # Execute each tool and add results to history
        for tc in result.tool_calls:
            print(f"    Calling {tc.name}({tc.arguments})")
            output = execute_tool(tc.name, tc.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": output,
            })

    return "[max turns reached]"


if __name__ == "__main__":
    answer = run_agent("List the files in /tmp and tell me how many there are.")
    print("\nFinal answer:", answer)
