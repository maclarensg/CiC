"""Internal utilities for prompt building and response parsing.

These helpers are not part of the public API but are tested directly
because the prompt/parse logic is critical to correct behaviour.
"""

from __future__ import annotations

import json
import re
from typing import Any

# System prompt injected at the top of every CiC call.
# Instructs the model to respond with exactly one JSON object.
_SYSTEM_PROMPT = """\
You are an AI agent. You have access to tools described below.

To call one or more tools, respond with EXACTLY this JSON (no markdown, no extra text):
{"tool_calls": [{"id": "call_1", "name": "tool_name", "arguments": {"param": "value"}}]}

You may include multiple tool calls in a single response.

When you have completed the task and need no more tools, respond with:
{"response": "your final answer here"}

NEVER mix tool_calls and response in the same JSON object.
"""

_TOOL_SECTION_TEMPLATE = """\
AVAILABLE TOOLS:
{tool_descriptions}

"""

_MAX_TOOL_ARG_CHARS = 200
_MAX_TOOL_RESULT_CHARS = 2000


def build_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> str:
    """Flatten messages and tool descriptions into a single prompt string.

    The claude CLI receives the full conversation as one prompt. This
    function serialises system instructions, available tools (if any), and
    the conversation history into that single string.

    Args:
        messages: OpenAI-format messages list.
        tools: OpenAI-format tool definitions, or None.

    Returns:
        A single UTF-8 string suitable for piping to ``claude --print``.
    """
    parts: list[str] = [_SYSTEM_PROMPT]

    if tools:
        desc = _build_tool_descriptions(tools)
        parts.append(_TOOL_SECTION_TEMPLATE.format(tool_descriptions=desc))

    parts.append("CONVERSATION HISTORY:")
    parts.append(_format_messages(messages))
    parts.append("\nWhat is your next action? Respond with JSON only.")

    return "\n".join(parts)


def _build_tool_descriptions(tools: list[dict[str, Any]]) -> str:
    """Convert OpenAI tool schemas into a compact, human-readable list.

    The model reads this to understand what tools are available and what
    parameters they accept.

    Args:
        tools: List of OpenAI-format tool dicts, each with a ``"function"`` key.

    Returns:
        A newline-separated string, one tool per line.
    """
    lines: list[str] = []
    for tool in tools:
        fn = tool.get("function", tool)  # Support both wrapped and unwrapped schemas
        name = fn.get("name", "unknown")
        description = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))

        param_parts: list[str] = []
        for param_name, param_schema in params.items():
            param_type = param_schema.get("type", "any")
            suffix = "" if param_name in required else "?"
            param_parts.append(f"{param_name}{suffix}: {param_type}")

        param_str = ", ".join(param_parts) if param_parts else ""
        lines.append(f"- {name}({param_str}): {description}")

    return "\n".join(lines)


def _format_messages(messages: list[dict[str, Any]]) -> str:
    """Serialise an OpenAI messages array to a readable conversation transcript.

    Tool results and long assistant arguments are truncated to keep the
    prompt from ballooning when a long agentic loop is replayed.

    Args:
        messages: OpenAI-format messages list.

    Returns:
        A double-newline-separated string representing the conversation.
    """
    parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            text = _extract_text(content)
            if text:
                parts.append(f"[System]: {text}")

        elif role == "user":
            text = _extract_text(content)
            parts.append(f"[User]: {text}")

        elif role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                call_strs: list[str] = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tc_name = fn.get("name", "unknown")
                    tc_args = fn.get("arguments", "{}")
                    if len(tc_args) > _MAX_TOOL_ARG_CHARS:
                        tc_args = tc_args[:_MAX_TOOL_ARG_CHARS] + "..."
                    call_strs.append(f"{tc_name}({tc_args})")
                parts.append(f"[Assistant called]: {', '.join(call_strs)}")
            elif content:
                text = _extract_text(content)
                parts.append(f"[Assistant]: {text}")

        elif role == "tool":
            tool_name = msg.get("name", "unknown")
            result_text = content if isinstance(content, str) else str(content)
            if len(result_text) > _MAX_TOOL_RESULT_CHARS:
                result_text = result_text[:_MAX_TOOL_RESULT_CHARS] + "\n[... truncated ...]"
            parts.append(f"[Tool Result ({tool_name})]: {result_text}")

    return "\n\n".join(parts)


def _extract_text(content: str | list[dict[str, Any]] | Any) -> str:
    """Extract plain text from an OpenAI content value.

    Handles both plain string content and multimodal content arrays.

    Args:
        content: Either a string or a list of content blocks.

    Returns:
        A plain text string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(texts)
    return str(content)


def parse_cli_output(stdout: str) -> dict[str, Any]:
    """Parse the claude CLI JSON envelope and extract an OpenAI-format response.

    The claude CLI with ``--output-format json`` returns an envelope like::

        {"type": "result", "result": "<Claude's actual output>", "is_error": false, ...}

    Claude's actual output is a JSON string (tool_calls or response object),
    which we parse and convert to OpenAI wire format.

    Args:
        stdout: Raw stdout from the claude subprocess.

    Returns:
        An OpenAI-format chat completion dict.

    Raises:
        ResponseParseError: If the envelope cannot be parsed.
        ClaudeSubprocessError: If ``is_error`` is True in the envelope.
    """
    # Import here to avoid circular at module level (utils imported by client)
    from .exceptions import ClaudeSubprocessError, ResponseParseError

    stdout = stdout.strip()
    if not stdout:
        return _make_content_response("[CiC] No output from claude subprocess")

    # Parse outer CLI envelope
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ResponseParseError(
            f"Could not parse CLI envelope: {exc}",
            raw=stdout[:500],
        ) from exc

    # Check for CLI-level error flag
    if envelope.get("is_error"):
        error_text = envelope.get("result", "Unknown CLI error")
        raise ClaudeSubprocessError(f"claude CLI error: {error_text[:500]}")

    # Extract the inner result string (what Claude actually wrote)
    result_text = envelope.get("result", "")
    if not result_text:
        return _make_content_response("[CiC] Empty result in CLI envelope")

    # Try to parse inner JSON
    inner: dict[str, Any] = {}
    try:
        # Claude sometimes wraps its response in markdown code fences; strip them.
        cleaned = _strip_code_fence(result_text)
        inner = json.loads(cleaned)
    except json.JSONDecodeError:
        # Not JSON — treat as a plain text answer
        return _make_content_response(result_text)

    # Convert to OpenAI format
    tool_calls_raw = inner.get("tool_calls")
    if tool_calls_raw and isinstance(tool_calls_raw, list):
        return _make_tool_call_response(tool_calls_raw)

    # Final answer
    response_text = inner.get("response", inner.get("result", str(inner)))
    return _make_content_response(response_text)


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences from a string if present.

    Some model outputs wrap JSON in triple backticks. This strips the fences
    so the inner content can be parsed as JSON.

    Args:
        text: Possibly fence-wrapped text.

    Returns:
        The inner content if fences are found, otherwise the original text.
    """
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if match:
        return match.group(1)
    return stripped


def _make_content_response(text: str) -> dict[str, Any]:
    """Build an OpenAI wire format response with plain text content.

    Args:
        text: The assistant's text response.

    Returns:
        An OpenAI chat completion dict.
    """
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": "stop",
        }],
    }


def _make_tool_call_response(tool_calls_raw: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert a CiC tool_calls list to OpenAI wire format.

    Args:
        tool_calls_raw: List of tool call dicts from Claude's JSON response.

    Returns:
        An OpenAI chat completion dict with tool_calls.
    """
    openai_tool_calls: list[dict[str, Any]] = []
    for i, tc in enumerate(tool_calls_raw):
        tc_id = tc.get("id", f"call_{i}")
        tc_name = tc.get("name", "unknown")
        tc_args = tc.get("arguments", {})

        if isinstance(tc_args, dict):
            tc_args_str = json.dumps(tc_args)
        elif isinstance(tc_args, str):
            tc_args_str = tc_args
        else:
            tc_args_str = json.dumps(tc_args)

        openai_tool_calls.append({
            "id": tc_id,
            "type": "function",
            "function": {
                "name": tc_name,
                "arguments": tc_args_str,
            },
        })

    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": openai_tool_calls,
            },
            "finish_reason": "tool_calls",
        }],
    }


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count.

    Rule of thumb: ~4 characters per token for English text.

    Args:
        text: Input text.

    Returns:
        Estimated token count (at least 1).
    """
    return max(1, len(text) // 4)


def extract_response_text(data: dict[str, Any]) -> str:
    """Extract the response text from an OpenAI-format dict for token estimation.

    Args:
        data: OpenAI chat completion dict.

    Returns:
        The text content or JSON-serialised tool calls string.
    """
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return json.dumps(tool_calls)
    return content
