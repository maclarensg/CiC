"""Internal utilities for prompt building and response parsing.

Hybrid mode: Claude executes file edits using its own built-in tools (Bash, Edit,
Read, Write). Custom tools (defined by the caller) are reported back via structured
output in the pending_tool_calls field. The agent loop executes those after Claude
finishes.

Non-hybrid mode: Claude has NO built-in tools (--tools ""). Instead, it outputs ONE
action per call via --json-schema with an action-enum schema. The caller executes
every tool — including file operations — and calls chat() again with the result.
This gives the caller full control and verifiability over every operation.

These helpers are not part of the public API but are tested directly
because the prompt/parse logic is critical to correct behaviour.
"""

from __future__ import annotations

import json
from typing import Any

from .exceptions import ClaudeSubprocessError, ResponseParseError

# System prompt for hybrid mode.
# Claude uses its own file tools — only custom (caller-defined) tools need description.
_SYSTEM_PROMPT = """\
You are an AI agent. You have access to your built-in file tools \
(Bash, Edit, Read, Write) — use them directly to make file edits, run \
commands, and read files.

For custom tools listed below, you CANNOT call them directly. Instead, report \
them in the pending_tool_calls field of your structured output so the calling \
application can execute them after you finish.

IMPORTANT:
1. Use your built-in tools (Bash, Edit, Read, Write) to make ALL file changes.
2. Do NOT describe what you will do — just do it using your tools.
3. After completing file work, report any custom tool calls needed in pending_tool_calls.
4. If you cannot complete the task, explain why in the blocked field.
"""

# System prompt for non-hybrid mode.
# Claude has NO built-in tools — every action must be expressed as a structured output.
_NON_HYBRID_SYSTEM_PROMPT = """\
You are an AI agent operating in tool-execution mode. You have NO built-in tools.
Every action you take must be expressed as a structured JSON output with exactly
one action per response.

The calling application will execute your chosen action and return the result.
You then decide the next action based on that result. Continue until the task
requires no more action. You MUST pick a tool — there is no "done" option.

IMPORTANT:
1. Output ONE action per response — not multiple.
2. Choose the action that directly moves the task forward.
3. For file edits: use file_read first to get the current content, then file_edit.
4. For file_edit, the old_string MUST exactly match the current file content.
5. Use "done" only when the task is fully complete.
6. Use "blocked" only when you cannot proceed (include the reason in reasoning).
"""

_TOOL_SECTION_TEMPLATE = """\
CUSTOM TOOLS (report in pending_tool_calls — do NOT describe as text):
{tool_descriptions}

"""

_NON_HYBRID_TOOL_SECTION_TEMPLATE = """\
AVAILABLE TOOLS (use these via the action field):
{tool_descriptions}

"""

# The JSON schema for structured output in hybrid mode.
# The caller fills this in with what Claude did and what custom tools to run next.
STRUCTURED_OUTPUT_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "What you did — files changed, commands run, outcomes",
        },
        "files_modified": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Absolute paths of files you created or modified",
        },
        "pending_tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
            "description": "Custom tools to execute after Claude finishes file work",
        },
        "test_results": {
            "type": "string",
            "description": "Test output if tests were run (optional)",
        },
        "blocked": {
            "type": "string",
            "description": "If you could not complete the task, explain why (optional)",
        },
    },
    "required": ["summary", "files_modified", "pending_tool_calls"],
})

_MAX_TOOL_ARG_CHARS = 200
_MAX_TOOL_RESULT_CHARS = 2000

# Tools that Claude handles natively in hybrid mode.
# These are NOT described in the prompt — Claude just uses them directly.
DEFAULT_NATIVE_TOOLS = frozenset({
    "file_read", "file_edit", "file_write", "file_append",
    "shell_exec", "run_tests", "content_search", "file_search", "list_directory",
    "read_file", "write_file", "edit_file", "exec_command",
})

# No terminal actions in enum — forces Claude to pick a real tool every time.
# When "done"/"blocked" are available, Claude uses them as escape hatches
# and never calls file_edit. Removing them achieves 100% tool call rate.
_NON_HYBRID_TERMINAL_ACTIONS: tuple[str, ...] = ()


def filter_custom_tools(
    tools: list[dict[str, Any]],
    native_tools: frozenset[str] = DEFAULT_NATIVE_TOOLS,
) -> list[dict[str, Any]]:
    """Return only custom (non-native) tools that Claude can't call directly.

    In hybrid mode, Claude handles file/shell operations through its built-in
    tools. Only custom tools defined by the caller need to be described so
    Claude knows to report them in pending_tool_calls.

    Args:
        tools: OpenAI-format tool definitions.
        native_tools: Set of tool names handled natively by Claude. Defaults to
            ``DEFAULT_NATIVE_TOOLS``.

    Returns:
        Filtered list containing only custom tools.
    """
    return [t for t in tools if _get_tool_name(t) not in native_tools]


def _get_tool_name(tool: dict[str, Any]) -> str:
    """Extract tool name from OpenAI-format or unwrapped tool dict."""
    fn = tool.get("function", tool)
    return fn.get("name", "")


def build_non_hybrid_schema(tools: list[dict[str, Any]]) -> str:
    """Build a --json-schema that constrains Claude to call only the provided tools.

    Non-hybrid mode uses an action-enum schema so every response is a single
    structured tool call decision. The schema enum is built from the caller's
    tool names only — no "done" or "blocked" escape hatches.

    Args:
        tools: OpenAI-format tool definitions.

    Returns:
        A JSON string suitable for passing to ``--json-schema``.
    """
    tool_names = [_get_tool_name(t) for t in tools if _get_tool_name(t)]
    enum_values = tool_names + list(_NON_HYBRID_TERMINAL_ACTIONS)

    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": enum_values,
                "description": "The tool to call. You MUST pick one.",
            },
            "arguments": {
                "type": "object",
                "description": (
                    "Tool arguments. For file_edit: {path, old_string, new_string}. "
                    "For file_read: {path}. For shell_exec: {command}. "
                    "For done/blocked: {} or {reason}."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Brief reasoning for this action choice.",
            },
        },
        "required": ["action", "arguments", "reasoning"],
    }
    return json.dumps(schema)


def build_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    native_tools: frozenset[str] | None = None,
    *,
    hybrid: bool = True,
) -> str:
    """Flatten messages and tool descriptions into a single prompt string.

    Hybrid mode: only describes custom (non-native) tools. Claude handles
    file/shell operations through its own built-in tools (Bash, Edit, Read, Write).
    Custom tools are described so Claude knows to report them in pending_tool_calls.

    Non-hybrid mode: describes ALL tools (Claude cannot call any of them directly).
    Each call returns one action for the caller to execute.

    Args:
        messages: OpenAI-format messages list.
        tools: OpenAI-format tool definitions, or None.
        native_tools: Set of tool names handled natively by Claude. Defaults to
            ``DEFAULT_NATIVE_TOOLS``. Ignored in non-hybrid mode.
        hybrid: When True (default), uses hybrid mode prompt. When False, uses
            non-hybrid mode prompt where all tools are described.

    Returns:
        A single UTF-8 string suitable for piping to ``claude --print``.
    """
    if not hybrid:
        return _build_non_hybrid_prompt(messages, tools)

    parts: list[str] = [_SYSTEM_PROMPT]

    if tools:
        _nt = native_tools if native_tools is not None else DEFAULT_NATIVE_TOOLS
        custom = filter_custom_tools(tools, _nt)
        if custom:
            desc = _build_tool_descriptions(custom)
            parts.append(_TOOL_SECTION_TEMPLATE.format(tool_descriptions=desc))

    parts.append("CONVERSATION HISTORY:")
    parts.append(_format_messages(messages))
    parts.append(
        "\nComplete the task using your built-in tools. "
        "Report any custom tool calls in pending_tool_calls."
    )

    return "\n".join(parts)


def _build_non_hybrid_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> str:
    """Build a prompt for non-hybrid mode.

    All tools are described — Claude cannot call any of them directly.
    The output is one action per call.

    Args:
        messages: OpenAI-format messages list.
        tools: OpenAI-format tool definitions, or None.

    Returns:
        A single UTF-8 string suitable for piping to ``claude --print``.
    """
    parts: list[str] = [_NON_HYBRID_SYSTEM_PROMPT]

    if tools:
        desc = _build_tool_descriptions_with_params(tools)
        parts.append(_NON_HYBRID_TOOL_SECTION_TEMPLATE.format(tool_descriptions=desc))

    parts.append("CONVERSATION HISTORY:")
    parts.append(_format_messages(messages))
    parts.append(
        "\nDecide the single next action to take. "
        "Output exactly one action in the structured JSON format."
    )

    return "\n".join(parts)


def _build_tool_descriptions(tools: list[dict[str, Any]]) -> str:
    """Convert OpenAI tool schemas into a compact, human-readable list.

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


def _build_tool_descriptions_with_params(tools: list[dict[str, Any]]) -> str:
    """Convert OpenAI tool schemas into a detailed, human-readable list.

    Used in non-hybrid mode where Claude needs full parameter information
    to produce correct arguments without being able to call tools directly.

    Args:
        tools: List of OpenAI-format tool dicts.

    Returns:
        A newline-separated string with name, description, and parameter details.
    """
    lines: list[str] = []
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "unknown")
        description = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))

        param_parts: list[str] = []
        for param_name, param_schema in params.items():
            param_type = param_schema.get("type", "any")
            param_desc = param_schema.get("description", "")
            suffix = "" if param_name in required else "?"
            if param_desc:
                param_parts.append(f"    {param_name}{suffix} ({param_type}): {param_desc}")
            else:
                param_parts.append(f"    {param_name}{suffix} ({param_type})")

        if param_parts:
            params_str = "\n" + "\n".join(param_parts)
        else:
            params_str = " (no parameters)"

        lines.append(f"- {name}: {description}{params_str}")

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
        content = msg.get("content") or ""

        if role == "system":
            text = _extract_text(content)
            if text:
                parts.append(f"[System]: {text}")

        elif role == "user":
            text = _extract_text(content)
            if text:
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


def parse_cli_output(stdout: str, *, hybrid: bool = True) -> dict[str, Any]:
    """Parse the claude CLI JSON envelope and extract an OpenAI-format response.

    Hybrid mode: the CLI envelope contains a ``structured_output`` field with
    ``summary``, ``files_modified``, and ``pending_tool_calls``.

    Non-hybrid mode: the CLI envelope contains a ``structured_output`` field with
    ``action``, ``arguments``, and ``reasoning`` (the action-enum schema).

    The claude CLI with ``--output-format json`` returns an envelope like::

        {
            "type": "result",
            "result": "",
            "structured_output": { ... },
            "is_error": false,
            ...
        }

    Args:
        stdout: Raw stdout from the claude subprocess.
        hybrid: When True (default), parses hybrid mode structured output.
            When False, parses non-hybrid mode action-enum structured output.

    Returns:
        An OpenAI-format chat completion dict.

    Raises:
        ResponseParseError: If the envelope cannot be parsed.
        ClaudeSubprocessError: If ``is_error`` is True in the envelope.
    """
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

    # Extract structured_output field (present when --json-schema is used)
    structured = envelope.get("structured_output")
    if structured and isinstance(structured, dict):
        if hybrid:
            return _parse_structured_output(structured)
        else:
            return _parse_non_hybrid_structured_output(structured)

    # Fallback: try the result field (non-structured response or old invocation style)
    result_text = envelope.get("result", "")
    if not result_text:
        return _make_content_response("[CiC] Empty result in CLI envelope")

    return _make_content_response(result_text)


def _parse_structured_output(structured: dict[str, Any]) -> dict[str, Any]:
    """Convert hybrid mode structured output fields to OpenAI wire format.

    - pending_tool_calls → OpenAI tool_calls (caller executes these custom tools)
    - blocked → content response indicating task failure
    - summary (no pending calls) → content response indicating completion

    Args:
        structured: The structured_output dict from the CLI envelope.

    Returns:
        An OpenAI chat completion dict.
    """
    summary = structured.get("summary", "")
    pending_tool_calls = structured.get("pending_tool_calls", [])
    blocked = structured.get("blocked", "")

    # Blocked = task failed, report as content
    if blocked:
        return _make_content_response(f"[CiC] BLOCKED: {blocked}")

    # pending_tool_calls → caller executes these custom tools
    if pending_tool_calls and isinstance(pending_tool_calls, list):
        return _make_tool_call_response(pending_tool_calls)

    # No pending tool calls → task is done, return summary as content
    return _make_content_response(summary or "[CiC] Task completed")


def _parse_non_hybrid_structured_output(structured: dict[str, Any]) -> dict[str, Any]:
    """Convert non-hybrid mode action-enum structured output to OpenAI wire format.

    Every response is a tool call — no "done" or "blocked" escape hatches.
    The caller (agent loop) handles termination via max iterations.

    Args:
        structured: The structured_output dict from the CLI envelope.
            Expected keys: action, arguments, reasoning.

    Returns:
        An OpenAI chat completion dict with exactly one tool_call.
    """
    action = structured.get("action", "")
    arguments = structured.get("arguments", {})
    reasoning = structured.get("reasoning", "")

    # Tool action → return as OpenAI tool_call
    args = arguments if isinstance(arguments, dict) else {}
    tool_call = {"name": action, "arguments": args}
    return _make_tool_call_response([tool_call])


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
    """Convert a pending_tool_calls list to OpenAI wire format.

    Args:
        tool_calls_raw: List of tool call dicts from Claude's structured output.

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
