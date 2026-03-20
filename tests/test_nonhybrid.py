"""Tests for non-hybrid mode (hybrid=False).

Non-hybrid mode: Claude has no built-in tools. Each chat() call returns ONE
action (a structured tool call decision). The caller executes every tool —
including file operations — and calls chat() again with the result.

Claude outputs action-enum structured JSON:
    {"action": "file_read", "arguments": {"path": "/tmp/x"}, "reasoning": "..."}

Terminal actions "done" and "blocked" are returned as content responses.
All other actions are returned as OpenAI tool_calls for the caller to execute.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cic.client import CiCClient
from cic.utils import (
    _NON_HYBRID_SYSTEM_PROMPT,
    _SYSTEM_PROMPT,
    build_non_hybrid_schema,
    build_prompt,
    parse_cli_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(**kwargs: Any) -> CiCClient:
    """Create a CiCClient with a patched claude path."""
    return CiCClient(claude_path="/usr/bin/claude", **kwargs)


def _non_hybrid_envelope(
    action: str,
    arguments: dict[str, Any] | None = None,
    reasoning: str = "test reasoning",
) -> str:
    """Build a fake CLI envelope with a non-hybrid action-enum structured output."""
    return json.dumps({
        "type": "result",
        "result": "",
        "is_error": False,
        "structured_output": {
            "action": action,
            "arguments": arguments if arguments is not None else {},
            "reasoning": reasoning,
        },
    })


def _mock_spawn(stdout: str, stderr: str = "") -> AsyncMock:
    """Return a coroutine mock that simulates _spawn_claude output."""
    mock = AsyncMock(return_value=(stdout, stderr))
    return mock


def _make_stream_proc(response_line: str, stderr: bytes = b"") -> MagicMock:
    """Build a mock asyncio subprocess that simulates stream-json output.

    Emits ``response_line`` as the single stdout line (the type:"result" event),
    then EOF. ``stdin`` supports write/drain/close.
    """
    proc = MagicMock()
    proc.returncode = 0

    stdin = MagicMock()
    stdin.write = MagicMock()
    stdin.drain = AsyncMock()
    stdin.close = MagicMock()
    proc.stdin = stdin

    _responses = iter([response_line.encode("utf-8") + b"\n", b""])
    stdout = MagicMock()
    stdout.readline = AsyncMock(side_effect=lambda: next(_responses))
    proc.stdout = stdout

    stderr_mock = MagicMock()
    stderr_mock.read = AsyncMock(return_value=stderr)
    proc.stderr = stderr_mock

    proc.wait = AsyncMock()
    proc.kill = MagicMock()
    proc.communicate = AsyncMock(return_value=(b"", stderr))
    return proc


# ---------------------------------------------------------------------------
# TestNonHybridSchema — build_non_hybrid_schema
# ---------------------------------------------------------------------------

class TestNonHybridSchema:
    def test_build_schema_from_tools(self):
        tools = [
            {"function": {"name": "file_read", "description": "Read", "parameters": {}}},
            {"function": {"name": "file_edit", "description": "Edit", "parameters": {}}},
        ]
        schema_str = build_non_hybrid_schema(tools)
        schema = json.loads(schema_str)

        enum_values = schema["properties"]["action"]["enum"]
        assert "file_read" in enum_values
        assert "file_edit" in enum_values

    def test_empty_tools_empty_enum(self):
        schema_str = build_non_hybrid_schema([])
        schema = json.loads(schema_str)
        enum_values = schema["properties"]["action"]["enum"]
        assert enum_values == []  # No tools, no done/blocked — empty enum

    def test_required_fields_in_schema(self):
        schema_str = build_non_hybrid_schema([])
        schema = json.loads(schema_str)
        assert set(schema["required"]) == {"action", "arguments", "reasoning"}

    def test_arguments_is_object_type(self):
        schema_str = build_non_hybrid_schema([])
        schema = json.loads(schema_str)
        assert schema["properties"]["arguments"]["type"] == "object"

    def test_reasoning_is_string_type(self):
        schema_str = build_non_hybrid_schema([])
        schema = json.loads(schema_str)
        assert schema["properties"]["reasoning"]["type"] == "string"

    def test_tool_names_from_wrapped_schema(self):
        """Tools with nested 'function' key are handled."""
        tools = [
            {
                "type": "function",
                "function": {"name": "shell_exec", "description": "Run", "parameters": {}},
            }
        ]
        schema_str = build_non_hybrid_schema(tools)
        schema = json.loads(schema_str)
        assert "shell_exec" in schema["properties"]["action"]["enum"]

    def test_tool_names_from_unwrapped_schema(self):
        """Tools without 'function' wrapper are also handled."""
        tools = [{"name": "run_tests", "description": "Run tests", "parameters": {}}]
        schema_str = build_non_hybrid_schema(tools)
        schema = json.loads(schema_str)
        assert "run_tests" in schema["properties"]["action"]["enum"]

    def test_schema_is_valid_json(self):
        tools = [
            {"function": {"name": "file_read", "description": "R", "parameters": {}}},
            {"function": {"name": "file_edit", "description": "E", "parameters": {}}},
            {"function": {"name": "shell_exec", "description": "S", "parameters": {}}},
        ]
        schema_str = build_non_hybrid_schema(tools)
        schema = json.loads(schema_str)
        assert schema["type"] == "object"

    def test_multiple_tools_all_in_enum(self):
        tools = [
            {"function": {"name": "tool_a", "description": "A", "parameters": {}}},
            {"function": {"name": "tool_b", "description": "B", "parameters": {}}},
            {"function": {"name": "tool_c", "description": "C", "parameters": {}}},
        ]
        schema_str = build_non_hybrid_schema(tools)
        schema = json.loads(schema_str)
        enum_values = schema["properties"]["action"]["enum"]
        assert "tool_a" in enum_values
        assert "tool_b" in enum_values
        assert "tool_c" in enum_values
        # Only tool names — no done/blocked
        assert set(enum_values) == {"tool_a", "tool_b", "tool_c"}


# ---------------------------------------------------------------------------
# TestNonHybridParsing — parse_cli_output with hybrid=False
# ---------------------------------------------------------------------------

class TestNonHybridParsing:
    def test_file_read_action_returns_tool_call(self):
        stdout = _non_hybrid_envelope(
            action="file_read",
            arguments={"path": "/tmp/test.py"},
        )
        data = parse_cli_output(stdout, hybrid=False)
        message = data["choices"][0]["message"]
        assert message["content"] is None
        assert len(message["tool_calls"]) == 1
        tc = message["tool_calls"][0]
        assert tc["function"]["name"] == "file_read"
        args = json.loads(tc["function"]["arguments"])
        assert args["path"] == "/tmp/test.py"

    def test_file_edit_action_returns_tool_call(self):
        stdout = _non_hybrid_envelope(
            action="file_edit",
            arguments={
                "path": "/tmp/test.py",
                "old_string": "old code",
                "new_string": "new code",
            },
        )
        data = parse_cli_output(stdout, hybrid=False)
        message = data["choices"][0]["message"]
        assert message["content"] is None
        tc = message["tool_calls"][0]
        assert tc["function"]["name"] == "file_edit"
        args = json.loads(tc["function"]["arguments"])
        assert args["old_string"] == "old code"
        assert args["new_string"] == "new code"

    def test_file_write_action_returns_tool_call(self):
        stdout = _non_hybrid_envelope(
            action="file_write",
            arguments={"path": "/tmp/new.py", "content": "hello"},
        )
        data = parse_cli_output(stdout, hybrid=False)
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "file_write"

    def test_shell_exec_action_returns_tool_call(self):
        stdout = _non_hybrid_envelope(
            action="shell_exec",
            arguments={"command": "pytest tests/"},
        )
        data = parse_cli_output(stdout, hybrid=False)
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "shell_exec"
        args = json.loads(tc["function"]["arguments"])
        assert args["command"] == "pytest tests/"

    def test_run_tests_action_returns_tool_call(self):
        stdout = _non_hybrid_envelope(
            action="run_tests",
            arguments={"path": "tests/"},
        )
        data = parse_cli_output(stdout, hybrid=False)
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "run_tests"

    def test_finish_reason_tool_calls_for_action(self):
        stdout = _non_hybrid_envelope(action="file_read", arguments={"path": "/x"})
        data = parse_cli_output(stdout, hybrid=False)
        assert data["choices"][0]["finish_reason"] == "tool_calls"

    def test_arguments_dict_preserved(self):
        """Arguments dict comes back as serialized JSON in OpenAI format."""
        args = {"path": "/tmp/f.py", "old_string": "a", "new_string": "b"}
        stdout = _non_hybrid_envelope(action="file_edit", arguments=args)
        data = parse_cli_output(stdout, hybrid=False)
        tc = data["choices"][0]["message"]["tool_calls"][0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == args

    def test_hybrid_true_parses_hybrid_format(self):
        """hybrid=True (default) still parses hybrid-format responses correctly."""
        stdout = json.dumps({
            "type": "result",
            "result": "",
            "is_error": False,
            "structured_output": {
                "summary": "Hybrid done",
                "files_modified": [],
                "pending_tool_calls": [],
            },
        })
        data = parse_cli_output(stdout, hybrid=True)
        content = data["choices"][0]["message"]["content"]
        assert "Hybrid done" in content


# ---------------------------------------------------------------------------
# TestNonHybridPrompt — build_prompt with hybrid=False
# ---------------------------------------------------------------------------

class TestNonHybridPrompt:
    def test_non_hybrid_uses_different_system_prompt(self):
        messages = [{"role": "user", "content": "Hi"}]
        prompt = build_prompt(messages, tools=None, hybrid=False)
        assert _NON_HYBRID_SYSTEM_PROMPT[:50] in prompt

    def test_hybrid_uses_hybrid_system_prompt(self):
        messages = [{"role": "user", "content": "Hi"}]
        prompt = build_prompt(messages, tools=None, hybrid=True)
        assert _SYSTEM_PROMPT[:50] in prompt

    def test_non_hybrid_describes_all_tools(self):
        """Non-hybrid mode describes ALL tools including native-like ones."""
        messages = [{"role": "user", "content": "Go"}]
        tools = [
            {"function": {"name": "file_read", "description": "Read", "parameters": {}}},
            {"function": {"name": "file_edit", "description": "Edit", "parameters": {}}},
        ]
        prompt = build_prompt(messages, tools=tools, hybrid=False)
        assert "file_read" in prompt
        assert "file_edit" in prompt

    def test_hybrid_strips_native_tools_from_prompt(self):
        """Hybrid mode strips native tools — Claude handles them directly."""
        messages = [{"role": "user", "content": "Go"}]
        tools = [
            {"function": {"name": "file_read", "description": "Read", "parameters": {}}},
            {"function": {"name": "my_custom_tool", "description": "Custom", "parameters": {}}},
        ]
        prompt = build_prompt(messages, tools=tools, hybrid=True)
        assert "file_read" not in prompt
        assert "my_custom_tool" in prompt

    def test_non_hybrid_includes_all_tools_even_native_ones(self):
        """Non-hybrid: no filtering — all tools go in the prompt."""
        messages = [{"role": "user", "content": "Go"}]
        tools = [
            {"function": {"name": "shell_exec", "description": "Exec", "parameters": {}}},
            {"function": {"name": "run_tests", "description": "Test", "parameters": {}}},
        ]
        prompt = build_prompt(messages, tools=tools, hybrid=False)
        assert "shell_exec" in prompt
        assert "run_tests" in prompt

    def test_non_hybrid_tool_section_header(self):
        messages = [{"role": "user", "content": "Go"}]
        tools = [
            {"function": {"name": "my_tool", "description": "Does stuff", "parameters": {}}},
        ]
        prompt = build_prompt(messages, tools=tools, hybrid=False)
        assert "AVAILABLE TOOLS" in prompt

    def test_hybrid_tool_section_header(self):
        messages = [{"role": "user", "content": "Go"}]
        tools = [
            {"function": {"name": "notify_done", "description": "Notify", "parameters": {}}},
        ]
        prompt = build_prompt(messages, tools=tools, hybrid=True)
        assert "CUSTOM TOOLS" in prompt

    def test_non_hybrid_tool_params_in_description(self):
        """Tool descriptions include param names and types in non-hybrid mode."""
        messages = [{"role": "user", "content": "Go"}]
        tools = [
            {
                "function": {
                    "name": "file_edit",
                    "description": "Edit a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["path", "old_string", "new_string"],
                    },
                }
            }
        ]
        prompt = build_prompt(messages, tools=tools, hybrid=False)
        assert "path" in prompt
        assert "old_string" in prompt
        assert "new_string" in prompt

    def test_non_hybrid_single_action_instruction(self):
        """Non-hybrid prompt instructs Claude to output one action."""
        messages = [{"role": "user", "content": "Do it"}]
        prompt = build_prompt(messages, tools=None, hybrid=False)
        assert "single" in prompt.lower() or "one action" in prompt.lower()

    def test_conversation_history_included_in_non_hybrid(self):
        messages = [
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "file_read", "arguments": '{"path": "/x"}'}}
            ]},
            {"role": "tool", "name": "file_read", "content": "def foo():\n    pass"},
        ]
        prompt = build_prompt(messages, tools=None, hybrid=False)
        assert "Fix the bug" in prompt
        assert "file_read" in prompt
        assert "def foo():" in prompt

    def test_hybrid_default_true(self):
        """hybrid=True is the default — existing callers are unaffected."""
        messages = [{"role": "user", "content": "Hi"}]
        # Both explicit hybrid=True and no hybrid arg should produce same result
        prompt_explicit = build_prompt(messages, tools=None, hybrid=True)
        prompt_default = build_prompt(messages, tools=None)
        assert prompt_explicit == prompt_default


# ---------------------------------------------------------------------------
# TestNonHybridClient — CiCClient with hybrid=False
# ---------------------------------------------------------------------------

class TestNonHybridClientInit:
    def test_hybrid_false_stored(self):
        client = _make_client(model="sonnet", hybrid=False)
        assert client._hybrid is False

    def test_hybrid_true_is_default(self):
        client = _make_client(model="sonnet")
        assert client._hybrid is True

    def test_hybrid_true_explicit(self):
        client = _make_client(model="sonnet", hybrid=True)
        assert client._hybrid is True

    def test_repr_shows_non_hybrid_mode(self):
        client = _make_client(model="sonnet", hybrid=False)
        assert "non-hybrid" in repr(client)

    def test_repr_shows_hybrid_mode(self):
        client = _make_client(model="sonnet", hybrid=True)
        assert "hybrid" in repr(client)


class TestNonHybridClientSubprocess:
    @pytest.mark.asyncio
    async def test_non_hybrid_uses_empty_tools_flag(self):
        """Non-hybrid mode must pass --tools "" to disable Claude's built-in tools."""
        client = _make_client(model="sonnet", hybrid=False)
        captured_cmd: list[str] = []

        async def _capture_exec(*args: Any, **kwargs: Any) -> Any:
            captured_cmd.extend(args)
            return _make_stream_proc(_non_hybrid_envelope("done"))

        with patch("asyncio.create_subprocess_exec", side_effect=_capture_exec):
            await client.achat([{"role": "user", "content": "hi"}])

        assert "--tools" in captured_cmd
        tools_idx = captured_cmd.index("--tools")
        assert captured_cmd[tools_idx + 1] == ""

    @pytest.mark.asyncio
    async def test_hybrid_uses_bash_edit_read_write_tools_flag(self):
        """Hybrid mode must pass --tools Bash,Edit,Read,Write."""
        client = _make_client(model="sonnet", hybrid=True)
        captured_cmd: list[str] = []

        hybrid_response = json.dumps({
            "type": "result",
            "result": "",
            "is_error": False,
            "structured_output": {
                "summary": "ok",
                "files_modified": [],
                "pending_tool_calls": [],
            },
        })

        async def _capture_exec(*args: Any, **kwargs: Any) -> Any:
            captured_cmd.extend(args)
            return _make_stream_proc(hybrid_response)

        with patch("asyncio.create_subprocess_exec", side_effect=_capture_exec):
            await client.achat([{"role": "user", "content": "hi"}])

        assert "--tools" in captured_cmd
        tools_idx = captured_cmd.index("--tools")
        assert captured_cmd[tools_idx + 1] == "Bash,Edit,Read,Write"

    @pytest.mark.asyncio
    async def test_non_hybrid_schema_contains_tool_names(self):
        """Non-hybrid --json-schema must include caller's tool names in enum."""
        client = _make_client(model="sonnet", hybrid=False)
        captured_cmd: list[str] = []

        async def _capture_exec(*args: Any, **kwargs: Any) -> Any:
            captured_cmd.extend(args)
            return _make_stream_proc(_non_hybrid_envelope("done"))

        tools = [
            {"function": {"name": "file_read", "description": "R", "parameters": {}}},
            {"function": {"name": "file_edit", "description": "E", "parameters": {}}},
        ]

        with patch("asyncio.create_subprocess_exec", side_effect=_capture_exec):
            await client.achat([{"role": "user", "content": "hi"}], tools=tools)

        assert "--json-schema" in captured_cmd
        schema_idx = captured_cmd.index("--json-schema")
        schema = json.loads(captured_cmd[schema_idx + 1])
        enum_values = schema["properties"]["action"]["enum"]
        assert "file_read" in enum_values
        assert "file_edit" in enum_values
        assert len(enum_values) > 0  # no done/blocked, only tool names

    @pytest.mark.asyncio
    async def test_non_hybrid_uses_non_hybrid_system_prompt(self):
        """Non-hybrid mode must use the non-hybrid system prompt."""
        client = _make_client(model="sonnet", hybrid=False)
        captured_cmd: list[str] = []

        async def _capture_exec(*args: Any, **kwargs: Any) -> Any:
            captured_cmd.extend(args)
            return _make_stream_proc(_non_hybrid_envelope("done"))

        with patch("asyncio.create_subprocess_exec", side_effect=_capture_exec):
            await client.achat([{"role": "user", "content": "hi"}])

        assert "--system-prompt" in captured_cmd
        sp_idx = captured_cmd.index("--system-prompt")
        system_prompt = captured_cmd[sp_idx + 1]
        assert "NO built-in tools" in system_prompt or "tool-execution mode" in system_prompt


# ---------------------------------------------------------------------------
# TestNonHybridChatResults — full achat round-trips
# ---------------------------------------------------------------------------

class TestNonHybridChatResults:
    @pytest.mark.asyncio
    async def test_file_read_action_returns_tool_call(self):
        client = _make_client(model="sonnet", hybrid=False)
        client._spawn_claude = _mock_spawn(
            _non_hybrid_envelope("file_read", {"path": "/tmp/main.py"})
        )
        result = await client.achat([{"role": "user", "content": "Read main.py"}])
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "file_read"
        assert result.tool_calls[0].arguments == {"path": "/tmp/main.py"}

    @pytest.mark.asyncio
    async def test_file_edit_action_returns_tool_call(self):
        client = _make_client(model="sonnet", hybrid=False)
        client._spawn_claude = _mock_spawn(
            _non_hybrid_envelope("file_edit", {
                "path": "/tmp/main.py",
                "old_string": "bug",
                "new_string": "fix",
            })
        )
        result = await client.achat([{"role": "user", "content": "Fix bug"}])
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "file_edit"
        assert result.tool_calls[0].arguments["old_string"] == "bug"
        assert result.tool_calls[0].arguments["new_string"] == "fix"

    @pytest.mark.asyncio
    async def test_shell_exec_action_returns_tool_call(self):
        client = _make_client(model="sonnet", hybrid=False)
        client._spawn_claude = _mock_spawn(
            _non_hybrid_envelope("shell_exec", {"command": "pytest tests/ -v"})
        )
        result = await client.achat([{"role": "user", "content": "Run tests"}])
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "shell_exec"
        assert result.tool_calls[0].arguments["command"] == "pytest tests/ -v"

    @pytest.mark.asyncio
    async def test_model_in_result(self):
        client = _make_client(model="sonnet", hybrid=False)
        client._spawn_claude = _mock_spawn(_non_hybrid_envelope("done"))
        result = await client.achat([{"role": "user", "content": "hi"}])
        assert result.model == "cic/sonnet"

    @pytest.mark.asyncio
    async def test_usage_estimated(self):
        client = _make_client(model="sonnet", hybrid=False)
        client._spawn_claude = _mock_spawn(_non_hybrid_envelope("done"))
        result = await client.achat([{"role": "user", "content": "hi"}])
        assert result.usage.prompt_tokens > 0


# ---------------------------------------------------------------------------
# TestNonHybridBackwardCompatibility — hybrid=True (default) is unchanged
# ---------------------------------------------------------------------------

class TestHybridBackwardCompatibility:
    def test_default_is_hybrid_true(self):
        """Creating a client without hybrid= uses hybrid mode."""
        client = _make_client(model="sonnet")
        assert client._hybrid is True

    @pytest.mark.asyncio
    async def test_hybrid_text_response_unchanged(self):
        """Existing hybrid mode response handling is unaffected."""
        client = _make_client(model="sonnet")
        response = json.dumps({
            "type": "result",
            "result": "",
            "is_error": False,
            "structured_output": {
                "summary": "Hybrid complete",
                "files_modified": ["/tmp/x.py"],
                "pending_tool_calls": [],
            },
        })
        client._spawn_claude = _mock_spawn(response)
        result = await client.achat([{"role": "user", "content": "hi"}])
        assert result.content == "Hybrid complete"
        assert not result.has_tool_calls

    @pytest.mark.asyncio
    async def test_hybrid_tool_call_response_unchanged(self):
        """Existing hybrid pending_tool_calls handling is unaffected."""
        client = _make_client(model="sonnet")
        response = json.dumps({
            "type": "result",
            "result": "",
            "is_error": False,
            "structured_output": {
                "summary": "Done",
                "files_modified": [],
                "pending_tool_calls": [
                    {"name": "notify_done", "arguments": {"message": "ok"}},
                ],
            },
        })
        client._spawn_claude = _mock_spawn(response)
        result = await client.achat([{"role": "user", "content": "hi"}])
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "notify_done"
