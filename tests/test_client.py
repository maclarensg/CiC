"""Tests for CiCClient — the main entry point.

All subprocess calls are mocked. We test the full client pipeline
(prompt building → subprocess → response parsing → ChatResult) without
actually spawning the claude CLI.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cic.client import CiCClient, _build_chat_result, _parse_arguments
from cic.exceptions import (
    ClaudeNotFoundError,
    ClaudeSubprocessError,
    ClaudeTimeoutError,
)
from cic.types import ChatResult, TokenUsage


# ---------------------------------------------------------------------------
# Helpers for building mock subprocess output
# ---------------------------------------------------------------------------

def _cli_envelope(result: str, is_error: bool = False) -> str:
    """Build a fake claude CLI JSON envelope."""
    return json.dumps({"type": "result", "result": result, "is_error": is_error})


def _text_response(text: str) -> str:
    """Build a fake CLI envelope with a plain text response."""
    return _cli_envelope(json.dumps({"response": text}))


def _tool_call_response(tool_calls: list[dict[str, Any]]) -> str:
    """Build a fake CLI envelope with tool calls."""
    return _cli_envelope(json.dumps({"tool_calls": tool_calls}))


def _make_client(**kwargs: Any) -> CiCClient:
    """Create a CiCClient with a patched claude path."""
    return CiCClient(claude_path="/usr/bin/claude", **kwargs)


# ---------------------------------------------------------------------------
# Subprocess mock helper
# ---------------------------------------------------------------------------

def _mock_spawn(stdout: str, stderr: str = "") -> AsyncMock:
    """Return a coroutine mock that simulates _spawn_claude output."""
    mock = AsyncMock(return_value=(stdout, stderr))
    return mock


# ---------------------------------------------------------------------------
# CiCClient construction
# ---------------------------------------------------------------------------

class TestCiCClientInit:
    def test_raises_when_claude_not_found(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(ClaudeNotFoundError):
                CiCClient()

    def test_accepts_explicit_claude_path(self):
        client = _make_client(model="sonnet")
        assert client._claude_path == "/usr/bin/claude"

    def test_fixed_model(self):
        client = _make_client(model="opus")
        assert client.active_model == "opus"

    def test_routing_model_default(self):
        client = _make_client()
        assert client.active_model == "sonnet"  # default complexity = "moderate"

    def test_custom_routing(self):
        client = _make_client(routing={"simple": "haiku", "moderate": "sonnet", "complex": "opus"})
        client.set_complexity("simple")
        assert client.active_model == "haiku"

    def test_repr_with_fixed_model(self):
        client = _make_client(model="opus")
        r = repr(client)
        assert "opus" in r

    def test_repr_with_routing(self):
        client = _make_client()
        r = repr(client)
        assert "routing" in r.lower() or "CiCClient" in r

    def test_context_manager_sync(self):
        client = _make_client(model="sonnet")
        with client as c:
            assert c is client

    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        client = _make_client(model="sonnet")
        async with client as c:
            assert c is client


# ---------------------------------------------------------------------------
# set_complexity / set_model
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_set_complexity_changes_model(self):
        client = _make_client()
        client.set_complexity("complex")
        assert client.active_model == "opus"

    def test_set_complexity_simple(self):
        client = _make_client()
        client.set_complexity("simple")
        assert client.active_model == "haiku"

    def test_set_model_overrides_routing(self):
        client = _make_client()
        client.set_model("haiku")
        assert client.active_model == "haiku"

    def test_set_model_persists(self):
        client = _make_client()
        client.set_complexity("complex")
        client.set_model("haiku")
        assert client.active_model == "haiku"


# ---------------------------------------------------------------------------
# achat — async interface
# ---------------------------------------------------------------------------

class TestAChatTextResponse:
    @pytest.mark.asyncio
    async def test_basic_text_response(self):
        client = _make_client(model="sonnet")
        client._spawn_claude = _mock_spawn(_text_response("Hello, world!"))
        result = await client.achat([{"role": "user", "content": "Hi"}])
        assert result.content == "Hello, world!"
        assert result.tool_calls == []
        assert not result.has_tool_calls

    @pytest.mark.asyncio
    async def test_model_in_result(self):
        client = _make_client(model="sonnet")
        client._spawn_claude = _mock_spawn(_text_response("Hi"))
        result = await client.achat([{"role": "user", "content": "Hi"}])
        assert result.model == "cic/sonnet"

    @pytest.mark.asyncio
    async def test_usage_estimated(self):
        client = _make_client(model="sonnet")
        client._spawn_claude = _mock_spawn(_text_response("Hello"))
        result = await client.achat([{"role": "user", "content": "Hi"}])
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_raw_openai_format_present(self):
        client = _make_client(model="sonnet")
        client._spawn_claude = _mock_spawn(_text_response("Hi"))
        result = await client.achat([{"role": "user", "content": "Hi"}])
        assert "choices" in result.raw


class TestAChatToolCalls:
    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        client = _make_client(model="sonnet")
        tool_calls = [{"id": "c1", "name": "read_file", "arguments": {"path": "/tmp/x"}}]
        client._spawn_claude = _mock_spawn(_tool_call_response(tool_calls))
        result = await client.achat(
            [{"role": "user", "content": "Read /tmp/x"}],
            tools=[{"name": "read_file", "description": "Read a file", "parameters": {}}],
        )
        assert result.content is None
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "/tmp/x"}

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        client = _make_client(model="sonnet")
        tool_calls = [
            {"id": "c1", "name": "read_file", "arguments": {"path": "/a"}},
            {"id": "c2", "name": "write_file", "arguments": {"path": "/b", "content": "hello"}},
        ]
        client._spawn_claude = _mock_spawn(_tool_call_response(tool_calls))
        result = await client.achat([{"role": "user", "content": "Do stuff"}])
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "c1"
        assert result.tool_calls[1].id == "c2"

    @pytest.mark.asyncio
    async def test_tool_call_id_auto_generated_when_missing(self):
        client = _make_client(model="sonnet")
        tool_calls = [{"name": "ping", "arguments": {}}]  # no id field
        client._spawn_claude = _mock_spawn(_tool_call_response(tool_calls))
        result = await client.achat([{"role": "user", "content": "ping"}])
        assert result.tool_calls[0].id.startswith("call_")

    @pytest.mark.asyncio
    async def test_arguments_as_json_string_parsed(self):
        """Arguments that arrive as a JSON string are materialised to dict."""
        client = _make_client(model="sonnet")
        tool_calls = [{"id": "c1", "name": "search", "arguments": '{"query": "test"}'}]
        client._spawn_claude = _mock_spawn(_tool_call_response(tool_calls))
        result = await client.achat([{"role": "user", "content": "search"}])
        assert result.tool_calls[0].arguments == {"query": "test"}


class TestAChatErrorHandling:
    @pytest.mark.asyncio
    async def test_timeout_raises_cic_error(self):
        client = _make_client(model="sonnet")

        async def _timeout_spawn(*args: Any, **kwargs: Any) -> tuple[str, str]:
            raise ClaudeTimeoutError(120.0)

        client._spawn_claude = _timeout_spawn
        with pytest.raises(ClaudeTimeoutError):
            await client.achat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_cli_error_raises(self):
        client = _make_client(model="sonnet")
        client._spawn_claude = _mock_spawn(
            _cli_envelope("Permission denied", is_error=True)
        )
        with pytest.raises(ClaudeSubprocessError):
            await client.achat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_non_json_response_treated_as_text(self):
        client = _make_client(model="sonnet")
        client._spawn_claude = _mock_spawn(
            _cli_envelope("This is not JSON, just text.")
        )
        result = await client.achat([{"role": "user", "content": "hi"}])
        assert result.content == "This is not JSON, just text."
        assert not result.has_tool_calls


# ---------------------------------------------------------------------------
# Synchronous chat() wrapper
# ---------------------------------------------------------------------------

class TestSyncChat:
    def test_chat_returns_result(self):
        client = _make_client(model="sonnet")
        client._spawn_claude = _mock_spawn(_text_response("Hello sync"))
        result = client.chat([{"role": "user", "content": "hi"}])
        assert result.content == "Hello sync"

    def test_chat_openai_format(self):
        client = _make_client(model="sonnet")
        client._spawn_claude = _mock_spawn(_text_response("raw result"))
        data = client.chat_openai_format([{"role": "user", "content": "hi"}])
        assert "choices" in data
        assert data["choices"][0]["message"]["content"] == "raw result"


# ---------------------------------------------------------------------------
# Smart routing in actual calls
# ---------------------------------------------------------------------------

class TestSmartRoutingInCalls:
    @pytest.mark.asyncio
    async def test_simple_complexity_uses_haiku(self):
        client = _make_client(routing={"simple": "haiku", "moderate": "sonnet", "complex": "opus"})
        client.set_complexity("simple")

        captured_model = None

        async def _capture_spawn(prompt: str, model: str) -> tuple[str, str]:
            nonlocal captured_model
            captured_model = model
            return _text_response("ok"), ""

        client._spawn_claude = _capture_spawn
        await client.achat([{"role": "user", "content": "easy"}])
        assert captured_model == "haiku"

    @pytest.mark.asyncio
    async def test_complex_complexity_uses_opus(self):
        client = _make_client(routing={"simple": "haiku", "moderate": "sonnet", "complex": "opus"})
        client.set_complexity("complex")

        captured_model = None

        async def _capture_spawn(prompt: str, model: str) -> tuple[str, str]:
            nonlocal captured_model
            captured_model = model
            return _text_response("done"), ""

        client._spawn_claude = _capture_spawn
        await client.achat([{"role": "user", "content": "hard problem"}])
        assert captured_model == "opus"


# ---------------------------------------------------------------------------
# _build_chat_result helper
# ---------------------------------------------------------------------------

class TestBuildChatResult:
    def test_text_response(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }]
        }
        result = _build_chat_result(data, model="cic/sonnet", usage=TokenUsage(1, 1))
        assert result.content == "hello"
        assert result.tool_calls == []

    def test_tool_call_response(self):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "greet", "arguments": '{"name": "Alice"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }]
        }
        result = _build_chat_result(data, model="cic/sonnet", usage=TokenUsage(1, 1))
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "greet"
        assert result.tool_calls[0].arguments == {"name": "Alice"}

    def test_no_choices_returns_fallback(self):
        result = _build_chat_result({}, model="cic/sonnet", usage=TokenUsage())
        assert "[CiC]" in result.content


# ---------------------------------------------------------------------------
# _parse_arguments helper
# ---------------------------------------------------------------------------

class TestParseArguments:
    def test_dict_passthrough(self):
        assert _parse_arguments({"key": "val"}) == {"key": "val"}

    def test_json_string_parsed(self):
        assert _parse_arguments('{"key": "val"}') == {"key": "val"}

    def test_invalid_json_returns_empty(self):
        assert _parse_arguments("not json") == {}

    def test_none_returns_empty(self):
        assert _parse_arguments(None) == {}

    def test_json_non_dict_returns_empty(self):
        assert _parse_arguments("[1, 2, 3]") == {}


# ---------------------------------------------------------------------------
# chat() event loop guard
# ---------------------------------------------------------------------------

class TestSyncChatEventLoopGuard:
    @pytest.mark.asyncio
    async def test_chat_raises_when_called_inside_event_loop(self):
        """chat() must raise RuntimeError when called from an already-running loop."""
        client = _make_client(model="sonnet")
        with pytest.raises(RuntimeError, match="event loop"):
            client.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_achat_works_inside_event_loop(self):
        """achat() is the correct method to use inside an async context."""
        client = _make_client(model="sonnet")
        client._spawn_claude = _mock_spawn(_text_response("async works"))
        result = await client.achat([{"role": "user", "content": "hi"}])
        assert result.content == "async works"
