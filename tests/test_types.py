"""Tests for CiC data types."""

import json
import pytest

from cic.types import ChatResult, ToolCall, TokenUsage


class TestTokenUsage:
    def test_total_computed_when_zero(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        assert usage.total_tokens == 15

    def test_total_not_overwritten_when_provided(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=99)
        assert usage.total_tokens == 99

    def test_defaults_are_zero(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_total_stays_zero_when_both_zero(self):
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        assert usage.total_tokens == 0


class TestToolCall:
    def test_arguments_json_roundtrips(self):
        tc = ToolCall(id="call_1", name="read_file", arguments={"path": "/tmp/test"})
        serialised = tc.arguments_json()
        assert json.loads(serialised) == {"path": "/tmp/test"}

    def test_empty_arguments(self):
        tc = ToolCall(id="call_1", name="ping")
        assert tc.arguments_json() == "{}"

    def test_nested_arguments(self):
        tc = ToolCall(
            id="x",
            name="query",
            arguments={"filter": {"field": "name", "value": "test"}},
        )
        parsed = json.loads(tc.arguments_json())
        assert parsed["filter"]["field"] == "name"


class TestChatResult:
    def test_has_tool_calls_true(self):
        tc = ToolCall(id="c1", name="tool_a")
        result = ChatResult(content=None, tool_calls=[tc])
        assert result.has_tool_calls is True

    def test_has_tool_calls_false_on_text_response(self):
        result = ChatResult(content="Hello!")
        assert result.has_tool_calls is False

    def test_has_tool_calls_false_on_empty_list(self):
        result = ChatResult(content="Hello!", tool_calls=[])
        assert result.has_tool_calls is False

    def test_to_openai_dict_returns_raw(self):
        raw = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        result = ChatResult(content="hi", raw=raw)
        assert result.to_openai_dict() is raw

    def test_default_usage_is_zero(self):
        result = ChatResult(content="hi")
        assert result.usage.total_tokens == 0

    def test_model_defaults_empty(self):
        result = ChatResult(content="hi")
        assert result.model == ""
