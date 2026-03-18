"""Tests for CiC prompt building and response parsing utilities."""

import json
import pytest

from cic.exceptions import ClaudeSubprocessError, ResponseParseError
from cic.utils import (
    build_prompt,
    estimate_tokens,
    extract_response_text,
    parse_cli_output,
    _strip_code_fence,
)


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_includes_system_prompt(self):
        messages = [{"role": "user", "content": "Hi"}]
        prompt = build_prompt(messages, tools=None)
        assert "tool_calls" in prompt  # system prompt instructions
        assert "CONVERSATION HISTORY" in prompt

    def test_includes_user_message(self):
        messages = [{"role": "user", "content": "What is 2+2?"}]
        prompt = build_prompt(messages, tools=None)
        assert "What is 2+2?" in prompt
        assert "[User]:" in prompt

    def test_tool_section_absent_when_no_tools(self):
        messages = [{"role": "user", "content": "Hello"}]
        prompt = build_prompt(messages, tools=None)
        assert "AVAILABLE TOOLS" not in prompt

    def test_tool_section_present_when_tools_provided(self):
        messages = [{"role": "user", "content": "Hello"}]
        tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]
        prompt = build_prompt(messages, tools=tools)
        assert "AVAILABLE TOOLS" in prompt
        assert "read_file" in prompt

    def test_optional_parameter_gets_question_mark(self):
        tools = [
            {
                "name": "search",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            }
        ]
        prompt = build_prompt([{"role": "user", "content": "go"}], tools=tools)
        assert "limit?: integer" in prompt
        assert "query: string" in prompt

    def test_system_message_included(self):
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello"},
        ]
        prompt = build_prompt(messages, tools=None)
        assert "You are a helper." in prompt

    def test_assistant_message_included(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        prompt = build_prompt(messages, tools=None)
        assert "Hi there!" in prompt
        assert "[Assistant]:" in prompt

    def test_assistant_tool_calls_included(self):
        messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
                }],
            },
        ]
        prompt = build_prompt(messages, tools=None)
        assert "read_file" in prompt
        assert "[Assistant called]:" in prompt

    def test_tool_result_included(self):
        messages = [
            {"role": "tool", "name": "read_file", "content": "file content here"},
        ]
        prompt = build_prompt(messages, tools=None)
        assert "file content here" in prompt
        assert "[Tool Result (read_file)]:" in prompt

    def test_long_tool_result_truncated(self):
        messages = [
            {"role": "tool", "name": "t", "content": "x" * 5000},
        ]
        prompt = build_prompt(messages, tools=None)
        assert "truncated" in prompt

    def test_multimodal_content_extracted(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
        prompt = build_prompt(messages, tools=None)
        assert "What do you see?" in prompt

    def test_wrapped_tool_schema(self):
        """Tools with a nested 'function' key are also handled."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "greet",
                    "description": "Greet someone",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
            }
        ]
        prompt = build_prompt([{"role": "user", "content": "hi"}], tools=tools)
        assert "greet" in prompt


# ---------------------------------------------------------------------------
# parse_cli_output
# ---------------------------------------------------------------------------

class TestParseCliOutput:
    def _make_envelope(self, result: str, is_error: bool = False) -> str:
        return json.dumps({"type": "result", "result": result, "is_error": is_error})

    def test_plain_text_response(self):
        inner = json.dumps({"response": "Hello, world!"})
        stdout = self._make_envelope(inner)
        data = parse_cli_output(stdout)
        content = data["choices"][0]["message"]["content"]
        assert content == "Hello, world!"

    def test_tool_call_response(self):
        inner = json.dumps({
            "tool_calls": [
                {"id": "call_1", "name": "read_file", "arguments": {"path": "/tmp/x"}}
            ]
        })
        stdout = self._make_envelope(inner)
        data = parse_cli_output(stdout)
        message = data["choices"][0]["message"]
        assert message["content"] is None
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "read_file"

    def test_multiple_tool_calls(self):
        inner = json.dumps({
            "tool_calls": [
                {"id": "c1", "name": "read_file", "arguments": {"path": "/a"}},
                {"id": "c2", "name": "write_file", "arguments": {"path": "/b", "content": "x"}},
            ]
        })
        stdout = self._make_envelope(inner)
        data = parse_cli_output(stdout)
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[1]["function"]["name"] == "write_file"

    def test_non_json_response_treated_as_text(self):
        stdout = self._make_envelope("This is plain text, not JSON.")
        data = parse_cli_output(stdout)
        content = data["choices"][0]["message"]["content"]
        assert content == "This is plain text, not JSON."

    def test_empty_stdout_returns_fallback(self):
        data = parse_cli_output("")
        content = data["choices"][0]["message"]["content"]
        assert "[CiC]" in content

    def test_cli_error_flag_raises(self):
        stdout = self._make_envelope("Something went wrong", is_error=True)
        with pytest.raises(ClaudeSubprocessError) as exc_info:
            parse_cli_output(stdout)
        assert "error" in str(exc_info.value).lower()

    def test_invalid_envelope_json_raises(self):
        with pytest.raises(ResponseParseError):
            parse_cli_output("{not valid json")

    def test_empty_result_field_returns_fallback(self):
        stdout = json.dumps({"type": "result", "result": "", "is_error": False})
        data = parse_cli_output(stdout)
        content = data["choices"][0]["message"]["content"]
        assert "[CiC]" in content

    def test_code_fence_stripped(self):
        inner = "```json\n" + json.dumps({"response": "Clean answer"}) + "\n```"
        stdout = self._make_envelope(inner)
        data = parse_cli_output(stdout)
        content = data["choices"][0]["message"]["content"]
        assert content == "Clean answer"

    def test_finish_reason_stop_for_text(self):
        inner = json.dumps({"response": "done"})
        stdout = self._make_envelope(inner)
        data = parse_cli_output(stdout)
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_finish_reason_tool_calls(self):
        inner = json.dumps({"tool_calls": [{"id": "c1", "name": "t", "arguments": {}}]})
        stdout = self._make_envelope(inner)
        data = parse_cli_output(stdout)
        assert data["choices"][0]["finish_reason"] == "tool_calls"

    def test_arguments_as_string_preserved(self):
        """Arguments that arrive as a JSON string stay valid JSON strings."""
        inner = json.dumps({
            "tool_calls": [
                {"id": "c1", "name": "fn", "arguments": '{"key": "value"}'}
            ]
        })
        stdout = self._make_envelope(inner)
        data = parse_cli_output(stdout)
        tc = data["choices"][0]["message"]["tool_calls"][0]
        # OpenAI format: arguments is a JSON string
        assert isinstance(tc["function"]["arguments"], str)
        assert json.loads(tc["function"]["arguments"]) == {"key": "value"}

    def test_result_field_with_alternative_key(self):
        """Inner JSON with 'result' key instead of 'response'."""
        inner = json.dumps({"result": "alternate key answer"})
        stdout = self._make_envelope(inner)
        data = parse_cli_output(stdout)
        content = data["choices"][0]["message"]["content"]
        assert content == "alternate key answer"


# ---------------------------------------------------------------------------
# _strip_code_fence
# ---------------------------------------------------------------------------

class TestStripCodeFence:
    def test_strips_json_fence(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        assert _strip_code_fence(text) == '{"key": "value"}'

    def test_strips_plain_fence(self):
        text = "```\nhello\n```"
        assert _strip_code_fence(text) == "hello"

    def test_passthrough_without_fence(self):
        text = "no fences here"
        assert _strip_code_fence(text) == "no fences here"

    def test_passthrough_partial_fence(self):
        text = "```only opening"
        assert _strip_code_fence(text) == "```only opening"


# ---------------------------------------------------------------------------
# estimate_tokens / extract_response_text
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_estimate_tokens_minimum_one(self):
        assert estimate_tokens("") == 1

    def test_estimate_tokens_approx(self):
        # 40 chars ≈ 10 tokens
        assert estimate_tokens("a" * 40) == 10

    def test_extract_text_content(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }]
        }
        assert extract_response_text(data) == "hello"

    def test_extract_tool_calls_as_json(self):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "name": "t"}],
                },
                "finish_reason": "tool_calls",
            }]
        }
        text = extract_response_text(data)
        assert "c1" in text

    def test_extract_empty_when_no_choices(self):
        assert extract_response_text({}) == ""


# ---------------------------------------------------------------------------
# Edge cases: empty inputs and boundary conditions
# ---------------------------------------------------------------------------

class TestBuildPromptEdgeCases:
    def test_empty_messages_list(self):
        """Empty messages list produces a valid prompt with no conversation lines."""
        prompt = build_prompt([], tools=None)
        assert "CONVERSATION HISTORY:" in prompt
        # No user/assistant lines, but the prompt itself is valid
        assert "[User]:" not in prompt

    def test_empty_tools_list_treated_as_no_tools(self):
        """tools=[] should not add the AVAILABLE TOOLS section."""
        prompt = build_prompt([{"role": "user", "content": "hi"}], tools=[])
        assert "AVAILABLE TOOLS" not in prompt

    def test_user_message_with_empty_content_omitted(self):
        """A user message with empty string content should not appear in the prompt."""
        messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "real message"},
        ]
        prompt = build_prompt(messages, tools=None)
        # Only the non-empty user message appears
        assert prompt.count("[User]:") == 1
        assert "real message" in prompt

    def test_user_message_with_none_content_omitted(self):
        """A user message with None content should not appear in the prompt."""
        messages = [{"role": "user", "content": None}]
        prompt = build_prompt(messages, tools=None)
        assert "[User]:" not in prompt

    def test_very_long_prompt(self):
        """Prompt building handles 100K+ character inputs without error."""
        big_content = "x" * 100_000
        messages = [{"role": "user", "content": big_content}]
        prompt = build_prompt(messages, tools=None)
        assert big_content in prompt


class TestParseCliOutputEdgeCases:
    def _make_envelope(self, result: str, is_error: bool = False) -> str:
        return json.dumps({"type": "result", "result": result, "is_error": is_error})

    def test_whitespace_only_stdout_returns_fallback(self):
        """Whitespace-only stdout is treated the same as empty."""
        data = parse_cli_output("   \n\t  ")
        content = data["choices"][0]["message"]["content"]
        assert "[CiC]" in content

    def test_envelope_with_no_result_key(self):
        """Envelope missing the 'result' key returns fallback."""
        stdout = json.dumps({"type": "result", "is_error": False})
        data = parse_cli_output(stdout)
        content = data["choices"][0]["message"]["content"]
        assert "[CiC]" in content

    def test_tool_calls_empty_list_treated_as_text(self):
        """tool_calls: [] is falsy — should fall through to text handling."""
        inner = json.dumps({"tool_calls": [], "response": "fallback text"})
        stdout = self._make_envelope(inner)
        data = parse_cli_output(stdout)
        content = data["choices"][0]["message"]["content"]
        assert content == "fallback text"

    def test_inner_json_with_unknown_keys_falls_back(self):
        """Inner JSON with no 'tool_calls' or 'response' key serialises the dict."""
        inner = json.dumps({"foo": "bar"})
        stdout = self._make_envelope(inner)
        data = parse_cli_output(stdout)
        # Falls back to str(inner_dict) — not a crash
        content = data["choices"][0]["message"]["content"]
        assert isinstance(content, str)

    def test_error_message_truncated_to_500_chars(self):
        """CLI error messages longer than 500 chars are truncated."""
        from cic.exceptions import ClaudeSubprocessError
        long_error = "E" * 1000
        stdout = json.dumps({"type": "result", "result": long_error, "is_error": True})
        with pytest.raises(ClaudeSubprocessError) as exc_info:
            parse_cli_output(stdout)
        # The raised message should not contain the full 1000-char error
        assert len(str(exc_info.value)) < 600
