"""Tests for CiC prompt building and response parsing utilities.

Hybrid mode: Claude executes file edits using its own built-in tools.
Custom tools are reported back via --json-schema structured output in
pending_tool_calls. The caller (agent loop) executes those custom tools.
"""

import json
import pytest

from cic.exceptions import ClaudeSubprocessError, ResponseParseError
from cic.utils import (
    DEFAULT_NATIVE_TOOLS,
    STRUCTURED_OUTPUT_SCHEMA,
    build_prompt,
    estimate_tokens,
    extract_response_text,
    filter_custom_tools,
    parse_cli_output,
)


# ---------------------------------------------------------------------------
# filter_custom_tools
# ---------------------------------------------------------------------------

class TestFilterCustomTools:
    def test_strips_native_file_tools(self):
        tools = [
            {"function": {"name": "file_read", "description": "Read", "parameters": {}}},
            {"function": {"name": "file_edit", "description": "Edit", "parameters": {}}},
            {"function": {"name": "shell_exec", "description": "Exec", "parameters": {}}},
            {"function": {"name": "notify_done", "description": "Notify", "parameters": {}}},
        ]
        filtered = filter_custom_tools(tools)
        names = [_get_name(t) for t in filtered]
        assert "file_read" not in names
        assert "file_edit" not in names
        assert "shell_exec" not in names
        assert "notify_done" in names

    def test_keeps_all_non_native_tools(self):
        tools = [
            {"function": {"name": "my_tool_a", "description": "A", "parameters": {}}},
            {"function": {"name": "my_tool_b", "description": "B", "parameters": {}}},
        ]
        assert len(filter_custom_tools(tools)) == 2

    def test_empty_input(self):
        assert filter_custom_tools([]) == []

    def test_all_native_returns_empty(self):
        tools = [
            {"function": {"name": "file_read", "description": "R", "parameters": {}}},
            {"function": {"name": "content_search", "description": "S", "parameters": {}}},
        ]
        assert filter_custom_tools(tools) == []

    def test_custom_native_set(self):
        tools = [
            {"function": {"name": "my_native", "description": "N", "parameters": {}}},
            {"function": {"name": "my_custom", "description": "C", "parameters": {}}},
        ]
        filtered = filter_custom_tools(tools, native_tools=frozenset({"my_native"}))
        names = [_get_name(t) for t in filtered]
        assert "my_native" not in names
        assert "my_custom" in names


def _get_name(tool):
    fn = tool.get("function", tool)
    return fn.get("name", "")


# ---------------------------------------------------------------------------
# build_prompt (hybrid mode)
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_includes_system_prompt(self):
        messages = [{"role": "user", "content": "Hi"}]
        prompt = build_prompt(messages, tools=None)
        assert "built-in" in prompt.lower() or "file tools" in prompt.lower()
        assert "CONVERSATION HISTORY" in prompt

    def test_includes_user_message(self):
        messages = [{"role": "user", "content": "What is 2+2?"}]
        prompt = build_prompt(messages, tools=None)
        assert "What is 2+2?" in prompt
        assert "[User]:" in prompt

    def test_no_custom_tool_section_when_all_native(self):
        """Native tools (file_read, shell_exec) are NOT described — Claude handles them."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [
            {
                "function": {
                    "name": "file_read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            }
        ]
        prompt = build_prompt(messages, tools=tools)
        assert "CUSTOM TOOLS" not in prompt
        assert "file_read" not in prompt

    def test_custom_tool_section_present_for_non_native_tools(self):
        messages = [{"role": "user", "content": "Hello"}]
        tools = [
            {
                "function": {
                    "name": "notify_done",
                    "description": "Notify completion",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                }
            }
        ]
        prompt = build_prompt(messages, tools=tools)
        assert "CUSTOM TOOLS" in prompt
        assert "notify_done" in prompt

    def test_native_tools_stripped_custom_tools_kept(self):
        """Mix of native and custom — only custom appear in prompt."""
        tools = [
            {"function": {"name": "file_read", "description": "Read", "parameters": {}}},
            {"function": {"name": "my_custom_tool", "description": "Custom", "parameters": {}}},
        ]
        prompt = build_prompt([{"role": "user", "content": "go"}], tools=tools)
        assert "my_custom_tool" in prompt
        assert "file_read" not in prompt

    def test_optional_parameter_gets_question_mark(self):
        tools = [
            {
                "function": {
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
                    "function": {"name": "notify_done", "arguments": '{"message": "done"}'},
                }],
            },
        ]
        prompt = build_prompt(messages, tools=None)
        assert "notify_done" in prompt
        assert "[Assistant called]:" in prompt

    def test_tool_result_included(self):
        messages = [
            {"role": "tool", "name": "notify_done", "content": "Notification sent"},
        ]
        prompt = build_prompt(messages, tools=None)
        assert "Notification sent" in prompt
        assert "[Tool Result (notify_done)]:" in prompt

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
# parse_cli_output (hybrid mode)
# ---------------------------------------------------------------------------

class TestParseCliOutput:
    def _make_envelope(
        self,
        result: str = "",
        is_error: bool = False,
        structured_output: dict | None = None,
    ) -> str:
        env = {"type": "result", "result": result, "is_error": is_error}
        if structured_output is not None:
            env["structured_output"] = structured_output
        return json.dumps(env)

    def _hybrid_envelope(
        self,
        summary: str = "Done",
        files_modified: list | None = None,
        pending_tool_calls: list | None = None,
        blocked: str | None = None,
    ) -> str:
        structured = {
            "summary": summary,
            "files_modified": files_modified or [],
            "pending_tool_calls": pending_tool_calls or [],
        }
        if blocked:
            structured["blocked"] = blocked
        return self._make_envelope(structured_output=structured)

    def test_structured_output_no_pending_calls_returns_text(self):
        stdout = self._hybrid_envelope(summary="All done.")
        data = parse_cli_output(stdout)
        content = data["choices"][0]["message"]["content"]
        assert "All done." in content

    def test_structured_output_with_pending_calls_returns_tool_calls(self):
        stdout = self._hybrid_envelope(
            summary="Card moved",
            pending_tool_calls=[
                {"name": "notify_done", "arguments": {"message": "Task complete"}}
            ],
        )
        data = parse_cli_output(stdout)
        message = data["choices"][0]["message"]
        assert message["content"] is None
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "notify_done"

    def test_structured_output_blocked_returns_error(self):
        stdout = self._hybrid_envelope(blocked="File not found")
        data = parse_cli_output(stdout)
        content = data["choices"][0]["message"]["content"]
        assert "BLOCKED" in content

    def test_structured_output_multiple_tool_calls(self):
        stdout = self._hybrid_envelope(
            summary="Multiple actions",
            pending_tool_calls=[
                {"name": "tool_a", "arguments": {"x": 1}},
                {"name": "tool_b", "arguments": {"y": 2}},
            ],
        )
        data = parse_cli_output(stdout)
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "tool_a"
        assert tool_calls[1]["function"]["name"] == "tool_b"

    def test_fallback_to_result_field_when_no_structured(self):
        """No structured_output → falls back to result field as plain text."""
        stdout = self._make_envelope(result="Fallback plain text")
        data = parse_cli_output(stdout)
        content = data["choices"][0]["message"]["content"]
        assert "Fallback plain text" in content

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

    def test_finish_reason_stop_for_text(self):
        stdout = self._hybrid_envelope(summary="done")
        data = parse_cli_output(stdout)
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_finish_reason_tool_calls(self):
        stdout = self._hybrid_envelope(
            pending_tool_calls=[{"name": "t", "arguments": {}}]
        )
        data = parse_cli_output(stdout)
        assert data["choices"][0]["finish_reason"] == "tool_calls"

    def test_arguments_serialized_to_json_string(self):
        """Arguments dict → JSON string in OpenAI format."""
        stdout = self._hybrid_envelope(
            pending_tool_calls=[{"name": "fn", "arguments": {"key": "value"}}]
        )
        data = parse_cli_output(stdout)
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert isinstance(tc["function"]["arguments"], str)
        assert json.loads(tc["function"]["arguments"]) == {"key": "value"}

    def test_error_message_truncated_to_500_chars(self):
        long_error = "E" * 1000
        stdout = json.dumps({"type": "result", "result": long_error, "is_error": True})
        with pytest.raises(ClaudeSubprocessError) as exc_info:
            parse_cli_output(stdout)
        assert len(str(exc_info.value)) < 600


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
# STRUCTURED_OUTPUT_SCHEMA is valid JSON
# ---------------------------------------------------------------------------

class TestStructuredOutputSchema:
    def test_schema_is_valid_json(self):
        schema = json.loads(STRUCTURED_OUTPUT_SCHEMA)
        assert schema["type"] == "object"
        assert "summary" in schema["properties"]
        assert "files_modified" in schema["properties"]
        assert "pending_tool_calls" in schema["properties"]

    def test_schema_required_fields(self):
        schema = json.loads(STRUCTURED_OUTPUT_SCHEMA)
        required = schema["required"]
        assert "summary" in required
        assert "files_modified" in required
        assert "pending_tool_calls" in required


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBuildPromptEdgeCases:
    def test_empty_messages_list(self):
        prompt = build_prompt([], tools=None)
        assert "CONVERSATION HISTORY:" in prompt
        assert "[User]:" not in prompt

    def test_empty_tools_list_treated_as_no_tools(self):
        prompt = build_prompt([{"role": "user", "content": "hi"}], tools=[])
        assert "CUSTOM TOOLS" not in prompt

    def test_user_message_with_empty_content_omitted(self):
        messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "real message"},
        ]
        prompt = build_prompt(messages, tools=None)
        assert prompt.count("[User]:") == 1
        assert "real message" in prompt

    def test_user_message_with_none_content_omitted(self):
        messages = [{"role": "user", "content": None}]
        prompt = build_prompt(messages, tools=None)
        assert "[User]:" not in prompt

    def test_very_long_prompt(self):
        big_content = "x" * 100_000
        messages = [{"role": "user", "content": big_content}]
        prompt = build_prompt(messages, tools=None)
        assert big_content in prompt


class TestParseCliOutputEdgeCases:
    def _make_envelope(self, result: str = "", is_error: bool = False) -> str:
        return json.dumps({"type": "result", "result": result, "is_error": is_error})

    def test_whitespace_only_stdout_returns_fallback(self):
        data = parse_cli_output("   \n\t  ")
        content = data["choices"][0]["message"]["content"]
        assert "[CiC]" in content

    def test_envelope_with_no_result_key_and_no_structured(self):
        stdout = json.dumps({"type": "result", "is_error": False})
        data = parse_cli_output(stdout)
        content = data["choices"][0]["message"]["content"]
        assert "[CiC]" in content

    def test_empty_pending_tool_calls_returns_summary(self):
        """pending_tool_calls: [] → fall through to summary as content."""
        env = {
            "type": "result",
            "result": "",
            "is_error": False,
            "structured_output": {
                "summary": "All done here",
                "files_modified": [],
                "pending_tool_calls": [],
            },
        }
        data = parse_cli_output(json.dumps(env))
        content = data["choices"][0]["message"]["content"]
        assert "All done here" in content
