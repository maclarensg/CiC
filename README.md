# CiC — Context in Claude Code

**A Python library for building AI agents on top of the Claude Code CLI.**

CiC provides a programmatic chat interface over the `claude` CLI, giving developers a clean way to integrate Claude Code into agent loops, automation scripts, and custom toolchains. It handles prompt construction, response parsing, tool call formatting, and multi-model routing — so you can focus on your agent logic.

```python
from cic import CiCClient

client = CiCClient(model="sonnet")
result = client.chat([{"role": "user", "content": "Hello!"}])
print(result.content)  # "Hello! How can I help you today?"
```

---

## Why CiC?

Claude Code's `--print` mode supports headless, non-interactive use — but working with raw subprocess I/O, JSON parsing, and tool call formatting is tedious. CiC wraps this into a clean Python interface:

- **Structured tool use** — Define tools, get back parsed `ToolCall` objects, feed results back. Standard agent loop pattern.
- **Smart routing** — Route simple tasks to Haiku, standard work to Sonnet, complex reasoning to Opus. Automatic model selection by task complexity.
- **OpenAI-compatible format** — Drop-in response format compatible with existing code that consumes OpenAI chat completions.
- **Fresh context per call** — Each invocation is a clean subprocess. No context window bloat across calls.
- **Async support** — Both sync `chat()` and async `achat()` interfaces.

---

## Important: Usage Terms

CiC uses the official `claude` CLI binary and respects Anthropic's authentication. It does **not** extract, proxy, or redistribute OAuth tokens.

Users are responsible for complying with [Anthropic's Consumer Terms of Service](https://www.anthropic.com/legal/consumer-terms) and the [Claude Code usage policies](https://code.claude.com/docs/en/legal-and-compliance). Key points:

- **Subscription limits apply.** Claude Pro/Max plans have rolling usage windows. CiC does not circumvent or modify these limits in any way.
- **Individual use.** Claude subscriptions are for individual use. Do not use CiC to pool, share, or resell subscription access.
- **Ordinary usage.** Anthropic's published limits assume "ordinary, individual usage of Claude Code." Sustained high-volume automated workloads may exceed what your plan is sized for.
- **For production/high-throughput workloads**, consider using the [Anthropic API](https://docs.anthropic.com/) with API key authentication, which is designed and priced for programmatic access at scale.

CiC is a developer tool that wraps the official CLI. It is not affiliated with or endorsed by Anthropic.

---

## Prerequisites

1. **Claude Code CLI** installed and authenticated:

```bash
npm install -g @anthropic-ai/claude-code
claude  # authenticate on first run
```

2. **Python 3.10+**

---

## Installation

```bash
# From source
git clone https://github.com/maclarensg/CiC
cd CiC
pip install -e .
```

---

## Quick Start

### Basic chat

```python
from cic import CiCClient

client = CiCClient(model="sonnet")
result = client.chat([{"role": "user", "content": "What is the capital of France?"}])
print(result.content)
```

### Multi-turn conversation

```python
messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What is my name?"},
]
result = client.chat(messages)
print(result.content)  # "Your name is Alice."
```

### Async

```python
import asyncio
from cic import CiCClient

async def main():
    client = CiCClient(model="sonnet")
    result = await client.achat([{"role": "user", "content": "Hello async!"}])
    print(result.content)

asyncio.run(main())
```

---

## Tool Use — Building Agents

CiC supports tool use for building agent loops. Define tools, pass them to `chat()`, execute what the model requests, and feed results back. Repeat until the model returns a final answer.

```python
from cic import CiCClient

tools = [
    {
        "name": "read_file",
        "description": "Read the contents of a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    }
]

client = CiCClient(model="sonnet")
messages = [{"role": "user", "content": "Read /etc/hostname and tell me what it says."}]

while True:
    result = client.chat(messages, tools=tools)

    if not result.has_tool_calls:
        print("Final answer:", result.content)
        break

    # Add assistant's tool call decisions to history
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

    # Execute tools and add results to history
    for tc in result.tool_calls:
        print(f"Calling {tc.name}({tc.arguments})")
        with open(tc.arguments["path"]) as f:
            output = f.read()
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": output,
        })
```

---

## Smart Routing

Route different task complexities to different models automatically.

```python
from cic import CiCClient

client = CiCClient(
    routing={
        "simple":   "haiku",   # Fast — lookups, classification, formatting
        "moderate": "sonnet",  # Balanced — most tasks
        "complex":  "opus",    # Powerful — analysis, multi-step reasoning
    }
)

# Set complexity before each call
client.set_complexity("simple")
result = client.chat([{"role": "user", "content": "Is 17 a prime number?"}])
# Uses Haiku

client.set_complexity("complex")
result = client.chat([{"role": "user", "content": "Write a detailed analysis of..."}])
# Uses Opus
```

---

## API Reference

### `CiCClient`

```python
CiCClient(
    model: str | None = None,               # Fixed model. Disables routing.
    routing: dict[str, str] | None = None,   # Complexity → model map
    timeout: float = 120.0,                  # Subprocess timeout in seconds
    claude_path: str | None = None,          # Explicit path to claude binary
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `chat(messages, *, tools=None) -> ChatResult` | Synchronous chat |
| `achat(messages, *, tools=None) -> ChatResult` | Async chat |
| `chat_openai_format(messages, *, tools=None) -> dict` | Returns raw OpenAI-compatible dict |
| `set_complexity(level: str)` | Set complexity for smart routing |
| `set_model(model: str)` | Override model for subsequent calls |
| `active_model -> str` | Property: which model will be used next |

### `ChatResult`

```python
@dataclass
class ChatResult:
    content: str | None        # Text response, or None if tool calls were made
    tool_calls: list[ToolCall] # Tool call decisions (empty if text response)
    model: str                 # Model used, e.g. "cic/sonnet"
    usage: TokenUsage          # Estimated token usage
    raw: dict                  # Raw OpenAI-format response dict

    has_tool_calls: bool       # Property: True if tool_calls is non-empty
    to_openai_dict() -> dict   # Returns raw (alias for drop-in compat)
```

### `ToolCall`

```python
@dataclass
class ToolCall:
    id: str                    # Tool call ID (e.g. "call_1")
    name: str                  # Tool name to invoke
    arguments: dict[str, Any]  # Parsed arguments

    arguments_json() -> str    # Serialise arguments to JSON string
```

### `TokenUsage`

```python
@dataclass
class TokenUsage:
    prompt_tokens: int         # Estimated prompt tokens (chars / 4)
    completion_tokens: int     # Estimated completion tokens
    total_tokens: int          # Sum of prompt + completion
```

### `CiCRouter`

```python
CiCRouter(
    simple: str = "haiku",
    moderate: str = "sonnet",
    complex: str = "opus",
    extra: dict[str, str] | None = None,  # Additional mappings
)

CiCRouter.from_dict(mapping: dict[str, str]) -> CiCRouter
router.model_for(complexity: str) -> str
router.as_dict() -> dict[str, str]
```

### Exceptions

| Exception | When raised |
|-----------|-------------|
| `CiCError` | Base class for all CiC errors |
| `ClaudeNotFoundError` | `claude` CLI not in PATH |
| `ClaudeTimeoutError` | Subprocess exceeded timeout |
| `ClaudeSubprocessError` | CLI returned `is_error: true` |
| `ResponseParseError` | Could not parse CLI envelope JSON |

---

## OpenAI Drop-In Compatibility

`chat_openai_format()` returns a dict that matches the OpenAI chat completion response structure:

```python
response = client.chat_openai_format(messages, tools=tools)
# response["choices"][0]["message"]["content"]
# response["choices"][0]["message"]["tool_calls"]
# response["choices"][0]["finish_reason"]
```

This makes it straightforward to integrate with code that already consumes OpenAI-format responses.

---

## Limitations

- **No streaming.** Each call waits for the full response. Not suitable for real-time UIs.
- **Subprocess overhead.** Each call spawns a process (~1–2s). Not designed for high-throughput production workloads.
- **Token estimates only.** Usage counts are approximated (chars / 4), not exact.
- **Tool call format.** CiC injects a system prompt instructing Claude to respond in JSON. Complex system prompts may interact with this.
- **Model names.** CiC passes model names directly to `claude --model`. Use the same names the CLI accepts (e.g. `"sonnet"`, `"opus"`, `"haiku"`).
- **Subscription limits apply.** Your Claude plan's usage limits are unchanged. CiC does not modify, bypass, or circumvent any rate limits or usage policies.

---

## How It Works

Each `chat()` call:

1. **Builds a prompt** from the messages array + tool descriptions
2. **Spawns** `claude --print --output-format json --tools "" --model <model>` as a subprocess
3. **Parses** the JSON response from the CLI envelope
4. **Converts** Claude's response to OpenAI wire format (tool_calls or text content)
5. **Returns** a structured `ChatResult`

The `--tools ""` flag disables Claude Code's built-in tools so Claude only reasons about your custom tools without executing anything on its own. Your code is in full control of tool execution.

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
PYTHONPATH=src pytest

# Run tests with verbose output
PYTHONPATH=src pytest -v
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Disclaimer

CiC is an independent open-source project. It is not affiliated with, endorsed by, or sponsored by Anthropic. "Claude" and "Claude Code" are trademarks of Anthropic, PBC. Users are solely responsible for ensuring their use of CiC complies with all applicable terms of service.
