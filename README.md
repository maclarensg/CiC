# CiC — Context in Claude Code

**Build AI agents using your Claude Pro/Max subscription. Zero per-token costs. No API keys.**

CiC is a Python library that wraps the `claude` CLI to provide a programmatic chat interface. Instead of calling Anthropic's REST API with per-token billing, it spawns `claude --print` subprocesses that use the caller's active Claude subscription.

```python
from cic import CiCClient

client = CiCClient(model="sonnet")
result = client.chat([{"role": "user", "content": "Hello!"}])
print(result.content)  # "Hello! How can I help you today?"
```

---

## Why CiC?

| | CiC | Anthropic API | OpenAI API |
|---|---|---|---|
| Cost | Free (uses subscription) | Per token | Per token |
| API key | Not required | Required | Required |
| Setup | Install claude CLI | Get API key | Get API key |
| Models | haiku, sonnet, opus | claude-3-x | gpt-4o, etc. |
| Streaming | Not supported | Yes | Yes |
| Throughput | Low (subprocess per call) | High | High |

CiC is for developers who have a Claude Pro or Max subscription and want to build agents, automation scripts, or hobby projects without accumulating API charges. It is not a replacement for direct API access in production systems with high throughput requirements.

---

## Prerequisites

1. A **Claude Pro or Max subscription** at [claude.ai](https://claude.ai)
2. **Claude Code CLI** installed and authenticated:

```bash
npm install -g @anthropic-ai/claude-code
claude  # authenticate on first run
```

3. **Python 3.10+**

---

## Installation

```bash
# From PyPI (once published)
pip install cic

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

CiC supports tool use. Define tools in OpenAI format, pass them to `chat()`, execute what the model requests, and feed results back. Repeat until the model returns a final answer.

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

Route different task complexities to different models. Faster/cheaper models handle simple tasks; powerful models handle heavy reasoning.

```python
from cic import CiCClient

client = CiCClient(
    routing={
        "simple":   "haiku",   # Fast — lookups, classification, formatting
        "moderate": "sonnet",  # Balanced — most tasks
        "complex":  "opus",    # Heavy — analysis, long-form reasoning
    }
)

# Pick the right model for each task
client.set_complexity("simple")
result = client.chat([{"role": "user", "content": "Is 17 a prime number?"}])
# Uses haiku

client.set_complexity("complex")
result = client.chat([{"role": "user", "content": "Write a detailed analysis of..."}])
# Uses opus
```

---

## API Reference

### `CiCClient`

```python
CiCClient(
    model: str | None = None,           # Fixed model. Disables routing.
    routing: dict[str, str] | None = None,  # Complexity → model map
    timeout: float = 120.0,             # Subprocess timeout in seconds
    claude_path: str | None = None,     # Explicit path to claude binary
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `chat(messages, *, tools=None) -> ChatResult` | Synchronous chat |
| `achat(messages, *, tools=None) -> ChatResult` | Async chat |
| `chat_openai_format(messages, *, tools=None) -> dict` | Returns raw OpenAI dict |
| `set_complexity(level: str)` | Set complexity for smart routing |
| `set_model(model: str)` | Override model for subsequent calls |
| `active_model -> str` | Property: which model will be used next |

### `ChatResult`

```python
@dataclass
class ChatResult:
    content: str | None       # Text response, or None if tool calls were made
    tool_calls: list[ToolCall] # Tool call decisions (empty if text response)
    model: str                # Model used, e.g. "cic/sonnet"
    usage: TokenUsage         # Estimated token usage
    raw: dict                 # Raw OpenAI-format response dict

    has_tool_calls: bool      # Property: True if tool_calls is non-empty
    to_openai_dict() -> dict  # Returns raw (alias for drop-in compat)
```

### `ToolCall`

```python
@dataclass
class ToolCall:
    id: str                   # Tool call ID (e.g. "call_1")
    name: str                 # Tool name to invoke
    arguments: dict[str, Any] # Parsed arguments

    arguments_json() -> str   # Serialise arguments to JSON string
```

### `TokenUsage`

```python
@dataclass
class TokenUsage:
    prompt_tokens: int        # Estimated prompt tokens (chars / 4)
    completion_tokens: int    # Estimated completion tokens
    total_tokens: int         # Sum of prompt + completion
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

`chat_openai_format()` returns a dict that matches the OpenAI chat completion wire format:

```python
response = client.chat_openai_format(messages, tools=tools)
# response["choices"][0]["message"]["content"]
# response["choices"][0]["message"]["tool_calls"]
# response["choices"][0]["finish_reason"]
```

This makes it straightforward to migrate code that already consumes OpenAI responses.

---

## Limitations

- **No streaming.** Each call waits for the full response. Unsuitable for real-time UIs.
- **Low throughput.** Each call spawns a subprocess (~1–2s overhead). Not suitable for high-volume production workloads.
- **Token estimates only.** Usage counts are approximated (chars / 4), not exact.
- **Tool call format.** CiC injects a system prompt instructing Claude to respond in JSON. Complex system prompts may conflict with this.
- **Model names.** CiC passes model names directly to `claude --model`. Use the same names the `claude` CLI accepts (e.g. `"sonnet"`, `"opus"`, `"haiku"`).
- **Subscription limits apply.** Claude Pro/Max plans have usage limits. CiC does not circumvent them.

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with verbose output
pytest -v
```

---

## License

MIT — see [LICENSE](LICENSE).
