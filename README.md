# CiC — Context in Claude Code

**A Python library for building AI agents on top of the Claude Code CLI.**

CiC gives you a clean programmatic interface to Claude Code. Define your own tools, let Claude decide which to call, and execute them yourself — the standard agent loop pattern, powered by the `claude` CLI.

---

## What CiC Does

```mermaid
flowchart LR
    A[Your Agent Code] -->|messages + tools| B[CiCClient]
    B -->|builds prompt| C["claude --print\n--tools &quot;&quot;\n--model sonnet"]
    C -->|JSON response| B
    B --> D{Tool calls?}
    D -->|Yes| E["result.tool_calls\n→ You execute"]
    D -->|No| F["result.content\n→ Final answer"]
    E -->|results back| A

    style B fill:#4a9eff,color:#fff
    style C fill:#f59e0b,color:#fff
    style E fill:#10b981,color:#fff
    style F fill:#8b5cf6,color:#fff
```

**Key insight:** Claude Code's built-in tools are disabled (`--tools ""`). Claude only *decides* which of **your** tools to call. **You** execute them. You stay in full control.

---

## The Agent Loop

This is the core pattern CiC enables:

```mermaid
flowchart TD
    A[Define your tools] --> B["client.chat(messages, tools)"]
    B --> C{Claude's response}
    C -->|tool_calls| D[Your code executes the tool]
    C -->|text content| E[Final answer - done!]
    D --> F[Append result to messages]
    F --> B

    style A fill:#6366f1,color:#fff
    style B fill:#4a9eff,color:#fff
    style D fill:#10b981,color:#fff
    style E fill:#8b5cf6,color:#fff
    style F fill:#f59e0b,color:#fff
```

```python
from cic import CiCClient

client = CiCClient(model="sonnet")
messages = [{"role": "user", "content": "Read config.yaml and summarize it"}]

while True:
    result = client.chat(messages, tools=my_tools)

    if not result.has_tool_calls:
        print("Done:", result.content)
        break

    # Claude decided — now YOU execute
    for tc in result.tool_calls:
        output = execute_tool(tc.name, tc.arguments)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
```

---

## Smart Routing

Different tasks need different models. CiC routes automatically based on complexity:

```mermaid
flowchart LR
    subgraph Your Code
        S["set_complexity()"]
    end

    subgraph CiC Router
        S --> R{Complexity?}
        R -->|simple| H["Haiku\nFast lookups"]
        R -->|moderate| SO["Sonnet\nBalanced"]
        R -->|complex| O["Opus\nDeep reasoning"]
    end

    H --> CLI["claude --model haiku"]
    SO --> CLI2["claude --model sonnet"]
    O --> CLI3["claude --model opus"]

    style H fill:#10b981,color:#fff
    style SO fill:#4a9eff,color:#fff
    style O fill:#8b5cf6,color:#fff
```

```python
client = CiCClient(routing={
    "simple":   "haiku",
    "moderate": "sonnet",
    "complex":  "opus",
})

client.set_complexity("simple")
result = client.chat(messages)   # Uses Haiku

client.set_complexity("complex")
result = client.chat(messages)   # Uses Opus
```

---

## How It Works Under the Hood

```mermaid
sequenceDiagram
    participant App as Your App
    participant CiC as CiCClient
    participant CLI as claude CLI

    App->>CiC: chat(messages, tools)

    Note over CiC: Build prompt:<br/>System + tool descriptions<br/>+ conversation history

    CiC->>CLI: stdin: prompt text
    Note over CLI: claude --print<br/>--output-format json<br/>--tools ""<br/>--model sonnet

    Note over CLI: Claude reasons<br/>about the task...<br/>Picks tools or answers

    CLI-->>CiC: stdout: {"type":"result", "result":"..."}

    Note over CiC: Parse JSON envelope<br/>Extract inner response<br/>Convert to OpenAI format

    CiC-->>App: ChatResult(tool_calls=[...])

    Note over App: Execute tools,<br/>append results,<br/>call chat() again
```

Each call is a **fresh subprocess** — no state leaks between calls. Your agent code maintains the conversation history in `messages[]` and passes it each time.

---

## Comparing Approaches

```mermaid
flowchart TD
    subgraph Traditional["Traditional: REST API"]
        A1[Your Code] -->|HTTP POST| A2[api.anthropic.com]
        A2 -->|JSON response| A1
        A3["Requires API key\nPay per token\nHigh throughput"]
    end

    subgraph CiC["CiC: Claude CLI"]
        B1[Your Code] -->|subprocess| B2[claude --print]
        B2 -->|JSON response| B1
        B3["Uses CLI auth\nSubscription limits apply\nFresh context per call"]
    end

    style Traditional fill:#1e293b,color:#fff
    style CiC fill:#1e293b,color:#fff
    style A2 fill:#f59e0b,color:#fff
    style B2 fill:#4a9eff,color:#fff
```

CiC is for developers building tools and agents on top of Claude Code. For production workloads with high throughput requirements, use the [Anthropic API](https://docs.anthropic.com/) directly.

---

## Quick Start

### Install

```bash
git clone https://github.com/maclarensg/CiC
cd CiC
pip install -e .
```

**Prerequisite:** [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated.

```bash
npm install -g @anthropic-ai/claude-code
claude  # authenticate on first run
```

### Basic chat

```python
from cic import CiCClient

client = CiCClient(model="sonnet")
result = client.chat([{"role": "user", "content": "What is the capital of France?"}])
print(result.content)
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

### Tool use agent

```python
from cic import CiCClient

tools = [
    {
        "name": "read_file",
        "description": "Read a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]

client = CiCClient(model="sonnet")
messages = [{"role": "user", "content": "Read /etc/hostname and tell me what it says."}]

while True:
    result = client.chat(messages, tools=tools)

    if not result.has_tool_calls:
        print("Answer:", result.content)
        break

    messages.append({
        "role": "assistant", "content": None,
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": tc.arguments_json()}}
            for tc in result.tool_calls
        ],
    })

    for tc in result.tool_calls:
        with open(tc.arguments["path"]) as f:
            output = f.read()
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": output})
```

---

## OpenAI Drop-In Compatibility

```python
response = client.chat_openai_format(messages, tools=tools)

# Same structure as OpenAI responses:
response["choices"][0]["message"]["content"]
response["choices"][0]["message"]["tool_calls"]
response["choices"][0]["finish_reason"]
```

---

## API Reference

### `CiCClient`

```python
CiCClient(
    model: str | None = None,               # Fixed model ("sonnet", "opus", "haiku")
    routing: dict[str, str] | None = None,   # Complexity -> model map
    timeout: float = 120.0,                  # Subprocess timeout (seconds)
    claude_path: str | None = None,          # Path to claude binary
)
```

| Method | Description |
|--------|-------------|
| `chat(messages, *, tools=None) -> ChatResult` | Synchronous chat |
| `achat(messages, *, tools=None) -> ChatResult` | Async chat |
| `chat_openai_format(messages, *, tools=None) -> dict` | Returns OpenAI-compatible dict |
| `set_complexity(level: str)` | Set complexity for smart routing |
| `set_model(model: str)` | Override model for next call |
| `active_model -> str` | Property: model that will be used |

### `ChatResult`

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str \| None` | Text response (None if tool calls) |
| `tool_calls` | `list[ToolCall]` | Tool call decisions |
| `has_tool_calls` | `bool` | True if tool_calls is non-empty |
| `model` | `str` | Model used (e.g. `"cic/sonnet"`) |
| `usage` | `TokenUsage` | Estimated token usage |
| `raw` | `dict` | Raw OpenAI-format dict |

### `ToolCall`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Call ID (e.g. `"call_1"`) |
| `name` | `str` | Tool name |
| `arguments` | `dict` | Parsed arguments |
| `arguments_json()` | `str` | Arguments as JSON string |

### Exceptions

| Exception | When |
|-----------|------|
| `ClaudeNotFoundError` | `claude` CLI not in PATH |
| `ClaudeTimeoutError` | Subprocess exceeded timeout |
| `ClaudeSubprocessError` | CLI returned an error |
| `ResponseParseError` | Could not parse response JSON |

---

## Limitations

- **No streaming** — each call waits for the full response
- **~1-2s overhead per call** — subprocess spawn time
- **Token estimates only** — usage is approximated (chars / 4)
- **Subscription limits apply** — your Claude plan's limits are unchanged; CiC does not modify, bypass, or circumvent any usage policies

---

## Important: Usage Terms

CiC uses the official `claude` CLI binary and respects Anthropic's authentication. It does **not** extract, proxy, or redistribute OAuth tokens.

Users are responsible for complying with [Anthropic's Consumer Terms of Service](https://www.anthropic.com/legal/consumer-terms) and the [Claude Code usage policies](https://code.claude.com/docs/en/legal-and-compliance):

- **Subscription limits apply.** Rolling usage windows are unchanged.
- **Individual use.** Do not pool, share, or resell subscription access.
- **For production/high-throughput**, consider the [Anthropic API](https://docs.anthropic.com/) with API key authentication.

---

## Development

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -v
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*CiC is an independent open-source project, not affiliated with or endorsed by Anthropic. "Claude" and "Claude Code" are trademarks of Anthropic, PBC.*
