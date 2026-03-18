# CiC — Context in Claude Code

**A Python library for building AI agents on top of the Claude Code CLI.**

CiC gives you a clean programmatic interface to Claude Code. Define your own tools, let Claude decide which to call, and execute them yourself — the standard agent loop pattern, powered by the `claude` CLI.

---

## What CiC Does

```
                         YOUR AGENT CODE
                              |
                    +---------v----------+
                    |     CiCClient      |
                    |                    |
                    |  messages + tools  |
                    +--------+-----------+
                             |
              Builds prompt, spawns subprocess
                             |
                    +--------v-----------+
                    |   claude --print   |
                    |   --tools ""       |
                    |   --model sonnet   |
                    +--------+-----------+
                             |
              Claude reasons, returns JSON
                             |
                    +--------v-----------+
                    |     CiCClient      |
                    |                    |
                    |  Parses response   |
                    |  into ChatResult   |
                    +--------+-----------+
                             |
               +-------------+-------------+
               |                           |
        Tool calls?                  Text response?
               |                           |
      +--------v--------+         +--------v--------+
      |   result.       |         |   result.       |
      |   tool_calls    |         |   content       |
      |                 |         |                 |
      |  You execute    |         |  Final answer   |
      |  the tools      |         |  from Claude    |
      +--------+--------+         +-----------------+
               |
        Feed results back
        into messages[]
               |
          Loop again
```

**Key insight:** Claude Code's built-in tools are disabled (`--tools ""`). Claude only *decides* which of **your** tools to call. **You** execute them. You stay in full control.

---

## The Agent Loop

This is the core pattern CiC enables:

```
    +------------------+
    |  Define tools    |  <-- Your tools: read_file, query_db, send_email, etc.
    +--------+---------+
             |
             v
    +------------------+
    |  Send task to    |  <-- client.chat(messages, tools=tools)
    |  CiCClient       |
    +--------+---------+
             |
             v
    +------------------+        +------------------+
    |  Claude thinks   | -----> |  "Call read_file  |
    |  about the task  |        |   with path X"   |
    +------------------+        +--------+---------+
                                         |
                                         v
                                +------------------+
                                |  YOUR code runs  |  <-- You execute the tool
                                |  read_file(X)    |
                                +--------+---------+
                                         |
                                    result text
                                         |
                                         v
                                +------------------+
                                |  Feed result     |  <-- Append to messages[]
                                |  back to Claude  |
                                +--------+---------+
                                         |
                                         v
                                +------------------+
                                |  Claude thinks   |  <-- May call another tool
                                |  again...        |      or return final answer
                                +------------------+
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

Different tasks need different models. CiC routes automatically:

```
    Task Complexity          Model Selected         Why
    ===============          ==============         ===================

    "Is X prime?"     ───>   Haiku                  Fast, simple lookup
    "Fix this bug"    ───>   Sonnet                 Balanced reasoning
    "Design a system" ───>   Opus                   Deep, multi-step logic


    +--------------------------------------------------+
    |              CiCClient with routing               |
    |                                                   |
    |   set_complexity("simple")  ──> haiku             |
    |   set_complexity("moderate") ──> sonnet           |
    |   set_complexity("complex") ──> opus              |
    +--------------------------------------------------+
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

```
    Your Python code                     claude CLI process
    ================                     ==================

    client.chat(msgs, tools)
           |
           |  1. Build prompt:
           |     - System: "You have these tools: ..."
           |     - History: [User said X, Assistant called Y, Tool returned Z]
           |     - "Respond with JSON tool_calls or final response"
           |
           +------- stdin -------->  claude --print
                                     --output-format json
                                     --tools ""
                                     --model sonnet
                                     --no-session-persistence

                                        Claude reads prompt
                                        Reasons about task
                                        Picks tools (or answers)

           <------ stdout ---------  {"type":"result",
                                      "result": "{\"tool_calls\":[...]}"}
           |
           |  2. Parse JSON envelope
           |  3. Extract inner response
           |  4. Convert to OpenAI format:
           |     {"choices":[{"message":{"tool_calls":[...]}}]}
           |
           v
    return ChatResult(
        tool_calls=[ToolCall(name="read_file", arguments={...})],
        content=None,
        model="cic/sonnet"
    )
```

Each call is a **fresh subprocess** — no state leaks between calls. Your agent code maintains the conversation history in `messages[]` and passes it each time.

---

## Quick Start

### Install

```bash
git clone https://github.com/maclarensg/CiC
cd CiC
pip install -e .
```

**Prerequisite:** [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated (`npm install -g @anthropic-ai/claude-code && claude`).

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

```
    Existing code using OpenAI format:

    response["choices"][0]["message"]["content"]
    response["choices"][0]["message"]["tool_calls"]
    response["choices"][0]["finish_reason"]

                        |
                  Just swap to:
                        |
                        v

    response = client.chat_openai_format(messages, tools=tools)

    # Same structure, same access pattern
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
- **~1-2s overhead per call** — subprocess spawn time; not for high-throughput
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
