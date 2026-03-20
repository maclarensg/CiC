"""CiCClient — the main entry point for building agents with CiC.

CiCClient wraps the ``claude`` CLI to provide a programmatic chat interface.
Instead of calling a REST API with per-token billing, it spawns
``claude --print`` subprocesses that use the caller's active Claude
Pro/Max subscription.

Two modes are available:

**Hybrid mode** (default, ``hybrid=True``): Claude keeps its built-in file tools
(Bash, Edit, Read, Write) and executes file edits ITSELF inside the subprocess.
Custom tools (defined by the caller) are reported back via ``--json-schema``
structured output in the ``pending_tool_calls`` field. The agent loop then
executes only those tools.

This fixes the phantom edit bug where ``--tools ""`` caused Claude to return tool
calls as narrative text (never executed), falsely marking tasks as done.

**Non-hybrid mode** (``hybrid=False``): Claude has NO built-in tools. Instead,
it outputs ONE action per call via ``--json-schema`` with an action-enum schema.
The caller executes every tool — including file operations — and calls ``chat()``
again with the result. This gives the caller full control and verifiability over
every operation at the cost of more round-trips.

Typical usage::

    from cic import CiCClient

    client = CiCClient(model="sonnet")
    result = client.chat([{"role": "user", "content": "Hello!"}])
    print(result.content)

For hybrid mode (agent loops with custom tools)::

    client = CiCClient(model="sonnet")
    # Only define custom tools — file tools are handled by Claude natively
    tools = [{"name": "notify_done", "description": "...", "parameters": {...}}]

    result = client.chat(messages, tools=tools)
    if result.has_tool_calls:
        # These are custom tool calls from pending_tool_calls
        for tc in result.tool_calls:
            output = execute_tool(tc.name, tc.arguments)
            messages.append({"role": "tool", "name": tc.name, "content": output})
        result = client.chat(messages, tools=tools)

For non-hybrid mode (full caller control)::

    client = CiCClient(model="sonnet", hybrid=False)
    # Define ALL tools — Claude cannot call any of them directly
    tools = [
        {"name": "file_read", "description": "Read a file", ...},
        {"name": "file_edit", "description": "Edit a file", ...},
        {"name": "shell_exec", "description": "Run a command", ...},
    ]

    result = client.chat(messages, tools=tools)
    while result.has_tool_calls:
        for tc in result.tool_calls:
            output = execute_tool(tc.name, tc.arguments)
            messages.append({"role": "tool", "name": tc.name, "content": output})
        result = client.chat(messages, tools=tools)
    # result.content is the final answer ("done" action)

For smart routing::

    client = CiCClient(routing={"simple": "haiku", "moderate": "sonnet", "complex": "opus"})
    client.set_complexity("complex")
    result = client.chat(messages)  # uses Opus
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from typing import Any

from .exceptions import ClaudeNotFoundError, ClaudeSubprocessError, ClaudeTimeoutError
from .routing import CiCRouter, DEFAULT_ROUTING
from .types import ChatResult, ToolCall, TokenUsage
from .utils import (
    STRUCTURED_OUTPUT_SCHEMA,
    _NON_HYBRID_SYSTEM_PROMPT,
    _SYSTEM_PROMPT,
    build_non_hybrid_schema,
    build_prompt,
    estimate_tokens,
    extract_response_text,
    parse_cli_output,
)

logger = logging.getLogger(__name__)


class CiCClient:
    """Chat client backed by the local ``claude`` CLI subprocess.

    Two modes:

    **Hybrid mode** (default, ``hybrid=True``): Claude executes file edits using
    its own built-in tools (Bash, Edit, Read, Write). Custom tools defined by the
    caller are reported back via ``--json-schema`` structured output and returned
    as ``tool_calls`` for the caller to execute.

    **Non-hybrid mode** (``hybrid=False``): Claude has NO built-in tools. Each
    ``chat()`` call returns ONE action (a tool call decision) for the caller to
    execute. The caller passes the result back in the next call. This continues
    until Claude outputs ``action: "done"`` (returns content) or ``action:
    "blocked"``. All tools — including file operations — are executed by the
    caller, giving full control and verifiability.

    Each call to ``chat()`` or ``achat()`` spawns a fresh ``claude --print``
    process, pipes the full conversation (system instructions + history +
    tool descriptions) as stdin, and parses the structured JSON response.

    Args:
        model: Fixed model name (e.g. ``"sonnet"``). When set, routing is
            disabled and every call uses this model.
        routing: A complexity → model mapping dict, e.g.
            ``{"simple": "haiku", "moderate": "sonnet", "complex": "opus"}``.
            Ignored when ``model`` is set.
        timeout: Subprocess timeout in seconds. Default: 120.
        claude_path: Explicit path to the ``claude`` binary. Auto-detected
            from PATH when not provided.
        hybrid: When True (default), uses hybrid mode — Claude edits files
            directly with its built-in tools. When False, uses non-hybrid mode —
            Claude decides actions, caller executes every tool.

    Raises:
        ClaudeNotFoundError: If ``claude`` is not found in PATH and
            ``claude_path`` is not provided.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        routing: dict[str, str] | None = None,
        timeout: float = 120.0,
        claude_path: str | None = None,
        hybrid: bool = True,
    ) -> None:
        self._fixed_model = model
        self._timeout = timeout
        self._current_complexity = "moderate"
        self._hybrid = hybrid

        # Set up routing
        if routing:
            self._router = CiCRouter.from_dict(routing)
        else:
            self._router = CiCRouter.from_dict(DEFAULT_ROUTING)

        # Locate the claude binary
        self._claude_path: str | None = claude_path or shutil.which("claude")
        if not self._claude_path:
            raise ClaudeNotFoundError(
                "claude CLI not found in PATH. "
                "Install Claude Code: npm install -g @anthropic-ai/claude-code"
            )

        mode = "hybrid" if self._hybrid else "non-hybrid"
        if self._fixed_model:
            logger.info(
                "[CiC] Fixed model: %s (routing disabled, mode=%s)",
                self._fixed_model,
                mode,
            )
        else:
            logger.info("[CiC] Smart routing enabled: %s (mode=%s)", self._router, mode)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def active_model(self) -> str:
        """The model that will be used for the next call.

        Determined by ``model`` (if fixed) or by the current complexity
        level and routing table.
        """
        if self._fixed_model:
            return self._fixed_model
        return self._router.model_for(self._current_complexity)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_complexity(self, level: str) -> None:
        """Set the task complexity for smart routing.

        The next call to ``chat()`` or ``achat()`` will select the model
        mapped to this complexity level.

        Args:
            level: One of ``"simple"``, ``"moderate"``, ``"complex"``, or any
                custom level added via the ``routing`` constructor argument.
        """
        self._current_complexity = level
        logger.debug(
            "[CiC] Complexity → %s (model: %s)", level, self.active_model
        )

    def set_model(self, model: str) -> None:
        """Override the model for subsequent calls.

        This sets the fixed model, disabling routing.

        Args:
            model: The model name to use, e.g. ``"opus"``.
        """
        self._fixed_model = model
        logger.debug("[CiC] Model override → %s", model)

    # ------------------------------------------------------------------
    # Synchronous interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        """Send a synchronous chat request via the claude CLI.

        This method blocks until the claude subprocess completes. It must
        not be called from within a running event loop — use ``achat()``
        instead in async code (e.g. inside FastAPI handlers or Jupyter notebooks).

        Args:
            messages: OpenAI-format messages list. Must contain at least one
                message with ``role: "user"``.
            tools: Optional list of tool definitions in OpenAI format. When
                provided, the model may return tool calls instead of a text
                response.

        Returns:
            A :class:`ChatResult` with either ``content`` or ``tool_calls``.

        Raises:
            ClaudeNotFoundError: If the claude CLI is not available.
            ClaudeTimeoutError: If the subprocess times out.
            ClaudeSubprocessError: If the CLI returns an error response.
            RuntimeError: If called from within an already-running event loop.
                Use ``achat()`` in async contexts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "CiCClient.chat() cannot be called from within a running event loop. "
                "Use 'await client.achat(...)' instead."
            )

        return asyncio.run(self.achat(messages, tools=tools))

    def chat_openai_format(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a chat request and return a raw OpenAI-format response dict.

        This is a convenience method for callers that already consume OpenAI
        responses and want a drop-in replacement.

        Args:
            messages: OpenAI-format messages list.
            tools: Optional tool definitions.

        Returns:
            An OpenAI chat completion dict (``choices``, ``message``, etc.).
        """
        result = self.chat(messages, tools=tools)
        return result.raw

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def achat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        """Send an async chat request via the claude CLI.

        Prefer this method in async code — it avoids the ``asyncio.run()``
        overhead of the sync ``chat()`` wrapper and works correctly inside
        running event loops (FastAPI, Jupyter, etc.).

        Args:
            messages: OpenAI-format messages list.
            tools: Optional tool definitions.

        Returns:
            A :class:`ChatResult` with either ``content`` or ``tool_calls``.

        Raises:
            ClaudeNotFoundError: If the claude CLI is not available.
            ClaudeTimeoutError: If the subprocess times out.
            ClaudeSubprocessError: If the CLI returns an error response.
        """
        model = self.active_model
        prompt = build_prompt(messages, tools, hybrid=self._hybrid)
        prompt_tokens = estimate_tokens(prompt)

        logger.info(
            "[CiC] Spawning %s (~%d tok prompt, complexity=%s, timeout=%ds, mode=%s)",
            model,
            prompt_tokens,
            self._current_complexity,
            int(self._timeout),
            "hybrid" if self._hybrid else "non-hybrid",
        )

        stdout, stderr = await self._spawn_claude(prompt, model, tools=tools)

        if stderr:
            logger.debug("[CiC] stderr: %.500s", stderr)

        data = parse_cli_output(stdout, hybrid=self._hybrid)
        response_text = extract_response_text(data)
        completion_tokens = estimate_tokens(response_text)

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        logger.info(
            "[CiC] Done — ~%dp + ~%dc tokens (model: %s)",
            prompt_tokens,
            completion_tokens,
            model,
        )

        return _build_chat_result(data, model=f"cic/{model}", usage=usage)

    # ------------------------------------------------------------------
    # Subprocess
    # ------------------------------------------------------------------

    async def _spawn_claude(
        self,
        prompt: str,
        model: str,
        tools: list[dict] | None = None,
    ) -> tuple[str, str]:
        """Spawn the claude CLI and return (stdout, stderr).

        Uses ``--output-format stream-json --verbose`` to receive line-by-line
        events as Claude works. An idle timer resets on each line received — the
        process is only killed if no output arrives for ``self._timeout`` seconds.
        This means Claude can run for as long as it needs on complex multi-step
        work without a wall-clock limit; the guard only fires on genuine stalls.

        In hybrid mode: uses ``--tools Bash,Edit,Read,Write`` and the hybrid
        structured output schema (summary + files_modified + pending_tool_calls).

        In non-hybrid mode: uses ``--tools ""`` (no built-in tools) and the
        action-enum schema built from the caller's tool definitions.

        Args:
            prompt: The full prompt string to pipe to the subprocess.
            model: The model flag value, e.g. ``"sonnet"``.
            tools: Tool definitions used to build the non-hybrid schema.
                Ignored in hybrid mode.

        Returns:
            A tuple of ``(stdout, stderr)`` as decoded strings. ``stdout`` is
            the JSON line with ``"type": "result"`` from the stream (or the last
            line if no explicit result event was received).

        Raises:
            ClaudeTimeoutError: If no output is received for ``self._timeout``
                seconds (idle timeout — not a wall-clock limit).
        """
        idle_timeout = self._timeout

        if self._hybrid:
            tools_flag = "Bash,Edit,Read,Write"
            schema = STRUCTURED_OUTPUT_SCHEMA
            system_prompt = _SYSTEM_PROMPT
        else:
            tools_flag = ""
            schema = build_non_hybrid_schema(tools or [])
            system_prompt = _NON_HYBRID_SYSTEM_PROMPT

        cmd = [
            self._claude_path,
            "--print",
            "--output-format", "stream-json",
            "--verbose",  # required for stream-json
            "--tools", tools_flag,
            "--json-schema", schema,
            "--no-session-persistence",
            "--dangerously-skip-permissions",
            "--model", model,
            # Reduce ~45K → ~3K token cache tax per call
            "--setting-sources", "user",
            "--system-prompt", system_prompt,
            # Strip ALL MCP tools (e.g. Google Calendar)
            "--strict-mcp-config",
        ]

        # Build a clean env:
        # 1. Strip CLAUDECODE — if set (e.g. running inside a Claude Code session),
        #    it blocks further claude subprocess spawning.
        # 2. Strip CLAUDE_CODE_ENTRY_POINT — prevents session nesting detection.
        env = {k: v for k, v in os.environ.items()
               if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRY_POINT")}

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Send prompt to stdin and close it so the subprocess can start reading.
        proc.stdin.write(prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        # Read stdout line-by-line with idle timeout.
        # Each line received resets the timer. Only kill if truly idle.
        lines: list[str] = []
        result_line: str = ""
        event_count = 0

        try:
            while True:
                try:
                    line_bytes = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=idle_timeout,
                    )
                except asyncio.TimeoutError:
                    # No output for idle_timeout seconds — Claude is stalled.
                    logger.warning(
                        "[CiC] Idle timeout (%ds no output, %d events received) — killing",
                        idle_timeout,
                        event_count,
                    )
                    proc.kill()
                    await proc.communicate()
                    raise ClaudeTimeoutError(idle_timeout) from None

                if not line_bytes:
                    # EOF — subprocess finished.
                    break

                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                lines.append(line)
                event_count += 1

                # Parse each line to identify event type for logging and result extraction.
                try:
                    event = json.loads(line)
                    etype = event.get("type", "")
                    if etype == "result":
                        result_line = line
                        logger.debug("[CiC] Stream: result event (final)")
                    elif etype == "assistant":
                        # Log tool_use events for operational visibility.
                        msg = event.get("message", {})
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "tool_use"
                                ):
                                    logger.info(
                                        "[CiC] Stream: tool_use %s",
                                        block.get("name", "?"),
                                    )
                except json.JSONDecodeError:
                    pass  # Not every line is valid JSON (e.g. debug output)

        finally:
            # Ensure the process is cleaned up even if we break out early.
            if proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()

        # Collect any remaining stderr.
        stderr = ""
        if proc.stderr:
            try:
                stderr_bytes = await asyncio.wait_for(proc.stderr.read(), timeout=5)
                stderr = stderr_bytes.decode("utf-8", errors="replace")
            except Exception:
                pass

        logger.info(
            "[CiC] Stream complete: %d events, result=%s",
            event_count,
            bool(result_line),
        )

        # The result line is the type:"result" event — the final structured output.
        # Fall back to the last line received if no explicit result event was emitted.
        if result_line:
            stdout = result_line
        elif lines:
            stdout = lines[-1]
        else:
            stdout = ""

        return stdout, stderr

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "CiCClient":
        return self

    def __exit__(self, *_: Any) -> None:
        pass  # No resources to release

    async def __aenter__(self) -> "CiCClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass  # No resources to release

    def __repr__(self) -> str:
        mode = "hybrid" if self._hybrid else "non-hybrid"
        if self._fixed_model:
            return (
                f"CiCClient(model={self._fixed_model!r}, "
                f"timeout={self._timeout}, mode={mode!r})"
            )
        return (
            f"CiCClient(routing={self._router!r}, "
            f"complexity={self._current_complexity!r}, "
            f"timeout={self._timeout}, mode={mode!r})"
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_chat_result(
    data: dict[str, Any],
    *,
    model: str,
    usage: TokenUsage,
) -> ChatResult:
    """Build a ChatResult from an OpenAI-format response dict.

    Args:
        data: OpenAI wire format dict from :func:`parse_cli_output`.
        model: The model identifier string.
        usage: Estimated token usage.

    Returns:
        A fully populated :class:`ChatResult`.
    """
    choices = data.get("choices", [])
    if not choices:
        return ChatResult(
            content="[CiC] No choices in response",
            model=model,
            usage=usage,
            raw=data,
        )

    message = choices[0].get("message", {})
    raw_tool_calls = message.get("tool_calls") or []

    if raw_tool_calls:
        tool_calls = [
            ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=tc.get("function", {}).get("name", "unknown"),
                arguments=_parse_arguments(tc.get("function", {}).get("arguments", "{}")),
            )
            for i, tc in enumerate(raw_tool_calls)
        ]
        return ChatResult(
            content=None,
            tool_calls=tool_calls,
            model=model,
            usage=usage,
            raw=data,
        )

    content = message.get("content", "")
    return ChatResult(
        content=content,
        model=model,
        usage=usage,
        raw=data,
    )


def _parse_arguments(arguments: str | dict[str, Any] | Any) -> dict[str, Any]:
    """Parse tool call arguments into a dict.

    OpenAI format stores arguments as a JSON string; we materialise it.

    Args:
        arguments: Either a JSON string, an already-parsed dict, or any other
            value (treated as empty arguments).

    Returns:
        A dict of argument name → value.
    """
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    try:
        parsed = json.loads(arguments)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
