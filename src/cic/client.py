"""CiCClient — the main entry point for building agents with CiC.

CiCClient wraps the ``claude`` CLI to provide a programmatic chat interface.
Instead of calling a REST API with per-token billing, it spawns
``claude --print`` subprocesses that use the caller's active Claude
Pro/Max subscription.

Typical usage::

    from cic import CiCClient

    client = CiCClient(model="sonnet")
    result = client.chat([{"role": "user", "content": "Hello!"}])
    print(result.content)

For tool use (agent loops)::

    client = CiCClient(model="sonnet")
    tools = [{"name": "read_file", "description": "...", "parameters": {...}}]

    result = client.chat(messages, tools=tools)
    if result.has_tool_calls:
        for tc in result.tool_calls:
            output = execute_tool(tc.name, tc.arguments)
            messages.append({"role": "tool", "name": tc.name, "content": output})
        result = client.chat(messages, tools=tools)

For smart routing::

    client = CiCClient(routing={"simple": "haiku", "moderate": "sonnet", "complex": "opus"})
    client.set_complexity("complex")
    result = client.chat(messages)  # uses Opus
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any

from .exceptions import ClaudeNotFoundError, ClaudeSubprocessError, ClaudeTimeoutError
from .routing import CiCRouter, DEFAULT_ROUTING
from .types import ChatResult, ToolCall, TokenUsage
from .utils import build_prompt, estimate_tokens, extract_response_text, parse_cli_output

logger = logging.getLogger(__name__)


class CiCClient:
    """Chat client backed by the local ``claude`` CLI subprocess.

    Each call to ``chat()`` or ``achat()`` spawns a fresh ``claude --print``
    process, pipes the full conversation (system instructions + history +
    tools) as stdin, and parses the JSON response.

    Args:
        model: Fixed model name (e.g. ``"sonnet"``). When set, routing is
            disabled and every call uses this model.
        routing: A complexity → model mapping dict, e.g.
            ``{"simple": "haiku", "moderate": "sonnet", "complex": "opus"}``.
            Ignored when ``model`` is set.
        timeout: Subprocess timeout in seconds. Default: 120.
        claude_path: Explicit path to the ``claude`` binary. Auto-detected
            from PATH when not provided.

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
    ) -> None:
        self._fixed_model = model
        self._timeout = timeout
        self._current_complexity = "moderate"

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

        if self._fixed_model:
            logger.info("[CiC] Fixed model: %s (routing disabled)", self._fixed_model)
        else:
            logger.info("[CiC] Smart routing enabled: %s", self._router)

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
        prompt = build_prompt(messages, tools)
        prompt_tokens = estimate_tokens(prompt)

        logger.info(
            "[CiC] Spawning %s (~%d tok prompt, complexity=%s, timeout=%ds)",
            model,
            prompt_tokens,
            self._current_complexity,
            int(self._timeout),
        )

        stdout, stderr = await self._spawn_claude(prompt, model)

        if stderr:
            logger.debug("[CiC] stderr: %.500s", stderr)

        data = parse_cli_output(stdout)
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
    ) -> tuple[str, str]:
        """Spawn the claude CLI and return (stdout, stderr).

        The prompt is piped to stdin. The CLI writes its JSON response to
        stdout. stderr carries diagnostic output (not part of the response).

        Args:
            prompt: The full prompt string to pipe to the subprocess.
            model: The model flag value, e.g. ``"sonnet"``.

        Returns:
            A tuple of ``(stdout, stderr)`` as decoded strings.

        Raises:
            ClaudeTimeoutError: If the subprocess does not complete within
                ``self._timeout`` seconds.
        """
        cmd = [
            self._claude_path,
            "--print",
            "--output-format", "json",
            "--tools", "",
            "--no-session-persistence",
            "--dangerously-skip-permissions",
            "--model", model,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise ClaudeTimeoutError(self._timeout) from None

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
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
        if self._fixed_model:
            return f"CiCClient(model={self._fixed_model!r}, timeout={self._timeout})"
        return (
            f"CiCClient(routing={self._router!r}, "
            f"complexity={self._current_complexity!r}, "
            f"timeout={self._timeout})"
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
