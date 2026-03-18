"""CiC exceptions.

All errors raised by the CiC library derive from CiCError, making it
easy for callers to catch any CiC-related failure with a single except clause.
"""

from __future__ import annotations


class CiCError(Exception):
    """Base class for all CiC errors."""


class ClaudeNotFoundError(CiCError):
    """Raised when the claude CLI binary cannot be found in PATH.

    Install Claude Code to fix:
        npm install -g @anthropic-ai/claude-code
    """


class ClaudeTimeoutError(CiCError):
    """Raised when a claude subprocess does not respond within the timeout.

    Args:
        timeout: The timeout in seconds that was exceeded.
    """

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(f"claude subprocess timed out after {timeout:.0f}s")


class ClaudeSubprocessError(CiCError):
    """Raised when the claude CLI returns an error response.

    Args:
        message: The error message from the CLI.
        stderr: Raw stderr output, if available.
    """

    def __init__(self, message: str, stderr: str = "") -> None:
        self.stderr = stderr
        super().__init__(message)


class ResponseParseError(CiCError):
    """Raised when the claude CLI output cannot be parsed into a valid response.

    Args:
        message: Description of what failed to parse.
        raw: The raw output that caused the failure.
    """

    def __init__(self, message: str, raw: str = "") -> None:
        self.raw = raw
        super().__init__(message)
