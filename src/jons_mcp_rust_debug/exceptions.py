"""Custom exceptions for the Rust debug MCP server."""

from __future__ import annotations


class DebuggerError(Exception):
    """Base exception for all debugger errors."""

    def __init__(self, message: str, is_retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.is_retryable = is_retryable


class SessionNotFoundError(DebuggerError):
    """Raised when a session ID is not found."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session not found: {session_id}")
        self.session_id = session_id


class ProcessNotRunningError(DebuggerError):
    """Raised when operation requires a running process."""

    def __init__(self) -> None:
        super().__init__("Process not running")


class NoActiveThreadError(DebuggerError):
    """Raised when no active thread is available."""

    def __init__(self) -> None:
        super().__init__("No active thread")


class NoActiveFrameError(DebuggerError):
    """Raised when no active frame is available."""

    def __init__(self) -> None:
        super().__init__("No active frame")


class BuildError(DebuggerError):
    """Raised when cargo build fails."""

    def __init__(self, stderr: str) -> None:
        super().__init__(f"Failed to build target: {stderr}")
        self.stderr = stderr


class LaunchError(DebuggerError):
    """Raised when process launch fails."""

    def __init__(self, error_string: str) -> None:
        super().__init__(f"Launch failed: {error_string}")
        self.error_string = error_string


class DebuggerNotPausedError(DebuggerError):
    """Raised when operation requires debugger to be paused."""

    def __init__(self, current_state: str) -> None:
        super().__init__(
            f"Cannot perform operation: debugger is {current_state}, not paused"
        )
        self.current_state = current_state


class BreakpointError(DebuggerError):
    """Raised when breakpoint operations fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class SourceNotFoundError(DebuggerError):
    """Raised when source file cannot be found."""

    def __init__(self, file_path: str) -> None:
        super().__init__(f"Source file not found: {file_path}")
        self.file_path = file_path


class EvaluationError(DebuggerError):
    """Raised when expression evaluation fails."""

    def __init__(self, expression: str, error_message: str) -> None:
        super().__init__(f"Failed to evaluate '{expression}': {error_message}")
        self.expression = expression
        self.error_message = error_message
