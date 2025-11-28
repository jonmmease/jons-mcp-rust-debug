"""Constants for the Rust debug MCP server."""

from __future__ import annotations

# Timeouts (in seconds)
RUN_TIMEOUT: float = 30.0
STEP_TIMEOUT: float = 5.0
FINISH_TIMEOUT: float = 10.0
CLEANUP_TIMEOUT: float = 0.5
POLL_INTERVAL: float = 0.1
EVENT_POLL_TIMEOUT: int = 1  # LLDB WaitForEvent timeout

# Pagination defaults
DEFAULT_PAGINATION_LIMIT: int | None = None
DEFAULT_PAGINATION_OFFSET: int = 0

# Source display defaults
DEFAULT_SOURCE_CONTEXT_LINES: int = 10

# LLDB stop reason strings
STOP_REASON_NONE = "none"
STOP_REASON_TRACE = "trace"
STOP_REASON_BREAKPOINT = "breakpoint"
STOP_REASON_WATCHPOINT = "watchpoint"
STOP_REASON_SIGNAL = "signal"
STOP_REASON_EXCEPTION = "exception"
STOP_REASON_EXEC = "exec"
STOP_REASON_PLAN_COMPLETE = "plan_complete"
STOP_REASON_UNKNOWN = "unknown"
