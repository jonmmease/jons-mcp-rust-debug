"""MCP server providing Rust debugging capabilities through LLDB."""

from .constants import (
    CLEANUP_TIMEOUT,
    DEFAULT_PAGINATION_LIMIT,
    DEFAULT_PAGINATION_OFFSET,
    DEFAULT_SOURCE_CONTEXT_LINES,
    EVENT_POLL_TIMEOUT,
    FINISH_TIMEOUT,
    POLL_INTERVAL,
    RUN_TIMEOUT,
    STEP_TIMEOUT,
)
from .exceptions import (
    BreakpointError,
    BuildError,
    DebuggerError,
    DebuggerNotPausedError,
    EvaluationError,
    LaunchError,
    NoActiveFrameError,
    NoActiveThreadError,
    ProcessNotRunningError,
    SessionNotFoundError,
    SourceNotFoundError,
)
from .lldb_client import (
    Breakpoint,
    Config,
    DebuggerState,
    DebugSession,
    RustDebugClient,
)
from .server import debug_client, ensure_debug_client, main, mcp
from .tools import (
    backtrace,
    check_debug_info,
    down,
    evaluate,
    finish,
    get_enum_info,
    get_test_summary,
    list_breakpoints,
    list_locals,
    list_sessions,
    list_source,
    list_threads,
    next,
    print_variable,
    remove_breakpoint,
    run,
    select_thread,
    session_diagnostics,
    set_breakpoint,
    start_debug,
    step,
    stop_debug,
    up,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Constants
    "RUN_TIMEOUT",
    "STEP_TIMEOUT",
    "FINISH_TIMEOUT",
    "CLEANUP_TIMEOUT",
    "POLL_INTERVAL",
    "EVENT_POLL_TIMEOUT",
    "DEFAULT_PAGINATION_LIMIT",
    "DEFAULT_PAGINATION_OFFSET",
    "DEFAULT_SOURCE_CONTEXT_LINES",
    # Exceptions
    "DebuggerError",
    "SessionNotFoundError",
    "ProcessNotRunningError",
    "NoActiveThreadError",
    "NoActiveFrameError",
    "BuildError",
    "LaunchError",
    "DebuggerNotPausedError",
    "BreakpointError",
    "SourceNotFoundError",
    "EvaluationError",
    # Client classes
    "Config",
    "Breakpoint",
    "DebugSession",
    "DebuggerState",
    "RustDebugClient",
    # Server
    "mcp",
    "debug_client",
    "ensure_debug_client",
    "main",
    # Tools - Session management
    "start_debug",
    "stop_debug",
    "list_sessions",
    # Tools - Breakpoints
    "set_breakpoint",
    "remove_breakpoint",
    "list_breakpoints",
    # Tools - Execution control
    "run",
    "step",
    "next",
    "finish",
    # Tools - Stack navigation
    "backtrace",
    "up",
    "down",
    # Tools - Inspection
    "list_source",
    "print_variable",
    "list_locals",
    "evaluate",
    # Tools - Thread management
    "list_threads",
    "select_thread",
    # Tools - Diagnostics
    "session_diagnostics",
    "check_debug_info",
    "get_test_summary",
    "get_enum_info",
]
