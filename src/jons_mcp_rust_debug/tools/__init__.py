"""MCP tools for Rust debugging."""

from .breakpoints import list_breakpoints, remove_breakpoint, set_breakpoint
from .diagnostics import (
    check_debug_info,
    get_enum_info,
    get_test_summary,
    session_diagnostics,
)
from .execution import continue_to_line, finish, next, run, step
from .inspection import (
    evaluate,
    list_locals,
    list_source,
    print_array,
    print_slice,
    print_variable,
    set_variable,
)
from .session import list_sessions, start_debug, stop_debug
from .stack import backtrace, down, up
from .threads import list_threads, select_thread
from .watchpoints import list_watchpoints, remove_watchpoint, set_watchpoint

__all__ = [
    # Session management
    "start_debug",
    "stop_debug",
    "list_sessions",
    # Breakpoints
    "set_breakpoint",
    "remove_breakpoint",
    "list_breakpoints",
    # Watchpoints
    "set_watchpoint",
    "remove_watchpoint",
    "list_watchpoints",
    # Execution control
    "run",
    "step",
    "next",
    "finish",
    "continue_to_line",
    # Stack navigation
    "backtrace",
    "up",
    "down",
    # Inspection
    "list_source",
    "print_variable",
    "print_array",
    "print_slice",
    "list_locals",
    "evaluate",
    "set_variable",
    # Thread management
    "list_threads",
    "select_thread",
    # Diagnostics
    "session_diagnostics",
    "check_debug_info",
    "get_test_summary",
    "get_enum_info",
]
