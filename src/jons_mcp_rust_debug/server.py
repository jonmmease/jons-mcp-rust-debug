"""FastMCP server for Rust debugging via LLDB."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

from mcp.server.fastmcp import FastMCP

from .exceptions import DebuggerError
from .lldb_client import RustDebugClient

if TYPE_CHECKING:
    pass

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "WARNING"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global debug client (set during lifespan)
debug_client: RustDebugClient | None = None


@asynccontextmanager
async def lifespan(mcp: FastMCP) -> AsyncIterator[None]:
    """Manage lifecycle of the LLDB debug client."""
    global debug_client

    debug_client = RustDebugClient()

    try:
        yield
    finally:
        if debug_client:
            debug_client._cleanup_all_sessions()
        debug_client = None


def ensure_debug_client() -> RustDebugClient:
    """Get the debug client or raise if not initialized.

    Returns:
        The global RustDebugClient instance.

    Raises:
        RuntimeError: If debug client is not initialized.
    """
    if debug_client is None:
        raise RuntimeError("Debug client not initialized")
    return debug_client


def handle_debugger_error(func):
    """Decorator to convert DebuggerError to error response dict."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return await func(*args, **kwargs)
        except DebuggerError as e:
            return {"status": "error", "error": str(e)}

    return wrapper


# Create FastMCP server with lifespan
mcp = FastMCP(
    "rust-debug-mcp",
    lifespan=lifespan,
)


def _register_tools() -> None:
    """Register all MCP tools with the server."""
    from .tools import (
        backtrace,
        check_debug_info,
        continue_to_line,
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
        list_watchpoints,
        next,
        print_array,
        print_slice,
        print_variable,
        remove_breakpoint,
        remove_watchpoint,
        run,
        select_thread,
        session_diagnostics,
        set_breakpoint,
        set_variable,
        set_watchpoint,
        start_debug,
        step,
        stop_debug,
        up,
    )

    # Session management (3 tools)
    mcp.tool()(start_debug)
    mcp.tool()(stop_debug)
    mcp.tool()(list_sessions)

    # Breakpoint management (3 tools)
    mcp.tool()(set_breakpoint)
    mcp.tool()(remove_breakpoint)
    mcp.tool()(list_breakpoints)

    # Watchpoint management (3 tools)
    mcp.tool()(set_watchpoint)
    mcp.tool()(remove_watchpoint)
    mcp.tool()(list_watchpoints)

    # Execution control (5 tools)
    mcp.tool()(run)
    mcp.tool()(step)
    mcp.tool()(next)
    mcp.tool()(finish)
    mcp.tool()(continue_to_line)

    # Stack navigation (3 tools)
    mcp.tool()(backtrace)
    mcp.tool()(up)
    mcp.tool()(down)

    # Inspection (7 tools)
    mcp.tool()(list_source)
    mcp.tool()(print_variable)
    mcp.tool()(print_array)
    mcp.tool()(print_slice)
    mcp.tool()(list_locals)
    mcp.tool()(evaluate)
    mcp.tool()(set_variable)

    # Thread management (2 tools)
    mcp.tool()(list_threads)
    mcp.tool()(select_thread)

    # Diagnostics (4 tools)
    mcp.tool()(session_diagnostics)
    mcp.tool()(check_debug_info)
    mcp.tool()(get_test_summary)
    mcp.tool()(get_enum_info)


# Register tools at module load time
_register_tools()


def main() -> None:
    """Main entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
