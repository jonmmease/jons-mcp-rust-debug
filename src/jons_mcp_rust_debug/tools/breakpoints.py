"""Breakpoint management tools for Rust debugging."""

from __future__ import annotations

import os
from typing import Any

from ..lldb_client import Breakpoint
from ..server import ensure_debug_client
from ..utils import validate_session


async def set_breakpoint(
    session_id: str,
    file: str | None = None,
    line: int | None = None,
    function: str | None = None,
    condition: str | None = None,
    temporary: bool = False,
) -> dict[str, Any]:
    """Set a breakpoint in the Rust program.

    Args:
        session_id: The session identifier
        file: Source file name. RECOMMENDED: Use filename only (e.g., "test_data.rs")
            rather than full paths. Absolute paths may cause resolution issues.
        line: Line number
        function: Function name (alternative to file/line)
        condition: Conditional expression
        temporary: If true, breakpoint is removed after first hit

    Returns:
        Dictionary with breakpoint details

    Examples:
        # Recommended - filename only
        set_breakpoint(session_id, file="my_test.rs", line=25)

        # Also works - relative path
        set_breakpoint(session_id, file="tests/my_test.rs", line=25)

        # By function name
        set_breakpoint(session_id, function="my_module::my_function")
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id, require_process=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Create breakpoint
    bp = None
    if function:
        bp = session.target.BreakpointCreateByName(function)
    elif file and line:
        # Try different path formats
        bp = session.target.BreakpointCreateByLocation(file, line)
        if not bp or not bp.IsValid() or bp.GetNumLocations() == 0:
            # Try with just filename
            filename = os.path.basename(file)
            bp = session.target.BreakpointCreateByLocation(filename, line)

    if not bp or not bp.IsValid():
        return {"status": "error", "error": "Failed to create breakpoint"}

    # Set properties
    if condition:
        bp.SetCondition(condition)
    if temporary:
        bp.SetOneShot(True)

    # Store breakpoint info
    breakpoint = Breakpoint(
        id=bp.GetID(),
        file=file or "",
        line=line or 0,
        function=function,
        condition=condition,
        temporary=temporary,
        enabled=True,
        lldb_breakpoint=bp,
    )
    session.breakpoints[bp.GetID()] = breakpoint

    # Get resolved location
    location = ""
    if bp.GetNumLocations() > 0:
        loc = bp.GetLocationAtIndex(0)
        addr = loc.GetAddress()
        if addr.IsValid():
            line_entry = addr.GetLineEntry()
            if line_entry.IsValid():
                file_spec = line_entry.GetFileSpec()
                location = f"{file_spec.GetFilename()}:{line_entry.GetLine()}"

    return {
        "breakpoint_id": bp.GetID(),
        "location": location or f"{file}:{line}" if file and line else function,
        "status": "set",
    }


async def remove_breakpoint(session_id: str, breakpoint_id: int) -> dict[str, Any]:
    """Remove a breakpoint.

    Args:
        session_id: The session identifier
        breakpoint_id: The breakpoint ID to remove

    Returns:
        Dictionary with status
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id, require_process=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if breakpoint_id not in session.breakpoints:
        return {"status": "error", "error": "Breakpoint not found"}

    # Remove from LLDB
    bp = session.breakpoints[breakpoint_id].lldb_breakpoint
    if bp and bp.IsValid():
        session.target.BreakpointDelete(bp.GetID())

    # Remove from session
    del session.breakpoints[breakpoint_id]

    return {"status": "removed"}


async def list_breakpoints(session_id: str) -> dict[str, Any]:
    """List all breakpoints in the session.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary with list of breakpoints
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id, require_process=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    breakpoints = []
    for bp_id, bp in session.breakpoints.items():
        lldb_bp = bp.lldb_breakpoint
        hit_count = lldb_bp.GetHitCount() if lldb_bp and lldb_bp.IsValid() else 0

        breakpoints.append(
            {
                "id": bp_id,
                "file": bp.file,
                "line": bp.line,
                "function": bp.function,
                "condition": bp.condition,
                "temporary": bp.temporary,
                "enabled": bp.enabled,
                "hit_count": hit_count,
            }
        )

    return {"breakpoints": breakpoints}
