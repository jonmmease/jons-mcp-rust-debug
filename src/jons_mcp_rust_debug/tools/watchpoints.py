"""Watchpoint management tools for Rust debugging."""

from __future__ import annotations

from typing import Any

import lldb

from ..lldb_client import Watchpoint
from ..server import ensure_debug_client
from ..utils import get_active_frame, validate_session


async def set_watchpoint(
    session_id: str,
    expression: str,
    watch_type: str = "write",
    size: int | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    """Set a watchpoint on a variable or memory address.

    Args:
        session_id: The session identifier
        expression: Variable name or memory address to watch
        watch_type: Type of watchpoint - "write", "read", or "read_write"
        size: Size in bytes to watch (optional, auto-detected from expression)
        condition: Conditional expression for the watchpoint
        temporary: If true, watchpoint is removed after first hit

    Returns:
        Dictionary with watchpoint details
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id, require_process=True)
        frame = get_active_frame(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Validate watch_type
    if watch_type not in ("write", "read", "read_write"):
        return {
            "status": "error",
            "error": f"Invalid watch_type '{watch_type}'. Must be 'write', 'read', or 'read_write'",
        }

    # Try to evaluate expression to get address
    value = frame.EvaluateExpression(expression)
    if not value.IsValid():
        return {
            "status": "error",
            "error": f"Failed to evaluate expression: {expression}",
        }

    # Get the memory address
    address = value.GetLoadAddress()
    if address == lldb.LLDB_INVALID_ADDRESS:
        return {
            "status": "error",
            "error": f"Could not get memory address for: {expression}",
        }

    # Determine size if not provided
    if size is None:
        size = value.GetByteSize()
        if size == 0:
            size = 8  # Default to 8 bytes if we can't determine size

    # Map watch_type to LLDB flags
    if watch_type == "write":
        read = False
        write = True
    elif watch_type == "read":
        read = True
        write = False
    else:  # read_write
        read = True
        write = True

    # Create the watchpoint
    error = lldb.SBError()
    wp = session.target.WatchAddress(address, size, read, write, error)

    if error.Fail() or not wp or not wp.IsValid():
        error_msg = error.GetCString() if error.Fail() else "Unknown error"
        return {
            "status": "error",
            "error": f"Failed to create watchpoint: {error_msg}",
        }

    # Set condition if provided
    if condition:
        wp.SetCondition(condition)

    # Store watchpoint info
    watchpoint = Watchpoint(
        id=wp.GetID(),
        address=address,
        size=size,
        watch_type=watch_type,
        expression=expression,
        condition=condition,
        enabled=True,
        lldb_watchpoint=wp,
    )
    session.watchpoints[wp.GetID()] = watchpoint

    return {
        "watchpoint_id": wp.GetID(),
        "address": hex(address),
        "size": size,
        "watch_type": watch_type,
        "expression": expression,
        "status": "set",
    }


async def list_watchpoints(session_id: str) -> dict[str, Any]:
    """List all watchpoints in the session.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary with list of watchpoints
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id, require_process=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    watchpoints = []
    for wp_id, wp in session.watchpoints.items():
        lldb_wp = wp.lldb_watchpoint
        hit_count = lldb_wp.GetHitCount() if lldb_wp and lldb_wp.IsValid() else 0

        watchpoints.append(
            {
                "id": wp_id,
                "address": hex(wp.address),
                "size": wp.size,
                "watch_type": wp.watch_type,
                "expression": wp.expression,
                "condition": wp.condition,
                "enabled": wp.enabled,
                "hit_count": hit_count,
            }
        )

    return {"watchpoints": watchpoints}


async def remove_watchpoint(session_id: str, watchpoint_id: int) -> dict[str, Any]:
    """Remove a watchpoint.

    Args:
        session_id: The session identifier
        watchpoint_id: The watchpoint ID to remove

    Returns:
        Dictionary with status
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id, require_process=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if watchpoint_id not in session.watchpoints:
        return {"status": "error", "error": "Watchpoint not found"}

    # Remove from LLDB
    success = session.target.DeleteWatchpoint(watchpoint_id)
    if not success:
        return {"status": "error", "error": "Failed to delete watchpoint from LLDB"}

    # Remove from session
    del session.watchpoints[watchpoint_id]

    return {"status": "removed"}
