"""Execution control tools for Rust debugging."""

from __future__ import annotations

import os
import threading
from typing import Any

import lldb

from ..constants import FINISH_TIMEOUT, RUN_TIMEOUT, STEP_TIMEOUT
from ..lldb_client import DebuggerState
from ..server import ensure_debug_client
from ..utils import get_active_thread, get_frame_location, paginate_text, validate_session


async def run(
    session_id: str,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any]:
    """Start or continue program execution.

    Args:
        session_id: The session identifier
        limit: Maximum number of characters to return (for pagination)
        offset: Starting character position (for pagination)

    Returns:
        Dictionary with execution status and stop reason
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id, require_process=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    action_taken = ""
    # Check if we need to launch a new process
    needs_launch = (
        not session.process
        or not session.process.IsValid()
        or session.process.GetState()
        in (lldb.eStateExited, lldb.eStateDetached)
    )

    if needs_launch:
        # Reset state for new launch
        session.state = DebuggerState.IDLE
        session.last_stop_reason = ""
        session.current_location = None

        # Launch the process
        error = lldb.SBError()
        working_dir = os.path.abspath(client.config.working_directory)
        session.process = session.target.Launch(
            session.listener,
            session.args,
            None,  # envp
            None,  # stdin_path
            None,  # stdout_path
            None,  # stderr_path
            working_dir,  # working directory
            0,  # launch flags
            False,  # stop at entry
            error,
        )

        if error.Fail():
            return {"status": "error", "error": f"Launch failed: {error.GetCString()}"}

        action_taken = "Started new execution"

        # Start event handler thread
        session.event_thread = threading.Thread(
            target=client._event_handler_thread,
            args=(session,),
            daemon=True,
        )
        session.event_thread.start()
    else:
        # Ensure event handler thread is running
        if not session.event_thread or not session.event_thread.is_alive():
            session.event_thread = threading.Thread(
                target=client._event_handler_thread,
                args=(session,),
                daemon=True,
            )
            session.event_thread.start()

        # Continue execution
        error = session.process.Continue()
        if error.Fail():
            return {"status": "error", "error": f"Continue failed: {error.GetCString()}"}
        action_taken = f"Continued from {session.current_location or 'breakpoint'}"

    # Wait for stop using event-based synchronization
    client.wait_for_stop(session, RUN_TIMEOUT)

    # Update stop info
    client._update_stop_info(session)

    # Get output if any
    output = session.output_buffer
    session.output_buffer = ""

    # Handle pagination
    pagination = paginate_text(output, limit, offset)

    return {
        "status": session.state.value,
        "action": action_taken,
        "stop_reason": session.last_stop_reason,
        "stopped_at": session.current_location or "",
        "current_location": session.current_location,
        "output": pagination["content"],
        "pagination": {
            "total_chars": pagination["total_chars"],
            "offset": pagination["offset"],
            "limit": pagination["limit"],
            "has_more": pagination["has_more"],
        },
    }


async def step(session_id: str) -> dict[str, Any]:
    """Step into the next line of code.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary with step result and location
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Step into
    thread.StepInto()

    # Wait for stop using event-based synchronization
    client.wait_for_stop(session, STEP_TIMEOUT)

    # Update stop info
    client._update_stop_info(session)

    # Get location info
    frame = thread.GetSelectedFrame()
    location_info = get_frame_location(frame)

    return {
        "status": session.state.value,
        "location": location_info["location"],
        "file": location_info["file"],
        "line": location_info["line"],
        "function": location_info["function"],
        "output": "",
        "message": "Stepped into next line" if location_info["location"] else "Step completed",
    }


async def next(session_id: str) -> dict[str, Any]:
    """Step over the next line of code.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary with step result and location
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Step over
    thread.StepOver()

    # Wait for stop using event-based synchronization
    client.wait_for_stop(session, STEP_TIMEOUT)

    # Update stop info
    client._update_stop_info(session)

    # Get location info
    frame = thread.GetSelectedFrame()
    location_info = get_frame_location(frame)

    return {
        "status": session.state.value,
        "location": location_info["location"],
        "file": location_info["file"],
        "line": location_info["line"],
        "function": location_info["function"],
        "output": "",
        "message": "Stepped over to next line" if location_info["location"] else "Step completed",
    }


async def finish(session_id: str) -> dict[str, Any]:
    """Continue execution until the current function returns.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary with return value if available
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Step out
    thread.StepOut()

    # Wait for stop using event-based synchronization
    client.wait_for_stop(session, FINISH_TIMEOUT)

    # Try to get return value
    return_value = ""
    frame = thread.GetSelectedFrame()
    if frame and frame.IsValid():
        # Check if we have a return value register
        # This is platform-specific
        return_reg = frame.FindRegister("rax")  # x86_64
        if not return_reg or not return_reg.IsValid():
            return_reg = frame.FindRegister("x0")  # ARM64

        if return_reg and return_reg.IsValid():
            return_value = return_reg.GetValue()

    return {"output": "", "return_value": return_value}
