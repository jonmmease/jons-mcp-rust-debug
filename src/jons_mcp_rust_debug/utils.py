"""Utility functions for the Rust debug MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lldb

from .constants import (
    STOP_REASON_BREAKPOINT,
    STOP_REASON_EXCEPTION,
    STOP_REASON_EXEC,
    STOP_REASON_NONE,
    STOP_REASON_PLAN_COMPLETE,
    STOP_REASON_SIGNAL,
    STOP_REASON_TRACE,
    STOP_REASON_UNKNOWN,
    STOP_REASON_WATCHPOINT,
)
from .exceptions import NoActiveFrameError, NoActiveThreadError

if TYPE_CHECKING:
    from .lldb_client import DebugSession


def get_active_thread(process: lldb.SBProcess) -> lldb.SBThread:
    """Get the active thread, preferring one with a stop reason.

    This eliminates the 14x duplicated pattern throughout the codebase.

    Args:
        process: The LLDB process to get a thread from.

    Returns:
        The selected thread with a stop reason, or the currently selected thread.

    Raises:
        NoActiveThreadError: If no valid thread is available.
    """
    if not process or not process.IsValid():
        raise NoActiveThreadError()

    thread = process.GetSelectedThread()
    if thread and thread.IsValid() and thread.GetStopReason() != lldb.eStopReasonNone:
        return thread

    # Search for a thread with a stop reason
    for i in range(process.GetNumThreads()):
        t = process.GetThreadAtIndex(i)
        if t and t.IsValid() and t.GetStopReason() != lldb.eStopReasonNone:
            process.SetSelectedThread(t)
            return t

    # Fallback to selected thread if valid
    if thread and thread.IsValid():
        return thread

    raise NoActiveThreadError()


def get_active_frame(process: lldb.SBProcess) -> lldb.SBFrame:
    """Get the active frame from the active thread.

    Args:
        process: The LLDB process to get a frame from.

    Returns:
        The selected frame from the active thread.

    Raises:
        NoActiveThreadError: If no valid thread is available.
        NoActiveFrameError: If no valid frame is available.
    """
    thread = get_active_thread(process)
    frame = thread.GetSelectedFrame()
    if not frame or not frame.IsValid():
        raise NoActiveFrameError()
    return frame


def stop_reason_to_string(reason: int) -> str:
    """Convert LLDB stop reason constant to human-readable string.

    Args:
        reason: The LLDB stop reason constant.

    Returns:
        Human-readable string representation of the stop reason.
    """
    reasons = {
        lldb.eStopReasonNone: STOP_REASON_NONE,
        lldb.eStopReasonTrace: STOP_REASON_TRACE,
        lldb.eStopReasonBreakpoint: STOP_REASON_BREAKPOINT,
        lldb.eStopReasonWatchpoint: STOP_REASON_WATCHPOINT,
        lldb.eStopReasonSignal: STOP_REASON_SIGNAL,
        lldb.eStopReasonException: STOP_REASON_EXCEPTION,
        lldb.eStopReasonExec: STOP_REASON_EXEC,
        lldb.eStopReasonPlanComplete: STOP_REASON_PLAN_COMPLETE,
    }
    return reasons.get(reason, STOP_REASON_UNKNOWN)


def paginate_text(
    text: str,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any]:
    """Paginate text output by character count.

    Args:
        text: The text to paginate.
        limit: Maximum number of characters to return.
        offset: Starting character position.

    Returns:
        Dictionary with content, pagination metadata.
    """
    total_chars = len(text)
    offset = offset or 0

    if limit is None:
        content = text[offset:]
        has_more = False
    else:
        end = offset + limit
        content = text[offset:end]
        has_more = end < total_chars

    return {
        "content": content,
        "total_chars": total_chars,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
    }


def get_frame_location(frame: lldb.SBFrame) -> dict[str, Any]:
    """Extract location info from a frame.

    Args:
        frame: The LLDB frame to extract location from.

    Returns:
        Dictionary with location, file, line, and function info.
    """
    location = ""
    file = ""
    line = 0
    function = ""

    if frame and frame.IsValid():
        line_entry = frame.GetLineEntry()
        if line_entry.IsValid():
            file_spec = line_entry.GetFileSpec()
            file = file_spec.GetFilename() or ""
            line = line_entry.GetLine()
            if file:
                location = f"{file}:{line}"

        func = frame.GetFunction()
        if func and func.IsValid():
            function = func.GetName() or ""

    return {
        "location": location,
        "file": file,
        "line": line,
        "function": function,
    }


def validate_session(
    sessions: dict[str, "DebugSession"],
    session_id: str,
    require_process: bool = True,
    require_paused: bool = False,
) -> "DebugSession":
    """Validate session exists and optionally has a running process.

    Args:
        sessions: Dictionary of session_id to DebugSession.
        session_id: The session ID to validate.
        require_process: If True, require process to be valid.
        require_paused: If True, require session to be in PAUSED state.

    Returns:
        The validated DebugSession.

    Raises:
        SessionNotFoundError: If session is not found.
        ProcessNotRunningError: If process is required but not running.
        DebuggerNotPausedError: If paused state is required but not paused.
    """
    from .exceptions import (
        DebuggerNotPausedError,
        ProcessNotRunningError,
        SessionNotFoundError,
    )
    from .lldb_client import DebuggerState

    if session_id not in sessions:
        raise SessionNotFoundError(session_id)

    session = sessions[session_id]

    if require_process and (not session.process or not session.process.IsValid()):
        raise ProcessNotRunningError()

    if require_paused and session.state != DebuggerState.PAUSED:
        raise DebuggerNotPausedError(session.state.value)

    return session


def format_error_response(error: str, **extra_fields: Any) -> dict[str, Any]:
    """Create consistent error response.

    Args:
        error: The error message.
        **extra_fields: Additional fields to include in response.

    Returns:
        Dictionary with status="error" and error message.
    """
    response: dict[str, Any] = {"status": "error", "error": error}
    response.update(extra_fields)
    return response
