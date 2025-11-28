"""Thread management tools for Rust debugging."""

from __future__ import annotations

from typing import Any

from ..server import ensure_debug_client
from ..utils import stop_reason_to_string, validate_session


async def list_threads(session_id: str) -> dict[str, Any]:
    """List all threads in the process.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary with list of threads
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    threads = []
    selected_thread = session.process.GetSelectedThread()
    selected_tid = selected_thread.GetThreadID() if selected_thread else None

    for i in range(session.process.GetNumThreads()):
        thread = session.process.GetThreadAtIndex(i)
        if thread and thread.IsValid():
            # Get thread info
            tid = thread.GetThreadID()
            name = thread.GetName() or f"Thread {i + 1}"

            # Get stop reason
            stop_reason = thread.GetStopReason()
            stop_reason_str = stop_reason_to_string(stop_reason)

            # Get current location
            location = ""
            frame = thread.GetFrameAtIndex(0)
            if frame and frame.IsValid():
                func = frame.GetFunction()
                function_name = func.GetName() if func and func.IsValid() else "??"
                line_entry = frame.GetLineEntry()
                if line_entry.IsValid():
                    file_spec = line_entry.GetFileSpec()
                    location = (
                        f"{function_name} at "
                        f"{file_spec.GetFilename()}:{line_entry.GetLine()}"
                    )
                else:
                    location = function_name

            threads.append(
                {
                    "index": i,
                    "thread_id": tid,
                    "name": name,
                    "stop_reason": stop_reason_str,
                    "location": location,
                    "selected": tid == selected_tid,
                }
            )

    return {"threads": threads}


async def select_thread(session_id: str, thread_index: int) -> dict[str, Any]:
    """Select a specific thread as the active thread.

    Args:
        session_id: The session identifier
        thread_index: Index of the thread to select

    Returns:
        Dictionary with selection status
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if thread_index >= session.process.GetNumThreads():
        return {"status": "error", "error": f"Thread index {thread_index} out of range"}

    thread = session.process.GetThreadAtIndex(thread_index)
    if thread and thread.IsValid():
        session.process.SetSelectedThread(thread)

        # Update stop info for the newly selected thread
        client._update_stop_info(session)

        return {
            "status": "selected",
            "thread_id": thread.GetThreadID(),
            "current_location": session.current_location,
        }

    return {"status": "error", "error": "Failed to select thread"}
