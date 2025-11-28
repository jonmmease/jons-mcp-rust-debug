"""Stack navigation tools for Rust debugging."""

from __future__ import annotations

from typing import Any

from ..server import ensure_debug_client
from ..utils import get_active_thread, paginate_text, validate_session


async def backtrace(
    session_id: str,
    frame_limit: int | None = None,
    char_limit: int | None = None,
    char_offset: int | None = None,
) -> dict[str, Any]:
    """Get the current call stack (backtrace).

    Args:
        session_id: The session identifier
        frame_limit: Maximum number of frames to return
        char_limit: Maximum characters for pagination
        char_offset: Character offset for pagination

    Returns:
        Dictionary with stack frames
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Get frames
    frames = []
    num_frames = thread.GetNumFrames()
    limit = min(frame_limit, num_frames) if frame_limit else num_frames

    for i in range(limit):
        frame = thread.GetFrameAtIndex(i)
        if not frame or not frame.IsValid():
            continue

        # Get frame info
        pc = frame.GetPC()
        func = frame.GetFunction()
        function_name = func.GetName() if func and func.IsValid() else "unknown"

        line_entry = frame.GetLineEntry()
        file = ""
        line = 0
        if line_entry.IsValid():
            file_spec = line_entry.GetFileSpec()
            file = file_spec.GetFilename()
            line = line_entry.GetLine()

        frames.append(
            {
                "index": i,
                "address": f"0x{pc:x}",
                "function": function_name,
                "file": file,
                "line": line,
            }
        )

    # Generate text output for pagination
    output_lines = []
    for f in frames:
        if f["file"]:
            output_lines.append(
                f"#{f['index']} {f['address']} in {f['function']} at {f['file']}:{f['line']}"
            )
        else:
            output_lines.append(f"#{f['index']} {f['address']} in {f['function']}")

    raw_output = "\n".join(output_lines)
    pagination = paginate_text(raw_output, char_limit, char_offset)

    return {
        "frames": frames,
        "raw_output": pagination["content"],
        "pagination": {
            "total_chars": pagination["total_chars"],
            "offset": pagination["offset"],
            "limit": pagination["limit"],
            "has_more": pagination["has_more"],
        },
    }


async def up(session_id: str, count: int = 1) -> dict[str, Any]:
    """Move up in the call stack.

    Args:
        session_id: The session identifier
        count: Number of frames to move up

    Returns:
        Dictionary with new frame info
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Get current frame index
    current_frame = thread.GetSelectedFrame()
    current_idx = current_frame.GetFrameID() if current_frame else 0

    # Move up
    new_idx = min(current_idx + count, thread.GetNumFrames() - 1)
    thread.SetSelectedFrame(new_idx)

    # Get new frame info
    frame = thread.GetFrameAtIndex(new_idx)
    frame_info = {}

    if frame and frame.IsValid():
        func = frame.GetFunction()
        function_name = func.GetName() if func and func.IsValid() else "unknown"

        line_entry = frame.GetLineEntry()
        file = ""
        line = 0
        if line_entry.IsValid():
            file_spec = line_entry.GetFileSpec()
            file = file_spec.GetFilename()
            line = line_entry.GetLine()

        frame_info = {
            "index": new_idx,
            "function": function_name,
            "file": file,
            "line": line,
        }

    return {"frame": frame_info, "output": ""}


async def down(session_id: str, count: int = 1) -> dict[str, Any]:
    """Move down in the call stack.

    Args:
        session_id: The session identifier
        count: Number of frames to move down

    Returns:
        Dictionary with new frame info
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Get current frame index
    current_frame = thread.GetSelectedFrame()
    current_idx = current_frame.GetFrameID() if current_frame else 0

    # Move down
    new_idx = max(current_idx - count, 0)
    thread.SetSelectedFrame(new_idx)

    # Get new frame info
    frame = thread.GetFrameAtIndex(new_idx)
    frame_info = {}

    if frame and frame.IsValid():
        func = frame.GetFunction()
        function_name = func.GetName() if func and func.IsValid() else "unknown"

        line_entry = frame.GetLineEntry()
        file = ""
        line = 0
        if line_entry.IsValid():
            file_spec = line_entry.GetFileSpec()
            file = file_spec.GetFilename()
            line = line_entry.GetLine()

        frame_info = {
            "index": new_idx,
            "function": function_name,
            "file": file,
            "line": line,
        }

    return {"frame": frame_info, "output": ""}
