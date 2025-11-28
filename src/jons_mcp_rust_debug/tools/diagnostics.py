"""Diagnostic tools for Rust debugging."""

from __future__ import annotations

from typing import Any

import lldb

from ..lldb_client import DebuggerState
from ..server import ensure_debug_client
from ..utils import stop_reason_to_string, validate_session


async def get_test_summary(session_id: str) -> dict[str, Any]:
    """Get test-specific information for test debugging sessions.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary with test summary
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id, require_process=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if session.target_type != "test":
        return {"status": "error", "error": "Not a test session"}

    # Get test functions from backtrace
    test_functions = []
    panic_info: dict[str, Any] = {}

    if session.process and session.process.IsValid():
        thread = session.process.GetSelectedThread()
        if thread and thread.IsValid():
            # Check frames for test functions
            for i in range(thread.GetNumFrames()):
                frame = thread.GetFrameAtIndex(i)
                if frame and frame.IsValid():
                    func = frame.GetFunction()
                    if func and func.IsValid():
                        func_name = func.GetName()
                        if (
                            "test" in func_name.lower()
                            or func_name.startswith(session.target_name)
                        ):
                            test_functions.append(func_name)

                        # Check for panic info
                        if "panic" in func_name.lower():
                            line_entry = frame.GetLineEntry()
                            if line_entry.IsValid():
                                file_spec = line_entry.GetFileSpec()
                                panic_info = {
                                    "function": func_name,
                                    "file": file_spec.GetFilename(),
                                    "line": line_entry.GetLine(),
                                }

    return {
        "session_type": "test",
        "target": session.target_name,
        "state": session.state.value,
        "last_stop_reason": session.last_stop_reason,
        "test_functions": test_functions,
        "panic_info": panic_info,
        "test_results": {},  # Would need to parse test output
    }


async def get_enum_info(session_id: str, type_name: str) -> dict[str, Any]:
    """Get enum variant information.

    Args:
        session_id: The session identifier
        type_name: Name of the enum type

    Returns:
        Dictionary with enum variants
    """
    client = ensure_debug_client()

    try:
        validate_session(client.sessions, session_id, require_process=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Try to get type info using expression evaluation
    variants: dict[str, Any] = {}

    # This is a simplified version - full implementation would need
    # to parse DWARF debug info or use more sophisticated LLDB APIs

    return {
        "type_name": type_name,
        "variants": variants,
        "raw_output": "Enum introspection not fully implemented in API version",
    }


async def session_diagnostics(session_id: str) -> dict[str, Any]:
    """Get detailed diagnostic information about the debugging session.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary with diagnostic information
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id, require_process=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Basic session info
    diag: dict[str, Any] = {
        "session_id": session_id,
        "debugger_type": "lldb-api",
        "state": session.state.value,
        "has_started": session.process is not None,
        "last_stop_reason": session.last_stop_reason,
        "current_location": session.current_location,
        "breakpoints": len(session.breakpoints),
    }

    # Process info
    if session.process and session.process.IsValid():
        diag["process_alive"] = True
        diag["process_state"] = lldb.SBDebugger.StateAsCString(
            session.process.GetState()
        )

        # Thread info
        thread = session.process.GetSelectedThread()
        if thread and thread.IsValid():
            diag["thread_info"] = {
                "thread_id": thread.GetThreadID(),
                "thread_index": thread.GetIndexID(),
                "stop_reason": stop_reason_to_string(thread.GetStopReason()),
            }

            # Frame info
            frame = thread.GetSelectedFrame()
            if frame and frame.IsValid():
                func = frame.GetFunction()
                diag["frame_info"] = {
                    "frame_index": frame.GetFrameID(),
                    "function": func.GetName() if func and func.IsValid() else "unknown",
                    "is_inlined": frame.IsInlined(),
                }

        # Program status
        exit_status = session.process.GetExitStatus()
        diag["program_status"] = {
            "exit_code": exit_status
            if session.process.GetState() == lldb.eStateExited
            else None,
            "signal": session.process.GetUnixSignals()
            if session.process.GetState() == lldb.eStateCrashed
            else None,
        }
    else:
        diag["process_alive"] = False

    # Test accessibility
    diag["context_tests"] = {
        "can_evaluate": session.state == DebuggerState.PAUSED,
        "has_source_info": session.current_location is not None,
    }

    return diag


async def check_debug_info(session_id: str) -> dict[str, Any]:
    """Check debug symbol and source mapping information.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary with debug info
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id, require_process=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    debug_info: dict[str, Any] = {}

    # Get loaded modules
    modules = []
    for i in range(session.target.GetNumModules()):
        module = session.target.GetModuleAtIndex(i)
        if module and module.IsValid():
            file_spec = module.GetFileSpec()
            modules.append(
                {
                    "path": f"{file_spec.GetDirectory()}/{file_spec.GetFilename()}",
                    "uuid": module.GetUUIDString(),
                    "has_symbols": module.GetNumSymbols() > 0,
                }
            )

    debug_info["modules"] = modules

    # Get current source info if stopped
    if session.process and session.process.IsValid():
        thread = session.process.GetSelectedThread()
        if thread and thread.IsValid():
            frame = thread.GetSelectedFrame()
            if frame and frame.IsValid():
                line_entry = frame.GetLineEntry()
                if line_entry.IsValid():
                    file_spec = line_entry.GetFileSpec()
                    debug_info["current_source"] = {
                        "file": f"{file_spec.GetDirectory()}/{file_spec.GetFilename()}",
                        "line": line_entry.GetLine(),
                        "column": line_entry.GetColumn(),
                    }

    return {"debug_info": debug_info}
