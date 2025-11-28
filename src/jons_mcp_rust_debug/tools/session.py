"""Session management tools for Rust debugging."""

from __future__ import annotations

from typing import Any

from ..server import ensure_debug_client


async def start_debug(
    target_type: str,
    target: str | None = None,
    args: list[str] | None = None,
    cargo_flags: list[str] | None = None,
    env: dict[str, str] | None = None,
    package: str | None = None,
) -> dict[str, Any]:
    """Start a new Rust debugging session using LLDB Python API.

    Args:
        target_type: Type of target to debug - "binary", "test", or "example"
        target: Name of the specific target (optional for binary if only one exists)
        args: Command line arguments to pass to the program
        cargo_flags: Additional cargo build flags
        env: Environment variables for the build process
        package: Specific package name for workspace projects

    Returns:
        Dictionary with session_id and status
    """
    try:
        client = ensure_debug_client()
        session_id = client.create_session(
            target_type=target_type,
            target=target or "",
            args=args or [],
            cargo_flags=cargo_flags,
            env=env,
            package=package,
        )
        return {
            "session_id": session_id,
            "status": "started",
            "debugger": "lldb-api",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def stop_debug(session_id: str) -> dict[str, Any]:
    """Stop an active debugging session.

    Args:
        session_id: The session identifier

    Returns:
        Dictionary with status
    """
    client = ensure_debug_client()

    if session_id not in client.sessions:
        return {"status": "error", "error": "Session not found"}

    try:
        client.stop_session_async(session_id)
        return {"status": "stopped"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def list_sessions() -> dict[str, Any]:
    """List all active debugging sessions.

    Returns:
        Dictionary with list of sessions
    """
    client = ensure_debug_client()

    sessions = []
    for session_id, session in client.sessions.items():
        sessions.append(
            {
                "session_id": session_id,
                "target_type": session.target_type,
                "target": session.target_name,
                "state": session.state.value,
                "debugger": "lldb-api",
            }
        )

    return {"sessions": sessions}
