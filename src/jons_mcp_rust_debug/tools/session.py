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
    root_directory: str | None = None,
) -> dict[str, Any]:
    """Start a new Rust debugging session using LLDB Python API.

    Args:
        target_type: Type of target to debug:
            - "binary": Debug a binary target
            - "test": Debug a test (lib test or integration test)
            - "example": Debug an example
        target: The target to debug:
            - For binaries/examples: the target name
            - For integration tests (tests/ dir): the file name without .rs
            - For lib tests: can be empty to build all tests
        args: Command line arguments to pass to the program. For tests, this is
            the test filter. IMPORTANT: The filter path differs by test type:

            Integration tests (tests/ directory):
            - Filter path starts from within the test file
            - Test directly in file: ["test_name", "--exact"]
            - Test in mod tests {}: ["tests::test_name", "--exact"]

            Lib tests (src/ with #[cfg(test)]):
            - Filter path includes the source module name
            - src/utils.rs mod tests {}: ["utils::tests::test_name", "--exact"]
            - src/foo/bar.rs mod tests {}: ["foo::bar::tests::test_name", "--exact"]

            Run `cargo test -- --list` to see all available test paths.
        cargo_flags: Additional cargo build flags (e.g., ["--release"])
        env: Environment variables for the build process
        package: Package name for workspace projects (e.g., "my-crate").
            When specified, the crate directory is automatically resolved
            from the workspace using cargo metadata.
        root_directory: Absolute path to the Rust project root:
            - Single crate: The project directory (where Cargo.toml is)
            - Workspace: The workspace root directory. Use with `package`
              to specify which crate to debug.

    Returns:
        Dictionary with session_id and status

    Notes:
        - Async tests (#[tokio::test], #[async_std::test]) are fully supported
        - The debugger automatically handles breakpoints on spawned threads

    Examples:
        # Single crate project
        start_debug(target_type="binary", target="my_app",
                    root_directory="/path/to/my-project")

        # Single crate - debug an integration test
        start_debug(target_type="test", target="my_test",
                    args=["test_name", "--exact"],
                    root_directory="/path/to/my-project")

        # Workspace - specify package, tool resolves crate directory
        start_debug(target_type="test", target="my_test",
                    package="my-crate",
                    args=["tests::test_name", "--exact"],
                    root_directory="/path/to/workspace")

        # Workspace - debug a binary in a specific package
        start_debug(target_type="binary", target="my_app",
                    package="my-crate",
                    root_directory="/path/to/workspace")
    """
    try:
        client = ensure_debug_client()

        # Set the root directory first (needed for package resolution)
        original_working_dir = None
        if root_directory:
            original_working_dir = client.config.working_directory
            client.config.working_directory = root_directory

        try:
            # If package is specified, resolve its directory from workspace
            if package:
                package_dir = client.resolve_package_directory(package)
                client.config.working_directory = package_dir

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
        finally:
            # Restore original working directory
            if original_working_dir is not None:
                client.config.working_directory = original_working_dir
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
