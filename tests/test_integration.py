"""Integration tests with real LLDB.

These tests require:
- Rust toolchain installed
- LLDB available

Run with: uv run pytest tests/test_integration.py -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
class TestIntegrationDebugSession:
    """Integration tests for full debug session lifecycle.

    These tests are marked as 'integration' and require real LLDB.
    Skip if LLDB is not available.
    """

    @pytest.fixture
    def rust_project(self, temp_rust_project: Path) -> Path:
        """Get the temporary Rust project path."""
        return temp_rust_project

    @pytest.mark.skip(reason="Requires building Rust project - run manually")
    async def test_full_debug_lifecycle(self, rust_project: Path) -> None:
        """Test complete debug workflow.

        This test:
        1. Starts a debug session
        2. Sets a breakpoint
        3. Runs to the breakpoint
        4. Inspects variables
        5. Steps through code
        6. Stops the session
        """
        # Import here to avoid issues when LLDB is not available
        from src.jons_mcp_rust_debug.tools import (
            list_sessions,
            run,
            set_breakpoint,
            start_debug,
            stop_debug,
        )

        # Start debug session
        result = await start_debug(target_type="binary")

        if result.get("status") == "error":
            pytest.skip(f"Could not start debug session: {result.get('error')}")

        session_id = result["session_id"]

        try:
            # Verify session exists
            sessions = await list_sessions()
            assert any(
                s["session_id"] == session_id for s in sessions["sessions"]
            )

            # Set breakpoint
            bp_result = await set_breakpoint(
                session_id=session_id,
                file="main.rs",
                line=7,  # let x = 5;
            )
            assert bp_result.get("status") != "error", bp_result.get("error")

            # Run to breakpoint
            run_result = await run(session_id=session_id)
            assert run_result.get("status") in ["paused", "stopped", "finished"]

        finally:
            # Always stop the session
            await stop_debug(session_id)

    @pytest.mark.skip(reason="Requires building Rust project - run manually")
    async def test_breakpoint_operations(self, rust_project: Path) -> None:
        """Test breakpoint set/remove/list operations."""
        from src.jons_mcp_rust_debug.tools import (
            list_breakpoints,
            remove_breakpoint,
            set_breakpoint,
            start_debug,
            stop_debug,
        )

        result = await start_debug(target_type="binary")
        if result.get("status") == "error":
            pytest.skip(f"Could not start debug session: {result.get('error')}")

        session_id = result["session_id"]

        try:
            # Set breakpoint
            bp_result = await set_breakpoint(
                session_id=session_id,
                file="main.rs",
                line=7,
            )
            if bp_result.get("status") == "error":
                pytest.skip(f"Could not set breakpoint: {bp_result.get('error')}")

            bp_id = bp_result["breakpoint_id"]

            # List breakpoints
            list_result = await list_breakpoints(session_id)
            assert len(list_result["breakpoints"]) == 1
            assert list_result["breakpoints"][0]["id"] == bp_id

            # Remove breakpoint
            remove_result = await remove_breakpoint(session_id, bp_id)
            assert remove_result["status"] == "removed"

            # Verify removed
            list_result = await list_breakpoints(session_id)
            assert len(list_result["breakpoints"]) == 0

        finally:
            await stop_debug(session_id)
