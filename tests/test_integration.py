"""Integration tests with real LLDB.

These tests require:
- Rust toolchain installed
- LLDB available

Run with: uv run pytest tests/test_integration.py -m integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    pass


@pytest.mark.integration
class TestDebugLifecycle:
    """Integration tests for debug session lifecycle.

    These tests are marked as 'integration' and require real LLDB.
    They are automatically skipped if LLDB or cargo is not available.
    """

    async def test_start_and_stop_session(
        self, debug_client_for_test_samples: Any
    ) -> None:
        """Test basic session start and stop.

        This test verifies:
        1. A debug session can be started against the test_samples binary
        2. The session appears in the list of sessions
        3. The session can be stopped cleanly
        """
        from src.jons_mcp_rust_debug.tools import list_sessions, start_debug, stop_debug

        # Start debug session for sample_program binary
        result = await start_debug(target_type="binary", target="sample_program")

        assert result.get("status") != "error", f"Failed to start: {result.get('error')}"
        assert "session_id" in result
        session_id = result["session_id"]

        try:
            # Verify session exists
            sessions = await list_sessions()
            assert any(
                s["session_id"] == session_id for s in sessions["sessions"]
            ), "Session not found in list"
        finally:
            # Always stop the session
            stop_result = await stop_debug(session_id)
            assert stop_result.get("status") == "stopped"

    async def test_set_breakpoint(
        self, debug_client_for_test_samples: Any
    ) -> None:
        """Test setting a breakpoint (without running to it).

        This test verifies breakpoint creation works.
        """
        from src.jons_mcp_rust_debug.tools import (
            list_breakpoints,
            set_breakpoint,
            start_debug,
            stop_debug,
        )

        # Start debug session
        result = await start_debug(target_type="binary", target="sample_program")
        assert result.get("status") != "error", f"Failed to start: {result.get('error')}"
        session_id = result["session_id"]

        try:
            # Set breakpoint at line 68: let mut person = Person::new(...)
            bp_result = await set_breakpoint(
                session_id=session_id,
                file="sample_program.rs",
                line=68,
            )
            assert bp_result.get("status") != "error", f"Failed to set breakpoint: {bp_result.get('error')}"
            assert "breakpoint_id" in bp_result

            # Verify breakpoint appears in list
            bp_list = await list_breakpoints(session_id)
            bp_ids = [bp["id"] for bp in bp_list["breakpoints"]]
            assert bp_result["breakpoint_id"] in bp_ids

        finally:
            await stop_debug(session_id)

    async def test_run_to_breakpoint(
        self, debug_client_for_test_samples: Any
    ) -> None:
        """Test running to a breakpoint.

        This test verifies the full debug flow:
        1. Start session
        2. Set breakpoint
        3. Run to breakpoint
        4. Verify stopped at correct location
        5. Clean stop
        """
        from src.jons_mcp_rust_debug.tools import (
            run,
            set_breakpoint,
            start_debug,
            stop_debug,
        )

        # Start debug session
        result = await start_debug(target_type="binary", target="sample_program")
        assert result.get("status") != "error", f"Failed to start: {result.get('error')}"
        session_id = result["session_id"]

        try:
            # Set breakpoint at line 68: let mut person = Person::new(...)
            bp_result = await set_breakpoint(
                session_id=session_id,
                file="sample_program.rs",
                line=68,
            )
            assert bp_result.get("status") != "error", f"Failed to set breakpoint: {bp_result.get('error')}"

            # Run to breakpoint
            run_result = await run(session_id)
            assert run_result.get("status") == "paused", f"Expected paused, got: {run_result}"
            assert "breakpoint" in run_result.get("stop_reason", ""), f"Expected breakpoint stop reason: {run_result}"
            assert run_result.get("current_location") == "sample_program.rs:68"

        finally:
            await stop_debug(session_id)

    async def test_inspect_variables(
        self, debug_client_for_test_samples: Any
    ) -> None:
        """Test variable inspection after hitting a breakpoint.

        This test verifies:
        1. Run to breakpoint
        2. Step to next line to initialize variable
        3. Inspect local variables
        """
        from src.jons_mcp_rust_debug.tools import (
            list_locals,
            next,
            run,
            set_breakpoint,
            start_debug,
            stop_debug,
        )

        # Start debug session
        result = await start_debug(target_type="binary", target="sample_program")
        assert result.get("status") != "error", f"Failed to start: {result.get('error')}"
        session_id = result["session_id"]

        try:
            # Set breakpoint at line 68: let mut person = Person::new(...)
            await set_breakpoint(
                session_id=session_id,
                file="sample_program.rs",
                line=68,
            )

            # Run to breakpoint
            await run(session_id)

            # Step to initialize the variable
            step_result = await next(session_id)
            assert step_result.get("status") != "error", f"Step failed: {step_result.get('error')}"

            # Now list local variables
            locals_result = await list_locals(session_id)
            assert "locals" in locals_result

            # Should have 'person' variable after the assignment
            var_names = list(locals_result["locals"].keys())
            assert "person" in var_names, f"Expected 'person' in variables: {var_names}"

        finally:
            await stop_debug(session_id)

    async def test_print_array(
        self, debug_client_for_test_samples: Any
    ) -> None:
        """Test print_array on a fixed-size array.

        This test verifies:
        1. Can set breakpoint and run to it
        2. print_array can display array contents
        3. Output contains expected array values
        """
        from src.jons_mcp_rust_debug.tools import (
            print_array,
            run,
            set_breakpoint,
            start_debug,
            stop_debug,
        )

        # Start debug session
        result = await start_debug(target_type="binary", target="sample_program")
        assert result.get("status") != "error", f"Failed to start: {result.get('error')}"
        session_id = result["session_id"]

        try:
            # Set breakpoint at line 95 (after fixed_array is initialized)
            bp_result = await set_breakpoint(
                session_id=session_id,
                file="sample_program.rs",
                line=95,
            )
            assert bp_result.get("status") != "error", f"Failed to set breakpoint: {bp_result.get('error')}"

            # Run to breakpoint
            run_result = await run(session_id)
            assert run_result.get("status") == "paused", f"Expected paused, got: {run_result}"

            # Call print_array on fixed_array (need pointer, so use &fixed_array[0])
            array_result = await print_array(
                session_id=session_id,
                expression="&fixed_array[0]",
                count=5,
            )
            assert array_result.get("status") != "error", f"Failed to print array: {array_result.get('error')}"
            assert "output" in array_result

            # Verify output contains expected values [10, 20, 30, 40, 50]
            output = array_result["output"]
            assert "10" in output, f"Expected '10' in output: {output}"
            assert "20" in output, f"Expected '20' in output: {output}"
            assert "30" in output, f"Expected '30' in output: {output}"
            assert "40" in output, f"Expected '40' in output: {output}"
            assert "50" in output, f"Expected '50' in output: {output}"

        finally:
            await stop_debug(session_id)

    async def test_print_slice(
        self, debug_client_for_test_samples: Any
    ) -> None:
        """Test print_slice on a Rust slice.

        This test verifies:
        1. Can set breakpoint and run to it
        2. print_slice can display slice contents using slice data_ptr
        3. Output contains expected slice values
        """
        from src.jons_mcp_rust_debug.tools import (
            print_array,
            run,
            set_breakpoint,
            start_debug,
            stop_debug,
        )

        # Start debug session
        result = await start_debug(target_type="binary", target="sample_program")
        assert result.get("status") != "error", f"Failed to start: {result.get('error')}"
        session_id = result["session_id"]

        try:
            # Set breakpoint at line 95 (after slice is initialized)
            bp_result = await set_breakpoint(
                session_id=session_id,
                file="sample_program.rs",
                line=95,
            )
            assert bp_result.get("status") != "error", f"Failed to set breakpoint: {bp_result.get('error')}"

            # Run to breakpoint
            run_result = await run(session_id)
            assert run_result.get("status") == "paused", f"Expected paused, got: {run_result}"

            # Call print_array to view slice contents (slice = &fixed_array[1..4] = [20, 30, 40])
            # We access the slice's data_ptr field directly since it's a fat pointer
            slice_result = await print_array(
                session_id=session_id,
                expression="slice.data_ptr",
                count=3,
            )
            assert slice_result.get("status") != "error", f"Failed to print slice: {slice_result.get('error')}"
            assert "output" in slice_result

            # Verify output contains expected slice values [20, 30, 40]
            output = slice_result["output"]
            assert "20" in output, f"Expected '20' in output: {output}"
            assert "30" in output, f"Expected '30' in output: {output}"
            assert "40" in output, f"Expected '40' in output: {output}"

        finally:
            await stop_debug(session_id)

    async def test_set_variable(
        self, debug_client_for_test_samples: Any
    ) -> None:
        """Test set_variable to modify a variable value.

        This test verifies:
        1. Can set breakpoint and run to it
        2. set_variable can change a variable's value
        3. Returns old and new values correctly
        """
        from src.jons_mcp_rust_debug.tools import (
            run,
            set_breakpoint,
            set_variable,
            start_debug,
            stop_debug,
        )

        # Start debug session
        result = await start_debug(target_type="binary", target="sample_program")
        assert result.get("status") != "error", f"Failed to start: {result.get('error')}"
        session_id = result["session_id"]

        try:
            # Set breakpoint at line 112 (after test_value = 42 and test_value += 10)
            bp_result = await set_breakpoint(
                session_id=session_id,
                file="sample_program.rs",
                line=112,
            )
            assert bp_result.get("status") != "error", f"Failed to set breakpoint: {bp_result.get('error')}"

            # Run to breakpoint
            run_result = await run(session_id)
            assert run_result.get("status") == "paused", f"Expected paused, got: {run_result}"

            # Call set_variable to change test_value
            set_result = await set_variable(
                session_id=session_id,
                variable="test_value",
                value="100",
            )
            assert set_result.get("status") != "error", f"Failed to set variable: {set_result.get('error')}"
            assert "old_value" in set_result
            assert "new_value" in set_result

            # Verify old_value is 52 (42 + 10)
            old_value = set_result["old_value"]
            assert "52" in old_value, f"Expected old_value to contain '52', got: {old_value}"

            # Verify new_value is 100
            new_value = set_result["new_value"]
            assert "100" in new_value, f"Expected new_value to contain '100', got: {new_value}"

        finally:
            await stop_debug(session_id)

    async def test_continue_to_line(
        self, debug_client_for_test_samples: Any
    ) -> None:
        """Test continue_to_line to jump to a specific line.

        This test verifies:
        1. Can set breakpoint and run to it
        2. continue_to_line can continue execution to a target line
        3. Execution stops at the correct line
        """
        from src.jons_mcp_rust_debug.tools import (
            continue_to_line,
            run,
            set_breakpoint,
            start_debug,
            stop_debug,
        )

        # Start debug session
        result = await start_debug(target_type="binary", target="sample_program")
        assert result.get("status") != "error", f"Failed to start: {result.get('error')}"
        session_id = result["session_id"]

        try:
            # Set breakpoint at line 89 (start of test data section)
            bp_result = await set_breakpoint(
                session_id=session_id,
                file="sample_program.rs",
                line=89,
            )
            assert bp_result.get("status") != "error", f"Failed to set breakpoint: {bp_result.get('error')}"

            # Run to breakpoint
            run_result = await run(session_id)
            assert run_result.get("status") == "paused", f"Expected paused, got: {run_result}"
            assert run_result.get("current_location") == "sample_program.rs:89"

            # Call continue_to_line to line 117
            continue_result = await continue_to_line(
                session_id=session_id,
                file="sample_program.rs",
                line=117,
            )
            assert continue_result.get("status") == "paused", f"Expected paused, got: {continue_result}"

            # Verify we reached the target line
            assert continue_result.get("current_location") == "sample_program.rs:117", \
                f"Expected location sample_program.rs:117, got: {continue_result.get('current_location')}"

        finally:
            await stop_debug(session_id)
