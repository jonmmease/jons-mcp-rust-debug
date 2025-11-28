"""Mock-based tests for MCP tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jons_mcp_rust_debug.lldb_client import DebuggerState


class TestSessionTools:
    """Tests for session management tools."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, mock_debug_client: MagicMock) -> None:
        """Test list_sessions with no active sessions."""
        mock_debug_client.sessions = {}

        with patch(
            "src.jons_mcp_rust_debug.tools.session.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.session import list_sessions

            result = await list_sessions()

        assert result["sessions"] == []

    @pytest.mark.asyncio
    async def test_list_sessions_with_session(
        self, mock_debug_client: MagicMock, mock_session: MagicMock
    ) -> None:
        """Test list_sessions with active session."""
        mock_debug_client.sessions = {"test_session_1": mock_session}

        with patch(
            "src.jons_mcp_rust_debug.tools.session.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.session import list_sessions

            result = await list_sessions()

        assert len(result["sessions"]) == 1
        assert result["sessions"][0]["session_id"] == "test_session_1"
        assert result["sessions"][0]["debugger"] == "lldb-api"

    @pytest.mark.asyncio
    async def test_stop_debug_not_found(self, mock_debug_client: MagicMock) -> None:
        """Test stop_debug with non-existent session."""
        mock_debug_client.sessions = {}

        with patch(
            "src.jons_mcp_rust_debug.tools.session.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.session import stop_debug

            result = await stop_debug("nonexistent")

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()


class TestBreakpointTools:
    """Tests for breakpoint management tools."""

    @pytest.mark.asyncio
    async def test_list_breakpoints_empty(
        self, mock_debug_client: MagicMock, mock_session: MagicMock
    ) -> None:
        """Test list_breakpoints with no breakpoints."""
        mock_session.breakpoints = {}
        mock_debug_client.sessions = {"test_session_1": mock_session}

        with patch(
            "src.jons_mcp_rust_debug.tools.breakpoints.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.breakpoints import list_breakpoints

            result = await list_breakpoints("test_session_1")

        assert result["breakpoints"] == []

    @pytest.mark.asyncio
    async def test_list_breakpoints_session_not_found(
        self, mock_debug_client: MagicMock
    ) -> None:
        """Test list_breakpoints with non-existent session."""
        mock_debug_client.sessions = {}

        with patch(
            "src.jons_mcp_rust_debug.tools.breakpoints.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.breakpoints import list_breakpoints

            result = await list_breakpoints("nonexistent")

        assert result["status"] == "error"


class TestStackTools:
    """Tests for stack navigation tools."""

    @pytest.mark.asyncio
    async def test_backtrace_session_not_found(
        self, mock_debug_client: MagicMock
    ) -> None:
        """Test backtrace with non-existent session."""
        mock_debug_client.sessions = {}

        with patch(
            "src.jons_mcp_rust_debug.tools.stack.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.stack import backtrace

            result = await backtrace("nonexistent")

        assert result["status"] == "error"


class TestInspectionTools:
    """Tests for inspection tools."""

    @pytest.mark.asyncio
    async def test_print_variable_not_paused(
        self, mock_debug_client: MagicMock, mock_session: MagicMock
    ) -> None:
        """Test print_variable when not paused."""
        mock_session.state = DebuggerState.RUNNING
        mock_debug_client.sessions = {"test_session_1": mock_session}

        with patch(
            "src.jons_mcp_rust_debug.tools.inspection.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.inspection import print_variable

            result = await print_variable("test_session_1", "x")

        assert "error" in result
        assert "not paused" in result["error"].lower()


class TestThreadTools:
    """Tests for thread management tools."""

    @pytest.mark.asyncio
    async def test_list_threads_session_not_found(
        self, mock_debug_client: MagicMock
    ) -> None:
        """Test list_threads with non-existent session."""
        mock_debug_client.sessions = {}

        with patch(
            "src.jons_mcp_rust_debug.tools.threads.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.threads import list_threads

            result = await list_threads("nonexistent")

        assert result["status"] == "error"


class TestDiagnosticTools:
    """Tests for diagnostic tools."""

    @pytest.mark.asyncio
    async def test_get_test_summary_not_test_session(
        self, mock_debug_client: MagicMock, mock_session: MagicMock
    ) -> None:
        """Test get_test_summary with non-test session."""
        mock_session.target_type = "binary"
        mock_debug_client.sessions = {"test_session_1": mock_session}

        with patch(
            "src.jons_mcp_rust_debug.tools.diagnostics.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.diagnostics import get_test_summary

            result = await get_test_summary("test_session_1")

        assert result["status"] == "error"
        assert "not a test session" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_session_diagnostics(
        self, mock_debug_client: MagicMock, mock_session: MagicMock
    ) -> None:
        """Test session_diagnostics returns expected fields."""
        mock_debug_client.sessions = {"test_session_1": mock_session}

        with patch(
            "src.jons_mcp_rust_debug.tools.diagnostics.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.diagnostics import session_diagnostics

            result = await session_diagnostics("test_session_1")

        assert "session_id" in result
        assert result["session_id"] == "test_session_1"
        assert result["debugger_type"] == "lldb-api"
        assert "state" in result
        assert "breakpoints" in result


class TestPrintArray:
    """Tests for print_array tool."""

    @pytest.mark.asyncio
    async def test_print_array_session_not_found(
        self, mock_debug_client: MagicMock
    ) -> None:
        """Test print_array with non-existent session."""
        mock_debug_client.sessions = {}

        with patch(
            "src.jons_mcp_rust_debug.tools.inspection.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.inspection import print_array

            result = await print_array("nonexistent", "arr", 10)

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_print_array_not_paused(
        self, mock_debug_client: MagicMock, mock_session: MagicMock
    ) -> None:
        """Test print_array when not paused."""
        mock_session.state = DebuggerState.RUNNING
        mock_debug_client.sessions = {"test_session_1": mock_session}

        with patch(
            "src.jons_mcp_rust_debug.tools.inspection.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.inspection import print_array

            result = await print_array("test_session_1", "arr", 10)

        assert result["status"] == "error"
        assert "not paused" in result["error"].lower()


class TestWatchpoints:
    """Tests for watchpoint tools."""

    @pytest.mark.asyncio
    async def test_set_watchpoint_session_not_found(
        self, mock_debug_client: MagicMock
    ) -> None:
        """Test set_watchpoint with non-existent session."""
        mock_debug_client.sessions = {}

        with patch(
            "src.jons_mcp_rust_debug.tools.watchpoints.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.watchpoints import set_watchpoint

            result = await set_watchpoint("nonexistent", "x")

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_list_watchpoints_session_not_found(
        self, mock_debug_client: MagicMock
    ) -> None:
        """Test list_watchpoints with non-existent session."""
        mock_debug_client.sessions = {}

        with patch(
            "src.jons_mcp_rust_debug.tools.watchpoints.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.watchpoints import list_watchpoints

            result = await list_watchpoints("nonexistent")

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_remove_watchpoint_session_not_found(
        self, mock_debug_client: MagicMock
    ) -> None:
        """Test remove_watchpoint with non-existent session."""
        mock_debug_client.sessions = {}

        with patch(
            "src.jons_mcp_rust_debug.tools.watchpoints.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.watchpoints import remove_watchpoint

            result = await remove_watchpoint("nonexistent", 1)

        assert result["status"] == "error"


class TestSetVariable:
    """Tests for set_variable tool."""

    @pytest.mark.asyncio
    async def test_set_variable_session_not_found(
        self, mock_debug_client: MagicMock
    ) -> None:
        """Test set_variable with non-existent session."""
        mock_debug_client.sessions = {}

        with patch(
            "src.jons_mcp_rust_debug.tools.inspection.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.inspection import set_variable

            result = await set_variable("nonexistent", "x", "42")

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_set_variable_not_paused(
        self, mock_debug_client: MagicMock, mock_session: MagicMock
    ) -> None:
        """Test set_variable when not paused."""
        mock_session.state = DebuggerState.RUNNING
        mock_debug_client.sessions = {"test_session_1": mock_session}

        with patch(
            "src.jons_mcp_rust_debug.tools.inspection.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.inspection import set_variable

            result = await set_variable("test_session_1", "x", "42")

        assert result["status"] == "error"
        assert "not paused" in result["error"].lower()


class TestContinueToLine:
    """Tests for continue_to_line tool."""

    @pytest.mark.asyncio
    async def test_continue_to_line_session_not_found(
        self, mock_debug_client: MagicMock
    ) -> None:
        """Test continue_to_line with non-existent session."""
        mock_debug_client.sessions = {}

        with patch(
            "src.jons_mcp_rust_debug.tools.execution.ensure_debug_client",
            return_value=mock_debug_client,
        ):
            from src.jons_mcp_rust_debug.tools.execution import continue_to_line

            result = await continue_to_line("nonexistent", "main.rs", 42)

        assert result["status"] == "error"
