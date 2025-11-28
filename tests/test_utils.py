"""Unit tests for utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.jons_mcp_rust_debug.constants import (
    STOP_REASON_BREAKPOINT,
    STOP_REASON_NONE,
    STOP_REASON_PLAN_COMPLETE,
    STOP_REASON_SIGNAL,
    STOP_REASON_UNKNOWN,
)
from src.jons_mcp_rust_debug.exceptions import (
    DebuggerNotPausedError,
    NoActiveThreadError,
    ProcessNotRunningError,
    SessionNotFoundError,
)
from src.jons_mcp_rust_debug.lldb_client import DebuggerState, DebugSession
from src.jons_mcp_rust_debug.utils import (
    format_error_response,
    get_frame_location,
    paginate_text,
    stop_reason_to_string,
    validate_session,
)


class TestPaginateText:
    """Tests for paginate_text function."""

    def test_no_limit(self) -> None:
        """Test pagination with no limit returns full text."""
        result = paginate_text("hello world")
        assert result["content"] == "hello world"
        assert result["total_chars"] == 11
        assert result["offset"] == 0
        assert result["limit"] is None
        assert result["has_more"] is False

    def test_with_limit(self) -> None:
        """Test pagination with limit."""
        result = paginate_text("hello world", limit=5, offset=0)
        assert result["content"] == "hello"
        assert result["total_chars"] == 11
        assert result["offset"] == 0
        assert result["limit"] == 5
        assert result["has_more"] is True

    def test_with_offset(self) -> None:
        """Test pagination with offset."""
        result = paginate_text("hello world", limit=5, offset=6)
        assert result["content"] == "world"
        assert result["total_chars"] == 11
        assert result["offset"] == 6
        assert result["has_more"] is False

    def test_offset_beyond_text(self) -> None:
        """Test pagination with offset beyond text length."""
        result = paginate_text("hello", offset=10)
        assert result["content"] == ""
        assert result["total_chars"] == 5
        assert result["has_more"] is False

    def test_empty_text(self) -> None:
        """Test pagination with empty text."""
        result = paginate_text("")
        assert result["content"] == ""
        assert result["total_chars"] == 0
        assert result["has_more"] is False


class TestStopReasonToString:
    """Tests for stop_reason_to_string function."""

    def test_known_reasons(self) -> None:
        """Test conversion of known stop reasons."""
        # Use real LLDB constants
        import lldb

        assert stop_reason_to_string(lldb.eStopReasonNone) == STOP_REASON_NONE
        assert stop_reason_to_string(lldb.eStopReasonBreakpoint) == STOP_REASON_BREAKPOINT
        assert stop_reason_to_string(lldb.eStopReasonSignal) == STOP_REASON_SIGNAL
        assert stop_reason_to_string(lldb.eStopReasonPlanComplete) == STOP_REASON_PLAN_COMPLETE

    def test_unknown_reason(self) -> None:
        """Test conversion of unknown stop reason."""
        assert stop_reason_to_string(999) == STOP_REASON_UNKNOWN


class TestGetFrameLocation:
    """Tests for get_frame_location function."""

    def test_valid_frame(self, mock_frame: MagicMock) -> None:
        """Test extracting location from valid frame."""
        result = get_frame_location(mock_frame)
        assert result["file"] == "main.rs"
        assert result["line"] == 42
        assert result["location"] == "main.rs:42"
        assert result["function"] == "test_function"

    def test_invalid_frame(self) -> None:
        """Test extracting location from invalid frame."""
        frame = MagicMock()
        frame.IsValid.return_value = False
        result = get_frame_location(frame)
        assert result["file"] == ""
        assert result["line"] == 0
        assert result["location"] == ""
        assert result["function"] == ""

    def test_none_frame(self) -> None:
        """Test extracting location from None frame."""
        result = get_frame_location(None)
        assert result["file"] == ""
        assert result["line"] == 0


class TestValidateSession:
    """Tests for validate_session function."""

    def test_session_not_found(self) -> None:
        """Test validation with non-existent session."""
        sessions: dict[str, DebugSession] = {}
        with pytest.raises(SessionNotFoundError):
            validate_session(sessions, "nonexistent")

    def test_process_not_running(self, mock_session: MagicMock) -> None:
        """Test validation when process is not running."""
        mock_session.process = None
        sessions = {"test": mock_session}
        with pytest.raises(ProcessNotRunningError):
            validate_session(sessions, "test", require_process=True)

    def test_not_paused(self, mock_session: MagicMock) -> None:
        """Test validation when not in paused state."""
        mock_session.state = DebuggerState.RUNNING
        sessions = {"test": mock_session}
        with pytest.raises(DebuggerNotPausedError):
            validate_session(sessions, "test", require_paused=True)

    def test_valid_session(self, mock_session: MagicMock) -> None:
        """Test validation with valid session."""
        sessions = {"test": mock_session}
        result = validate_session(sessions, "test")
        assert result == mock_session


class TestFormatErrorResponse:
    """Tests for format_error_response function."""

    def test_basic_error(self) -> None:
        """Test basic error response."""
        result = format_error_response("Something went wrong")
        assert result["status"] == "error"
        assert result["error"] == "Something went wrong"

    def test_with_extra_fields(self) -> None:
        """Test error response with extra fields."""
        result = format_error_response("Error", session_id="123", code=500)
        assert result["status"] == "error"
        assert result["error"] == "Error"
        assert result["session_id"] == "123"
        assert result["code"] == 500
