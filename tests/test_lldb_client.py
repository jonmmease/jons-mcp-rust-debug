"""Unit tests for LLDB client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jons_mcp_rust_debug.lldb_client import (
    Breakpoint,
    Config,
    DebuggerState,
    DebugSession,
)


class TestConfig:
    """Tests for Config dataclass."""

    def test_defaults(self) -> None:
        """Test Config default values."""
        config = Config()
        assert config.cargo_path is None
        assert config.working_directory == "."
        assert config.environment == {}
        assert config.cargo_args == []

    def test_custom_values(self) -> None:
        """Test Config with custom values."""
        config = Config(
            cargo_path="/usr/bin/cargo",
            working_directory="/project",
            environment={"RUST_BACKTRACE": "1"},
            cargo_args=["--release"],
        )
        assert config.cargo_path == "/usr/bin/cargo"
        assert config.working_directory == "/project"
        assert config.environment == {"RUST_BACKTRACE": "1"}
        assert config.cargo_args == ["--release"]


class TestBreakpoint:
    """Tests for Breakpoint dataclass."""

    def test_minimal(self) -> None:
        """Test Breakpoint with minimal fields."""
        bp = Breakpoint(id=1, file="main.rs", line=10)
        assert bp.id == 1
        assert bp.file == "main.rs"
        assert bp.line == 10
        assert bp.function is None
        assert bp.condition is None
        assert bp.temporary is False
        assert bp.enabled is True
        assert bp.hit_count == 0
        assert bp.lldb_breakpoint is None

    def test_full(self) -> None:
        """Test Breakpoint with all fields."""
        mock_bp = MagicMock()
        bp = Breakpoint(
            id=5,
            file="lib.rs",
            line=42,
            function="my_func",
            condition="x > 10",
            temporary=True,
            enabled=False,
            hit_count=3,
            lldb_breakpoint=mock_bp,
        )
        assert bp.id == 5
        assert bp.function == "my_func"
        assert bp.condition == "x > 10"
        assert bp.temporary is True
        assert bp.enabled is False
        assert bp.hit_count == 3
        assert bp.lldb_breakpoint == mock_bp


class TestDebuggerState:
    """Tests for DebuggerState enum."""

    def test_values(self) -> None:
        """Test DebuggerState enum values."""
        assert DebuggerState.IDLE.value == "idle"
        assert DebuggerState.RUNNING.value == "running"
        assert DebuggerState.PAUSED.value == "paused"
        assert DebuggerState.FINISHED.value == "finished"
        assert DebuggerState.ERROR.value == "error"


class TestDebugSession:
    """Tests for DebugSession dataclass."""

    def test_minimal(self) -> None:
        """Test DebugSession with minimal fields."""
        mock_debugger = MagicMock()
        mock_target = MagicMock()

        session = DebugSession(
            session_id="session_1",
            debugger=mock_debugger,
            target=mock_target,
        )

        assert session.session_id == "session_1"
        assert session.debugger == mock_debugger
        assert session.target == mock_target
        assert session.process is None
        assert session.state == DebuggerState.IDLE
        assert session.breakpoints == {}
        assert session.target_type == "binary"
        assert session.args == []

    def test_state_changed_event(self) -> None:
        """Test DebugSession has state_changed event."""
        mock_debugger = MagicMock()
        mock_target = MagicMock()

        session = DebugSession(
            session_id="session_1",
            debugger=mock_debugger,
            target=mock_target,
        )

        # Event should not be set initially
        assert not session.state_changed.is_set()

        # Can set and clear the event
        session.state_changed.set()
        assert session.state_changed.is_set()

        session.state_changed.clear()
        assert not session.state_changed.is_set()

    def test_lock(self) -> None:
        """Test DebugSession has a lock for thread safety."""
        mock_debugger = MagicMock()
        mock_target = MagicMock()

        session = DebugSession(
            session_id="session_1",
            debugger=mock_debugger,
            target=mock_target,
        )

        # Can acquire and release lock
        acquired = session._lock.acquire(blocking=False)
        assert acquired
        session._lock.release()
