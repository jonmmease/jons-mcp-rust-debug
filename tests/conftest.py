"""Pytest fixtures for jons-mcp-rust-debug tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def reset_globals() -> Generator[None, None, None]:
    """Reset global state between tests."""
    # Import here to avoid circular imports
    from src.jons_mcp_rust_debug import server as server_module

    original_client = server_module.debug_client
    yield
    server_module.debug_client = original_client


@pytest.fixture
def mock_lldb() -> MagicMock:
    """Create a mock LLDB module."""
    mock = MagicMock()

    # Mock stop reasons
    mock.eStopReasonNone = 0
    mock.eStopReasonTrace = 1
    mock.eStopReasonBreakpoint = 2
    mock.eStopReasonWatchpoint = 3
    mock.eStopReasonSignal = 4
    mock.eStopReasonException = 5
    mock.eStopReasonExec = 6
    mock.eStopReasonPlanComplete = 7

    # Mock states
    mock.eStateStopped = 5
    mock.eStateRunning = 6
    mock.eStateExited = 10
    mock.eStateCrashed = 8
    mock.eStateDetached = 11

    return mock


@pytest.fixture
def mock_debugger(mock_lldb: MagicMock) -> MagicMock:
    """Create a mock SBDebugger."""
    debugger = MagicMock()
    debugger.IsValid.return_value = True
    return debugger


@pytest.fixture
def mock_target(mock_lldb: MagicMock) -> MagicMock:
    """Create a mock SBTarget."""
    target = MagicMock()
    target.IsValid.return_value = True
    return target


@pytest.fixture
def mock_process(mock_lldb: MagicMock) -> MagicMock:
    """Create a mock SBProcess."""
    process = MagicMock()
    process.IsValid.return_value = True
    process.GetState.return_value = mock_lldb.eStateStopped
    process.GetNumThreads.return_value = 1
    return process


@pytest.fixture
def mock_thread(mock_lldb: MagicMock) -> MagicMock:
    """Create a mock SBThread."""
    thread = MagicMock()
    thread.IsValid.return_value = True
    thread.GetStopReason.return_value = mock_lldb.eStopReasonBreakpoint
    thread.GetThreadID.return_value = 12345
    thread.GetNumFrames.return_value = 5
    return thread


@pytest.fixture
def mock_frame() -> MagicMock:
    """Create a mock SBFrame."""
    frame = MagicMock()
    frame.IsValid.return_value = True
    frame.GetFrameID.return_value = 0

    # Mock line entry
    line_entry = MagicMock()
    line_entry.IsValid.return_value = True
    line_entry.GetLine.return_value = 42

    file_spec = MagicMock()
    file_spec.GetFilename.return_value = "main.rs"
    file_spec.GetDirectory.return_value = "/src"
    line_entry.GetFileSpec.return_value = file_spec

    frame.GetLineEntry.return_value = line_entry

    # Mock function
    func = MagicMock()
    func.IsValid.return_value = True
    func.GetName.return_value = "test_function"
    frame.GetFunction.return_value = func

    return frame


@pytest.fixture
def mock_session(
    mock_debugger: MagicMock,
    mock_target: MagicMock,
    mock_process: MagicMock,
    mock_thread: MagicMock,
    mock_frame: MagicMock,
) -> MagicMock:
    """Create a mock DebugSession with all dependencies."""
    from src.jons_mcp_rust_debug.lldb_client import DebuggerState

    session = MagicMock()
    session.session_id = "test_session_1"
    session.debugger = mock_debugger
    session.target = mock_target
    session.process = mock_process
    session.state = DebuggerState.PAUSED
    session.breakpoints = {}
    session.target_type = "binary"
    session.target_name = "test_binary"
    session.args = []
    session.last_stop_reason = "breakpoint_1"
    session.current_location = "main.rs:42"
    session.output_buffer = ""

    # Wire up thread and frame
    mock_process.GetSelectedThread.return_value = mock_thread
    mock_thread.GetSelectedFrame.return_value = mock_frame
    mock_thread.GetFrameAtIndex.return_value = mock_frame
    mock_process.GetThreadAtIndex.return_value = mock_thread

    return session


@pytest.fixture
def mock_debug_client(mock_session: MagicMock) -> MagicMock:
    """Create a mock RustDebugClient."""
    client = MagicMock()
    client.sessions = {"test_session_1": mock_session}
    client.config = MagicMock()
    client.config.working_directory = "."
    return client


@pytest.fixture
def temp_rust_project(tmp_path: Path) -> Path:
    """Create a temporary Rust project for integration tests."""
    # Create Cargo.toml
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_toml.write_text(
        """\
[package]
name = "test_project"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "test_project"
path = "src/main.rs"
"""
    )

    # Create src directory
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create main.rs with a simple debuggable program
    main_rs = src_dir / "main.rs"
    main_rs.write_text(
        """\
fn add(a: i32, b: i32) -> i32 {
    let result = a + b;
    result
}

fn main() {
    let x = 5;
    let y = 10;
    let sum = add(x, y);
    println!("Sum: {}", sum);
}
"""
    )

    return tmp_path
