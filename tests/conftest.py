"""Pytest fixtures for jons-mcp-rust-debug tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Prerequisite Detection Fixtures
# =============================================================================


def _check_lldb_available() -> bool:
    """Check if LLDB is available on the system."""
    return shutil.which("lldb") is not None


def _check_cargo_available() -> bool:
    """Check if cargo (Rust toolchain) is available on the system."""
    return shutil.which("cargo") is not None


# Cache the results to avoid repeated checks
_LLDB_AVAILABLE = _check_lldb_available()
_CARGO_AVAILABLE = _check_cargo_available()


@pytest.fixture(scope="session")
def lldb_available() -> bool:
    """Check if LLDB is available for integration tests."""
    return _LLDB_AVAILABLE


@pytest.fixture(scope="session")
def cargo_available() -> bool:
    """Check if cargo is available for integration tests."""
    return _CARGO_AVAILABLE


@pytest.fixture(scope="session")
def integration_prerequisites(lldb_available: bool, cargo_available: bool) -> bool:
    """Check all prerequisites for integration tests."""
    return lldb_available and cargo_available


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    # Register the integration marker (already in pyproject.toml, but good to have here too)
    config.addinivalue_line(
        "markers", "integration: Integration tests with real LLDB process"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip integration tests if prerequisites are not met."""
    skip_lldb = pytest.mark.skip(reason="LLDB not available on this system")
    skip_cargo = pytest.mark.skip(reason="Cargo/Rust toolchain not available")

    for item in items:
        if "integration" in item.keywords:
            if not _LLDB_AVAILABLE:
                item.add_marker(skip_lldb)
            elif not _CARGO_AVAILABLE:
                item.add_marker(skip_cargo)


# =============================================================================
# Integration Test Fixtures
# =============================================================================

# Get the path to test_samples directory
_TEST_SAMPLES_DIR = Path(__file__).parent.parent / "test_samples"
_BINARY_BUILT = False
_BINARY_PATH: Path | None = None


@pytest.fixture(scope="session")
def test_samples_dir() -> Path:
    """Get the path to the test_samples directory."""
    return _TEST_SAMPLES_DIR


@pytest.fixture(scope="session")
def built_sample_binary(
    integration_prerequisites: bool,
) -> Generator[Path | None, None, None]:
    """Build the test_samples binary and return the path to it.

    This is a session-scoped fixture that builds once and caches the result.
    Returns None if build fails or prerequisites are not met.
    """
    global _BINARY_BUILT, _BINARY_PATH

    if not integration_prerequisites:
        yield None
        return

    if _BINARY_BUILT:
        yield _BINARY_PATH
        return

    # Build the sample program
    result = subprocess.run(
        ["cargo", "build"],
        cwd=_TEST_SAMPLES_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Failed to build test_samples: {result.stderr}")
        _BINARY_BUILT = True
        _BINARY_PATH = None
        yield None
        return

    # Find the built binary
    binary_path = _TEST_SAMPLES_DIR / "target" / "debug" / "sample_program"
    if not binary_path.exists():
        print(f"Binary not found at {binary_path}")
        _BINARY_BUILT = True
        _BINARY_PATH = None
        yield None
        return

    _BINARY_BUILT = True
    _BINARY_PATH = binary_path
    yield binary_path


@pytest.fixture(scope="function")
def debug_client_for_test_samples(
    built_sample_binary: Path | None,
) -> Generator[Any, None, None]:
    """Create a debug client configured for test_samples.

    This initializes the global debug_client for integration tests.
    """
    if built_sample_binary is None:
        pytest.skip("test_samples binary not available")

    from src.jons_mcp_rust_debug import server as server_module
    from src.jons_mcp_rust_debug.lldb_client import RustDebugClient

    # Create a real debug client
    client = RustDebugClient()
    # Override working directory to test_samples
    client.config.working_directory = str(_TEST_SAMPLES_DIR)

    # Set the global client
    original_client = server_module.debug_client
    server_module.debug_client = client

    yield client

    # Cleanup: stop all sessions and restore original client
    for session_id in list(client.sessions.keys()):
        try:
            client._stop_session(session_id)
        except Exception:
            pass

    server_module.debug_client = original_client


# =============================================================================
# Unit Test Fixtures (Mock-based)
# =============================================================================


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
    session.watchpoints = {}
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


