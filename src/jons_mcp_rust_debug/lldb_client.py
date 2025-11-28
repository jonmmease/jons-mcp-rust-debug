"""LLDB client for managing Rust debugger sessions."""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lldb

from .constants import (
    CLEANUP_TIMEOUT,
    EVENT_POLL_TIMEOUT,
)
from .exceptions import BuildError, LaunchError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DebuggerState(Enum):
    """States of the debugger."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class Config:
    """Configuration for Rust Debug MCP server."""

    cargo_path: str | None = None
    working_directory: str = "."
    environment: dict[str, str] = field(default_factory=dict)
    cargo_args: list[str] = field(default_factory=list)


@dataclass
class Breakpoint:
    """Represents a breakpoint."""

    id: int
    file: str
    line: int
    function: str | None = None
    condition: str | None = None
    temporary: bool = False
    enabled: bool = True
    hit_count: int = 0
    lldb_breakpoint: lldb.SBBreakpoint | None = None


@dataclass
class DebugSession:
    """Represents a debugging session using LLDB Python API."""

    session_id: str
    debugger: lldb.SBDebugger
    target: lldb.SBTarget
    process: lldb.SBProcess | None = None
    listener: lldb.SBListener | None = None
    event_thread: threading.Thread | None = None
    state: DebuggerState = DebuggerState.IDLE
    breakpoints: dict[int, Breakpoint] = field(default_factory=dict)
    target_type: str = "binary"  # "binary", "test", "example"
    target_name: str = ""
    args: list[str] = field(default_factory=list)
    last_stop_reason: str = ""
    current_location: str | None = None
    output_buffer: str = ""
    created_time: float = field(default_factory=time.time)
    # Thread synchronization
    state_changed: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)


class RustDebugClient:
    """Client for managing Rust debugger sessions using LLDB Python API."""

    def __init__(self) -> None:
        self.sessions: dict[str, DebugSession] = {}
        self.lock = threading.Lock()
        self.session_counter = 0
        self.config = self._load_config()
        atexit.register(self._cleanup_all_sessions)

        # Initialize LLDB
        lldb.SBDebugger.Initialize()

    def _load_config(self) -> Config:
        """Load configuration from rustdebugconfig.json if it exists."""
        config_path = Path("rustdebugconfig.json")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
                # Remove debugger field as we're using LLDB API
                data.pop("debugger", None)
                data.pop("prefer_rust_wrappers", None)
                return Config(**data)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return Config()

    def _cleanup_all_sessions(self) -> None:
        """Clean up all active sessions on exit."""
        for session_id in list(self.sessions.keys()):
            try:
                self._stop_session(session_id)
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")

        # Terminate LLDB
        lldb.SBDebugger.Terminate()

    def _find_cargo_executable(self) -> str:
        """Find cargo executable."""
        if self.config.cargo_path:
            return self.config.cargo_path

        cargo = shutil.which("cargo")
        if not cargo:
            raise RuntimeError(
                "cargo not found. Please install Rust or specify cargo_path in configuration"
            )

        return cargo

    def _build_target(
        self,
        target_type: str,
        target: str,
        cargo_flags: list[str] | None = None,
        env: dict[str, str] | None = None,
        package: str | None = None,
    ) -> str:
        """Build the target and return the path to the binary."""
        cargo = self._find_cargo_executable()

        # Build command based on target type
        if target_type == "test":
            cmd = [cargo, "test", "--no-run"]
            if target:
                cmd.extend(["--test", target])
        elif target_type == "example":
            cmd = [cargo, "build", "--example", target]
        else:  # binary
            cmd = [cargo, "build"]
            if target:
                cmd.extend(["--bin", target])

        # Add package specification for workspace support
        if package:
            cmd.extend(["-p", package])

        # Add any additional cargo args from config
        cmd.extend(self.config.cargo_args)

        # Add any runtime cargo flags
        if cargo_flags:
            cmd.extend(cargo_flags)

        # Merge environment variables
        build_env = {**os.environ, **self.config.environment}
        if env:
            build_env.update(env)

        # Run cargo build
        logger.info(f"Building target: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=self.config.working_directory,
            capture_output=True,
            text=True,
            env=build_env,
        )

        if result.returncode != 0:
            raise BuildError(result.stderr)

        # Parse cargo output to find the binary path
        binary_pattern = re.compile(r"Executable.+\((.+)\)")
        for line in result.stderr.split("\n"):
            match = binary_pattern.search(line)
            if match:
                return match.group(1)

        # Fallback: guess the binary location
        build_mode = (
            "release" if cargo_flags and "--release" in cargo_flags else "debug"
        )
        target_dir = Path(self.config.working_directory) / "target" / build_mode

        if target_type == "test":
            deps_dir = target_dir / "deps"
            if deps_dir.exists():
                test_bins = (
                    list(deps_dir.glob(f"{target}-*"))
                    if target
                    else list(deps_dir.glob("*"))
                )
                test_bins = [
                    b
                    for b in test_bins
                    if b.is_file() and not b.suffix and not b.name.endswith(".dSYM")
                ]
                if test_bins:
                    return str(max(test_bins, key=lambda p: p.stat().st_mtime))
        elif target_type == "example":
            example_bin = target_dir / "examples" / target
            if example_bin.exists():
                return str(example_bin)
        else:
            # Regular binary
            if target:
                bin_path = target_dir / target
            else:
                # Find the project name from Cargo.toml
                cargo_toml = Path(self.config.working_directory) / "Cargo.toml"
                if cargo_toml.exists():
                    import tomllib

                    with open(cargo_toml, "rb") as f:
                        cargo_data = tomllib.load(f)
                        project_name = cargo_data.get("package", {}).get("name", "")
                        if project_name:
                            bin_path = target_dir / project_name.replace("-", "_")
                        else:
                            raise RuntimeError(
                                "Could not determine binary name from Cargo.toml"
                            )
                else:
                    raise RuntimeError("No Cargo.toml found")

            if bin_path.exists():
                return str(bin_path)

        raise RuntimeError(f"Could not find built binary for {target_type} {target}")

    def _event_handler_thread(self, session: DebugSession) -> None:
        """Thread to handle LLDB events."""
        try:
            while session.process and session.process.IsValid():
                event = lldb.SBEvent()
                if session.listener and session.listener.WaitForEvent(
                    EVENT_POLL_TIMEOUT, event
                ):
                    if lldb.SBProcess.EventIsProcessEvent(event):
                        state = lldb.SBProcess.GetStateFromEvent(event)

                        # Use lock for thread-safe state updates
                        with session._lock:
                            if state == lldb.eStateStopped:
                                session.state = DebuggerState.PAUSED
                                self._update_stop_info(session)
                                session.state_changed.set()
                            elif state == lldb.eStateRunning:
                                session.state = DebuggerState.RUNNING
                                session.state_changed.clear()
                            elif state == lldb.eStateExited:
                                session.state = DebuggerState.FINISHED
                                session.state_changed.set()
                            elif state == lldb.eStateCrashed:
                                session.state = DebuggerState.ERROR
                                session.state_changed.set()

        except Exception as e:
            logger.error(f"Event handler error: {e}")
            with session._lock:
                session.state = DebuggerState.ERROR
                session.state_changed.set()

    def _update_stop_info(self, session: DebugSession) -> None:
        """Update stop information when process stops."""
        if not session.process or not session.process.IsValid():
            return

        # Find the thread that caused the stop (e.g., hit a breakpoint)
        thread = None
        for i in range(session.process.GetNumThreads()):
            t = session.process.GetThreadAtIndex(i)
            if t and t.IsValid():
                reason = t.GetStopReason()
                if reason != lldb.eStopReasonNone:
                    thread = t
                    # Select this thread for future operations
                    session.process.SetSelectedThread(t)
                    break

        if not thread:
            # Fallback to selected thread
            thread = session.process.GetSelectedThread()

        if not thread or not thread.IsValid():
            return

        # Get stop reason
        stop_reason = thread.GetStopReason()
        if stop_reason == lldb.eStopReasonBreakpoint:
            bp_id = thread.GetStopReasonDataAtIndex(0)
            session.last_stop_reason = f"breakpoint_{bp_id}"
        elif stop_reason == lldb.eStopReasonSignal:
            session.last_stop_reason = "signal"
        elif stop_reason == lldb.eStopReasonException:
            session.last_stop_reason = "exception"
        elif stop_reason == lldb.eStopReasonPlanComplete:
            session.last_stop_reason = "step"
        else:
            session.last_stop_reason = "unknown"

        # Get current location - ensure we're at the right frame
        frame = thread.GetFrameAtIndex(0)  # Get frame 0 which is where we stopped
        if frame and frame.IsValid():
            thread.SetSelectedFrame(0)  # Make sure frame 0 is selected
            line_entry = frame.GetLineEntry()
            if line_entry.IsValid():
                file_spec = line_entry.GetFileSpec()
                if file_spec.IsValid():
                    session.current_location = (
                        f"{file_spec.GetFilename()}:{line_entry.GetLine()}"
                    )

    def create_session(
        self,
        target_type: str,
        target: str,
        args: list[str],
        cargo_flags: list[str] | None = None,
        env: dict[str, str] | None = None,
        package: str | None = None,
    ) -> str:
        """Create a new debugging session."""
        with self.lock:
            self.session_counter += 1
            session_id = f"session_{self.session_counter}"

            # Build the target
            binary_path = self._build_target(
                target_type, target, cargo_flags, env, package
            )

            # Create debugger instance
            debugger = lldb.SBDebugger.Create()
            debugger.SetAsync(True)  # Enable async mode

            # Create listener for events
            listener = lldb.SBListener("rust-debug-listener")

            # Create target
            error = lldb.SBError()
            target_obj = debugger.CreateTarget(binary_path, None, None, True, error)

            if not target_obj or error.Fail():
                raise LaunchError(
                    error.GetCString() or "Failed to create target"
                )

            # Create session
            session = DebugSession(
                session_id=session_id,
                debugger=debugger,
                target=target_obj,
                listener=listener,
                target_type=target_type,
                target_name=target,
                args=args,
            )

            # Set Rust-specific settings
            debugger.HandleCommand(
                "settings set target.process.thread.step-avoid-regexp '^std::'"
            )
            debugger.HandleCommand(
                "settings set target.process.thread.step-in-avoid-code std"
            )

            # Set breakpoint on panic handlers
            session.target.BreakpointCreateByName("rust_panic")
            session.target.BreakpointCreateByName("rust_begin_unwind")

            # Test-specific setup
            if target_type == "test":
                session.target.BreakpointCreateByName("core::panicking::panic")

            session.state = DebuggerState.PAUSED
            self.sessions[session_id] = session

            return session_id

    def _stop_session(self, session_id: str) -> None:
        """Stop a debugging session."""
        session = self.sessions.get(session_id)
        if not session:
            return

        try:
            # Stop the process
            if session.process and session.process.IsValid():
                session.process.Kill()

            # Destroy the debugger
            if session.debugger:
                lldb.SBDebugger.Destroy(session.debugger)
        except Exception as e:
            logger.error(f"Error stopping session {session_id}: {e}")

        del self.sessions[session_id]

    def stop_session_async(self, session_id: str) -> None:
        """Stop a session with timeout to avoid hanging."""
        cleanup_thread = threading.Thread(
            target=self._stop_session, args=(session_id,), daemon=True
        )
        cleanup_thread.start()
        cleanup_thread.join(timeout=CLEANUP_TIMEOUT)

    def wait_for_stop(
        self, session: DebugSession, timeout: float
    ) -> bool:
        """Wait for session to reach a stopped state.

        Args:
            session: The debug session to wait on.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if stopped within timeout, False otherwise.
        """
        session.state_changed.clear()
        return session.state_changed.wait(timeout=timeout)
