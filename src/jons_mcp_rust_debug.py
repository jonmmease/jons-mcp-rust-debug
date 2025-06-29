#!/usr/bin/env python3
"""
FastMCP server that provides Rust debugging capabilities through LLDB Python API.

This server uses LLDB's Python API for direct, reliable debugging control.
Run with: uv run jons_mcp_rust_debug.py
"""

import subprocess
import sys
import os
import json
import re
import threading
import queue
import time
import signal
import atexit
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import shutil
import platform

# Import LLDB
try:
    import lldb
except ImportError:
    print("Error: LLDB Python module not found.")
    print("Please install lldb-python: uv pip install lldb-python --prerelease=allow")
    print("Or ensure LLDB is installed with Python bindings.")
    sys.exit(1)

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "WARNING"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("rust-debug-mcp")


class DebuggerState(Enum):
    """States of the debugger"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class Config:
    """Configuration for Rust Debug MCP server"""
    cargo_path: Optional[str] = None
    working_directory: str = "."
    environment: Dict[str, str] = field(default_factory=dict)
    cargo_args: List[str] = field(default_factory=list)


@dataclass
class Breakpoint:
    """Represents a breakpoint"""
    id: int
    file: str
    line: int
    function: Optional[str] = None
    condition: Optional[str] = None
    temporary: bool = False
    enabled: bool = True
    hit_count: int = 0
    lldb_breakpoint: Optional[lldb.SBBreakpoint] = None


@dataclass
class DebugSession:
    """Represents a debugging session using LLDB Python API"""
    session_id: str
    debugger: lldb.SBDebugger
    target: lldb.SBTarget
    process: Optional[lldb.SBProcess] = None
    listener: Optional[lldb.SBListener] = None
    event_thread: Optional[threading.Thread] = None
    state: DebuggerState = DebuggerState.IDLE
    breakpoints: Dict[int, Breakpoint] = field(default_factory=dict)
    target_type: str = "binary"  # "binary", "test", "example"
    target_name: str = ""
    args: List[str] = field(default_factory=list)
    last_stop_reason: str = ""
    current_location: Optional[str] = None
    output_buffer: str = ""
    created_time: float = field(default_factory=time.time)


class RustDebugClient:
    """Client for managing Rust debugger sessions using LLDB Python API"""

    def __init__(self):
        self.sessions: Dict[str, DebugSession] = {}
        self.lock = threading.Lock()
        self.session_counter = 0
        self.config = self._load_config()
        atexit.register(self._cleanup_all_sessions)
        
        # Initialize LLDB
        lldb.SBDebugger.Initialize()

    def _load_config(self) -> Config:
        """Load configuration from rustdebugconfig.json if it exists"""
        config_path = Path("rustdebugconfig.json")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
                # Remove debugger field as we're using LLDB API
                data.pop('debugger', None)
                data.pop('prefer_rust_wrappers', None)
                return Config(**data)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return Config()

    def _cleanup_all_sessions(self):
        """Clean up all active sessions on exit"""
        for session_id in list(self.sessions.keys()):
            try:
                self._stop_session(session_id)
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
        
        # Terminate LLDB
        lldb.SBDebugger.Terminate()

    def _find_cargo_executable(self) -> str:
        """Find cargo executable"""
        if self.config.cargo_path:
            return self.config.cargo_path
        
        cargo = shutil.which("cargo")
        if not cargo:
            raise RuntimeError("cargo not found. Please install Rust or specify cargo_path in configuration")
        
        return cargo

    def _build_target(self, target_type: str, target: str, cargo_flags: List[str] = None, 
                      env: Dict[str, str] = None, package: Optional[str] = None) -> str:
        """Build the target and return the path to the binary"""
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
            env=build_env
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build target: {result.stderr}")
        
        # Parse cargo output to find the binary path
        binary_pattern = re.compile(r"Executable.+\((.+)\)")
        for line in result.stderr.split("\n"):
            match = binary_pattern.search(line)
            if match:
                return match.group(1)
        
        # Fallback: guess the binary location
        build_mode = "release" if cargo_flags and "--release" in cargo_flags else "debug"
        target_dir = Path(self.config.working_directory) / "target" / build_mode
        
        if target_type == "test":
            deps_dir = target_dir / "deps"
            if deps_dir.exists():
                test_bins = list(deps_dir.glob(f"{target}-*")) if target else list(deps_dir.glob("*"))
                test_bins = [b for b in test_bins if b.is_file() and not b.suffix and not b.name.endswith(".dSYM")]
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
                            raise RuntimeError("Could not determine binary name from Cargo.toml")
                else:
                    raise RuntimeError("No Cargo.toml found")
            
            if bin_path.exists():
                return str(bin_path)
        
        raise RuntimeError(f"Could not find built binary for {target_type} {target}")

    def _event_handler_thread(self, session: DebugSession):
        """Thread to handle LLDB events"""
        try:
            while session.state not in (DebuggerState.FINISHED, DebuggerState.ERROR):
                # Check if process is still valid
                if not session.process or not session.process.IsValid():
                    break
                    
                event = lldb.SBEvent()
                # Use a shorter timeout to be more responsive to state changes
                if session.listener.WaitForEvent(0.5, event):
                    if lldb.SBProcess.EventIsProcessEvent(event):
                        state = lldb.SBProcess.GetStateFromEvent(event)
                        
                        if state == lldb.eStateStopped:
                            session.state = DebuggerState.PAUSED
                            self._update_stop_info(session)
                        elif state == lldb.eStateRunning:
                            session.state = DebuggerState.RUNNING
                        elif state == lldb.eStateExited:
                            session.state = DebuggerState.FINISHED
                            break  # Exit the loop
                        elif state == lldb.eStateCrashed:
                            session.state = DebuggerState.ERROR
                            break  # Exit the loop
                        elif state == lldb.eStateDetached:
                            session.state = DebuggerState.FINISHED
                            break  # Exit the loop
                            
                    # Handle other event types if needed
                    elif event.GetType() == lldb.SBTarget.eBroadcastBitBreakpointChanged:
                        # Breakpoint hit
                        pass
                        
        except Exception as e:
            logger.error(f"Event handler error: {e}")
            session.state = DebuggerState.ERROR
        finally:
            logger.debug(f"Event handler thread exiting for session {session.session_id}")

    def _update_stop_info(self, session: DebugSession):
        """Update stop information when process stops"""
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
                    session.current_location = f"{file_spec.GetFilename()}:{line_entry.GetLine()}"

    def create_session(self, target_type: str, target: str, args: List[str], 
                      cargo_flags: List[str] = None, env: Dict[str, str] = None,
                      package: Optional[str] = None) -> str:
        """Create a new debugging session"""
        with self.lock:
            self.session_counter += 1
            session_id = f"session_{self.session_counter}"
            
            # Build the target
            binary_path = self._build_target(target_type, target, cargo_flags, env, package)
            
            # Create debugger instance
            debugger = lldb.SBDebugger.Create()
            debugger.SetAsync(True)  # Enable async mode
            
            # Create listener for events
            listener = lldb.SBListener("rust-debug-listener")
            
            # Create target
            error = lldb.SBError()
            target_obj = debugger.CreateTarget(binary_path, None, None, True, error)
            
            if not target_obj or error.Fail():
                raise RuntimeError(f"Failed to create target: {error.GetCString()}")
            
            # Create session
            session = DebugSession(
                session_id=session_id,
                debugger=debugger,
                target=target_obj,
                listener=listener,
                target_type=target_type,
                target_name=target,
                args=args
            )
            
            # Set Rust-specific settings
            debugger.HandleCommand("settings set target.process.thread.step-avoid-regexp '^std::'")
            debugger.HandleCommand("settings set target.process.thread.step-in-avoid-code std")
            
            # Set breakpoint on panic handlers
            session.target.BreakpointCreateByName("rust_panic")
            session.target.BreakpointCreateByName("rust_begin_unwind")
            
            # Test-specific setup
            if target_type == "test":
                session.target.BreakpointCreateByName("core::panicking::panic")
            
            session.state = DebuggerState.PAUSED
            self.sessions[session_id] = session
            
            return session_id

    def _stop_session(self, session_id: str):
        """Stop a debugging session"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        try:
            # Mark session as finished to signal event thread to exit
            session.state = DebuggerState.FINISHED
            
            # Stop the process if it's running
            if session.process and session.process.IsValid():
                state = session.process.GetState()
                if state != lldb.eStateExited and state != lldb.eStateDetached:
                    # Try to stop gracefully first
                    error = session.process.Stop()
                    if error.Success():
                        # Give it a moment to stop
                        time.sleep(0.1)
                    
                    # Then kill if still running
                    if session.process.IsValid():
                        session.process.Kill()
                        # Wait a bit for the kill to take effect
                        time.sleep(0.1)
            
            # Wait for event thread to finish (with timeout)
            if session.event_thread and session.event_thread.is_alive():
                session.event_thread.join(timeout=2.0)
                if session.event_thread.is_alive():
                    logger.warning(f"Event thread for session {session_id} did not terminate cleanly")
            
            # Clear the listener
            if session.listener:
                session.listener.Clear()
            
            # Destroy the debugger
            if session.debugger:
                # Clear any remaining commands
                session.debugger.HandleCommand("quit")
                # Small delay to let quit process
                time.sleep(0.1)
                lldb.SBDebugger.Destroy(session.debugger)
                
        except Exception as e:
            logger.error(f"Error stopping session {session_id}: {e}")
        finally:
            # Always remove from sessions dict
            if session_id in self.sessions:
                del self.sessions[session_id]
    
    def _stop_reason_to_string(self, reason: int) -> str:
        """Convert LLDB stop reason to string."""
        reasons = {
            lldb.eStopReasonNone: "none",
            lldb.eStopReasonTrace: "trace",
            lldb.eStopReasonBreakpoint: "breakpoint",
            lldb.eStopReasonWatchpoint: "watchpoint",
            lldb.eStopReasonSignal: "signal",
            lldb.eStopReasonException: "exception",
            lldb.eStopReasonExec: "exec",
            lldb.eStopReasonPlanComplete: "plan_complete"
        }
        return reasons.get(reason, "unknown")


# Global debug client instance
debug_client = RustDebugClient()


# Helper function for pagination
def paginate_text(text: str, limit: Optional[int] = None, offset: Optional[int] = None) -> Dict[str, Any]:
    """Paginate text output by character count"""
    total_chars = len(text)
    offset = offset or 0
    
    if limit is None:
        content = text[offset:]
        has_more = False
    else:
        end = offset + limit
        content = text[offset:end]
        has_more = end < total_chars
    
    return {
        "content": content,
        "total_chars": total_chars,
        "offset": offset,
        "limit": limit,
        "has_more": has_more
    }


# MCP Tools - Session Management

@mcp.tool()
async def start_debug(
    target_type: str, 
    target: Optional[str] = None, 
    args: Optional[List[str]] = None,
    cargo_flags: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    package: Optional[str] = None
) -> Dict[str, Any]:
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
        session_id = debug_client.create_session(
            target_type=target_type,
            target=target or "",
            args=args or [],
            cargo_flags=cargo_flags,
            env=env,
            package=package
        )
        return {
            "session_id": session_id,
            "status": "started",
            "debugger": "lldb-api"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool()
async def stop_debug(session_id: str) -> Dict[str, Any]:
    """Stop an active debugging session.
    
    Args:
        session_id: The session identifier
        
    Returns:
        Dictionary with status
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    try:
        debug_client._stop_session(session_id)
        return {"status": "stopped"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def list_sessions() -> Dict[str, Any]:
    """List all active debugging sessions.
    
    Returns:
        Dictionary with list of sessions
    """
    sessions = []
    for session_id, session in debug_client.sessions.items():
        sessions.append({
            "session_id": session_id,
            "target_type": session.target_type,
            "target": session.target_name,
            "state": session.state.value,
            "debugger": "lldb-api"
        })
    
    return {"sessions": sessions}


# MCP Tools - Breakpoint Management

@mcp.tool()
async def set_breakpoint(
    session_id: str,
    file: Optional[str] = None,
    line: Optional[int] = None,
    function: Optional[str] = None,
    condition: Optional[str] = None,
    temporary: bool = False
) -> Dict[str, Any]:
    """Set a breakpoint in the Rust program.
    
    Args:
        session_id: The session identifier
        file: Source file path
        line: Line number
        function: Function name
        condition: Conditional expression
        temporary: If true, breakpoint is removed after first hit
        
    Returns:
        Dictionary with breakpoint details
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Create breakpoint
    bp = None
    if function:
        bp = session.target.BreakpointCreateByName(function)
    elif file and line:
        # Try different path formats
        bp = session.target.BreakpointCreateByLocation(file, line)
        if not bp or not bp.IsValid() or bp.GetNumLocations() == 0:
            # Try with just filename
            filename = os.path.basename(file)
            bp = session.target.BreakpointCreateByLocation(filename, line)
    
    if not bp or not bp.IsValid():
        return {"status": "error", "error": "Failed to create breakpoint"}
    
    # Set properties
    if condition:
        bp.SetCondition(condition)
    if temporary:
        bp.SetOneShot(True)
    
    # Store breakpoint info
    breakpoint = Breakpoint(
        id=bp.GetID(),
        file=file or "",
        line=line or 0,
        function=function,
        condition=condition,
        temporary=temporary,
        enabled=True,
        lldb_breakpoint=bp
    )
    session.breakpoints[bp.GetID()] = breakpoint
    
    # Get resolved location
    location = ""
    if bp.GetNumLocations() > 0:
        loc = bp.GetLocationAtIndex(0)
        addr = loc.GetAddress()
        if addr.IsValid():
            line_entry = addr.GetLineEntry()
            if line_entry.IsValid():
                file_spec = line_entry.GetFileSpec()
                location = f"{file_spec.GetFilename()}:{line_entry.GetLine()}"
    
    return {
        "breakpoint_id": bp.GetID(),
        "location": location or f"{file}:{line}" if file and line else function,
        "status": "set"
    }


@mcp.tool()
async def remove_breakpoint(session_id: str, breakpoint_id: int) -> Dict[str, Any]:
    """Remove a breakpoint."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if breakpoint_id not in session.breakpoints:
        return {"status": "error", "error": "Breakpoint not found"}
    
    # Remove from LLDB
    bp = session.breakpoints[breakpoint_id].lldb_breakpoint
    if bp and bp.IsValid():
        session.target.BreakpointDelete(bp.GetID())
    
    # Remove from session
    del session.breakpoints[breakpoint_id]
    
    return {"status": "removed"}


@mcp.tool()
async def list_breakpoints(session_id: str) -> Dict[str, Any]:
    """List all breakpoints in the session."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    breakpoints = []
    for bp_id, bp in session.breakpoints.items():
        lldb_bp = bp.lldb_breakpoint
        hit_count = lldb_bp.GetHitCount() if lldb_bp and lldb_bp.IsValid() else 0
        
        breakpoints.append({
            "id": bp_id,
            "file": bp.file,
            "line": bp.line,
            "function": bp.function,
            "condition": bp.condition,
            "temporary": bp.temporary,
            "enabled": bp.enabled,
            "hit_count": hit_count
        })
    
    return {"breakpoints": breakpoints}


# MCP Tools - Execution Control

@mcp.tool()
async def run(
    session_id: str,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> Dict[str, Any]:
    """Start or continue program execution.
    
    Args:
        session_id: The session identifier
        limit: Maximum number of characters to return (for pagination)
        offset: Starting character position (for pagination)
        
    Returns:
        Dictionary with execution status and stop reason
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    action_taken = ""
    # Check if we need to launch a new process
    needs_launch = (not session.process or 
                   not session.process.IsValid() or
                   session.process.GetState() in (lldb.eStateExited, lldb.eStateDetached))
    
    if needs_launch:
        # Reset state for new launch
        session.state = DebuggerState.IDLE
        session.last_stop_reason = ""
        session.current_location = None
        
        # Launch the process
        error = lldb.SBError()
        working_dir = os.path.abspath(debug_client.config.working_directory)
        session.process = session.target.Launch(
            session.listener,
            session.args,
            None,  # envp
            None,  # stdin_path
            None,  # stdout_path
            None,  # stderr_path
            working_dir,  # working directory
            0,     # launch flags
            False, # stop at entry
            error
        )
        
        if error.Fail():
            return {"status": "error", "error": f"Launch failed: {error.GetCString()}"}
        
        action_taken = "Started new execution"
        
        # Start event handler thread
        session.event_thread = threading.Thread(
            target=debug_client._event_handler_thread,
            args=(session,),
            daemon=True
        )
        session.event_thread.start()
    else:
        # Ensure event handler thread is running
        if not session.event_thread or not session.event_thread.is_alive():
            session.event_thread = threading.Thread(
                target=debug_client._event_handler_thread,
                args=(session,),
                daemon=True
            )
            session.event_thread.start()
        
        # Continue execution
        error = session.process.Continue()
        if error.Fail():
            return {"status": "error", "error": f"Continue failed: {error.GetCString()}"}
        action_taken = f"Continued from {session.current_location or 'breakpoint'}"
    
    # Wait for stop or timeout
    timeout = 30.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        if session.state != DebuggerState.RUNNING:
            break
        time.sleep(0.1)
    
    # Update stop info
    debug_client._update_stop_info(session)
    
    # Get output if any (would need to capture stdout/stderr)
    output = session.output_buffer
    session.output_buffer = ""
    
    # Handle pagination
    pagination = paginate_text(output, limit, offset)
    
    return {
        "status": session.state.value,
        "action": action_taken,
        "stop_reason": session.last_stop_reason,
        "stopped_at": session.current_location or "",
        "current_location": session.current_location,
        "output": pagination["content"],
        "pagination": {
            "total_chars": pagination["total_chars"],
            "offset": pagination["offset"],
            "limit": pagination["limit"],
            "has_more": pagination["has_more"]
        }
    }


@mcp.tool()
async def step(session_id: str) -> Dict[str, Any]:
    """Step into the next line of code."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if not session.process or not session.process.IsValid():
        return {"status": "error", "error": "Process not running"}
    
    # Ensure we have the correct thread selected
    thread = session.process.GetSelectedThread()
    if not thread or not thread.IsValid():
        # Try to find a thread with a stop reason
        for i in range(session.process.GetNumThreads()):
            t = session.process.GetThreadAtIndex(i)
            if t and t.IsValid() and t.GetStopReason() != lldb.eStopReasonNone:
                thread = t
                session.process.SetSelectedThread(t)
                break
                
    if not thread or not thread.IsValid():
        return {"status": "error", "error": "No active thread"}
    
    # Step into
    thread.StepInto()
    
    # Wait for stop
    timeout = 5.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        if session.state == DebuggerState.PAUSED:
            break
        time.sleep(0.1)
    
    # Update stop info
    debug_client._update_stop_info(session)
    
    # Get location info
    frame = thread.GetSelectedFrame()
    location = ""
    file = ""
    line = 0
    function = ""
    
    if frame and frame.IsValid():
        line_entry = frame.GetLineEntry()
        if line_entry.IsValid():
            file_spec = line_entry.GetFileSpec()
            file = file_spec.GetFilename()
            line = line_entry.GetLine()
            location = f"{file}:{line}"
        
        func = frame.GetFunction()
        if func and func.IsValid():
            function = func.GetName()
    
    return {
        "status": session.state.value,
        "location": location,
        "file": file,
        "line": line,
        "function": function,
        "output": "",
        "message": "Stepped into next line" if location else "Step completed"
    }


@mcp.tool()
async def next(session_id: str) -> Dict[str, Any]:
    """Step over the next line of code."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if not session.process or not session.process.IsValid():
        return {"status": "error", "error": "Process not running"}
    
    # Ensure we have the correct thread selected
    thread = session.process.GetSelectedThread()
    if not thread or not thread.IsValid():
        # Try to find a thread with a stop reason
        for i in range(session.process.GetNumThreads()):
            t = session.process.GetThreadAtIndex(i)
            if t and t.IsValid() and t.GetStopReason() != lldb.eStopReasonNone:
                thread = t
                session.process.SetSelectedThread(t)
                break
                
    if not thread or not thread.IsValid():
        return {"status": "error", "error": "No active thread"}
    
    # Step over
    thread.StepOver()
    
    # Wait for stop
    timeout = 5.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        if session.state == DebuggerState.PAUSED:
            break
        time.sleep(0.1)
    
    # Update stop info
    debug_client._update_stop_info(session)
    
    # Get location info (same as step)
    frame = thread.GetSelectedFrame()
    location = ""
    file = ""
    line = 0
    function = ""
    
    if frame and frame.IsValid():
        line_entry = frame.GetLineEntry()
        if line_entry.IsValid():
            file_spec = line_entry.GetFileSpec()
            file = file_spec.GetFilename()
            line = line_entry.GetLine()
            location = f"{file}:{line}"
        
        func = frame.GetFunction()
        if func and func.IsValid():
            function = func.GetName()
    
    return {
        "status": session.state.value,
        "location": location,
        "file": file,
        "line": line,
        "function": function,
        "output": "",
        "message": "Stepped over to next line" if location else "Step completed"
    }


@mcp.tool()
async def finish(session_id: str) -> Dict[str, Any]:
    """Continue execution until the current function returns."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if not session.process or not session.process.IsValid():
        return {"status": "error", "error": "Process not running"}
    
    # Ensure we have the correct thread selected
    thread = session.process.GetSelectedThread()
    if not thread or not thread.IsValid():
        # Try to find a thread with a stop reason
        for i in range(session.process.GetNumThreads()):
            t = session.process.GetThreadAtIndex(i)
            if t and t.IsValid() and t.GetStopReason() != lldb.eStopReasonNone:
                thread = t
                session.process.SetSelectedThread(t)
                break
                
    if not thread or not thread.IsValid():
        return {"status": "error", "error": "No active thread"}
    
    # Step out
    thread.StepOut()
    
    # Wait for stop
    timeout = 10.0
    start_time = time.time()
    while time.time() - start_time < timeout:
        if session.state == DebuggerState.PAUSED:
            break
        time.sleep(0.1)
    
    # Try to get return value
    return_value = ""
    frame = thread.GetSelectedFrame()
    if frame and frame.IsValid():
        # Check if we have a return value register
        # This is platform-specific
        return_reg = frame.FindRegister("rax")  # x86_64
        if not return_reg or not return_reg.IsValid():
            return_reg = frame.FindRegister("x0")  # ARM64
        
        if return_reg and return_reg.IsValid():
            return_value = return_reg.GetValue()
    
    return {
        "output": "",
        "return_value": return_value
    }


# MCP Tools - Stack Navigation

@mcp.tool()
async def backtrace(
    session_id: str, 
    frame_limit: Optional[int] = None,
    char_limit: Optional[int] = None,
    char_offset: Optional[int] = None
) -> Dict[str, Any]:
    """Get the current call stack (backtrace)."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if not session.process or not session.process.IsValid():
        return {"status": "error", "error": "Process not running"}
    
    # Ensure we have the correct thread selected
    thread = session.process.GetSelectedThread()
    if not thread or not thread.IsValid():
        # Try to find a thread with a stop reason
        for i in range(session.process.GetNumThreads()):
            t = session.process.GetThreadAtIndex(i)
            if t and t.IsValid() and t.GetStopReason() != lldb.eStopReasonNone:
                thread = t
                session.process.SetSelectedThread(t)
                break
                
    if not thread or not thread.IsValid():
        return {"status": "error", "error": "No active thread"}
    
    # Get frames
    frames = []
    num_frames = thread.GetNumFrames()
    limit = min(frame_limit, num_frames) if frame_limit else num_frames
    
    for i in range(limit):
        frame = thread.GetFrameAtIndex(i)
        if not frame or not frame.IsValid():
            continue
        
        # Get frame info
        pc = frame.GetPC()
        func = frame.GetFunction()
        function_name = func.GetName() if func and func.IsValid() else "unknown"
        
        line_entry = frame.GetLineEntry()
        file = ""
        line = 0
        if line_entry.IsValid():
            file_spec = line_entry.GetFileSpec()
            file = file_spec.GetFilename()
            line = line_entry.GetLine()
        
        frames.append({
            "index": i,
            "address": f"0x{pc:x}",
            "function": function_name,
            "file": file,
            "line": line
        })
    
    # Generate text output for pagination
    output_lines = []
    for f in frames:
        if f["file"]:
            output_lines.append(f"#{f['index']} {f['address']} in {f['function']} at {f['file']}:{f['line']}")
        else:
            output_lines.append(f"#{f['index']} {f['address']} in {f['function']}")
    
    raw_output = "\n".join(output_lines)
    pagination = paginate_text(raw_output, char_limit, char_offset)
    
    return {
        "frames": frames,
        "raw_output": pagination["content"],
        "pagination": {
            "total_chars": pagination["total_chars"],
            "offset": pagination["offset"],
            "limit": pagination["limit"],
            "has_more": pagination["has_more"]
        }
    }


@mcp.tool()
async def up(session_id: str, count: int = 1) -> Dict[str, Any]:
    """Move up in the call stack."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if not session.process or not session.process.IsValid():
        return {"status": "error", "error": "Process not running"}
    
    # Ensure we have the correct thread selected
    thread = session.process.GetSelectedThread()
    if not thread or not thread.IsValid():
        # Try to find a thread with a stop reason
        for i in range(session.process.GetNumThreads()):
            t = session.process.GetThreadAtIndex(i)
            if t and t.IsValid() and t.GetStopReason() != lldb.eStopReasonNone:
                thread = t
                session.process.SetSelectedThread(t)
                break
                
    if not thread or not thread.IsValid():
        return {"status": "error", "error": "No active thread"}
    
    # Get current frame index
    current_frame = thread.GetSelectedFrame()
    current_idx = current_frame.GetFrameID() if current_frame else 0
    
    # Move up
    new_idx = min(current_idx + count, thread.GetNumFrames() - 1)
    thread.SetSelectedFrame(new_idx)
    
    # Get new frame info
    frame = thread.GetFrameAtIndex(new_idx)
    frame_info = {}
    
    if frame and frame.IsValid():
        func = frame.GetFunction()
        function_name = func.GetName() if func and func.IsValid() else "unknown"
        
        line_entry = frame.GetLineEntry()
        file = ""
        line = 0
        if line_entry.IsValid():
            file_spec = line_entry.GetFileSpec()
            file = file_spec.GetFilename()
            line = line_entry.GetLine()
        
        frame_info = {
            "index": new_idx,
            "function": function_name,
            "file": file,
            "line": line
        }
    
    return {"frame": frame_info, "output": ""}


@mcp.tool()
async def down(session_id: str, count: int = 1) -> Dict[str, Any]:
    """Move down in the call stack."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if not session.process or not session.process.IsValid():
        return {"status": "error", "error": "Process not running"}
    
    # Ensure we have the correct thread selected
    thread = session.process.GetSelectedThread()
    if not thread or not thread.IsValid():
        # Try to find a thread with a stop reason
        for i in range(session.process.GetNumThreads()):
            t = session.process.GetThreadAtIndex(i)
            if t and t.IsValid() and t.GetStopReason() != lldb.eStopReasonNone:
                thread = t
                session.process.SetSelectedThread(t)
                break
                
    if not thread or not thread.IsValid():
        return {"status": "error", "error": "No active thread"}
    
    # Get current frame index
    current_frame = thread.GetSelectedFrame()
    current_idx = current_frame.GetFrameID() if current_frame else 0
    
    # Move down
    new_idx = max(current_idx - count, 0)
    thread.SetSelectedFrame(new_idx)
    
    # Get new frame info (same as up)
    frame = thread.GetFrameAtIndex(new_idx)
    frame_info = {}
    
    if frame and frame.IsValid():
        func = frame.GetFunction()
        function_name = func.GetName() if func and func.IsValid() else "unknown"
        
        line_entry = frame.GetLineEntry()
        file = ""
        line = 0
        if line_entry.IsValid():
            file_spec = line_entry.GetFileSpec()
            file = file_spec.GetFilename()
            line = line_entry.GetLine()
        
        frame_info = {
            "index": new_idx,
            "function": function_name,
            "file": file,
            "line": line
        }
    
    return {"frame": frame_info, "output": ""}


# MCP Tools - Inspection

@mcp.tool()
async def list_source(
    session_id: str, 
    line: Optional[int] = None, 
    count: int = 10,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> Dict[str, Any]:
    """Show source code around current or specified line."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if not session.process or not session.process.IsValid():
        return {"status": "error", "error": "Process not running"}
    
    # Ensure we have the correct thread selected
    thread = session.process.GetSelectedThread()
    if not thread or not thread.IsValid():
        # Try to find a thread with a stop reason
        for i in range(session.process.GetNumThreads()):
            t = session.process.GetThreadAtIndex(i)
            if t and t.IsValid() and t.GetStopReason() != lldb.eStopReasonNone:
                thread = t
                session.process.SetSelectedThread(t)
                break
                
    if not thread or not thread.IsValid():
        return {"status": "error", "error": "No active thread"}
    
    frame = thread.GetSelectedFrame()
    if not frame or not frame.IsValid():
        return {"status": "error", "error": "No active frame"}
    
    # Get current location
    line_entry = frame.GetLineEntry()
    if not line_entry.IsValid():
        return {"status": "error", "error": "No source information available"}
    
    file_spec = line_entry.GetFileSpec()
    current_line = line if line else line_entry.GetLine()
    
    # Try to read the source file
    file_path = os.path.join(file_spec.GetDirectory() or "", file_spec.GetFilename() or "")
    
    # If the path doesn't exist, try to resolve it relative to working directory
    if not os.path.exists(file_path):
        # Try just the filename in case it's a relative path
        alt_path = file_spec.GetFilename()
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            # Try to find it in the source directories
            for root, dirs, files in os.walk(debug_client.config.working_directory):
                if file_spec.GetFilename() in files:
                    file_path = os.path.join(root, file_spec.GetFilename())
                    break
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Calculate range
        start = max(0, current_line - count // 2)
        end = min(len(lines), current_line + count // 2)
        
        # Format output
        output_lines = []
        for i in range(start, end):
            prefix = "=>" if i + 1 == current_line else "  "
            output_lines.append(f"{prefix} {i+1:4d} {lines[i].rstrip()}")
        
        output = "\n".join(output_lines)
    except Exception as e:
        return {"status": "error", "error": f"Failed to read source: {e}"}
    
    # Handle pagination
    pagination = paginate_text(output, limit, offset)
    
    return {
        "source": pagination["content"],
        "current_line": current_line,
        "pagination": {
            "total_chars": pagination["total_chars"],
            "offset": pagination["offset"],
            "limit": pagination["limit"],
            "has_more": pagination["has_more"]
        }
    }


@mcp.tool()
async def print_variable(
    session_id: str, 
    expression: str,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    depth: Optional[int] = None
) -> Dict[str, Any]:
    """Print the value of a variable or expression."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if not session.process or not session.process.IsValid():
        return {"status": "error", "error": "Process not running"}
    
    if session.state != DebuggerState.PAUSED:
        return {
            "value": "",
            "type": "",
            "expression": expression,
            "error": f"Cannot print variable: debugger is {session.state.value}, not paused"
        }
    
    # Ensure we have the correct thread selected
    thread = session.process.GetSelectedThread()
    if not thread or not thread.IsValid():
        # Try to find a thread with a stop reason
        for i in range(session.process.GetNumThreads()):
            t = session.process.GetThreadAtIndex(i)
            if t and t.IsValid() and t.GetStopReason() != lldb.eStopReasonNone:
                thread = t
                session.process.SetSelectedThread(t)
                break
                
    if not thread or not thread.IsValid():
        return {"status": "error", "error": "No active thread"}
    
    frame = thread.GetSelectedFrame()
    if not frame or not frame.IsValid():
        return {"status": "error", "error": "No active frame"}
    
    # Evaluate expression
    options = lldb.SBExpressionOptions()
    if depth:
        # This would need custom formatting
        pass
    
    result = frame.EvaluateExpression(expression, options)
    
    if not result or result.GetError().Fail():
        error_msg = result.GetError().GetCString() if result else "Unknown error"
        return {
            "value": "",
            "type": "",
            "expression": expression,
            "error": error_msg
        }
    
    # Get value and type
    value = result.GetValue() or str(result)
    type_name = result.GetTypeName() or "unknown"
    
    # Handle pagination
    value_pagination = paginate_text(value, limit, offset)
    type_pagination = paginate_text(type_name, limit, offset)
    
    return {
        "value": value_pagination["content"],
        "type": type_pagination["content"],
        "expression": expression,
        "pagination": {
            "value": {
                "total_chars": value_pagination["total_chars"],
                "offset": value_pagination["offset"],
                "limit": value_pagination["limit"],
                "has_more": value_pagination["has_more"]
            },
            "type": {
                "total_chars": type_pagination["total_chars"],
                "offset": type_pagination["offset"],
                "limit": type_pagination["limit"],
                "has_more": type_pagination["has_more"]
            }
        }
    }


@mcp.tool()
async def list_locals(
    session_id: str,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> Dict[str, Any]:
    """List all local variables in current scope."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if not session.process or not session.process.IsValid():
        return {"status": "error", "error": "Process not running"}
    
    # Ensure we have the correct thread selected
    thread = session.process.GetSelectedThread()
    if not thread or not thread.IsValid():
        # Try to find a thread with a stop reason
        for i in range(session.process.GetNumThreads()):
            t = session.process.GetThreadAtIndex(i)
            if t and t.IsValid() and t.GetStopReason() != lldb.eStopReasonNone:
                thread = t
                session.process.SetSelectedThread(t)
                break
                
    if not thread or not thread.IsValid():
        return {"status": "error", "error": "No active thread"}
    
    frame = thread.GetSelectedFrame()
    if not frame or not frame.IsValid():
        return {"status": "error", "error": "No active frame"}
    
    # Get local variables
    locals_dict = {}
    variables = frame.GetVariables(True, True, False, True)  # args, locals, statics, in_scope_only
    
    for var in variables:
        if var.IsValid():
            name = var.GetName()
            value = var.GetValue() or str(var)
            type_name = var.GetTypeName()
            locals_dict[name] = f"({type_name}) {value}"
    
    # Format output
    output_lines = []
    for name, info in locals_dict.items():
        output_lines.append(f"{name} = {info}")
    
    output = "\n".join(output_lines)
    
    # Handle pagination
    pagination = paginate_text(output, limit, offset)
    
    return {
        "locals": locals_dict,
        "raw_output": pagination["content"],
        "pagination": {
            "total_chars": pagination["total_chars"],
            "offset": pagination["offset"],
            "limit": pagination["limit"],
            "has_more": pagination["has_more"]
        }
    }


@mcp.tool()
async def evaluate(session_id: str, expression: str) -> Dict[str, Any]:
    """Evaluate a Rust expression in the current context."""
    # This is essentially the same as print_variable but with simpler output
    result = await print_variable(session_id, expression)
    
    if "error" in result:
        return {
            "result": "",
            "error": result["error"],
            "expression": expression
        }
    
    return {
        "result": result["value"],
        "error": None,
        "expression": expression
    }


@mcp.tool()
async def list_threads(session_id: str) -> Dict[str, Any]:
    """List all threads in the process."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if not session.process or not session.process.IsValid():
        return {"status": "error", "error": "Process not running"}
    
    threads = []
    selected_thread = session.process.GetSelectedThread()
    selected_tid = selected_thread.GetThreadID() if selected_thread else None
    
    for i in range(session.process.GetNumThreads()):
        thread = session.process.GetThreadAtIndex(i)
        if thread and thread.IsValid():
            # Get thread info
            tid = thread.GetThreadID()
            name = thread.GetName() or f"Thread {i+1}"
            
            # Get stop reason
            stop_reason = thread.GetStopReason()
            stop_reason_str = debug_client._stop_reason_to_string(stop_reason)
            
            # Get current location
            location = ""
            frame = thread.GetFrameAtIndex(0)
            if frame and frame.IsValid():
                func = frame.GetFunction()
                function_name = func.GetName() if func and func.IsValid() else "??"
                line_entry = frame.GetLineEntry()
                if line_entry.IsValid():
                    file_spec = line_entry.GetFileSpec()
                    location = f"{function_name} at {file_spec.GetFilename()}:{line_entry.GetLine()}"
                else:
                    location = function_name
            
            threads.append({
                "index": i,
                "thread_id": tid,
                "name": name,
                "stop_reason": stop_reason_str,
                "location": location,
                "selected": tid == selected_tid
            })
    
    return {"threads": threads}


@mcp.tool()
async def select_thread(session_id: str, thread_index: int) -> Dict[str, Any]:
    """Select a specific thread as the active thread."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if not session.process or not session.process.IsValid():
        return {"status": "error", "error": "Process not running"}
    
    if thread_index >= session.process.GetNumThreads():
        return {"status": "error", "error": f"Thread index {thread_index} out of range"}
    
    thread = session.process.GetThreadAtIndex(thread_index)
    if thread and thread.IsValid():
        session.process.SetSelectedThread(thread)
        
        # Update stop info for the newly selected thread
        debug_client._update_stop_info(session)
        
        return {
            "status": "selected",
            "thread_id": thread.GetThreadID(),
            "current_location": session.current_location
        }
    
    return {"status": "error", "error": "Failed to select thread"}


@mcp.tool()
async def get_test_summary(session_id: str) -> Dict[str, Any]:
    """Get test-specific information for test debugging sessions."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if session.target_type != "test":
        return {"status": "error", "error": "Not a test session"}
    
    # Get test functions from backtrace
    test_functions = []
    panic_info = {}
    
    if session.process and session.process.IsValid():
        thread = session.process.GetSelectedThread()
        if thread and thread.IsValid():
            # Check frames for test functions
            for i in range(thread.GetNumFrames()):
                frame = thread.GetFrameAtIndex(i)
                if frame and frame.IsValid():
                    func = frame.GetFunction()
                    if func and func.IsValid():
                        func_name = func.GetName()
                        if "test" in func_name.lower() or func_name.startswith(session.target_name):
                            test_functions.append(func_name)
                        
                        # Check for panic info
                        if "panic" in func_name.lower():
                            line_entry = frame.GetLineEntry()
                            if line_entry.IsValid():
                                file_spec = line_entry.GetFileSpec()
                                panic_info = {
                                    "function": func_name,
                                    "file": file_spec.GetFilename(),
                                    "line": line_entry.GetLine()
                                }
    
    return {
        "session_type": "test",
        "target": session.target_name,
        "state": session.state.value,
        "last_stop_reason": session.last_stop_reason,
        "test_functions": test_functions,
        "panic_info": panic_info,
        "test_results": {}  # Would need to parse test output
    }


@mcp.tool()
async def get_enum_info(session_id: str, type_name: str) -> Dict[str, Any]:
    """Get enum variant information."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Try to get type info using expression evaluation
    variants = {}
    
    # This is a simplified version - full implementation would need
    # to parse DWARF debug info or use more sophisticated LLDB APIs
    
    return {
        "type_name": type_name,
        "variants": variants,
        "raw_output": "Enum introspection not fully implemented in API version"
    }


@mcp.tool()
async def session_diagnostics(session_id: str) -> Dict[str, Any]:
    """Get detailed diagnostic information about the debugging session."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Basic session info
    diag = {
        "session_id": session_id,
        "debugger_type": "lldb-api",
        "state": session.state.value,
        "has_started": session.process is not None,
        "last_stop_reason": session.last_stop_reason,
        "current_location": session.current_location,
        "breakpoints": len(session.breakpoints)
    }
    
    # Process info
    if session.process and session.process.IsValid():
        diag["process_alive"] = True
        diag["process_state"] = lldb.SBDebugger.StateAsCString(session.process.GetState())
        
        # Thread info
        thread = session.process.GetSelectedThread()
        if thread and thread.IsValid():
            diag["thread_info"] = {
                "thread_id": thread.GetThreadID(),
                "thread_index": thread.GetIndexID(),
                "stop_reason": lldb.SBDebugger.GetStopReasonString(thread.GetStopReason())
            }
            
            # Frame info
            frame = thread.GetSelectedFrame()
            if frame and frame.IsValid():
                func = frame.GetFunction()
                diag["frame_info"] = {
                    "frame_index": frame.GetFrameID(),
                    "function": func.GetName() if func and func.IsValid() else "unknown",
                    "is_inlined": frame.IsInlined()
                }
        
        # Program status
        exit_status = session.process.GetExitStatus()
        diag["program_status"] = {
            "exit_code": exit_status if session.process.GetState() == lldb.eStateExited else None,
            "signal": session.process.GetUnixSignals() if session.process.GetState() == lldb.eStateCrashed else None
        }
    else:
        diag["process_alive"] = False
    
    # Test accessibility
    diag["context_tests"] = {
        "can_evaluate": session.state == DebuggerState.PAUSED,
        "has_source_info": session.current_location is not None
    }
    
    return diag


@mcp.tool()
async def check_debug_info(session_id: str) -> Dict[str, Any]:
    """Check debug symbol and source mapping information."""
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    debug_info = {}
    
    # Get loaded modules
    modules = []
    for i in range(session.target.GetNumModules()):
        module = session.target.GetModuleAtIndex(i)
        if module and module.IsValid():
            file_spec = module.GetFileSpec()
            modules.append({
                "path": f"{file_spec.GetDirectory()}/{file_spec.GetFilename()}",
                "uuid": module.GetUUIDString(),
                "has_symbols": module.GetNumSymbols() > 0
            })
    
    debug_info["modules"] = modules
    
    # Get current source info if stopped
    if session.process and session.process.IsValid():
        thread = session.process.GetSelectedThread()
        if thread and thread.IsValid():
            frame = thread.GetSelectedFrame()
            if frame and frame.IsValid():
                line_entry = frame.GetLineEntry()
                if line_entry.IsValid():
                    file_spec = line_entry.GetFileSpec()
                    debug_info["current_source"] = {
                        "file": f"{file_spec.GetDirectory()}/{file_spec.GetFilename()}",
                        "line": line_entry.GetLine(),
                        "column": line_entry.GetColumn()
                    }
    
    return {"debug_info": debug_info}


def main():
    """Main entry point for the MCP server"""
    import sys
    
    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()