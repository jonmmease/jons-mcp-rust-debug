#!/usr/bin/env python3
"""
FastMCP server that provides Rust debugging capabilities through gdb/lldb.

This server manages gdb/lldb subprocess debugging sessions and exposes debugger features through MCP tools.
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

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "WARNING"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("rust-debug-mcp")

# Constants for debugger prompts and patterns
GDB_PROMPT = "(gdb) "
LLDB_PROMPT = "(lldb) "
GDB_PROMPT_PATTERN = re.compile(r"\(gdb\)\s*$", re.MULTILINE)
LLDB_PROMPT_PATTERN = re.compile(r"\(lldb\)\s*$", re.MULTILINE)

# GDB patterns
GDB_BREAKPOINT_SET_PATTERN = re.compile(r"Breakpoint (\d+) at (0x[0-9a-fA-F]+): file (.+), line (\d+)")
GDB_CURRENT_LOCATION_PATTERN = re.compile(r"^(0x[0-9a-fA-F]+) in (.+) \(.*\) at (.+):(\d+)")
GDB_STACK_FRAME_PATTERN = re.compile(r"^#(\d+)\s+(0x[0-9a-fA-F]+) in (.+) \(.*\) at (.+):(\d+)")

# LLDB patterns  
LLDB_BREAKPOINT_SET_PATTERN = re.compile(r"Breakpoint (\d+): where = .+ at (.+):(\d+)")
LLDB_CURRENT_LOCATION_PATTERN = re.compile(r"frame #\d+: 0x[0-9a-fA-F]+ .+`(.+) at (.+):(\d+)")
LLDB_STACK_FRAME_PATTERN = re.compile(r"frame #(\d+): 0x[0-9a-fA-F]+ .+`(.+) at (.+):(\d+)")


class DebuggerType(Enum):
    """Type of debugger being used"""
    GDB = "gdb"
    LLDB = "lldb"
    RUST_GDB = "rust-gdb"
    RUST_LLDB = "rust-lldb"


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
    debugger: Optional[str] = None  # gdb, lldb, rust-gdb, rust-lldb, or path
    cargo_path: Optional[str] = None
    working_directory: str = "."
    environment: Dict[str, str] = field(default_factory=dict)
    cargo_args: List[str] = field(default_factory=list)
    prefer_rust_wrappers: bool = True


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
    address: Optional[str] = None


@dataclass
class StackFrame:
    """Represents a stack frame"""
    index: int
    file: str
    line: int
    function: str
    address: str
    locals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebugSession:
    """Represents a debugging session"""
    session_id: str
    process: Optional[subprocess.Popen] = None
    state: DebuggerState = DebuggerState.IDLE
    current_frame: Optional[StackFrame] = None
    breakpoints: Dict[int, Breakpoint] = field(default_factory=dict)
    target_type: str = "binary"  # "binary", "test", "example"
    target: str = ""
    args: List[str] = field(default_factory=list)
    output_queue: queue.Queue = field(default_factory=queue.Queue)
    reader_thread: Optional[threading.Thread] = None
    writer_thread: Optional[threading.Thread] = None
    command_queue: queue.Queue = field(default_factory=queue.Queue)
    last_output: str = ""
    debugger_type: DebuggerType = DebuggerType.GDB
    debugger_path: str = ""


class RustDebugClient:
    """Client for managing Rust debugger subprocess sessions"""

    def __init__(self):
        self.sessions: Dict[str, DebugSession] = {}
        self.lock = threading.Lock()
        self.session_counter = 0
        self.config = self._load_config()
        atexit.register(self._cleanup_all_sessions)

    def _load_config(self) -> Config:
        """Load configuration from rustdebugconfig.json if it exists"""
        config_path = Path("rustdebugconfig.json")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
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

    def _find_debugger(self) -> Tuple[DebuggerType, str]:
        """Find the appropriate debugger executable"""
        # Check config first
        if self.config.debugger:
            if self.config.debugger in ["gdb", "lldb", "rust-gdb", "rust-lldb"]:
                # Try to find the specified debugger
                path = shutil.which(self.config.debugger)
                if path:
                    return DebuggerType(self.config.debugger.replace("rust-", "")), path
            elif Path(self.config.debugger).exists():
                # Custom path provided
                debugger_name = Path(self.config.debugger).name
                if "lldb" in debugger_name:
                    return DebuggerType.LLDB, self.config.debugger
                else:
                    return DebuggerType.GDB, self.config.debugger

        # Platform-specific defaults
        system = platform.system()
        
        # Try rust-specific wrappers first if preferred
        if self.config.prefer_rust_wrappers:
            if system == "Darwin":  # macOS
                for debugger in ["rust-lldb", "lldb", "rust-gdb", "gdb"]:
                    path = shutil.which(debugger)
                    if path:
                        dbg_type = DebuggerType.RUST_LLDB if "rust-lldb" in debugger else (
                            DebuggerType.LLDB if "lldb" in debugger else (
                                DebuggerType.RUST_GDB if "rust-gdb" in debugger else DebuggerType.GDB
                            )
                        )
                        return dbg_type, path
            else:  # Linux/Windows
                for debugger in ["rust-gdb", "gdb", "rust-lldb", "lldb"]:
                    path = shutil.which(debugger)
                    if path:
                        dbg_type = DebuggerType.RUST_GDB if "rust-gdb" in debugger else (
                            DebuggerType.GDB if "gdb" in debugger else (
                                DebuggerType.RUST_LLDB if "rust-lldb" in debugger else DebuggerType.LLDB
                            )
                        )
                        return dbg_type, path

        raise RuntimeError(
            "No debugger found. Please install gdb or lldb, or specify debugger path in rustdebugconfig.json"
        )

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
        # Look for lines like: "Executable unittests src/main.rs (target/debug/deps/myapp-123abc)"
        binary_pattern = re.compile(r"Executable.+\((.+)\)")
        for line in result.stderr.split("\n"):
            match = binary_pattern.search(line)
            if match:
                return match.group(1)
        
        # Fallback: guess the binary location
        # Check if --release flag was used
        build_mode = "release" if cargo_flags and "--release" in cargo_flags else "debug"
        target_dir = Path(self.config.working_directory) / "target" / build_mode
        if target_type == "test":
            # Test binaries are in deps/ with hash suffix
            deps_dir = target_dir / "deps"
            if deps_dir.exists():
                # Find the most recent test binary
                test_bins = list(deps_dir.glob(f"{target}-*")) if target else list(deps_dir.glob("*"))
                # Filter out .d and .dSYM files
                test_bins = [b for b in test_bins if b.is_file() and not b.suffix and not b.name.endswith(".dSYM")]
                if test_bins:
                    # Return the most recently modified
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

    def _reader_thread(self, session: DebugSession):
        """Thread for reading output from debugger subprocess"""
        try:
            buffer = ""
            prompt = GDB_PROMPT if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB] else LLDB_PROMPT
            prompt_pattern = GDB_PROMPT_PATTERN if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB] else LLDB_PROMPT_PATTERN
            
            while session.process and session.process.poll() is None:
                # Read one character at a time to handle prompts without newlines
                char = session.process.stdout.read(1)
                if not char:
                    continue

                buffer += char

                # If we hit a newline, send the complete line
                if char == "\n":
                    session.output_queue.put(buffer)
                    session.last_output += buffer
                    buffer = ""
                # Check if we have a prompt
                elif buffer.endswith(prompt):
                    session.output_queue.put(buffer)
                    session.last_output += buffer
                    session.state = DebuggerState.PAUSED
                    buffer = ""

                # Check for state changes
                if "Program exited" in session.last_output or "Process exited" in session.last_output:
                    session.state = DebuggerState.FINISHED
                elif "Program received signal" in session.last_output:
                    session.state = DebuggerState.PAUSED

            # Send any remaining buffer content
            if buffer:
                session.output_queue.put(buffer)
                session.last_output += buffer
        except Exception as e:
            logger.error(f"Reader thread error: {e}")
            session.state = DebuggerState.ERROR

    def _writer_thread(self, session: DebugSession):
        """Thread for writing commands to debugger subprocess"""
        try:
            while session.process and session.process.poll() is None:
                try:
                    command = session.command_queue.get(timeout=0.1)
                    if command:
                        logger.debug(f"Sending command to debugger: {command}")
                        session.process.stdin.write(command + "\n")
                        session.process.stdin.flush()
                except queue.Empty:
                    continue
        except Exception as e:
            logger.error(f"Writer thread error: {e}")

    def _wait_for_prompt(self, session: DebugSession, timeout: float = 5.0) -> bool:
        """Wait for debugger prompt to appear"""
        start_time = time.time()
        accumulated_output = ""
        
        prompt_pattern = GDB_PROMPT_PATTERN if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB] else LLDB_PROMPT_PATTERN
        
        # Check if we already have a prompt
        if prompt_pattern.search(session.last_output):
            return True

        while time.time() - start_time < timeout:
            try:
                output = session.output_queue.get(timeout=0.1)
                accumulated_output += output
                session.last_output += output

                if prompt_pattern.search(accumulated_output):
                    return True
            except queue.Empty:
                continue

        return False

    def _send_command(self, session: DebugSession, command: str, wait_for_response: bool = True) -> str:
        """Send command to debugger and optionally wait for response"""
        # Clear the output buffer
        session.last_output = ""
        
        # Send command
        session.command_queue.put(command)
        
        if not wait_for_response:
            return ""
        
        # Wait for prompt
        if not self._wait_for_prompt(session):
            raise TimeoutError(f"Timeout waiting for debugger response to command: {command}")
        
        # Return the output (excluding the prompt)
        prompt = GDB_PROMPT if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB] else LLDB_PROMPT
        output = session.last_output.replace(prompt, "").strip()
        return output

    def create_session(self, target_type: str, target: str, args: List[str], 
                      cargo_flags: List[str] = None, env: Dict[str, str] = None,
                      package: Optional[str] = None) -> str:
        """Create a new debugging session"""
        with self.lock:
            self.session_counter += 1
            session_id = f"session_{self.session_counter}"
            
            # Find debugger
            debugger_type, debugger_path = self._find_debugger()
            
            # Build the target
            binary_path = self._build_target(target_type, target, cargo_flags, env, package)
            
            # Create session
            session = DebugSession(
                session_id=session_id,
                target_type=target_type,
                target=target,
                args=args,
                debugger_type=debugger_type,
                debugger_path=debugger_path
            )
            
            # Start debugger process
            cmd = [debugger_path, binary_path]
            
            logger.info(f"Starting debugger: {' '.join(cmd)}")
            
            session.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                cwd=self.config.working_directory,
                env={**os.environ, **self.config.environment}
            )
            
            # Start reader and writer threads
            session.reader_thread = threading.Thread(
                target=self._reader_thread,
                args=(session,),
                daemon=True
            )
            session.writer_thread = threading.Thread(
                target=self._writer_thread,
                args=(session,),
                daemon=True
            )
            
            session.reader_thread.start()
            session.writer_thread.start()
            
            # Wait for initial prompt
            if not self._wait_for_prompt(session):
                session.process.terminate()
                raise RuntimeError("Failed to get initial debugger prompt")
            
            # Initialize debugger settings
            if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
                # GDB settings
                self._send_command(session, "set pagination off")
                self._send_command(session, "set print pretty on")
                self._send_command(session, "set print object on")
                self._send_command(session, "set print static-members on")
                self._send_command(session, "set print vtbl on")
                self._send_command(session, "set print demangle on")
                self._send_command(session, "set demangle-style rust")
            else:
                # LLDB settings
                self._send_command(session, "settings set target.process.thread.step-avoid-regexp '^std::'")
                self._send_command(session, "settings set target.process.thread.step-in-avoid-code std")
            
            # Set breakpoint on rust_panic to catch panics
            if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
                self._send_command(session, "break rust_panic")
            else:
                self._send_command(session, "b rust_panic")
            
            # Set arguments if provided
            if args:
                args_str = " ".join(args)
                if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
                    self._send_command(session, f"set args {args_str}")
                else:
                    self._send_command(session, f"settings set target.run-args {args_str}")
            
            session.state = DebuggerState.PAUSED
            self.sessions[session_id] = session
            
            return session_id

    def _stop_session(self, session_id: str):
        """Stop a debugging session"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        try:
            if session.process and session.process.poll() is None:
                # Try graceful exit first
                try:
                    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
                        self._send_command(session, "quit", wait_for_response=False)
                    else:
                        self._send_command(session, "exit", wait_for_response=False)
                    session.process.wait(timeout=2)
                except:
                    # Force terminate if graceful exit fails
                    session.process.terminate()
                    session.process.wait(timeout=2)
        except Exception as e:
            logger.error(f"Error stopping session {session_id}: {e}")
            if session.process:
                session.process.kill()
        
        del self.sessions[session_id]


# Global debug client instance
debug_client = RustDebugClient()


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
    """Start a new Rust debugging session.
    
    Args:
        target_type: Type of target to debug - "binary", "test", or "example"
        target: Name of the specific target (optional for binary if only one exists)
        args: Command line arguments to pass to the program
        cargo_flags: Additional cargo build flags (e.g., ["--no-default-features", "--features", "test-only"])
        env: Environment variables for the build process (e.g., {"RUST_TEST_THREADS": "1"})
        package: Specific package name for workspace projects (e.g., "my-crate")
        
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
            "debugger": debug_client.sessions[session_id].debugger_type.value
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
            "target": session.target,
            "state": session.state.value,
            "debugger": session.debugger_type.value
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
        file: Source file path (optional if function specified)
        line: Line number (optional if function specified) 
        function: Function name (optional if file/line specified)
        condition: Conditional expression for breakpoint
        temporary: If true, breakpoint is removed after first hit
        
    Returns:
        Dictionary with breakpoint details
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Build breakpoint command
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        if function:
            cmd = f"{'tbreak' if temporary else 'break'} {function}"
        elif file and line:
            cmd = f"{'tbreak' if temporary else 'break'} {file}:{line}"
        else:
            return {"status": "error", "error": "Must specify either function or file:line"}
        
        if condition:
            cmd += f" if {condition}"
    else:  # LLDB
        if function:
            cmd = f"breakpoint set --name {function}"
        elif file and line:
            cmd = f"breakpoint set --file {file} --line {line}"
        else:
            return {"status": "error", "error": "Must specify either function or file:line"}
        
        if condition:
            cmd += f" --condition '{condition}'"
        if temporary:
            cmd += " --one-shot true"
    
    # Send command
    output = debug_client._send_command(session, cmd)
    
    # Parse breakpoint ID
    breakpoint_id = None
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        match = re.search(r"Breakpoint (\d+)", output)
        if match:
            breakpoint_id = int(match.group(1))
    else:  # LLDB
        match = re.search(r"Breakpoint (\d+):", output)
        if match:
            breakpoint_id = int(match.group(1))
    
    if breakpoint_id is None:
        return {"status": "error", "error": f"Failed to parse breakpoint ID from: {output}"}
    
    # Store breakpoint info
    bp = Breakpoint(
        id=breakpoint_id,
        file=file or "",
        line=line or 0,
        function=function,
        condition=condition,
        temporary=temporary,
        enabled=True
    )
    session.breakpoints[breakpoint_id] = bp
    
    return {
        "breakpoint_id": breakpoint_id,
        "location": f"{file}:{line}" if file and line else function,
        "status": "set"
    }


@mcp.tool()
async def remove_breakpoint(session_id: str, breakpoint_id: int) -> Dict[str, Any]:
    """Remove a breakpoint.
    
    Args:
        session_id: The session identifier
        breakpoint_id: The breakpoint identifier
        
    Returns:
        Dictionary with status
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    if breakpoint_id not in session.breakpoints:
        return {"status": "error", "error": "Breakpoint not found"}
    
    # Send delete command
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        cmd = f"delete {breakpoint_id}"
    else:  # LLDB
        cmd = f"breakpoint delete {breakpoint_id}"
    
    debug_client._send_command(session, cmd)
    
    # Remove from session
    del session.breakpoints[breakpoint_id]
    
    return {"status": "removed"}


@mcp.tool()
async def list_breakpoints(session_id: str) -> Dict[str, Any]:
    """List all breakpoints in the session.
    
    Args:
        session_id: The session identifier
        
    Returns:
        Dictionary with list of breakpoints
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    breakpoints = []
    for bp_id, bp in session.breakpoints.items():
        breakpoints.append({
            "id": bp_id,
            "file": bp.file,
            "line": bp.line,
            "function": bp.function,
            "condition": bp.condition,
            "temporary": bp.temporary,
            "enabled": bp.enabled,
            "hit_count": bp.hit_count
        })
    
    return {"breakpoints": breakpoints}


# MCP Tools - Execution Control

@mcp.tool()
async def run(session_id: str) -> Dict[str, Any]:
    """Start or continue program execution.
    
    Args:
        session_id: The session identifier
        
    Returns:
        Dictionary with execution status and stop reason
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    session.state = DebuggerState.RUNNING
    
    # Send run/continue command
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        cmd = "run" if "Starting program" not in session.last_output else "continue"
    else:  # LLDB
        cmd = "run" if "Process" not in session.last_output else "continue"
    
    output = debug_client._send_command(session, cmd)
    
    # Parse stop reason
    stop_reason = "unknown"
    stopped_at = ""
    
    if "Breakpoint" in output:
        stop_reason = "breakpoint"
        # Extract location
        if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
            match = GDB_CURRENT_LOCATION_PATTERN.search(output)
            if match:
                stopped_at = f"{match.group(3)}:{match.group(4)}"
        else:
            match = LLDB_CURRENT_LOCATION_PATTERN.search(output)
            if match:
                stopped_at = f"{match.group(2)}:{match.group(3)}"
    elif "Program exited" in output or "Process exited" in output:
        stop_reason = "exited"
        session.state = DebuggerState.FINISHED
    elif "received signal" in output:
        stop_reason = "signal"
        if "SIGSEGV" in output:
            stop_reason = "segfault"
        elif "SIGABRT" in output:
            stop_reason = "abort"
    
    return {
        "status": session.state.value,
        "stop_reason": stop_reason,
        "stopped_at": stopped_at,
        "output": output
    }


@mcp.tool()
async def step(session_id: str) -> Dict[str, Any]:
    """Step into the next line of code (enters functions).
    
    Args:
        session_id: The session identifier
        
    Returns:
        Dictionary with new location
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send step command
    output = debug_client._send_command(session, "step")
    
    # Parse location
    location = ""
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        match = GDB_CURRENT_LOCATION_PATTERN.search(output)
        if match:
            location = f"{match.group(3)}:{match.group(4)}"
            session.current_frame = StackFrame(
                index=0,
                file=match.group(3),
                line=int(match.group(4)),
                function=match.group(2),
                address=match.group(1)
            )
    else:
        match = LLDB_CURRENT_LOCATION_PATTERN.search(output)
        if match:
            location = f"{match.group(2)}:{match.group(3)}"
            session.current_frame = StackFrame(
                index=0,
                file=match.group(2),
                line=int(match.group(3)),
                function=match.group(1),
                address=""
            )
    
    return {
        "location": location,
        "output": output
    }


@mcp.tool()
async def next(session_id: str) -> Dict[str, Any]:
    """Step over the next line of code (doesn't enter functions).
    
    Args:
        session_id: The session identifier
        
    Returns:
        Dictionary with new location
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send next command
    output = debug_client._send_command(session, "next")
    
    # Parse location (same as step)
    location = ""
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        match = GDB_CURRENT_LOCATION_PATTERN.search(output)
        if match:
            location = f"{match.group(3)}:{match.group(4)}"
            session.current_frame = StackFrame(
                index=0,
                file=match.group(3),
                line=int(match.group(4)),
                function=match.group(2),
                address=match.group(1)
            )
    else:
        match = LLDB_CURRENT_LOCATION_PATTERN.search(output)
        if match:
            location = f"{match.group(2)}:{match.group(3)}"
            session.current_frame = StackFrame(
                index=0,
                file=match.group(2),
                line=int(match.group(3)),
                function=match.group(1),
                address=""
            )
    
    return {
        "location": location,
        "output": output
    }


@mcp.tool()
async def finish(session_id: str) -> Dict[str, Any]:
    """Continue execution until the current function returns.
    
    Args:
        session_id: The session identifier
        
    Returns:
        Dictionary with return location and value
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send finish command
    output = debug_client._send_command(session, "finish")
    
    # Parse return value if available
    return_value = ""
    if "Value returned" in output:
        # Extract return value
        match = re.search(r"Value returned is \$\d+ = (.+)", output)
        if match:
            return_value = match.group(1)
    
    return {
        "output": output,
        "return_value": return_value
    }


# MCP Tools - Stack Navigation

@mcp.tool()
async def backtrace(session_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Get the current call stack (backtrace).
    
    Args:
        session_id: The session identifier
        limit: Maximum number of frames to return
        
    Returns:
        Dictionary with stack frames
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send backtrace command
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        cmd = f"backtrace {limit}" if limit else "backtrace"
    else:  # LLDB
        cmd = f"thread backtrace -c {limit}" if limit else "thread backtrace"
    
    output = debug_client._send_command(session, cmd)
    
    # Parse stack frames
    frames = []
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        for match in GDB_STACK_FRAME_PATTERN.finditer(output):
            frames.append({
                "index": int(match.group(1)),
                "address": match.group(2),
                "function": match.group(3),
                "file": match.group(4),
                "line": int(match.group(5))
            })
    else:  # LLDB
        for match in LLDB_STACK_FRAME_PATTERN.finditer(output):
            frames.append({
                "index": int(match.group(1)),
                "function": match.group(2),
                "file": match.group(3),
                "line": int(match.group(4))
            })
    
    return {"frames": frames}


@mcp.tool()
async def up(session_id: str, count: int = 1) -> Dict[str, Any]:
    """Move up in the call stack (to caller).
    
    Args:
        session_id: The session identifier
        count: Number of frames to move up
        
    Returns:
        Dictionary with new frame info
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send up command
    cmd = f"up {count}" if count > 1 else "up"
    output = debug_client._send_command(session, cmd)
    
    # Parse new frame
    frame_info = {}
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        match = re.search(r"#(\d+)\s+(.+) at (.+):(\d+)", output)
        if match:
            frame_info = {
                "index": int(match.group(1)),
                "function": match.group(2),
                "file": match.group(3),
                "line": int(match.group(4))
            }
    else:  # LLDB
        match = re.search(r"frame #(\d+):.+`(.+) at (.+):(\d+)", output)
        if match:
            frame_info = {
                "index": int(match.group(1)),
                "function": match.group(2),
                "file": match.group(3),
                "line": int(match.group(4))
            }
    
    return {"frame": frame_info, "output": output}


@mcp.tool()
async def down(session_id: str, count: int = 1) -> Dict[str, Any]:
    """Move down in the call stack.
    
    Args:
        session_id: The session identifier
        count: Number of frames to move down
        
    Returns:
        Dictionary with new frame info
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send down command
    cmd = f"down {count}" if count > 1 else "down"
    output = debug_client._send_command(session, cmd)
    
    # Parse new frame (same as up)
    frame_info = {}
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        match = re.search(r"#(\d+)\s+(.+) at (.+):(\d+)", output)
        if match:
            frame_info = {
                "index": int(match.group(1)),
                "function": match.group(2),
                "file": match.group(3),
                "line": int(match.group(4))
            }
    else:  # LLDB
        match = re.search(r"frame #(\d+):.+`(.+) at (.+):(\d+)", output)
        if match:
            frame_info = {
                "index": int(match.group(1)),
                "function": match.group(2),
                "file": match.group(3),
                "line": int(match.group(4))
            }
    
    return {"frame": frame_info, "output": output}


# MCP Tools - Inspection

@mcp.tool()
async def list_source(session_id: str, line: Optional[int] = None, count: int = 10) -> Dict[str, Any]:
    """Show source code around current or specified line.
    
    Args:
        session_id: The session identifier
        line: Line number to center on (uses current if not specified)
        count: Number of lines to show
        
    Returns:
        Dictionary with source code
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send list command
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        if line:
            cmd = f"list {line}"
        else:
            cmd = "list"
    else:  # LLDB
        cmd = f"source list -l {line}" if line else "source list"
    
    output = debug_client._send_command(session, cmd)
    
    return {
        "source": output,
        "current_line": session.current_frame.line if session.current_frame else None
    }


@mcp.tool()
async def print_variable(session_id: str, expression: str) -> Dict[str, Any]:
    """Print the value of a variable or expression.
    
    Args:
        session_id: The session identifier
        expression: Variable name or expression to evaluate
        
    Returns:
        Dictionary with the value and type
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send print command
    output = debug_client._send_command(session, f"print {expression}")
    
    # Also get type info
    type_output = ""
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        type_output = debug_client._send_command(session, f"ptype {expression}")
    else:  # LLDB
        type_output = debug_client._send_command(session, f"expression -T -- {expression}")
    
    return {
        "value": output,
        "type": type_output,
        "expression": expression
    }


@mcp.tool()
async def list_locals(session_id: str) -> Dict[str, Any]:
    """List all local variables in the current scope.
    
    Args:
        session_id: The session identifier
        
    Returns:
        Dictionary with local variables
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send info locals command
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        output = debug_client._send_command(session, "info locals")
    else:  # LLDB
        output = debug_client._send_command(session, "frame variable")
    
    return {"locals": output}


@mcp.tool()
async def evaluate(session_id: str, expression: str) -> Dict[str, Any]:
    """Evaluate an arbitrary Rust expression in the current context.
    
    Args:
        session_id: The session identifier
        expression: Rust expression to evaluate
        
    Returns:
        Dictionary with the result
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send expression command
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        cmd = f"print {expression}"
    else:  # LLDB
        cmd = f"expression -- {expression}"
    
    output = debug_client._send_command(session, cmd)
    
    # Check for errors
    error = ""
    if "error:" in output.lower() or "no symbol" in output.lower():
        error = output
    
    return {
        "result": output if not error else "",
        "error": error,
        "expression": expression
    }


def main():
    """Main entry point for the MCP server"""
    import sys
    import asyncio
    
    # Run the MCP server
    asyncio.run(mcp.run())


if __name__ == "__main__":
    main()