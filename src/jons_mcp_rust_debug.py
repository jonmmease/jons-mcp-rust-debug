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
import select
import fcntl

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
LLDB_BREAKPOINT_SET_PATTERN = re.compile(r"^Breakpoint (\d+): (?:where = .+ at |file = '?)(.+?)(?:'?, line = |:)(\d+)", re.MULTILINE)
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
    has_started: bool = False  # Track if program has been started
    last_stop_reason: str = ""  # Track why we stopped
    current_location: Optional[str] = None  # Current file:line
    pty_master: Optional[int] = None  # PTY master file descriptor for LLDB
    created_time: float = field(default_factory=time.time)  # Track session creation time


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

        
        # Check if lldb-mi is available as fallback
        lldb_mi = shutil.which("lldb-mi")
        if lldb_mi:
            logger.info("Found lldb-mi, consider using MI interface for better reliability")
        
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
            
            # Handle PTY reading for LLDB
            if session.pty_master is not None:
                import select
                
                # During initialization, accumulate more output
                init_period = time.time() < session.created_time + 15.0 if hasattr(session, 'created_time') else False
                
                while session.process and session.process.poll() is None:
                    # Check if data available
                    ready, _, _ = select.select([session.pty_master], [], [], 0.1)
                    if ready:
                        try:
                            # Read available data
                            chunk = os.read(session.pty_master, 4096)
                            if chunk:
                                # Decode and process
                                text = chunk.decode('utf-8', errors='replace')
                                buffer += text
                                
                                # During initialization, wait for more complete output
                                if init_period and prompt in buffer:
                                    # Wait a bit for more output after prompt
                                    time.sleep(0.05)
                                    try:
                                        extra = os.read(session.pty_master, 1024)
                                        if extra:
                                            buffer += extra.decode('utf-8', errors='replace')
                                    except:
                                        pass
                                
                                # Check for prompts
                                while prompt in buffer:
                                    # Find the last complete prompt
                                    idx = buffer.rfind(prompt)
                                    if idx >= 0:
                                        # Output everything before the last prompt
                                        if idx > 0:
                                            output = buffer[:idx]
                                            session.output_queue.put(output)
                                            session.last_output += output
                                        session.state = DebuggerState.PAUSED
                                        # Signal prompt found
                                        session.output_queue.put("")
                                        # Keep rest after prompt for next iteration
                                        buffer = buffer[idx + len(prompt):]
                                        break
                                        
                        except (OSError, IOError) as e:
                            if e.errno == 5:  # I/O error - PTY closed
                                break
                            logger.debug(f"PTY read error: {e}")
                            
            else:
                # Regular pipe reading for GDB
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
                        
                        if session.pty_master is not None:
                            # Write to PTY
                            os.write(session.pty_master, (command + "\n").encode('utf-8'))
                        else:
                            # Write to regular pipe
                            session.process.stdin.write(command + "\n")
                            session.process.stdin.flush()
                except queue.Empty:
                    continue
        except Exception as e:
            logger.error(f"Writer thread error: {e}")

    def _stderr_reader(self, session: DebugSession):
        """Thread for reading stderr output from debugger subprocess"""
        try:
            while session.process and session.process.stderr and session.process.poll() is None:
                line = session.process.stderr.readline()
                if line:
                    # Log stderr output for debugging but don't mix with stdout
                    logger.debug(f"Debugger stderr: {line.strip()}")
                else:
                    break
        except Exception as e:
            logger.error(f"Stderr reader thread error: {e}")

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
                    # Update stop information if we stopped
                    self._update_stop_info(session, accumulated_output)
                    return True
            except queue.Empty:
                continue

        return False
    
    def _update_stop_info(self, session: DebugSession, output: str):
        """Update session stop information from debugger output"""
        # For LLDB, check for stop reason in output
        if session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB]:
            # First check if process stopped at all
            process_stopped = re.search(r"Process \d+ stopped", output)
            if process_stopped:
                session.state = DebuggerState.PAUSED
                
                # Now look for explicit stop reason in multiple patterns
                # Pattern 1: Direct stop reason
                reason_match = re.search(r"stop reason = (.+?)(?:\r?\n|$)", output)
                if not reason_match:
                    # Pattern 2: In thread info format
                    reason_match = re.search(r", stop reason = (.+?)(?:\r?\n|$)", output)
                
                if reason_match:
                    reason = reason_match.group(1).strip()
                    if "breakpoint" in reason:
                        # Extract breakpoint number
                        bp_match = re.search(r"breakpoint (\d+(?:\.\d+)?)", reason)
                        if bp_match:
                            session.last_stop_reason = f"breakpoint_{bp_match.group(1).replace('.', '_')}"
                        else:
                            session.last_stop_reason = "breakpoint"
                    else:
                        session.last_stop_reason = reason
                else:
                    # Process stopped but no explicit reason - check context
                    if "thread #" in output and "stop reason" not in output:
                        # Stopped but no reason given - likely at entry or after step
                        if not session.has_started:
                            session.last_stop_reason = "entry"
                        else:
                            session.last_stop_reason = "step"
                    else:
                        session.last_stop_reason = "stopped"
            
            # Update location - try multiple patterns
            # Pattern 1: at file:line
            loc_match = re.search(r" at ([^:]+):(\d+)", output)
            if loc_match:
                session.current_location = f"{loc_match.group(1)}:{loc_match.group(2)}"
            else:
                # Pattern 2: frame info format
                frame_match = re.search(r"frame #\d+:.*?`.*? at ([^:]+):(\d+)", output)
                if frame_match:
                    session.current_location = f"{frame_match.group(1)}:{frame_match.group(2)}"
        
        # Also check for GDB stop reasons
        elif session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
            if "Breakpoint" in output:
                bp_match = re.search(r"Breakpoint (\d+)", output)
                if bp_match:
                    session.last_stop_reason = f"breakpoint_{bp_match.group(1)}"
                else:
                    session.last_stop_reason = "breakpoint"
                session.state = DebuggerState.PAUSED
            
            # Check for location in GDB format
            loc_match = re.search(r" at ([^:]+):(\d+)", output)
            if loc_match:
                session.current_location = f"{loc_match.group(1)}:{loc_match.group(2)}"

    def _clear_output_buffer(self, session: DebugSession):
        """Clear any pending output from the debugger"""
        try:
            while True:
                session.output_queue.get_nowait()
        except queue.Empty:
            pass
        session.last_output = ""

    def _send_command(self, session: DebugSession, command: str, wait_for_response: bool = True, timeout: float = 5.0) -> str:
        """Send command to debugger and optionally wait for response"""
        # Clear any pending output first
        self._clear_output_buffer(session)
        
        # Send command with proper handling for LLDB
        # For LLDB, we need to ensure clean command transmission
        if session.pty_master is not None and session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB]:
            # Send directly to PTY with proper line ending
            command_bytes = (command + '\r\n').encode('utf-8')
            os.write(session.pty_master, command_bytes)
            os.fsync(session.pty_master)  # Force flush
            
            # Give LLDB time to process the command
            time.sleep(0.05)
        else:
            # Regular queue-based sending
            session.command_queue.put(command)
        
        # Wait for prompt with custom timeout
        if not self._wait_for_prompt(session, timeout=timeout):
            # For LLDB, sometimes we get output without a prompt
            if session.last_output and session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB]:
                # Return what we have
                return session.last_output.strip()
            raise TimeoutError(f"Timeout waiting for debugger response to command: {command}")
        
        # Return the output (excluding the prompt and command echo)
        prompt = GDB_PROMPT if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB] else LLDB_PROMPT
        output = session.last_output.replace(prompt, "").strip()
        
        # Remove command echo if present
        lines = output.split('\n')
        filtered_lines = []
        skip_echo = True
        
        for i, line in enumerate(lines):
            # Skip the command echo (first occurrence of the command)
            if skip_echo and line.strip() == command:
                skip_echo = False
                continue
            
            # Skip empty lines immediately after echo
            if not skip_echo and line.strip():
                filtered_lines.append(line)
        
        output = '\n'.join(filtered_lines).strip()
        
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
            
            # Use PTY for LLDB on macOS/Unix systems
            if session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB] and hasattr(os, 'openpty'):
                import pty
                
                # Create pseudo-terminal with proper settings for LLDB
                import pty
                import termios
                import tty
                
                master_fd, slave_fd = pty.openpty()
                
                # Configure the PTY for LLDB - disable echo and set proper modes
                # This prevents command echoing issues
                attrs = termios.tcgetattr(slave_fd)
                # Disable echo (ECHO), canonical mode (ICANON), and signals (ISIG)
                attrs[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG)
                # Set VMIN and VTIME for non-blocking reads
                attrs[6][termios.VMIN] = 0
                attrs[6][termios.VTIME] = 0
                termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
                
                # Store master fd for reading/writing
                session.pty_master = master_fd
                
                # Make non-blocking
                import fcntl
                flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
                fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                
                logger.info("Using PTY for LLDB communication")
            else:
                # Regular pipes for GDB
                session.process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,  # Separate stderr to avoid mixing with stdout
                    text=True,
                    bufsize=0,
                    cwd=self.config.working_directory,
                    env={**os.environ, **self.config.environment}
                )
                session.pty_master = None
            
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
            # Add stderr reader to consume stderr output
            stderr_thread = threading.Thread(
                target=self._stderr_reader,
                args=(session,),
                daemon=True
            )
            
            session.reader_thread.start()
            session.writer_thread.start()
            stderr_thread.start()
            
            # Wait for initial prompt
            if not self._wait_for_prompt(session, timeout=10.0):
                session.process.terminate()
                raise RuntimeError("Failed to get initial debugger prompt")
            
            # For rust-lldb, handle initialization differently with PTY
            if session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB]:
                if session.pty_master is not None:
                    # With PTY, we need to wait for initialization to complete
                    # rust-lldb runs many initialization commands
                    logger.info("Waiting for LLDB initialization...")
                    
                    # Collect all initialization output
                    init_output = ""
                    init_start_time = time.time()
                    prompt_count = 0
                    last_prompt_time = 0
                    
                    # Wait for initialization prompts
                    # LLDB runs many commands during init, we need to wait for them all
                    while time.time() - init_start_time < 10.0:
                        try:
                            output = session.output_queue.get(timeout=0.5)
                            init_output += output
                            
                            # Count prompts
                            new_prompts = output.count(LLDB_PROMPT)
                            if new_prompts > 0:
                                prompt_count += new_prompts
                                last_prompt_time = time.time()
                                
                            # Check if initialization is complete by looking for specific patterns
                            if "Executing commands in" in output and ".lldbinit" in output:
                                # Still initializing
                                continue
                            
                            # If we see the type formatters being added, we're still initializing
                            if "type summary add" in output or "type synthetic add" in output:
                                continue
                                
                        except queue.Empty:
                            # Check if we've seen enough prompts and they've stopped coming
                            if prompt_count >= 5 and time.time() - last_prompt_time > 1.0:
                                # Additional check: send a test command to verify readiness
                                test_cmd = "script print('ready')"
                                os.write(session.pty_master, (test_cmd + '\r\n').encode())
                                time.sleep(0.2)
                                
                                # Try to read the response
                                try:
                                    test_chunk = os.read(session.pty_master, 1024)
                                    if b'ready' in test_chunk:
                                        break
                                except:
                                    pass
                    
                    logger.info(f"LLDB initialization complete after {prompt_count} prompts")
                    
                    # Clear the output buffer after initialization
                    self._clear_output_buffer(session)
                    
                    # Now send a test command to ensure we're ready
                    try:
                        test_output = self._send_command(session, "version", timeout=2.0)
                        if "lldb" in test_output.lower():
                            logger.debug("LLDB ready - version command successful")
                    except:
                        logger.warning("LLDB may not be fully initialized")
                else:
                    # Regular pipe behavior
                    time.sleep(0.5)
                    self._clear_output_buffer(session)
            
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
                # CRITICAL: Prevent LLDB from sharing terminal with inferior
                self._send_command(session, "settings set target.input-path /dev/null")
                self._send_command(session, "settings set target.output-path /dev/null")
                self._send_command(session, "settings set target.error-path /dev/null")
                
                # Disable syntax highlighting and other terminal features
                self._send_command(session, "settings set use-color false")
                self._send_command(session, "settings set prompt '(lldb) '")
                self._send_command(session, "settings set stop-line-count-before 0")
                self._send_command(session, "settings set stop-line-count-after 0")
                
                # Set async mode for better command handling
                self._send_command(session, "settings set target.async true")
                
                # Original settings
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
            
            # For LLDB, set the working directory to help with source resolution
            if session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB]:
                self._send_command(session, f"settings set target.run-args {' '.join(args) if args else ''}")
                self._send_command(session, f"platform settings -w {self.config.working_directory}")
            
            # Test-specific setup
            if target_type == "test":
                # Set test-specific breakpoints
                if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
                    # Break on test assertions
                    self._send_command(session, "break rust_begin_unwind")
                    self._send_command(session, "break core::panicking::panic")
                else:
                    # LLDB equivalents
                    self._send_command(session, "b rust_begin_unwind")
                    self._send_command(session, "b core::panicking::panic")
                
                # Set environment for better test output
                if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
                    self._send_command(session, "set environment RUST_BACKTRACE=1")
                else:
                    self._send_command(session, "settings set target.env-vars RUST_BACKTRACE=1")
            
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
        
        # Close PTY if used
        if session.pty_master is not None:
            try:
                os.close(session.pty_master)
            except:
                pass
        
        del self.sessions[session_id]


# Global debug client instance
debug_client = RustDebugClient()


# Helper function for pretty-printing Rust types
def pretty_print_rust_value(raw_output: str, enum_hints: Optional[Dict[int, str]] = None) -> str:
    """Attempt to pretty-print Rust debugger output.
    
    Args:
        raw_output: Raw debugger output
        enum_hints: Optional dict mapping discriminant values to variant names
    """
    try:
        # First, remove duplicate lines (common LLDB issue)
        lines = raw_output.split('\n')
        seen_lines = set()
        unique_lines = []
        
        for line in lines:
            # Skip lines that are debugger commands
            if 'expression -T --' in line or 'print' in line and line.strip().startswith('print'):
                continue
            
            stripped = line.strip()
            if stripped and stripped not in seen_lines:
                unique_lines.append(line)
                seen_lines.add(stripped)
        
        # Now format with proper indentation
        result = []
        indent = 0
        
        for line in unique_lines:
            line = line.strip()
            if not line:
                continue
            
            # Decrease indent for closing braces
            if line.startswith('}'):
                indent = max(0, indent - 2)
            
            # Add indented line
            result.append(' ' * indent + line)
            
            # Increase indent after opening braces
            if line.endswith('{'):
                indent += 2
        
        output = '\n'.join(result)
        
        # Try to resolve enum discriminants
        if '$discr$' in output:
            # Extract discriminant value
            discr_match = re.search(r'\$discr\$ = (\d+)', output)
            if discr_match:
                discr_value = int(discr_match.group(1))
                
                # Map common Rust enum discriminants
                variant_names = {
                    # Common Option/Result variants
                    0: "None/Ok",
                    1: "Some/Err",
                    # For other enums, we'd need type info
                }
                
                # Use provided hints if available
                if enum_hints and discr_value in enum_hints:
                    variant_name = enum_hints[discr_value]
                elif discr_value in variant_names:
                    variant_name = variant_names[discr_value]
                else:
                    variant_name = f"variant_{discr_value}"
                
                output = re.sub(r'\$discr\$ = \d+', f'variant: {variant_name}', output)
        
        return output
    except Exception as e:
        logger.debug(f"Pretty-printing failed: {e}")
        # If pretty-printing fails, return original
        return raw_output


# Helper function for pagination
def paginate_text(text: str, limit: Optional[int] = None, offset: Optional[int] = None) -> Dict[str, Any]:
    """Paginate text output by character count.
    
    Args:
        text: The full text to paginate
        limit: Maximum number of characters to return (None for all)
        offset: Starting character position (None or 0 for beginning)
        
    Returns:
        Dictionary with:
            - content: The requested text segment
            - total_chars: Total number of characters
            - offset: The offset used
            - limit: The limit used
            - has_more: Whether there's more content after this page
    """
    total_chars = len(text)
    offset = offset or 0
    
    # If no limit, return everything from offset
    if limit is None:
        content = text[offset:]
        has_more = False
    else:
        # Return the requested segment
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
    
    # Try to normalize the file path for better compatibility
    if file and not function:
        # If file doesn't start with /, try to make it relative to working directory
        if not file.startswith('/'):
            # Try different path formats
            possible_paths = [
                file,  # As provided
                f"./{file}",  # Relative with ./
                str(Path(debug_client.config.working_directory) / file),  # Full path
            ]
            
            # For workspace projects, also try without package prefix
            if '/' in file and file.count('/') > 1:
                # e.g., "vegafusion-runtime/src/..." -> "src/..."
                parts = file.split('/', 1)
                if len(parts) == 2:
                    possible_paths.append(parts[1])
        else:
            possible_paths = [file]
    else:
        possible_paths = None
    
    # Build breakpoint command
    commands_to_try = []
    
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        if function:
            base_cmd = f"{'tbreak' if temporary else 'break'} {function}"
            if condition:
                base_cmd += f" if {condition}"
            commands_to_try.append(base_cmd)
        elif file and line:
            for path in possible_paths:
                base_cmd = f"{'tbreak' if temporary else 'break'} {path}:{line}"
                if condition:
                    base_cmd += f" if {condition}"
                commands_to_try.append(base_cmd)
    else:  # LLDB
        if function:
            base_cmd = f"breakpoint set --name {function}"
            if condition:
                base_cmd += f" --condition '{condition}'"
            if temporary:
                base_cmd += " --one-shot true"
            commands_to_try.append(base_cmd)
            
            # Also try with regex for LLDB
            regex_cmd = f"breakpoint set --func-regex {function}"
            if condition:
                regex_cmd += f" --condition '{condition}'"
            if temporary:
                regex_cmd += " --one-shot true"
            commands_to_try.append(regex_cmd)
        elif file and line:
            for path in possible_paths:
                base_cmd = f"breakpoint set --file {path} --line {line}"
                if condition:
                    base_cmd += f" --condition '{condition}'"
                if temporary:
                    base_cmd += " --one-shot true"
                commands_to_try.append(base_cmd)
                
                # Also try the simpler format
                simple_cmd = f"b {path}:{line}"
                commands_to_try.append(simple_cmd)
    
    # Try each command until one succeeds
    output = ""
    successful_cmd = None
    all_outputs = []
    
    for cmd in commands_to_try:
        logger.info(f"Trying breakpoint command: {cmd}")
        try:
            output = debug_client._send_command(session, cmd)
            logger.debug(f"Breakpoint command output: {output}")
            all_outputs.append((cmd, output))
            
            # Check if this command succeeded by looking for breakpoint ID
            if "Breakpoint" in output and (":" in output or "at" in output):
                successful_cmd = cmd
                break
        except Exception as e:
            logger.error(f"Command failed: {cmd}, error: {e}")
            all_outputs.append((cmd, str(e)))
    
    # Parse breakpoint ID
    breakpoint_id = None
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        match = re.search(r"Breakpoint (\d+)", output)
        if match:
            breakpoint_id = int(match.group(1))
    else:  # LLDB
        # LLDB output format: "Breakpoint 1: where = ..." or "Breakpoint 1: file = '...', line = ..."
        # Make sure we're not matching type formatter output
        lines = output.strip().split('\n')
        for line in lines:
            # Only look for lines that start with "Breakpoint X:"
            if line.strip().startswith("Breakpoint ") and ":" in line:
                match = re.match(r"^\s*Breakpoint (\d+):", line)
                if match:
                    breakpoint_id = int(match.group(1))
                    logger.debug(f"Found breakpoint ID {breakpoint_id} in line: {line}")
                    break
    
    if breakpoint_id is None:
        # Log the full output for debugging
        logger.error(f"Failed to parse breakpoint ID from any command")
        logger.error(f"All attempts:")
        for cmd, out in all_outputs:
            logger.error(f"  Command: {cmd}")
            logger.error(f"  Output: {repr(out)}")
        
        # Build detailed error message
        error_details = {
            "error": "Failed to set breakpoint",
            "attempted_commands": [cmd for cmd, _ in all_outputs],
            "responses": [out for _, out in all_outputs],
            "suggestions": [
                "Ensure the file path is correct relative to the project root",
                "Check that debug symbols are included (use 'cargo build' not 'cargo build --release' unless you added debug symbols)",
                "For workspace projects, try using just 'src/...' without the package name",
                "Verify the line number contains executable code (not comments or blank lines)"
            ]
        }
        
        # Try to provide more specific guidance based on the output
        if any("No source file named" in out for _, out in all_outputs):
            error_details["likely_cause"] = "Source file not found. Check the path format."
        elif any("No locations found" in out for _, out in all_outputs):
            error_details["likely_cause"] = "No executable code at specified line or function not found."
        elif any("pending" in out.lower() for _, out in all_outputs):
            error_details["likely_cause"] = "Breakpoint is pending. The binary might not have debug symbols."
        
        return {"status": "error", **error_details}
    
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
    
    # Parse the actual location from the output to get resolved path
    resolved_location = ""
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        # GDB: "Breakpoint 1 at 0x1234: file src/main.rs, line 42."
        loc_match = re.search(r"file (.+), line (\d+)", output)
        if loc_match:
            resolved_location = f"{loc_match.group(1)}:{loc_match.group(2)}"
    else:  # LLDB
        # LLDB: "Breakpoint 1: where = ... at src/main.rs:42"
        loc_match = re.search(r" at ([^:]+):(\d+)", output)
        if loc_match:
            resolved_location = f"{loc_match.group(1)}:{loc_match.group(2)}"
    
    return {
        "breakpoint_id": breakpoint_id,
        "location": resolved_location or (f"{file}:{line}" if file and line else function),
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
        Dictionary with execution status, stop reason, and pagination info
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Determine which command to use and track action
    action_taken = ""
    if not session.has_started:
        # First time running
        cmd = "run"
        action_taken = "Started new execution"
        session.has_started = True
    else:
        # Program has been started before
        if session.state == DebuggerState.PAUSED:
            cmd = "continue"
            action_taken = f"Continued from {session.current_location or 'breakpoint'}"
        elif session.state == DebuggerState.FINISHED:
            cmd = "run"
            action_taken = "Restarted execution"
        elif session.state == DebuggerState.RUNNING:
            # Already running
            return {
                "status": "running",
                "action": "Already running",
                "message": "Program is already executing",
                "pagination": {
                    "total_chars": 0,
                    "offset": 0,
                    "limit": limit,
                    "has_more": False
                }
            }
        else:
            cmd = "continue"
            action_taken = "Continued execution"
    
    # Now set state to running
    session.state = DebuggerState.RUNNING
    
    # Use longer timeout for run commands as they might take time
    output = debug_client._send_command(session, cmd, timeout=30.0)
    
    # Update stop info first - this will parse stop reasons and locations
    debug_client._update_stop_info(session, output)
    
    # For LLDB, if we still don't have a stop reason, try getting it from bt
    if (session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB] and 
        session.state == DebuggerState.PAUSED and
        (not session.last_stop_reason or session.last_stop_reason == "unknown")):
        
        # Get backtrace which often contains stop reason
        try:
            bt_output = debug_client._send_command(session, "bt", timeout=2.0)
            debug_client._update_stop_info(session, bt_output)
        except:
            pass
    
    # Special handling for LLDB: if we stopped at assembly main, step into Rust main
    if (session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB] and 
        session.state == DebuggerState.PAUSED and
        "frame #0:" in output and "sample_program`main:" in output and
        not " at " in output):  # No source location means we're at assembly
        
        logger.info("Detected stop at assembly entry point, stepping into Rust main")
        
        # Step a few times to get into actual Rust code
        for _ in range(3):
            step_output = debug_client._send_command(session, "step", timeout=2.0)
            if " at " in step_output and ".rs:" in step_output:
                # We've reached Rust source code
                output = step_output
                debug_client._update_stop_info(session, step_output)
                break
    
    # Use the parsed values as defaults
    stop_reason = session.last_stop_reason if session.last_stop_reason else "unknown"
    stopped_at = session.current_location if session.current_location else ""
    
    # Log output for debugging
    logger.debug(f"Run/continue output: {output}")
    
    # Check various stop conditions
    if "Breakpoint" in output or "breakpoint" in output:
        stop_reason = "breakpoint"
        session.state = DebuggerState.PAUSED
        
        # Extract breakpoint number
        if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
            bp_match = re.search(r"Breakpoint (\d+)", output)
            if bp_match:
                stop_reason = f"breakpoint_{bp_match.group(1)}"
        else:  # LLDB
            # LLDB format: "stop reason = breakpoint 1.1"
            reason_match = re.search(r"stop reason = breakpoint (\d+)", output)
            if reason_match:
                stop_reason = f"breakpoint_{reason_match.group(1)}"
            else:
                # Alternative: "Breakpoint N: ..."
                bp_match = re.search(r"Breakpoint (\d+):", output)
                if bp_match:
                    stop_reason = f"breakpoint_{bp_match.group(1)}"
        
        # Extract location
        if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
            match = GDB_CURRENT_LOCATION_PATTERN.search(output)
            if match:
                stopped_at = f"{match.group(3)}:{match.group(4)}"
        else:  # LLDB
            # Try multiple patterns for LLDB
            # Pattern 1: frame #0: ... at file:line
            match = re.search(r"frame #0:.*? at ([^:]+):(\d+)", output)
            if match:
                stopped_at = f"{match.group(1)}:{match.group(2)}"
            else:
                # Pattern 2: Standard location pattern
                match = LLDB_CURRENT_LOCATION_PATTERN.search(output)
                if match:
                    stopped_at = f"{match.group(2)}:{match.group(3)}"
    
    elif "Program exited" in output or "Process exited" in output or "exited with" in output:
        stop_reason = "exited"
        session.state = DebuggerState.FINISHED
        # Try to extract exit code
        exit_match = re.search(r"exited (?:with code|normally) \[(\d+)\]", output)
        if not exit_match:
            # LLDB format: "Process N exited with status = N"
            exit_match = re.search(r"exited with status = (\d+)", output)
        if exit_match:
            stop_reason = f"exited_code_{exit_match.group(1)}"
    
    elif "received signal" in output or "signal" in output.lower():
        stop_reason = "signal"
        session.state = DebuggerState.PAUSED
        if "SIGSEGV" in output:
            stop_reason = "segfault"
        elif "SIGABRT" in output:
            stop_reason = "abort"
        elif "SIGINT" in output:
            stop_reason = "interrupted"
    
    elif "panic" in output.lower() or "rust_panic" in output:
        stop_reason = "panic"
        session.state = DebuggerState.PAUSED
    
    # If we're still at unknown but paused, check for LLDB stop reason
    if stop_reason == "unknown" and session.state == DebuggerState.PAUSED:
        reason_match = re.search(r"stop reason = (.+)", output)
        if reason_match:
            reason_text = reason_match.group(1).strip()
            if "step" in reason_text:
                stop_reason = "step"
            elif "breakpoint" in reason_text:
                stop_reason = "breakpoint"
            else:
                stop_reason = reason_text
    
    # Update session tracking
    session.last_stop_reason = stop_reason
    session.current_location = stopped_at if stopped_at else session.current_location
    
    # If we found a location but not through normal means, update it
    if not stopped_at and session.state == DebuggerState.PAUSED:
        # Try to get current location with a separate command
        try:
            if session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB]:
                frame_output = debug_client._send_command(session, "frame info")
                frame_match = re.search(r" at ([^:]+):(\d+)", frame_output)
                if frame_match:
                    stopped_at = f"{frame_match.group(1)}:{frame_match.group(2)}"
                    session.current_location = stopped_at
        except:
            pass
    
    # Handle pagination
    pagination = paginate_text(output, limit, offset)
    
    return {
        "status": session.state.value,
        "action": action_taken,
        "stop_reason": stop_reason,
        "stopped_at": stopped_at,
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
    function = ""
    file = ""
    line = 0
    
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        match = GDB_CURRENT_LOCATION_PATTERN.search(output)
        if match:
            file = match.group(3)
            line = int(match.group(4))
            function = match.group(2)
            location = f"{file}:{line}"
            session.current_frame = StackFrame(
                index=0,
                file=file,
                line=line,
                function=function,
                address=match.group(1)
            )
    else:
        match = LLDB_CURRENT_LOCATION_PATTERN.search(output)
        if match:
            file = match.group(2)
            line = int(match.group(3))
            function = match.group(1)
            location = f"{file}:{line}"
            session.current_frame = StackFrame(
                index=0,
                file=file,
                line=line,
                function=function,
                address=""
            )
    
    # Update session location
    session.current_location = location if location else session.current_location
    session.state = DebuggerState.PAUSED
    
    # If we couldn't parse location, try to get it explicitly
    if not location:
        try:
            if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
                where_output = debug_client._send_command(session, "where 1")
            else:
                where_output = debug_client._send_command(session, "thread backtrace -c 1")
            
            # Try to parse from where output
            logger.debug(f"Where output for location: {where_output}")
        except:
            pass
    
    return {
        "status": session.state.value,
        "location": location,
        "file": file,
        "line": line,
        "function": function,
        "output": output,
        "message": "Stepped into next line" if location else "Step completed but location unknown"
    }


@mcp.tool()
async def next(session_id: str) -> Dict[str, Any]:
    """Step over the next line of code (doesn't enter functions).
    
    Args:
        session_id: The session identifier
        
    Returns:
        Dictionary with new location and state
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send next command
    output = debug_client._send_command(session, "next")
    
    # Parse location (same as step)
    location = ""
    function = ""
    file = ""
    line = 0
    
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        match = GDB_CURRENT_LOCATION_PATTERN.search(output)
        if match:
            file = match.group(3)
            line = int(match.group(4))
            function = match.group(2)
            location = f"{file}:{line}"
            session.current_frame = StackFrame(
                index=0,
                file=file,
                line=line,
                function=function,
                address=match.group(1)
            )
    else:
        match = LLDB_CURRENT_LOCATION_PATTERN.search(output)
        if match:
            file = match.group(2)
            line = int(match.group(3))
            function = match.group(1)
            location = f"{file}:{line}"
            session.current_frame = StackFrame(
                index=0,
                file=file,
                line=line,
                function=function,
                address=""
            )
    
    # Update session location
    session.current_location = location if location else session.current_location
    session.state = DebuggerState.PAUSED
    
    # If we couldn't parse location, try to get it explicitly
    if not location:
        try:
            if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
                where_output = debug_client._send_command(session, "where 1")
            else:
                where_output = debug_client._send_command(session, "thread backtrace -c 1")
            
            # Try to parse from where output
            logger.debug(f"Where output for location: {where_output}")
        except:
            pass
    
    return {
        "status": session.state.value,
        "location": location,
        "file": file,
        "line": line,
        "function": function,
        "output": output,
        "message": "Stepped over to next line" if location else "Step completed but location unknown"
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
async def backtrace(
    session_id: str, 
    frame_limit: Optional[int] = None,
    char_limit: Optional[int] = None,
    char_offset: Optional[int] = None
) -> Dict[str, Any]:
    """Get the current call stack (backtrace).
    
    Args:
        session_id: The session identifier
        frame_limit: Maximum number of frames to retrieve from debugger
        char_limit: Maximum number of characters to return (for pagination)
        char_offset: Starting character position (for pagination)
        
    Returns:
        Dictionary with stack frames and pagination info
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Send backtrace command
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        cmd = f"backtrace {frame_limit}" if frame_limit else "backtrace"
    else:  # LLDB
        cmd = f"thread backtrace -c {frame_limit}" if frame_limit else "thread backtrace"
    
    output = debug_client._send_command(session, cmd)
    
    # Log raw output for debugging
    logger.info(f"Backtrace raw output: {repr(output[:500])}")
    
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
        # Try multiple patterns for LLDB output
        # Pattern 1: Standard format
        for match in LLDB_STACK_FRAME_PATTERN.finditer(output):
            frames.append({
                "index": int(match.group(1)),
                "function": match.group(2),
                "file": match.group(3),
                "line": int(match.group(4))
            })
        
        # If no frames found, try alternative patterns
        if not frames:
            # Pattern 2: Without file info
            alt_pattern = re.compile(r"frame #(\d+):\s*0x[0-9a-fA-F]+\s+(.+?)(?:\s+\+|$)", re.MULTILINE)
            for match in alt_pattern.finditer(output):
                frames.append({
                    "index": int(match.group(1)),
                    "function": match.group(2).strip(),
                    "file": "",
                    "line": 0
                })
    
    # If still no frames and we have output, try more flexible parsing
    if not frames and output.strip():
        # Split by lines and look for frame-like patterns
        lines = output.strip().split('\n')
        for line in lines:
            # Skip empty lines and prompts
            if not line.strip() or 'gdb)' in line or 'lldb)' in line:
                continue
            
            # Look for any line with frame number
            frame_match = re.search(r'#(\d+)', line)
            if frame_match:
                frame_num = int(frame_match.group(1))
                
                # Try to extract function name
                func_match = re.search(r'in\s+(\S+)', line) or re.search(r'`(\S+)', line)
                function = func_match.group(1) if func_match else "unknown"
                
                # Try to extract file:line
                file_match = re.search(r'(\S+\.rs):(\d+)', line)
                if file_match:
                    file = file_match.group(1)
                    line_num = int(file_match.group(2))
                else:
                    file = ""
                    line_num = 0
                
                frames.append({
                    "index": frame_num,
                    "function": function,
                    "file": file,
                    "line": line_num
                })
    
    # Log parsing results
    logger.info(f"Parsed {len(frames)} frames from backtrace")
    if not frames and output.strip():
        logger.warning(f"No frames parsed from non-empty backtrace output. First 200 chars: {repr(output[:200])}")
    
    # Handle pagination of raw output
    pagination = paginate_text(output, char_limit, char_offset)
    
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
async def list_source(
    session_id: str, 
    line: Optional[int] = None, 
    count: int = 10,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> Dict[str, Any]:
    """Show source code around current or specified line.
    
    Args:
        session_id: The session identifier
        line: Line number to center on (uses current if not specified)
        count: Number of lines to show
        limit: Maximum number of characters to return (for pagination)
        offset: Starting character position (for pagination)
        
    Returns:
        Dictionary with source code and pagination info
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # For LLDB, ensure we're in the right context
    if session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB]:
        # First check our current location
        frame_info = debug_client._send_command(session, "frame info", timeout=1.0)
        
        # If we're at assembly level, try to get source differently
        if " at " not in frame_info:
            # Try stepping if we haven't yet
            if "sample_program`main:" in frame_info:
                logger.info("At assembly level in list_source, attempting to step")
                for _ in range(3):
                    step_output = debug_client._send_command(session, "step", timeout=2.0)
                    if " at " in step_output and ".rs:" in step_output:
                        frame_info = debug_client._send_command(session, "frame info", timeout=1.0)
                        break
    
    # Send list command
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        if line:
            cmd = f"list {line}"
        else:
            cmd = "list"
    else:  # LLDB
        # For LLDB, first ensure we have proper context
        if session.state != DebuggerState.PAUSED:
            return {
                "source": "",
                "current_line": None,
                "pagination": {
                    "total_chars": 0,
                    "offset": 0,
                    "limit": limit,
                    "has_more": False
                },
                "error": "Debugger must be paused to list source"
            }
        
        # Try different approaches for LLDB
        if line:
            cmd = f"source list -l {line} -c {count}"
        else:
            cmd = f"source list -c {count}"
    
    output = debug_client._send_command(session, cmd)
    
    # Log the output for debugging
    logger.info(f"Source list command '{cmd}' output: {repr(output[:200])}")
    
    # Check if output is empty, just the command, or an error
    if not output or output.strip() == cmd or output.strip() == "source list" or len(output.strip()) < 10:
        logger.warning(f"Source list returned minimal output: {repr(output)}")
        
        # For LLDB, try alternative approaches
        if session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB]:
            # First, ensure we have correct file and line info
            frame_info = debug_client._send_command(session, "frame info")
            logger.info(f"Frame info for source context: {repr(frame_info[:200])}")
            
            # Try to extract file and line from frame info
            file_match = re.search(r' at ([^:]+):(\d+)', frame_info)
            if file_match:
                file_path = file_match.group(1)
                target_line = int(file_match.group(2)) if not line else line
                
                # Make sure we have an absolute path
                if not file_path.startswith('/'):
                    # Try to resolve relative path
                    abs_path = os.path.abspath(os.path.join(debug_client.config.working_directory, file_path))
                    if os.path.exists(abs_path):
                        file_path = abs_path
                
                # Try multiple source list variants
                attempts = [
                    f"source list -f {file_path} -l {target_line} -c {count}",
                    f"source list -f \"{file_path}\" -l {target_line} -c {count}",
                    f"source list -n main -c {count}",  # Try by function name as fallback
                    f"source info"  # Get source info as diagnostic
                ]
                
                for attempt_cmd in attempts:
                    logger.info(f"Trying alternative command: {attempt_cmd}")
                    alt_output = debug_client._send_command(session, attempt_cmd)
                    if alt_output and len(alt_output.strip()) > len(output.strip()) and "error" not in alt_output.lower():
                        output = alt_output
                        logger.info(f"Alternative command succeeded with output length: {len(alt_output)}")
                        break
                    else:
                        logger.info(f"Alternative command failed or returned minimal output: {repr(alt_output[:100])}")
    
    # Final fallback: if we still have no source, try reading the file directly
    if (not output or len(output.strip()) < 10) and session.current_frame and session.current_frame.file:
        try:
            file_path = session.current_frame.file
            if not file_path.startswith('/'):
                file_path = os.path.join(debug_client.config.working_directory, file_path)
            
            if os.path.exists(file_path):
                logger.info(f"Reading source directly from file: {file_path}")
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                # Determine line range
                if line:
                    center_line = line
                elif session.current_frame:
                    center_line = session.current_frame.line
                else:
                    center_line = 1
                
                # Calculate range
                start = max(1, center_line - count // 2)
                end = min(len(lines), start + count)
                
                # Format output similar to debugger
                output_lines = []
                for i in range(start, end + 1):
                    if i <= len(lines):
                        prefix = "=>" if i == center_line else "  "
                        output_lines.append(f"{prefix} {i:4d}   {lines[i-1].rstrip()}")
                
                output = "\n".join(output_lines)
                logger.info(f"Successfully read {len(output_lines)} lines from file")
        except Exception as e:
            logger.error(f"Failed to read source file directly: {e}")
    
    # Handle pagination
    pagination = paginate_text(output, limit, offset)
    
    return {
        "source": pagination["content"],
        "current_line": session.current_frame.line if session.current_frame else None,
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
    """Print the value of a variable or expression.
    
    Args:
        session_id: The session identifier
        expression: Variable name or expression to evaluate
        limit: Maximum number of characters to return (for pagination)
        offset: Starting character position (for pagination)
        depth: Maximum depth for nested structures (LLDB only, default is 1)
        
    Returns:
        Dictionary with the value, type, and pagination info
    """
    if session_id not in debug_client.sessions:
        return {"status": "error", "error": "Session not found"}
    
    session = debug_client.sessions[session_id]
    
    # Check if we're actually stopped first
    if session.state != DebuggerState.PAUSED:
        return {
            "value": "",
            "type": "",
            "expression": expression,
            "error": f"Cannot print variable: debugger is {session.state.value}, not paused",
            "pagination": {
                "value": {"total_chars": 0, "offset": 0, "limit": limit, "has_more": False},
                "type": {"total_chars": 0, "offset": 0, "limit": limit, "has_more": False}
            }
        }
    
    # First check if we're in a valid debugging context
    if session.debugger_type in [DebuggerType.LLDB, DebuggerType.RUST_LLDB]:
        # Make sure we have valid context by checking frame
        frame_check = debug_client._send_command(session, "frame info", timeout=1.0)
        if "sample_program`main:" in frame_check and " at " not in frame_check:
            # We're at assembly level, try to step into Rust code
            logger.info("At assembly level, attempting to step into Rust code")
            for _ in range(3):
                step_output = debug_client._send_command(session, "step", timeout=2.0)
                if " at " in step_output and ".rs:" in step_output:
                    break
    
    # Send print command with depth support
    if session.debugger_type in [DebuggerType.GDB, DebuggerType.RUST_GDB]:
        # GDB supports print settings for depth
        if depth is not None:
            # Save current settings
            old_depth = debug_client._send_command(session, "show print max-depth")
            # Set new depth
            debug_client._send_command(session, f"set print max-depth {depth}")
        
        output = debug_client._send_command(session, f"print {expression}")
        # Get type info separately
        type_output = debug_client._send_command(session, f"ptype {expression}")
        
        # Restore depth setting if changed
        if depth is not None:
            # Extract old value and restore
            match = re.search(r"is (\d+)", old_depth)
            if match:
                debug_client._send_command(session, f"set print max-depth {match.group(1)}")
    else:  # LLDB
        # For LLDB, try frame variable first for simple variables
        if re.match(r'^\w+$', expression) and not expression.startswith('*'):
            # Simple variable name
            output = debug_client._send_command(session, f"frame variable {expression}")
            if output and "error" not in output.lower() and len(output.strip()) > 0:
                # Parse frame variable output: (type) name = value
                match = re.match(r'^\(([^)]+)\)\s*\w+\s*=\s*(.+)', output.strip(), re.DOTALL)
                if match:
                    type_output = f"type = {match.group(1)}"
                    output = match.group(2).strip()
                else:
                    type_output = "type = unknown"
            else:
                # Fallback to expression
                output = debug_client._send_command(session, f"expression -- {expression}")
                type_output = "type = unknown"
        else:
            # Complex expression or dereference
            if expression.startswith('*'):
                # For dereference, try p command first
                p_output = debug_client._send_command(session, f"p {expression}")
                if p_output and "error" not in p_output.lower():
                    output = p_output
                else:
                    # Try frame variable with dereference
                    output = debug_client._send_command(session, f"frame variable {expression}")
                    if not output or "error" in output.lower():
                        # Last resort
                        output = debug_client._send_command(session, f"expression -- {expression}")
            else:
                # Regular expression
                output = debug_client._send_command(session, f"expression -- {expression}")
            
            # Try to get type information
            if output and "error" not in output.lower():
                # Try to extract type from output
                type_match = re.match(r'^\(([^)]+)\)', output)
                if type_match:
                    type_output = f"type = {type_match.group(1)}"
                else:
                    type_output = "type = unknown"
            else:
                type_output = "type = unknown"
    
    # Parse the output to extract value and type if needed
    value_text = output.strip()
    type_text = type_output.strip()
    
    # Remove command echo if present
    if value_text.startswith(expression):
        value_text = value_text[len(expression):].strip()
    
    # Handle pagination for value and type
    value_pagination = paginate_text(value_text, limit, offset)
    type_pagination = paginate_text(type_text, limit, offset)
    
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