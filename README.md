# Jon's MCP Rust Debug Server

A Model Context Protocol (MCP) server that provides Rust debugging capabilities through subprocess-based gdb/lldb integration.

## Overview

The MCP Rust Debug Server implements the Model Context Protocol to expose Rust debugger (gdb/lldb) functionality. It manages debugger subprocess sessions, enabling MCP clients to control debugging of Rust binaries, tests, and examples through a standardized interface.

## Architecture

This implementation uses a subprocess-based architecture where:
- Each debug session spawns a separate debugger process (gdb or lldb)
- Communication happens via stdin/stdout pipes
- Thread-based I/O handling ensures non-blocking operations
- Sessions are managed independently, allowing multiple concurrent debugging sessions
- Automatically uses rust-gdb/rust-lldb wrappers when available for better Rust support

## Features

- **Multi-Debugger Support**: Works with gdb, lldb, rust-gdb, and rust-lldb
- **Cargo Integration**: Automatically builds targets before debugging
- **Session Management**: Create and manage multiple debugging sessions concurrently
- **Breakpoint Control**: Set, remove, and list breakpoints with conditions
- **Execution Control**: Run, step, next, finish commands with clear action feedback
- **Stack Navigation**: Navigate up and down the call stack, view backtraces
- **Variable Inspection**: Examine variables with pretty-printing, evaluate expressions, list locals
- **Source Code Display**: View source code around current execution point
- **Rust Panic Handling**: Automatically sets breakpoints on rust_panic
- **Platform Detection**: Automatically selects the best debugger for your platform
- **Test Debugging**: Special support for debugging Rust tests with test summary
- **Smart Timeouts**: Different timeout values for different operations
- **Session State Tracking**: Track execution state and stop reasons

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/jonmmease/jons-mcp-rust-debug.git
cd jons-mcp-rust-debug

# Install and run
uv run python src/jons_mcp_rust_debug.py
```

### Direct from GitHub

```bash
# Run directly from GitHub
uvx --from git+https://github.com/jonmmease/jons-mcp-rust-debug jons-mcp-rust-debug
```

### Adding to Claude Code as MCP Server

To use this with Claude Code, add it using the CLI:

```bash
claude mcp add jons-mcp-rust-debug uvx -- --from git+https://github.com/jonmmease/jons-mcp-rust-debug jons-mcp-rust-debug
```

The server will be available in Claude Code for debugging Rust applications.

## Pagination Support

Many tools support pagination to handle large outputs efficiently. Pagination works with character-based `limit` and `offset` parameters:

- `limit`: Maximum number of characters to return
- `offset`: Starting character position (0-based)

Tools with pagination support return a `pagination` object containing:
- `total_chars`: Total characters available
- `offset`: Current offset
- `limit`: Limit used
- `has_more`: Whether more content is available

Supported tools:
- `backtrace` - Stack traces can be very long
- `run` - Program output can be extensive
- `list_source` - Source files can be large
- `print_variable` - Complex data structures may have long representations
- `list_locals` - Many local variables can produce long output
- `check_debug_info` - Debug information can be very detailed

Example usage:
```python
# Get first 1000 characters of backtrace
result = await backtrace(session_id="session_1", char_limit=1000, char_offset=0)

# Get next page if more content available
if result["pagination"]["has_more"]:
    next_result = await backtrace(session_id="session_1", char_limit=1000, char_offset=1000)
```

## Configuration

Create a `rustdebugconfig.json` file in your project root:

```json
{
  "debugger": null,              // "gdb", "lldb", "rust-gdb", "rust-lldb", or path
  "cargo_path": null,            // Path to cargo executable (null = auto-detect)
  "working_directory": ".",      // Working directory for debugging
  "environment": {               // Additional environment variables
    "RUST_BACKTRACE": "1"
  },
  "cargo_args": ["--release"],   // Additional cargo build arguments
  "prefer_rust_wrappers": true   // Prefer rust-gdb/rust-lldb over plain gdb/lldb
}
```

## MCP Tools

### Session Management

#### start_debug
Initialize a debugging session.
```
Args:
  target_type: "binary", "test", or "example" (required)
  target: Name of the specific target (optional for single binary)
  args: Command line arguments (optional)
  cargo_flags: Additional cargo build flags (optional)
    Example: ["--no-default-features", "--features", "test-only"]
  env: Environment variables for the build process (optional)
    Example: {"RUST_TEST_THREADS": "1", "CARGO_BUILD_TARGET": "x86_64-unknown-linux-gnu"}
  package: Specific package name for workspace projects (optional)
    Example: "my-crate" to build only that crate in a workspace
Returns:
  session_id: Unique session identifier
  status: "started" or error message
  debugger: The debugger being used (gdb/lldb/rust-gdb/rust-lldb)
```

#### stop_debug
Terminate an active debugging session.
```
Args:
  session_id: The session identifier
Returns:
  status: "stopped" or error message
```

#### list_sessions
List all active debugging sessions.
```
Returns:
  sessions: Array of session objects with details
```

### Breakpoint Management

#### set_breakpoint
Set a breakpoint at specified location.
```
Args:
  session_id: The session identifier
  file: Source file path (optional if function specified)
  line: Line number (optional if function specified)
  function: Function name (optional if file/line specified)
  condition: Conditional expression (optional)
  temporary: One-time breakpoint (default: false)
Returns:
  breakpoint_id: Unique breakpoint identifier
  location: Resolved location
  status: "set" or error message
```

#### remove_breakpoint
Remove a breakpoint.
```
Args:
  session_id: The session identifier
  breakpoint_id: Breakpoint identifier
Returns:
  status: "removed" or error message
```

#### list_breakpoints
List all breakpoints.
```
Args:
  session_id: The session identifier
Returns:
  breakpoints: Array of breakpoint objects
```

### Execution Control

#### run
Start or continue program execution.
```
Args:
  session_id: The session identifier
  limit: Max characters to return (optional, for pagination)
  offset: Starting character position (optional, for pagination)
Returns:
  status: Current execution state
  stop_reason: "breakpoint", "signal", "exited", etc.
  stopped_at: Location where execution stopped
  output: Debugger output
  pagination: Pagination info (if limit specified)
```

#### step
Step into function calls (execute next line).
```
Args:
  session_id: The session identifier
Returns:
  location: New execution position
  output: Debugger output
```

#### next
Step over function calls (execute next line in current function).
```
Args:
  session_id: The session identifier
Returns:
  location: New execution position
  output: Debugger output
```

#### finish
Continue until current function returns.
```
Args:
  session_id: The session identifier
Returns:
  output: Debugger output
  return_value: Value being returned (if available)
```

### Stack Navigation

#### backtrace
Get current stack trace.
```
Args:
  session_id: The session identifier
  frame_limit: Maximum frames to return (optional)
  char_limit: Max characters to return (optional, for pagination)
  char_offset: Starting character position (optional, for pagination)
Returns:
  frames: Array of stack frames
  raw_output: Raw debugger output (paginated if limit specified)
  pagination: Pagination info (if char_limit specified)
```

#### up
Move up in the stack (to caller).
```
Args:
  session_id: The session identifier
  count: Number of frames to move (default: 1)
Returns:
  frame: New current frame information
  output: Debugger output
```

#### down
Move down in the stack.
```
Args:
  session_id: The session identifier
  count: Number of frames to move (default: 1)
Returns:
  frame: New current frame information
  output: Debugger output
```

### Inspection

#### list_source
Show source code around current or specified position.
```
Args:
  session_id: The session identifier
  line: Center line (optional, defaults to current)
  count: Number of lines to show (default: 10)
Returns:
  source: Source code
  current_line: Currently executing line
```

#### print_variable
Print the value of a variable or expression.
```
Args:
  session_id: The session identifier
  expression: Variable name or expression
Returns:
  value: String representation
  type: Type information
  expression: The evaluated expression
```

#### list_locals
List all local variables in current scope.
```
Args:
  session_id: The session identifier
Returns:
  locals: Local variables with values
```

#### evaluate
Evaluate a Rust expression in the current context.
```
Args:
  session_id: The session identifier
  expression: Rust expression
Returns:
  result: Evaluation result
  error: Error message if evaluation failed
  expression: The evaluated expression
```

#### get_test_summary
Get test-specific information (test sessions only).
```
Args:
  session_id: The session identifier
Returns:
  session_type: "test"
  target: Test target name
  state: Current debugger state
  last_stop_reason: Why execution stopped
  test_functions: List of test functions in backtrace
  panic_info: Details about panic if stopped on assertion
  test_results: Summary of passed/failed/ignored tests
```

#### get_enum_info
Get enum variant information for better understanding of discriminant values.
```
Args:
  session_id: The session identifier
  type_name: The enum type name (e.g., "Option<i32>", "MyEnum")
Returns:
  type_name: The requested type name
  variants: Dictionary mapping discriminant values to variant names
  raw_output: Raw debugger output
```

#### session_diagnostics
Get detailed diagnostic information about the debugging session.
```
Args:
  session_id: The session identifier
Returns:
  session_id: The session ID
  debugger_type: Type of debugger (gdb/lldb/rust-gdb/rust-lldb)
  state: Current state (idle/running/paused/finished/error)
  has_started: Whether the program has been run
  last_stop_reason: Why execution stopped
  current_location: Current file:line if stopped
  breakpoints: Number of breakpoints set
  process_alive: Whether debugger process is running
  thread_info: Thread information
  frame_info: Current frame information
  program_status: Program execution status
  is_stopped: Whether actually stopped
  actual_stop_reason: Real stop reason from debugger
  context_tests: Results of context accessibility tests
```

#### check_debug_info
Check debug symbol and source mapping information.
```
Args:
  session_id: The session identifier
Returns:
  debug_info: Dictionary containing:
    - images: Loaded binary images (LLDB)
    - source_map: Source mapping settings (LLDB)
    - source_info: Current source information (LLDB)
    - loaded_files: Loaded files (GDB)
    - sources: Available source files (GDB)
```

## Platform Support

- **Linux**: Best GDB support, rust-gdb recommended
- **macOS**: LLDB preferred, rust-lldb recommended
- **Windows**: GDB support through MinGW/MSYS2

## Debugger Selection

The tool automatically selects the best debugger in this order:
1. Configured debugger in `rustdebugconfig.json`
2. Platform-specific preference (LLDB on macOS, GDB on Linux)
3. Rust-specific wrappers (rust-lldb, rust-gdb) if available
4. Standard debuggers (lldb, gdb)

## Requirements

- Rust toolchain installed (cargo, rustc)
- One of: gdb, lldb, rust-gdb, or rust-lldb
- Python 3.10+

## Usage Examples

### Basic Binary Debugging
```python
# Start debugging a simple binary
await start_debug("binary", "myapp")
```

### Debugging with Custom Features
```python
# Debug with specific features enabled
await start_debug(
    target_type="test",
    target="integration_test",
    cargo_flags=["--no-default-features", "--features", "test-only"],
    env={"RUST_TEST_THREADS": "1"}
)
```

### Workspace-Specific Debugging
```python
# Debug a specific crate in a workspace
await start_debug(
    target_type="binary",
    target="server",
    package="backend-crate",
    cargo_flags=["--release"],
    env={"RUST_LOG": "debug"}
)
```

### Cross-Platform Debugging
```python
# Debug for a specific target platform
await start_debug(
    target_type="binary",
    target="myapp",
    env={"CARGO_BUILD_TARGET": "x86_64-unknown-linux-musl"}
)
```

## Recent Improvements

Based on user feedback, the following enhancements have been added:

### Latest Updates (Round 3)

1. **Session Diagnostics Tool**: New `session_diagnostics` tool provides comprehensive debugging state information
2. **Improved Variable Printing**: Better handling of LLDB output with fallback to `frame variable` for simple variables
3. **Enhanced Source Listing**: Fixed minimal output issue with better context handling and file path resolution
4. **Better Enum Type Lookup**: Multiple approaches for finding enum type information in LLDB
5. **Automatic Stop Info Updates**: Stop reason and location now updated automatically when prompt detected

### Previous Updates (Round 2)

1. **Fixed Pretty-Printing Duplication**: Resolved issue where LLDB output was duplicated, now shows clean formatted output
2. **Enhanced Stop Reason Detection**: Better detection of breakpoint stops in LLDB with "stop reason = breakpoint N" parsing
3. **Improved Location Tracking**: Current location now properly tracked when stopped at breakpoints
4. **Enum Variant Support**: New `get_enum_info` tool to lookup enum variant names from discriminant values
5. **Cleaner Value Display**: Type information separated from value output for cleaner display

### Previous Updates

1. **Enhanced Command Flow**: The `run` command now returns clear action messages indicating what happened (e.g., "Started new execution", "Continued from breakpoint")
2. **Improved Stop Reasons**: More detailed stop reasons including specific breakpoint numbers and exit codes
3. **Better Variable Display**: Pretty-printing for Rust values with proper indentation
4. **Session State Tracking**: Track whether program has started, last stop reason, and current location
5. **Test-Specific Features**: Automatic breakpoints on test assertions and panic handlers, plus `get_test_summary` tool
6. **Flexible Timeouts**: Different timeout values for different operations (30s for run, 15s for evaluate, 5s default)
7. **Enhanced Backtrace Parsing**: More flexible parsing to handle various output formats

## Limitations

- Subprocess communication may have slight delays
- Some advanced debugger features may not be exposed
- Pretty-printing quality depends on debugger and Rust wrapper availability
- Windows support may require additional setup

## Troubleshooting

### Common Issues

1. **Debugger not found**: Install gdb or lldb, or specify path in configuration
2. **Build failures**: Ensure cargo can build your project normally
3. **Permission denied on macOS**: LLDB may require developer tools or code signing
4. **Missing rust-gdb/rust-lldb**: Install with `rustup component add rust-src`

### Debugging the Debugger

If you're experiencing issues with the debugger not working as expected:

1. **Use session_diagnostics**: Run this tool first to understand the current state
   ```python
   result = await session_diagnostics(session_id)
   print(result["is_stopped"])  # Check if actually stopped
   print(result["actual_stop_reason"])  # See real stop reason
   print(result["context_tests"])  # Check what's accessible
   ```

2. **Variables showing as "unknown"**:
   - Ensure the program is actually stopped at a breakpoint (not just set)
   - Run the program first with the `run` command
   - Check `session_diagnostics` to verify the debugger state

3. **Empty enum info**:
   - Try with just the base type name without generics (e.g., "Option" instead of "Option<i32>")
   - Some complex generic types may not be fully supported

4. **Minimal source listing**:
   - Make sure debug symbols are included in your build
   - Try specifying an explicit line number
   - Check that source files are accessible from the working directory

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
uv run python -m pytest

# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/
```

## License

MIT