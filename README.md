# Jon's MCP Rust Debug Server

A Model Context Protocol (MCP) server that provides Rust debugging capabilities through LLDB Python API integration.

## Overview

The MCP Rust Debug Server implements the Model Context Protocol to expose Rust debugger functionality using LLDB's Python API. This provides direct, reliable debugging control through LLDB's native Python bindings.

The server enables MCP clients to control debugging of Rust binaries, tests, and examples through a standardized interface.

## Architecture

This implementation uses LLDB's Python API for direct debugging control:
- Each debug session uses lldb.SBDebugger for direct API access
- No subprocess or PTY communication needed
- Event-driven architecture with LLDB's listener/event system
- Sessions are managed independently, allowing multiple concurrent debugging sessions
- Direct access to debugging objects (breakpoints, variables, frames)

## Features

- **LLDB Python API**: Direct, reliable debugging through LLDB's native Python bindings
- **Cargo Integration**: Automatically builds targets before debugging
- **Session Management**: Create and manage multiple debugging sessions concurrently
- **Breakpoint Control**: Set, remove, and list breakpoints with conditions
- **Execution Control**: Run, step, next, finish commands with clear action feedback
- **Stack Navigation**: Navigate up and down the call stack, view backtraces
- **Variable Inspection**: Examine variables with direct API access, evaluate expressions, list locals
- **Source Code Display**: View source code around current execution point
- **Rust Panic Handling**: Automatically sets breakpoints on rust_panic
- **Test Debugging**: Special support for debugging Rust tests with test summary
- **Event-Driven**: Uses LLDB's event system for responsive debugging
- **Session State Tracking**: Track execution state and stop reasons

## Installation

### Prerequisites

- Rust toolchain (cargo, rustc)
- Python 3.10-3.12 (required by lldb-python package)

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/jonmmease/jons-mcp-rust-debug.git
cd jons-mcp-rust-debug

# Run the server
uv run jons-mcp-rust-debug
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
  "cargo_path": null,            // Path to cargo executable (null = auto-detect)
  "working_directory": ".",      // Working directory for debugging
  "environment": {               // Additional environment variables
    "RUST_BACKTRACE": "1"
  },
  "cargo_args": ["--release"]    // Additional cargo build arguments
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
  limit: Max characters to return (optional, for pagination)
  offset: Starting character position (optional, for pagination)
  depth: Maximum depth for nested structures (optional)
Returns:
  value: String representation
  type: Type information
  expression: The evaluated expression
  pagination: Pagination info for value and type
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

- **macOS**: Best support with lldb-python package
- **Linux**: Good support with lldb-python package  
- **Windows**: Limited support - LLDB availability varies

## Requirements

- Rust toolchain installed (cargo, rustc)
- Python 3.10-3.12
- lldb-python package (automatically installed)

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

This implementation uses LLDB's Python API for direct, reliable debugging:

### Key Benefits

1. **Direct API Access**:
   - No subprocess or PTY communication needed
   - Direct access to debugging objects and state
   - Type-safe Python bindings
   - Used by professional tools like CodeLLDB

2. **Reliable Operation**:
   - ✅ Variable dereferencing works correctly
   - ✅ Stop reasons are always accurate
   - ✅ Source listing shows proper code
   - ✅ Type information is complete
   - ✅ No command echo or parsing issues

3. **Better Performance**:
   - No PTY overhead or terminal emulation
   - Direct API calls are faster
   - Event-driven architecture
   - Non-blocking operations

4. **Enhanced Features**:
   - Rich object inspection
   - Accurate breakpoint management
   - Proper thread and frame navigation
   - Direct expression evaluation

## Limitations

- Requires LLDB with Python bindings (no GDB support)
- Windows support is limited due to LLDB availability
- Some LLDB features may not be fully exposed through the API

## Troubleshooting

### Common Issues

1. **"LLDB Python module not found"**: 
   - Run: `uv pip install lldb-python --prerelease=allow`
   - Ensure you're using Python 3.10-3.12
2. **Build failures**: Ensure cargo can build your project normally
3. **Permission denied on macOS**: LLDB may require developer tools or code signing

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