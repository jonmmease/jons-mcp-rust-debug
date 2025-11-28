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

## How It Works

The server uses the `lldb-python` package to provide direct access to LLDB's debugging capabilities through its Python API:

1. **Session Creation**: When you start debugging, the server:
   - Builds your Rust target using `cargo build`
   - Creates an LLDB debugger instance (`lldb.SBDebugger`)
   - Loads the compiled binary as a target
   - Sets up an event listener for asynchronous debugging events

2. **Event-Driven Control**: A background thread monitors LLDB events:
   - Process state changes (running, stopped, exited)
   - Breakpoint hits
   - Thread state updates
   - The event handler updates session state in real-time

3. **Direct API Access**: All debugging operations use LLDB's structured API:
   - Breakpoints: `target.BreakpointCreateByLocation()`
   - Execution: `process.Continue()`, `thread.StepOver()`
   - Inspection: `frame.EvaluateExpression()`, `value.GetSummary()`
   - No text parsing or command interpretation needed

4. **Multi-threaded Debugging**: The server properly handles multi-threaded programs:
   - Automatically selects the thread that hit a breakpoint
   - Allows manual thread switching
   - Maintains correct context for variable inspection

5. **Clean Shutdown**: Sessions can be stopped gracefully:
   - Process termination handled in background thread
   - Resources cleaned up without blocking
   - Event threads exit cleanly

## Features

- **LLDB Python API**: Direct, reliable debugging through LLDB's native Python bindings
- **Cargo Integration**: Automatically builds targets before debugging
- **Session Management**: Create and manage multiple debugging sessions concurrently
- **Breakpoint Control**: Set, remove, and list breakpoints with conditions
- **Watchpoint Support**: Monitor memory locations and variables for changes with read/write watchpoints
- **Execution Control**: Run, step, next, finish, and continue-to-line commands with clear action feedback
- **Stack Navigation**: Navigate up and down the call stack, view backtraces
- **Variable Inspection**: Examine variables with direct API access, evaluate expressions, list locals
- **Variable Modification**: Set variables to new values during debugging sessions
- **Array Inspection**: Print array and slice contents with automatic type detection
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
- `print_array` - Array contents can be very large
- `print_slice` - Slice contents can be very large
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
  "environment": {
    "RUST_BACKTRACE": "1"
  }
}
```

All fields are optional:
- `cargo_path`: Path to cargo executable (default: auto-detect)
- `working_directory`: Working directory for debugging (default: directory containing the config file)
- `environment`: Additional environment variables
- `cargo_args`: Additional cargo build arguments (e.g., `["--release"]`)

## MCP Tools

### Session Management

#### start_debug
Initialize a debugging session.
```
Args:
  target_type: "binary", "test", or "example" (required)
  target: Name of the specific target (optional for single binary)
  args: Command line arguments (optional)
    For tests: the test filter, e.g., ["test_name", "--exact"]
    IMPORTANT: Use full module path for tests in a module: ["tests::test_name", "--exact"]
    Run `cargo test --test <file> -- --list` to find the correct filter
  cargo_flags: Additional cargo build flags (optional)
    Example: ["--no-default-features", "--features", "test-only"]
  env: Environment variables for the build process (optional)
    Example: {"RUST_TEST_THREADS": "1", "CARGO_BUILD_TARGET": "x86_64-unknown-linux-gnu"}
  package: Package name for workspace projects (optional)
    When specified, the crate directory is automatically resolved from cargo metadata
    Example: "my-crate"
  root_directory: Absolute path to the Rust project root (optional)
    - Single crate: The project directory (where Cargo.toml is)
    - Workspace: The workspace root. Use with `package` to specify which crate
Returns:
  session_id: Unique session identifier
  status: "started" or error message
  debugger: The debugger being used (lldb-api)
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

### Watchpoint Management

#### set_watchpoint
Set a watchpoint on a variable or memory address to pause execution when the value changes.
```
Args:
  session_id: The session identifier
  expression: Variable name or memory address to watch
  watch_type: Type of watchpoint - "write", "read", or "read_write" (default: "write")
  size: Size in bytes to watch (optional, auto-detected from expression)
  condition: Conditional expression for the watchpoint (optional)
Returns:
  watchpoint_id: Unique watchpoint identifier
  address: Memory address being watched (hex format)
  size: Size in bytes being monitored
  watch_type: Type of watchpoint set
  expression: Expression being watched
  status: "set" or error message
```

#### list_watchpoints
List all watchpoints in the session.
```
Args:
  session_id: The session identifier
Returns:
  watchpoints: Array of watchpoint objects with details:
    - id: Watchpoint identifier
    - address: Memory address (hex format)
    - size: Size in bytes
    - watch_type: "write", "read", or "read_write"
    - expression: Original expression
    - condition: Conditional expression (if set)
    - enabled: Whether watchpoint is active
    - hit_count: Number of times triggered
```

#### remove_watchpoint
Remove a watchpoint.
```
Args:
  session_id: The session identifier
  watchpoint_id: Watchpoint identifier
Returns:
  status: "removed" or error message
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

#### continue_to_line
Continue execution until the specified line is hit.
```
Args:
  session_id: The session identifier
  file: Source file path (relative or absolute)
  line: Target line number
  timeout_ms: Optional timeout in milliseconds
Returns:
  status: Current execution state
  stop_reason: Why execution stopped
  stopped_at: Location where execution stopped
  current_location: Current file:line position
  target_location: The requested file:line
  reached_target: Boolean indicating if the target was reached
  file: File path where stopped
  line: Line number where stopped
  function: Function name where stopped
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

#### set_variable
Set a variable to a new value during debugging.
```
Args:
  session_id: The session identifier
  variable: Variable name or path (e.g., "x", "config.timeout")
  value: New value as string (e.g., "42", "true", "3.14")
  frame_index: Stack frame index (default: 0 = current frame)
Returns:
  status: "success" or "error"
  variable: Variable name
  old_value: Previous value
  new_value: New value after assignment
  type: Variable type
  error: Error message if modification failed
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

#### print_array
Print elements of an array or pointer range using LLDB's parray command.
```
Args:
  session_id: The session identifier
  expression: Array/pointer expression to index (e.g., "arr", "ptr", "&vec[0]")
  count: Number of elements to print
  start: Starting index (default: 0)
  element_type: Optional type for pointer casting (e.g., "i32")
  limit: Max characters to return (optional, for pagination)
  offset: Starting character position (optional, for pagination)
Returns:
  output: Formatted array elements
  expression: The evaluated expression
  start: Starting index used
  count: Number of elements printed
  element_type: Type used (if specified)
  pagination: Pagination info (if limit specified)
```

#### print_slice
Print elements of a Rust slice, Vec, or Box<[T]>.

Use print_slice for Rust types (Vec, &[T], Box<[T]>, arrays).
Use print_array for raw pointers or pointer expressions.
```
Args:
  session_id: The session identifier
  expression: Slice expression (e.g., "my_vec", "&slice[..]", "my_array")
  count: Number of elements to print (required)
  start: Starting index (default: 0)
  limit: Max characters to return (optional, for pagination)
  offset: Starting character position (optional, for pagination)
Returns:
  output: Formatted slice elements
  expression: The evaluated expression
  start: Starting index used
  count: Number of elements printed
  detected_length: Total length of the slice (if detectable)
  detected_type: Rust type of the slice
  data_ptr_expression: Internal pointer expression used
  pagination: Pagination info (if limit specified)
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
await start_debug(
    target_type="binary",
    target="myapp",
    root_directory="/path/to/my-project"
)
```

### Debugging with Custom Features
```python
# Debug with specific features enabled
await start_debug(
    target_type="test",
    target="integration_test",
    cargo_flags=["--no-default-features", "--features", "test-only"],
    env={"RUST_TEST_THREADS": "1"},
    root_directory="/path/to/my-project"
)
```

### Workspace-Specific Debugging
```python
# Debug a specific crate in a workspace
# The tool automatically resolves the crate directory from cargo metadata
await start_debug(
    target_type="binary",
    target="server",
    package="backend-crate",
    cargo_flags=["--release"],
    env={"RUST_LOG": "debug"},
    root_directory="/path/to/workspace"
)
```

### Cross-Platform Debugging
```python
# Debug for a specific target platform
await start_debug(
    target_type="binary",
    target="myapp",
    env={"CARGO_BUILD_TARGET": "x86_64-unknown-linux-musl"},
    root_directory="/path/to/my-project"
)
```

## Debugging Tests Guide

This section covers common patterns and gotchas when debugging Rust tests.

### Single Crate vs Workspace

For **single crate** projects, just specify `root_directory`:
```python
start_debug(target_type="test", target="my_test",
            root_directory="/path/to/my-project")
```

For **workspace** projects, specify both `root_directory` (workspace root) and `package`:
```python
start_debug(target_type="test", target="my_test",
            package="my-crate",
            root_directory="/path/to/workspace")
```

The tool automatically resolves the crate's directory using `cargo metadata`.

### Test Target Naming

**Integration tests** (in `tests/` directory):
- `target`: the test file name without `.rs` extension
- `args`: filter path starts from *within* the test file

```rust
// tests/test_data.rs

// Test directly in file:
#[test]
fn my_test() { }
// → args: ["my_test", "--exact"]

// Test inside mod tests {}:
mod tests {
    #[test]
    fn my_test() { }
}
// → args: ["tests::my_test", "--exact"]
```

**Lib tests** (in `src/` with `#[cfg(test)]`):
- `target`: leave empty (builds all lib tests)
- `args`: filter path *includes the source module name*

```rust
// src/lyon_utils.rs
mod tests {
    #[test]
    fn test_multi_polygon() { }
}
// → args: ["lyon_utils::tests::test_multi_polygon", "--exact"]

// src/foo/bar.rs
mod tests {
    #[test]
    fn test_something() { }
}
// → args: ["foo::bar::tests::test_something", "--exact"]
```

This is the key difference: integration test filters start from within the file, while lib test filters include the full module path from the crate root.

**To find the correct filter**, run:
```bash
# For integration tests
cargo test --test <test_file> -- --list

# For lib tests
cargo test --lib -- --list
```

### Breakpoint File Paths

Use the **filename only** for breakpoints, not the full path:

```python
# Recommended - filename only
set_breakpoint(session_id, file="test_data.rs", line=117)

# Also works
set_breakpoint(session_id, file="tests/test_data.rs", line=117)

# May cause issues - avoid absolute paths
set_breakpoint(session_id, file="/full/absolute/path/to/test_data.rs", line=117)
```

### Async Test Support

Async tests using `#[tokio::test]`, `#[async_std::test]`, etc. are fully supported. The debugger automatically handles breakpoints on spawned threads.

```rust
#[tokio::test]
async fn test_async_operation() {
    let result = some_async_fn().await;  // ← breakpoint here works
}
```

No special configuration needed.

### Complete Example: Debugging an Integration Test

**Single crate:**
```python
# 1. Start session
result = await start_debug(
    target_type="test",
    target="integration",
    args=["tests::test_foo", "--exact"],
    root_directory="/path/to/my-crate"
)
session_id = result["session_id"]

# 2. Set breakpoint (filename only)
await set_breakpoint(session_id, file="integration.rs", line=25)

# 3. Run to breakpoint
await run(session_id)  # → stops at breakpoint
```

**Workspace:**
```python
# Specify package - tool automatically resolves crate directory
result = await start_debug(
    target_type="test",
    target="integration",
    package="my-crate",
    args=["tests::test_foo", "--exact"],
    root_directory="/path/to/workspace"
)
```

## Technical Implementation

### LLDB Python API Benefits

The server leverages LLDB's Python API (via the `lldb-python` package) which provides several advantages:

- **Type-safe bindings**: Direct access to LLDB objects without text parsing
- **Reliable operation**: No issues with command echoing or output capture
- **Better performance**: No subprocess overhead or PTY emulation
- **Rich functionality**: Full access to LLDB's debugging capabilities
- **Professional-grade**: Same approach used by tools like CodeLLDB

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

# Format code
uv run black src/

# Lint code
uv run ruff check src/
```

### Testing

The test suite needs to be rewritten for the LLDB Python API implementation. The previous tests were written for the subprocess-based approach and are no longer applicable.

## License

MIT