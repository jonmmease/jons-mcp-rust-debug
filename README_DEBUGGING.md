# Debugging LLDB Output Issues

This document explains the diagnostic approach for fixing LLDB output capture issues in the MCP.

## Test Scripts

### 1. `debug_lldb_output.py`
Tests the MCP's print_variable, stop reason detection, and source listing through the actual MCP interface. This helps identify where the MCP is failing.

```bash
python debug_lldb_output.py
```

### 2. `test_lldb_raw.py`
Tests LLDB subprocess communication directly, bypassing the MCP layer. This shows what LLDB actually outputs and helps identify timing/buffering issues.

```bash
python test_lldb_raw.py
```

### 3. `test_output_capture.py`
Compares different methods of capturing subprocess output:
- Thread-based reading (current MCP approach)
- Select-based non-blocking I/O
- Asyncio subprocess
- PTY (pseudo-terminal)

```bash
python test_output_capture.py
```

### 4. `fix_lldb_issues.py`
Demonstrates working solutions for each problematic operation using PTY and proper output parsing.

```bash
python fix_lldb_issues.py
```

## Key Findings

### 1. Dereferencing Issue (`*expr`)
**Problem**: The MCP sends the correct commands but doesn't capture/parse output properly.

**Root Causes**:
- LLDB frame variable syntax was wrong (`-varname` instead of `*varname`)
- Output parsing doesn't handle multi-line LLDB format
- Command echo contaminating the output

**Fix**:
```python
# Correct syntax
"frame variable *varname"  # NOT "frame variable -varname"

# Multiple fallbacks
"expression -- *varname"
"p *varname"

# Parse multi-line output correctly
re.match(r'^\(([^)]+)\)\s*\w+\s*=\s*(.+)', output, re.DOTALL)
```

### 2. Stop Reason Detection
**Problem**: "Process XXXXX stopped" is visible but stop_reason remains "unknown"

**Root Causes**:
- Only looking for "stop reason =" pattern
- Not checking process status first
- Missing "Process stopped" detection

**Fix**:
```python
# Check multiple sources
"process status"  # Primary source
"thread info"     # Fallback

# Detect "Process X stopped" pattern
if "Process" in output and "stopped" in output:
    # Now look for stop reason
```

### 3. Source Listing
**Problem**: Returns empty despite valid location

**Root Causes**:
- LLDB might return just command echo
- File paths might need to be absolute
- Output capture timing issue

**Fix**:
```python
# Try multiple approaches
f"source list -f {file_path} -l {line} -c {count}"
f"source list -l {line} -c {count}"
"source list"

# Fallback to reading file directly
if os.path.exists(file_path):
    with open(file_path) as f:
        # Format like debugger output
```

### 4. Output Capture Issues
**Problem**: LLDB output not being captured reliably

**Root Causes**:
- Buffering issues with subprocess pipes
- LLDB expects interactive terminal
- Timing issues with prompt detection

**Solutions**:
1. **Use PTY (Pseudo-Terminal)**:
   ```python
   import pty
   master, slave = pty.openpty()
   proc = subprocess.Popen([lldb, binary], stdin=slave, stdout=slave)
   ```

2. **Non-blocking I/O with select()**:
   ```python
   fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
   ready, _, _ = select.select([proc.stdout], [], [], timeout)
   ```

3. **Proper prompt detection**:
   ```python
   # Wait for "(lldb) " not just any output
   # Handle partial prompts at buffer boundaries
   ```

## Recommended MCP Fixes

1. **Switch to PTY for LLDB**:
   - Better interactive behavior
   - More reliable output
   - Handles ANSI codes properly

2. **Enhanced Output Parsing**:
   - Remove command echoes
   - Handle multi-line output
   - Parse multiple output formats

3. **Multiple Command Fallbacks**:
   - Try different command syntaxes
   - Check error indicators
   - Fall back to file reading for source

4. **Better State Tracking**:
   - Always check process status first
   - Update state from multiple sources
   - Cache parsed information

5. **Platform-Specific Handling**:
   - Test on macOS ARM64 with rust-lldb
   - Handle LLDB version differences
   - Account for platform-specific output formats

## Testing the Fixes

To verify fixes are working:

```bash
# 1. Run diagnostic tests
python debug_lldb_output.py > diagnostics.log

# 2. Check specific operations
python -c "
import asyncio
from jons_mcp_rust_debug import start_debug, print_variable, session_diagnostics

async def test():
    r = await start_debug('binary', 'sample_program')
    sid = r['session_id']
    
    # Test each operation
    diag = await session_diagnostics(sid)
    print(f'Stop reason: {diag['last_stop_reason']}')
    
    var = await print_variable(sid, '*args')
    print(f'Deref type: {var['type']}')

asyncio.run(test())
"
```

## Implementation Priority

1. **Critical**: Fix output capture with PTY
2. **Critical**: Fix frame variable syntax for dereferencing  
3. **High**: Add "Process stopped" detection
4. **High**: Add source file reading fallback
5. **Medium**: Improve output parsing robustness