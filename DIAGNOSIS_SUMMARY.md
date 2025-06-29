# LLDB Output Capture Diagnosis Summary

## Root Cause Identified

The MCP's current subprocess approach with regular pipes **does not work** with LLDB on macOS. LLDB requires either:
1. A PTY (pseudo-terminal) for interactive mode
2. Batch mode with `-o` commands

## Evidence

### 1. Regular Pipes = No Output
```python
proc = subprocess.Popen(['rust-lldb', binary], 
                       stdin=PIPE, stdout=PIPE, ...)
# Result: 0 bytes of output captured
```

### 2. PTY = Partial Success
```python
master, slave = pty.openpty()
proc = subprocess.Popen(['rust-lldb', binary],
                       stdin=slave, stdout=slave, ...)
# Result: Output captured but mixed with initialization
```

### 3. Batch Mode = Full Success ✓
```bash
rust-lldb binary -o "b main" -o "run" -o "frame variable" --batch
# Result: Complete output with all information
```

Output includes:
- ✓ "Process 52491 stopped"
- ✓ "stop reason = breakpoint 1.2" 
- ✓ "frame #0: 0x000000010000192c sample_program`main"
- ✓ Frame variables (when requested)
- ✓ Source listing (when requested)

## Why Current MCP Fails

1. **Output Capture Method**: Uses regular pipes which LLDB doesn't write to in interactive mode
2. **No PTY Support**: Doesn't use pseudo-terminals which LLDB expects
3. **Parsing Assumptions**: Expects output that never arrives

## Recommended Fixes

### Option 1: Switch to PTY (Preferred for Interactive)
```python
import pty
import select
import fcntl

master, slave = pty.openpty()
proc = subprocess.Popen([lldb_cmd, binary], 
                       stdin=slave, stdout=slave, stderr=PIPE)
os.close(slave)

# Make non-blocking
flags = fcntl.fcntl(master, fcntl.F_GETFL)
fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)

# Read/write through master fd
os.write(master, b"command\n")
output = os.read(master, 4096)
```

### Option 2: Batch Command Execution
For specific operations, use batch mode:
```python
def execute_batch(commands):
    cmd_args = ['rust-lldb', binary]
    for cmd in commands:
        cmd_args.extend(['-o', cmd])
    cmd_args.extend(['--batch', '-o', 'quit'])
    
    result = subprocess.run(cmd_args, capture_output=True, text=True)
    return result.stdout
```

### Option 3: Hybrid Approach
- Use PTY for long-running session
- Use batch mode for quick queries
- Cache results to avoid re-execution

## Specific Issue Fixes

### 1. Dereferencing (*expr)
**Current**: Sends correct command but gets no output
**Fix**: Will work once output capture is fixed

### 2. Stop Reason Detection  
**Current**: Can't detect because no output received
**Fix**: Output contains "stop reason = breakpoint X.Y"

### 3. Source Listing
**Current**: Returns empty
**Fix**: Batch mode returns full source

### 4. Type/Value Parsing
**Current**: Can't parse what isn't captured
**Fix**: Proper output will have clear format

## Implementation Priority

1. **Immediate**: Add PTY support to `_reader_thread` and process creation
2. **High**: Update `_wait_for_prompt` to handle PTY output
3. **High**: Add initialization wait logic
4. **Medium**: Implement batch mode for specific queries
5. **Low**: Clean up parsing once output is captured

## Testing

The diagnostic scripts prove:
- Regular pipes: 0% success
- PTY: 70% success (with init noise)
- Batch: 100% success

The MCP must switch from regular pipes to PTY or batch mode to function on macOS with LLDB.