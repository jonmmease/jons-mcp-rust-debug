#!/usr/bin/env python3
"""
Manual test script for the Rust Debug MCP server
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from jons_mcp_rust_debug import (
    start_debug, stop_debug, list_sessions,
    set_breakpoint, list_breakpoints,
    run, step, next,
    backtrace, list_source, print_variable, list_locals
)


async def test_debugging():
    """Run a manual debugging test"""
    print("=== Rust Debug MCP Manual Test ===\n")
    
    # Change to test_samples directory
    os.chdir("test_samples")
    
    try:
        # Start debug session
        print("1. Starting debug session...")
        result = await start_debug("binary", "sample_program")
        if result["status"] != "started":
            print(f"Failed to start: {result}")
            return
        
        session_id = result["session_id"]
        print(f"Session started: {session_id}")
        print(f"Debugger: {result['debugger']}\n")
        
        # List sessions
        print("2. Listing sessions...")
        sessions = await list_sessions()
        print(f"Active sessions: {sessions}\n")
        
        # Set breakpoints
        print("3. Setting breakpoints...")
        bp1 = await set_breakpoint(session_id, function="main")
        print(f"Breakpoint 1: {bp1}")
        
        bp2 = await set_breakpoint(session_id, function="factorial")
        print(f"Breakpoint 2: {bp2}")
        
        bp3 = await set_breakpoint(session_id, file="sample_program.rs", line=50)
        print(f"Breakpoint 3: {bp3}\n")
        
        # List breakpoints
        print("4. Listing breakpoints...")
        bps = await list_breakpoints(session_id)
        print(f"Breakpoints: {bps}\n")
        
        # Run program
        print("5. Running program...")
        run_result = await run(session_id)
        print(f"Run result: {run_result}\n")
        
        # Show source
        print("6. Showing source code...")
        source = await list_source(session_id)
        print("Source code:")
        print(source["source"][:500] + "...\n")
        
        # List locals
        print("7. Listing local variables...")
        locals_result = await list_locals(session_id)
        print(f"Locals:\n{locals_result['locals']}\n")
        
        # Step into
        print("8. Stepping into next line...")
        step_result = await step(session_id)
        print(f"Stepped to: {step_result['location']}\n")
        
        # Print variable
        print("9. Printing variables...")
        var_result = await print_variable(session_id, "args")
        print(f"args = {var_result['value']}")
        print(f"Type: {var_result['type']}\n")
        
        # Backtrace
        print("10. Getting backtrace...")
        bt_result = await backtrace(session_id, limit=5)
        print("Stack frames:")
        for frame in bt_result["frames"]:
            print(f"  #{frame['index']} {frame['function']} at {frame['file']}:{frame['line']}")
        print()
        
        # Continue execution
        print("11. Continuing execution...")
        cont_result = await run(session_id)
        print(f"Stopped at: {cont_result.get('stopped_at', 'unknown')}")
        print(f"Reason: {cont_result['stop_reason']}\n")
        
        # Stop session
        print("12. Stopping debug session...")
        stop_result = await stop_debug(session_id)
        print(f"Session stopped: {stop_result}\n")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to cleanup
        try:
            sessions = await list_sessions()
            for session in sessions["sessions"]:
                await stop_debug(session["session_id"])
        except:
            pass


async def test_test_debugging():
    """Test debugging a Rust test"""
    print("\n=== Testing Rust Test Debugging ===\n")
    
    # Change to test_samples directory
    os.chdir("test_samples")
    
    try:
        # Start debug session for test
        print("1. Starting test debug session...")
        result = await start_debug("test", "sample_tests")
        if result["status"] != "started":
            print(f"Failed to start: {result}")
            return
        
        session_id = result["session_id"]
        print(f"Test session started: {session_id}\n")
        
        # Set breakpoint in test
        print("2. Setting breakpoint in test...")
        bp = await set_breakpoint(session_id, function="test_calculator_add")
        print(f"Breakpoint: {bp}\n")
        
        # Run tests
        print("3. Running tests...")
        run_result = await run(session_id)
        print(f"Run result: {run_result}\n")
        
        # Stop session
        print("4. Stopping test session...")
        stop_result = await stop_debug(session_id)
        print(f"Session stopped: {stop_result}\n")
        
    except Exception as e:
        print(f"\nError during test debugging: {e}")


async def test_advanced_features():
    """Test debugging with advanced features"""
    print("\n=== Testing Advanced Features ===\n")
    
    # Change to test_samples directory
    os.chdir("test_samples")
    
    try:
        # Start debug session with custom flags
        print("1. Starting debug session with custom cargo flags...")
        result = await start_debug(
            target_type="binary",
            target="sample_program",
            args=["test-arg"],
            cargo_flags=["--release"],
            env={"RUST_LOG": "debug", "RUST_BACKTRACE": "full"}
        )
        
        if result["status"] != "started":
            print(f"Failed to start: {result}")
            return
        
        session_id = result["session_id"]
        print(f"Advanced session started: {session_id}")
        print(f"Built with --release flag and custom env vars\n")
        
        # Set a breakpoint
        print("2. Setting breakpoint...")
        bp = await set_breakpoint(session_id, function="main")
        print(f"Breakpoint set: {bp}\n")
        
        # Run and check environment
        print("3. Running with custom environment...")
        run_result = await run(session_id)
        print(f"Stopped at: {run_result.get('stopped_at', 'unknown')}\n")
        
        # Clean up
        print("4. Stopping session...")
        await stop_debug(session_id)
        print("Advanced features test completed!\n")
        
    except Exception as e:
        print(f"\nError during advanced features test: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point"""
    print("Testing basic debugging...")
    await test_debugging()
    
    print("\n" + "="*50 + "\n")
    
    print("Testing test debugging...")
    await test_test_debugging()
    
    print("\n" + "="*50 + "\n")
    
    print("Testing advanced features...")
    await test_advanced_features()


if __name__ == "__main__":
    asyncio.run(main())