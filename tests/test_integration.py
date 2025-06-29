"""
Integration tests for the Rust Debug MCP server
"""
import pytest
import asyncio
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.jons_mcp_rust_debug import debug_client, start_debug, stop_debug, set_breakpoint, run, list_locals


class TestIntegration:
    """Integration tests with real Rust binaries"""

    @pytest.fixture
    def rust_project(self, tmp_path):
        """Create a simple Rust project for testing"""
        # Create project structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        
        # Create Cargo.toml
        cargo_toml = tmp_path / "Cargo.toml"
        cargo_toml.write_text("""
[package]
name = "test_project"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "test_binary"
path = "src/main.rs"
""")
        
        # Create main.rs
        main_rs = src_dir / "main.rs"
        main_rs.write_text("""
fn add(a: i32, b: i32) -> i32 {
    let result = a + b;
    result
}

fn main() {
    let x = 5;
    let y = 10;
    let sum = add(x, y);
    println!("Sum: {}", sum);
}
""")
        
        # Set working directory for debug client
        debug_client.config.working_directory = str(tmp_path)
        
        yield tmp_path
        
        # Cleanup sessions
        for session_id in list(debug_client.sessions.keys()):
            try:
                debug_client._stop_session(session_id)
            except:
                pass

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not shutil.which("gdb") and not shutil.which("lldb"),
        reason="No debugger available"
    )
    async def test_basic_debugging_session(self, rust_project):
        """Test a basic debugging session"""
        # Start debug session
        result = await start_debug("binary", "test_binary")
        assert result["status"] == "started"
        session_id = result["session_id"]
        
        try:
            # Set breakpoint in main
            bp_result = await set_breakpoint(session_id, function="main")
            assert bp_result["status"] == "set"
            
            # Run program
            run_result = await run(session_id)
            assert "breakpoint" in run_result["stop_reason"].lower()
            
            # Check locals
            locals_result = await list_locals(session_id)
            assert "locals" in locals_result
            
        finally:
            # Stop session
            await stop_debug(session_id)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not shutil.which("cargo"),
        reason="Cargo not available"
    )
    async def test_cargo_build_integration(self, rust_project):
        """Test that cargo build works correctly"""
        try:
            # This should trigger cargo build
            result = await start_debug("binary", "test_binary")
            
            if "error" not in result["status"]:
                session_id = result["session_id"]
                await stop_debug(session_id)
                
                # Check that binary was created
                target_dir = rust_project / "target" / "debug"
                assert target_dir.exists()
                
                binary = target_dir / "test_binary"
                assert binary.exists() or (target_dir / "test_binary.exe").exists()
        except Exception as e:
            # If debugger is not available, at least check cargo build
            if "No debugger found" in str(e):
                pytest.skip("No debugger available")
            else:
                raise

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not shutil.which("gdb") and not shutil.which("lldb"),
        reason="No debugger available"
    )
    async def test_test_target_debugging(self, rust_project):
        """Test debugging a test target"""
        # Create a test file
        tests_dir = rust_project / "tests"
        tests_dir.mkdir()
        
        test_file = tests_dir / "integration_test.rs"
        test_file.write_text("""
#[test]
fn test_addition() {
    let a = 2;
    let b = 3;
    assert_eq!(a + b, 5);
}
""")
        
        # Start debug session for test
        result = await start_debug("test", "integration_test")
        
        if result["status"] == "started":
            session_id = result["session_id"]
            
            try:
                # Set breakpoint in test function
                bp_result = await set_breakpoint(session_id, function="test_addition")
                # Note: test function names might be mangled
                
            finally:
                await stop_debug(session_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])