"""
Tests for MCP tool functions
"""
import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.jons_mcp_rust_debug import (
    start_debug, stop_debug, list_sessions,
    set_breakpoint, remove_breakpoint, list_breakpoints,
    run, step, next, finish,
    backtrace, up, down,
    list_source, print_variable, list_locals, evaluate,
    debug_client
)


class TestMCPTools:
    """Test MCP tool functions"""

    @pytest.mark.asyncio
    async def test_start_debug_success(self):
        """Test successful debug session start"""
        with patch.object(debug_client, 'create_session') as mock_create:
            mock_create.return_value = "session_1"
            debug_client.sessions["session_1"] = Mock(
                debugger_type=Mock(value="rust-gdb")
            )
            
            result = await start_debug("binary", "myapp", ["arg1", "arg2"])
            
            assert result["session_id"] == "session_1"
            assert result["status"] == "started"
            assert result["debugger"] == "rust-gdb"
            mock_create.assert_called_once_with(
                target_type="binary",
                target="myapp",
                args=["arg1", "arg2"]
            )

    @pytest.mark.asyncio
    async def test_start_debug_failure(self):
        """Test debug session start failure"""
        with patch.object(debug_client, 'create_session') as mock_create:
            mock_create.side_effect = RuntimeError("Failed to start debugger")
            
            result = await start_debug("binary", "myapp")
            
            assert result["status"] == "error"
            assert "Failed to start debugger" in result["error"]

    @pytest.mark.asyncio
    async def test_stop_debug_success(self):
        """Test successful debug session stop"""
        debug_client.sessions["session_1"] = Mock()
        
        with patch.object(debug_client, '_stop_session') as mock_stop:
            result = await stop_debug("session_1")
            
            assert result["status"] == "stopped"
            mock_stop.assert_called_once_with("session_1")

    @pytest.mark.asyncio
    async def test_stop_debug_not_found(self):
        """Test stopping non-existent session"""
        result = await stop_debug("non_existent")
        
        assert result["status"] == "error"
        assert "Session not found" in result["error"]

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        """Test listing active sessions"""
        mock_session1 = Mock(
            target_type="binary",
            target="app1",
            state=Mock(value="paused"),
            debugger_type=Mock(value="gdb")
        )
        mock_session2 = Mock(
            target_type="test",
            target="test1",
            state=Mock(value="running"),
            debugger_type=Mock(value="lldb")
        )
        
        debug_client.sessions = {
            "session_1": mock_session1,
            "session_2": mock_session2
        }
        
        result = await list_sessions()
        
        assert len(result["sessions"]) == 2
        assert result["sessions"][0]["session_id"] in ["session_1", "session_2"]
        assert any(s["target"] == "app1" for s in result["sessions"])
        assert any(s["target"] == "test1" for s in result["sessions"])

    @pytest.mark.asyncio
    async def test_set_breakpoint_by_line(self):
        """Test setting breakpoint by file and line"""
        mock_session = Mock(
            debugger_type=Mock(GDB=True, RUST_GDB=True),
            breakpoints={}
        )
        debug_client.sessions["session_1"] = mock_session
        
        with patch.object(debug_client, '_send_command') as mock_send:
            mock_send.return_value = "Breakpoint 1 at 0x12345: file src/main.rs, line 42."
            
            result = await set_breakpoint("session_1", file="src/main.rs", line=42)
            
            assert result["breakpoint_id"] == 1
            assert result["location"] == "src/main.rs:42"
            assert result["status"] == "set"
            assert 1 in mock_session.breakpoints
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_breakpoint_by_function(self):
        """Test setting breakpoint by function name"""
        mock_session = Mock(
            debugger_type=Mock(GDB=False, LLDB=True),
            breakpoints={}
        )
        debug_client.sessions["session_1"] = mock_session
        
        with patch.object(debug_client, '_send_command') as mock_send:
            mock_send.return_value = "Breakpoint 1: where = myapp`main at main.rs:10"
            
            result = await set_breakpoint("session_1", function="main")
            
            assert result["breakpoint_id"] == 1
            assert result["location"] == "main"
            assert result["status"] == "set"

    @pytest.mark.asyncio
    async def test_run_command(self):
        """Test run/continue command"""
        mock_session = Mock(
            debugger_type=Mock(GDB=True),
            state=Mock(value="paused"),
            last_output=""
        )
        debug_client.sessions["session_1"] = mock_session
        
        with patch.object(debug_client, '_send_command') as mock_send:
            mock_send.return_value = "Starting program: /path/to/binary\nBreakpoint 1, main () at src/main.rs:10"
            
            result = await run("session_1")
            
            assert result["status"] == "running"
            assert result["stop_reason"] == "breakpoint"
            assert "main.rs:10" in result["stopped_at"]

    @pytest.mark.asyncio
    async def test_step_command(self):
        """Test step command"""
        mock_session = Mock(
            debugger_type=Mock(GDB=True),
            current_frame=None
        )
        debug_client.sessions["session_1"] = mock_session
        
        with patch.object(debug_client, '_send_command') as mock_send:
            mock_send.return_value = "0x0000555555559b1a in calculate (x=5) at src/calc.rs:15"
            
            result = await step("session_1")
            
            assert "calc.rs:15" in result["location"]
            assert mock_session.current_frame is not None
            assert mock_session.current_frame.function == "calculate (x=5)"

    @pytest.mark.asyncio
    async def test_backtrace_command(self):
        """Test backtrace command"""
        mock_session = Mock(debugger_type=Mock(GDB=True))
        debug_client.sessions["session_1"] = mock_session
        
        with patch.object(debug_client, '_send_command') as mock_send:
            mock_send.return_value = """#0  0x0000555555559b1a in main () at src/main.rs:42
#1  0x0000555555559c2b in start () at src/lib.rs:10
#2  0x00007ffff7dc7083 in __libc_start_main () from /lib/x86_64-linux-gnu/libc.so.6"""
            
            result = await backtrace("session_1")
            
            assert len(result["frames"]) == 3
            assert result["frames"][0]["index"] == 0
            assert result["frames"][0]["function"] == "main ()"
            assert result["frames"][0]["file"] == "src/main.rs"
            assert result["frames"][0]["line"] == 42

    @pytest.mark.asyncio
    async def test_print_variable(self):
        """Test printing variable value"""
        mock_session = Mock(debugger_type=Mock(GDB=True))
        debug_client.sessions["session_1"] = mock_session
        
        with patch.object(debug_client, '_send_command') as mock_send:
            mock_send.side_effect = [
                "$1 = 42",
                "type = i32"
            ]
            
            result = await print_variable("session_1", "x")
            
            assert result["value"] == "$1 = 42"
            assert result["type"] == "type = i32"
            assert result["expression"] == "x"

    @pytest.mark.asyncio
    async def test_evaluate_expression(self):
        """Test evaluating expression"""
        mock_session = Mock(debugger_type=Mock(LLDB=True))
        debug_client.sessions["session_1"] = mock_session
        
        with patch.object(debug_client, '_send_command') as mock_send:
            mock_send.return_value = "(i32) $0 = 15"
            
            result = await evaluate("session_1", "x + y")
            
            assert result["result"] == "(i32) $0 = 15"
            assert result["expression"] == "x + y"
            assert result["error"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])