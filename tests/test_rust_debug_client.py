"""
Tests for the RustDebugClient class
"""
import pytest
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.jons_mcp_rust_debug import RustDebugClient, Config, DebuggerType, DebuggerState


class TestRustDebugClient:
    """Test suite for RustDebugClient"""

    def setup_method(self):
        """Setup for each test"""
        self.client = RustDebugClient()

    def teardown_method(self):
        """Cleanup after each test"""
        # Stop all sessions
        for session_id in list(self.client.sessions.keys()):
            try:
                self.client._stop_session(session_id)
            except:
                pass

    def test_load_config_default(self):
        """Test loading default configuration"""
        config = self.client._load_config()
        assert isinstance(config, Config)
        assert config.debugger is None
        assert config.cargo_path is None
        assert config.working_directory == "."
        assert config.prefer_rust_wrappers is True

    def test_load_config_from_file(self, tmp_path):
        """Test loading configuration from file"""
        config_data = {
            "debugger": "rust-lldb",
            "cargo_path": "/usr/local/bin/cargo",
            "working_directory": "/tmp",
            "environment": {"RUST_BACKTRACE": "full"},
            "cargo_args": ["--release"],
            "prefer_rust_wrappers": False
        }
        
        config_file = tmp_path / "rustdebugconfig.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        
        # Change to tmp directory to load config
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            client = RustDebugClient()
            config = client.config
            
            assert config.debugger == "rust-lldb"
            assert config.cargo_path == "/usr/local/bin/cargo"
            assert config.working_directory == "/tmp"
            assert config.environment["RUST_BACKTRACE"] == "full"
            assert config.cargo_args == ["--release"]
            assert config.prefer_rust_wrappers is False
        finally:
            os.chdir(original_cwd)

    @patch('shutil.which')
    def test_find_debugger_config_specified(self, mock_which):
        """Test finding debugger when specified in config"""
        self.client.config.debugger = "rust-gdb"
        mock_which.return_value = "/usr/bin/rust-gdb"
        
        dbg_type, dbg_path = self.client._find_debugger()
        
        assert dbg_type == DebuggerType.GDB
        assert dbg_path == "/usr/bin/rust-gdb"
        mock_which.assert_called_with("rust-gdb")

    @patch('shutil.which')
    @patch('platform.system')
    def test_find_debugger_macos(self, mock_system, mock_which):
        """Test finding debugger on macOS"""
        mock_system.return_value = "Darwin"
        
        # Simulate rust-lldb not found, but lldb found
        def which_side_effect(cmd):
            if cmd == "rust-lldb":
                return None
            elif cmd == "lldb":
                return "/usr/bin/lldb"
            return None
        
        mock_which.side_effect = which_side_effect
        
        dbg_type, dbg_path = self.client._find_debugger()
        
        assert dbg_type == DebuggerType.LLDB
        assert dbg_path == "/usr/bin/lldb"

    @patch('shutil.which')
    @patch('platform.system')
    def test_find_debugger_linux(self, mock_system, mock_which):
        """Test finding debugger on Linux"""
        mock_system.return_value = "Linux"
        
        # Simulate rust-gdb found
        mock_which.return_value = "/usr/bin/rust-gdb"
        
        dbg_type, dbg_path = self.client._find_debugger()
        
        assert dbg_type == DebuggerType.RUST_GDB
        assert dbg_path == "/usr/bin/rust-gdb"

    @patch('shutil.which')
    def test_find_debugger_not_found(self, mock_which):
        """Test error when no debugger is found"""
        mock_which.return_value = None
        
        with pytest.raises(RuntimeError, match="No debugger found"):
            self.client._find_debugger()

    @patch('shutil.which')
    def test_find_cargo_executable(self, mock_which):
        """Test finding cargo executable"""
        mock_which.return_value = "/usr/local/bin/cargo"
        
        cargo_path = self.client._find_cargo_executable()
        
        assert cargo_path == "/usr/local/bin/cargo"
        mock_which.assert_called_with("cargo")

    @patch('shutil.which')
    def test_find_cargo_not_found(self, mock_which):
        """Test error when cargo is not found"""
        mock_which.return_value = None
        
        with pytest.raises(RuntimeError, match="cargo not found"):
            self.client._find_cargo_executable()

    @patch('subprocess.run')
    @patch.object(RustDebugClient, '_find_cargo_executable')
    def test_build_target_binary(self, mock_find_cargo, mock_run):
        """Test building a binary target"""
        mock_find_cargo.return_value = "cargo"
        mock_run.return_value = Mock(
            returncode=0,
            stderr="   Compiling myapp v0.1.0\n    Finished dev [unoptimized] target(s)"
        )
        
        # Create a mock target directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "target" / "debug"
            target_dir.mkdir(parents=True)
            binary_path = target_dir / "myapp"
            binary_path.touch()
            
            self.client.config.working_directory = tmpdir
            
            result = self.client._build_target("binary", "myapp")
            
            assert result == str(binary_path)
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args == ["cargo", "build", "--bin", "myapp"]

    @patch('subprocess.run')
    @patch.object(RustDebugClient, '_find_cargo_executable')
    def test_build_target_test(self, mock_find_cargo, mock_run):
        """Test building a test target"""
        mock_find_cargo.return_value = "cargo"
        mock_run.return_value = Mock(
            returncode=0,
            stderr="   Compiling tests\n    Executable unittests src/lib.rs (target/debug/deps/myapp-123abc)"
        )
        
        result = self.client._build_target("test", "myapp")
        
        assert result == "target/debug/deps/myapp-123abc"
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["cargo", "test", "--no-run", "--test", "myapp"]

    @patch('subprocess.run')
    @patch.object(RustDebugClient, '_find_cargo_executable')
    def test_build_target_failed(self, mock_find_cargo, mock_run):
        """Test build failure"""
        mock_find_cargo.return_value = "cargo"
        mock_run.return_value = Mock(
            returncode=1,
            stderr="error: could not compile `myapp`"
        )
        
        with pytest.raises(RuntimeError, match="Failed to build target"):
            self.client._build_target("binary", "myapp")

    def test_create_session_no_debugger(self):
        """Test session creation fails without debugger"""
        with patch.object(self.client, '_find_debugger') as mock_find:
            mock_find.side_effect = RuntimeError("No debugger found")
            
            with pytest.raises(RuntimeError, match="No debugger found"):
                self.client.create_session("binary", "myapp", [])

    def test_session_cleanup_on_exit(self):
        """Test that sessions are cleaned up on exit"""
        # Create mock sessions
        mock_session1 = Mock(session_id="session_1")
        mock_session2 = Mock(session_id="session_2")
        self.client.sessions = {
            "session_1": mock_session1,
            "session_2": mock_session2
        }
        
        with patch.object(self.client, '_stop_session') as mock_stop:
            self.client._cleanup_all_sessions()
            
            assert mock_stop.call_count == 2
            mock_stop.assert_any_call("session_1")
            mock_stop.assert_any_call("session_2")


class TestDebuggerCommands:
    """Test debugger command generation and parsing"""

    def test_gdb_breakpoint_parsing(self):
        """Test parsing GDB breakpoint output"""
        from src.jons_mcp_rust_debug import GDB_BREAKPOINT_SET_PATTERN
        
        output = "Breakpoint 1 at 0x12345: file src/main.rs, line 42."
        match = GDB_BREAKPOINT_SET_PATTERN.search(output)
        
        assert match is not None
        assert match.group(1) == "1"
        assert match.group(2) == "0x12345"
        assert match.group(3) == "src/main.rs"
        assert match.group(4) == "42"

    def test_lldb_breakpoint_parsing(self):
        """Test parsing LLDB breakpoint output"""
        from src.jons_mcp_rust_debug import LLDB_BREAKPOINT_SET_PATTERN
        
        output = "Breakpoint 1: where = myapp`main + 16 at main.rs:10:5"
        match = LLDB_BREAKPOINT_SET_PATTERN.search(output)
        
        assert match is not None
        assert match.group(1) == "1"
        assert match.group(2) == "main.rs"
        assert match.group(3) == "10"

    def test_gdb_stack_frame_parsing(self):
        """Test parsing GDB stack frame output"""
        from src.jons_mcp_rust_debug import GDB_STACK_FRAME_PATTERN
        
        output = "#0  0x0000555555559b1a in main () at src/main.rs:42"
        match = GDB_STACK_FRAME_PATTERN.search(output)
        
        assert match is not None
        assert match.group(1) == "0"
        assert match.group(2) == "0x0000555555559b1a"
        assert match.group(3) == "main ()"
        assert match.group(4) == "src/main.rs"
        assert match.group(5) == "42"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])