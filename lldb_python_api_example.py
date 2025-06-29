#!/usr/bin/env python3
"""Example of using LLDB's Python API instead of PTY control"""

import lldb
import os
import sys

class LLDBPythonDebugger:
    def __init__(self, target_path):
        # Create debugger instance
        self.debugger = lldb.SBDebugger.Create()
        self.debugger.SetAsync(True)  # Non-blocking mode
        
        # Create target
        self.target = self.debugger.CreateTarget(target_path)
        if not self.target:
            raise RuntimeError(f"Failed to create target for {target_path}")
            
        # Command interpreter for fallback commands
        self.interpreter = self.debugger.GetCommandInterpreter()
        
    def set_breakpoint(self, location):
        """Set breakpoint at function or file:line"""
        if ':' in location:
            file, line = location.split(':')
            bp = self.target.BreakpointCreateByLocation(file, int(line))
        else:
            bp = self.target.BreakpointCreateByName(location)
        return bp.GetID() if bp else None
        
    def run(self, args=None):
        """Launch the process"""
        error = lldb.SBError()
        self.process = self.target.Launch(
            self.debugger.GetListener(),
            args or [],
            None,  # envp
            None,  # stdin_path
            None,  # stdout_path
            None,  # stderr_path
            os.getcwd(),  # working directory
            0,     # launch flags
            False, # stop at entry
            error
        )
        
        if error.Fail():
            raise RuntimeError(f"Launch failed: {error}")
            
        # Wait for stop
        self.wait_for_stop()
        return self.get_stop_reason()
        
    def wait_for_stop(self):
        """Wait for process to stop"""
        listener = self.debugger.GetListener()
        event = lldb.SBEvent()
        
        while True:
            if listener.WaitForEvent(1, event):
                if lldb.SBProcess.EventIsProcessEvent(event):
                    state = lldb.SBProcess.GetStateFromEvent(event)
                    if state == lldb.eStateStopped:
                        break
                    elif state == lldb.eStateExited:
                        return "exited"
                        
    def get_stop_reason(self):
        """Get why we stopped"""
        thread = self.process.GetSelectedThread()
        if thread:
            reason = thread.GetStopReason()
            if reason == lldb.eStopReasonBreakpoint:
                return f"breakpoint_{thread.GetStopReasonDataAtIndex(0)}"
            elif reason == lldb.eStopReasonSignal:
                return "signal"
            elif reason == lldb.eStopReasonException:
                return "exception"
            else:
                return str(reason)
        return "unknown"
        
    def get_variables(self):
        """Get local variables"""
        thread = self.process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        
        variables = {}
        for var in frame.GetVariables(True, True, True, True):
            name = var.GetName()
            value = var.GetValue()
            type_name = var.GetTypeName()
            variables[name] = {
                "value": value,
                "type": type_name
            }
        return variables
        
    def print_variable(self, expr):
        """Evaluate expression"""
        thread = self.process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        
        result = frame.EvaluateExpression(expr)
        return {
            "value": result.GetValue() or str(result),
            "type": result.GetTypeName(),
            "error": result.GetError().GetCString() if result.GetError().Fail() else None
        }
        
    def get_source_context(self, count=10):
        """Get source code around current location"""
        thread = self.process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        
        line_entry = frame.GetLineEntry()
        file_spec = line_entry.GetFileSpec()
        line = line_entry.GetLine()
        
        if file_spec.IsValid():
            file_path = file_spec.GetDirectory() + "/" + file_spec.GetFilename()
            
            # Read source directly
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    start = max(0, line - count // 2)
                    end = min(len(lines), line + count // 2)
                    
                    result = []
                    for i in range(start, end):
                        prefix = "=>" if i + 1 == line else "  "
                        result.append(f"{prefix} {i+1:4d} {lines[i].rstrip()}")
                    return "\n".join(result)
            except:
                pass
                
        return ""

# Example usage
if __name__ == "__main__":
    debugger = LLDBPythonDebugger("test_samples/target/debug/sample_program")
    
    # Set breakpoint
    bp_id = debugger.set_breakpoint("main")
    print(f"Breakpoint set: {bp_id}")
    
    # Run
    stop_reason = debugger.run()
    print(f"Stopped: {stop_reason}")
    
    # Get variables
    vars = debugger.get_variables()
    print(f"Variables: {vars}")
    
    # Print expression
    result = debugger.print_variable("2 + 2")
    print(f"Expression result: {result}")
    
    # Get source
    source = debugger.get_source_context()
    print(f"Source:\n{source}")