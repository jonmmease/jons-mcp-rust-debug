"""Inspection tools for Rust debugging."""

from __future__ import annotations

import os
from typing import Any

import lldb

from ..lldb_client import DebuggerState
from ..server import ensure_debug_client
from ..utils import get_active_thread, paginate_text, validate_session


async def list_source(
    session_id: str,
    line: int | None = None,
    count: int = 10,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any]:
    """Show source code around current or specified line.

    Args:
        session_id: The session identifier
        line: Line number to center on (None for current)
        count: Number of lines to show
        limit: Character limit for pagination
        offset: Character offset for pagination

    Returns:
        Dictionary with source code
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    frame = thread.GetSelectedFrame()
    if not frame or not frame.IsValid():
        return {"status": "error", "error": "No active frame"}

    # Get current location
    line_entry = frame.GetLineEntry()
    if not line_entry.IsValid():
        return {"status": "error", "error": "No source information available"}

    file_spec = line_entry.GetFileSpec()
    current_line = line if line else line_entry.GetLine()

    # Try to read the source file
    file_path = os.path.join(
        file_spec.GetDirectory() or "", file_spec.GetFilename() or ""
    )

    # If the path doesn't exist, try to resolve it relative to working directory
    if not os.path.exists(file_path):
        # Try just the filename in case it's a relative path
        alt_path = file_spec.GetFilename()
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            # Try to find it in the source directories
            for root, dirs, files in os.walk(client.config.working_directory):
                if file_spec.GetFilename() in files:
                    file_path = os.path.join(root, file_spec.GetFilename())
                    break

    try:
        with open(file_path) as f:
            lines = f.readlines()

        # Calculate range
        start = max(0, current_line - count // 2)
        end = min(len(lines), current_line + count // 2)

        # Format output
        output_lines = []
        for i in range(start, end):
            prefix = "=>" if i + 1 == current_line else "  "
            output_lines.append(f"{prefix} {i + 1:4d} {lines[i].rstrip()}")

        output = "\n".join(output_lines)
    except Exception as e:
        return {"status": "error", "error": f"Failed to read source: {e}"}

    # Handle pagination
    pagination = paginate_text(output, limit, offset)

    return {
        "source": pagination["content"],
        "current_line": current_line,
        "pagination": {
            "total_chars": pagination["total_chars"],
            "offset": pagination["offset"],
            "limit": pagination["limit"],
            "has_more": pagination["has_more"],
        },
    }


async def print_variable(
    session_id: str,
    expression: str,
    limit: int | None = None,
    offset: int | None = None,
    depth: int | None = None,
) -> dict[str, Any]:
    """Print the value of a variable or expression.

    Args:
        session_id: The session identifier
        expression: Variable name or expression to evaluate
        limit: Character limit for pagination
        offset: Character offset for pagination
        depth: Display depth for nested structures

    Returns:
        Dictionary with value and type
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if session.state != DebuggerState.PAUSED:
        return {
            "value": "",
            "type": "",
            "expression": expression,
            "error": f"Cannot print variable: debugger is {session.state.value}, not paused",
        }

    try:
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    frame = thread.GetSelectedFrame()
    if not frame or not frame.IsValid():
        return {"status": "error", "error": "No active frame"}

    # Evaluate expression
    options = lldb.SBExpressionOptions()
    if depth:
        # This would need custom formatting
        pass

    result = frame.EvaluateExpression(expression, options)

    if not result or result.GetError().Fail():
        error_msg = result.GetError().GetCString() if result else "Unknown error"
        return {
            "value": "",
            "type": "",
            "expression": expression,
            "error": error_msg,
        }

    # Get value and type
    value = result.GetValue() or str(result)
    type_name = result.GetTypeName() or "unknown"

    # Handle pagination
    value_pagination = paginate_text(value, limit, offset)
    type_pagination = paginate_text(type_name, limit, offset)

    return {
        "value": value_pagination["content"],
        "type": type_pagination["content"],
        "expression": expression,
        "pagination": {
            "value": {
                "total_chars": value_pagination["total_chars"],
                "offset": value_pagination["offset"],
                "limit": value_pagination["limit"],
                "has_more": value_pagination["has_more"],
            },
            "type": {
                "total_chars": type_pagination["total_chars"],
                "offset": type_pagination["offset"],
                "limit": type_pagination["limit"],
                "has_more": type_pagination["has_more"],
            },
        },
    }


async def list_locals(
    session_id: str,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any]:
    """List all local variables in current scope.

    Args:
        session_id: The session identifier
        limit: Character limit for pagination
        offset: Character offset for pagination

    Returns:
        Dictionary with local variables
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    frame = thread.GetSelectedFrame()
    if not frame or not frame.IsValid():
        return {"status": "error", "error": "No active frame"}

    # Get local variables
    locals_dict = {}
    variables = frame.GetVariables(
        True, True, False, True
    )  # args, locals, statics, in_scope_only

    for var in variables:
        if var.IsValid():
            name = var.GetName()
            value = var.GetValue() or str(var)
            type_name = var.GetTypeName()
            locals_dict[name] = f"({type_name}) {value}"

    # Format output
    output_lines = []
    for name, info in locals_dict.items():
        output_lines.append(f"{name} = {info}")

    output = "\n".join(output_lines)

    # Handle pagination
    pagination = paginate_text(output, limit, offset)

    return {
        "locals": locals_dict,
        "raw_output": pagination["content"],
        "pagination": {
            "total_chars": pagination["total_chars"],
            "offset": pagination["offset"],
            "limit": pagination["limit"],
            "has_more": pagination["has_more"],
        },
    }


async def evaluate(session_id: str, expression: str) -> dict[str, Any]:
    """Evaluate a Rust expression in the current context.

    Args:
        session_id: The session identifier
        expression: Expression to evaluate

    Returns:
        Dictionary with evaluation result
    """
    # This is essentially the same as print_variable but with simpler output
    result = await print_variable(session_id, expression)

    if "error" in result and result.get("error"):
        return {
            "result": "",
            "error": result["error"],
            "expression": expression,
        }

    return {
        "result": result["value"],
        "error": None,
        "expression": expression,
    }


async def print_array(
    session_id: str,
    expression: str,
    count: int,
    start: int = 0,
    element_type: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any]:
    """Print elements of an array or pointer range using LLDB's parray command.

    Args:
        session_id: The session identifier
        expression: Array/pointer expression to index (e.g., "arr", "ptr", "&vec[0]")
        count: Number of elements to print
        start: Starting index (default 0)
        element_type: Optional type for pointer casting (e.g., "i32")
        limit: Character limit for pagination
        offset: Character offset for pagination

    Returns:
        Dictionary with array elements output and pagination info
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if session.state != DebuggerState.PAUSED:
        return {
            "status": "error",
            "error": f"Cannot print array: debugger is {session.state.value}, not paused",
        }

    try:
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    frame = thread.GetSelectedFrame()
    if not frame or not frame.IsValid():
        return {"status": "error", "error": "No active frame"}

    # Build the parray command
    adjusted_expression = expression

    # Apply type cast if provided
    if element_type:
        adjusted_expression = f"({element_type}*)({adjusted_expression})"

    # Apply offset if start > 0
    if start > 0:
        adjusted_expression = f"({adjusted_expression})+{start}"

    command = f"parray {count} {adjusted_expression}"

    # Get command interpreter and execute
    interpreter = session.debugger.GetCommandInterpreter()
    result = lldb.SBCommandReturnObject()
    interpreter.HandleCommand(command, result)

    if not result.Succeeded():
        error_msg = result.GetError() or "parray command failed"
        return {
            "status": "error",
            "error": error_msg,
            "expression": expression,
            "start": start,
            "count": count,
        }

    # Get output
    output = result.GetOutput() or ""

    # Handle pagination
    pagination = paginate_text(output, limit, offset)

    return {
        "output": pagination["content"],
        "expression": expression,
        "start": start,
        "count": count,
        "element_type": element_type,
        "pagination": {
            "total_chars": pagination["total_chars"],
            "offset": pagination["offset"],
            "limit": pagination["limit"],
            "has_more": pagination["has_more"],
        },
    }


async def set_variable(
    session_id: str,
    variable: str,
    value: str,
    frame_index: int = 0,
) -> dict[str, Any]:
    """Set a variable to a new value.

    Args:
        session_id: The session identifier
        variable: Variable name or path (e.g., "x", "config.timeout")
        value: New value as string (e.g., "42", "true", "3.14")
        frame_index: Stack frame index (0 = current frame)

    Returns:
        Dictionary with old_value, new_value, type, and status
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if session.state != DebuggerState.PAUSED:
        return {
            "status": "error",
            "error": f"Cannot set variable: debugger is {session.state.value}, not paused",
        }

    try:
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Get the frame at the specified index
    frame = thread.GetFrameAtIndex(frame_index)
    if not frame or not frame.IsValid():
        return {
            "status": "error",
            "error": f"Invalid frame at index {frame_index}",
        }

    # Get old value first
    old_result = frame.EvaluateExpression(variable)
    if not old_result or not old_result.IsValid():
        return {
            "status": "error",
            "error": f"Variable '{variable}' not found",
        }

    # Check if there was an error evaluating the variable
    if old_result.GetError().Fail():
        error_msg = old_result.GetError().GetCString()
        return {
            "status": "error",
            "error": f"Failed to access variable '{variable}': {error_msg}",
        }

    old_value = old_result.GetValue() or str(old_result)
    type_name = old_result.GetTypeName() or "unknown"

    # Set new value using assignment expression
    assign_expr = f"{variable} = {value}"
    options = lldb.SBExpressionOptions()
    result = frame.EvaluateExpression(assign_expr, options)

    if not result or result.GetError().Fail():
        error_msg = result.GetError().GetCString() if result else "Unknown error"
        # Common errors include type mismatches, const variables, optimized out variables
        return {
            "status": "error",
            "error": f"Failed to set variable '{variable}': {error_msg}",
            "old_value": old_value,
            "type": type_name,
        }

    # Verify the new value was set
    verify = frame.EvaluateExpression(variable)
    if not verify or verify.GetError().Fail():
        return {
            "status": "error",
            "error": f"Assignment succeeded but verification failed for '{variable}'",
            "old_value": old_value,
            "type": type_name,
        }

    new_value = verify.GetValue() or str(verify)

    return {
        "status": "success",
        "variable": variable,
        "old_value": old_value,
        "new_value": new_value,
        "type": type_name,
    }


async def print_slice(
    session_id: str,
    expression: str,
    count: int,
    start: int = 0,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any]:
    """Print elements of a Rust slice, Vec, or Box<[T]>.

    This tool automatically extracts the data pointer from STANDARD Rust types:
    - Vec<T>
    - &[T] and &mut [T]
    - Box<[T]>
    - [T; N] fixed-size arrays

    For custom structs with pointer+length fields (e.g., path.points where
    you see data_pointer and length fields), use print_array instead with
    the data pointer expression (e.g., "path.points.data_pointer").

    Args:
        session_id: The session identifier
        expression: Slice expression (e.g., "my_vec", "&slice[..]", "my_array")
        count: Number of elements to print (required)
        start: Starting index (default 0)
        limit: Character limit for pagination
        offset: Character offset for pagination

    Returns:
        Dictionary with slice elements output and pagination info
    """
    client = ensure_debug_client()

    try:
        session = validate_session(client.sessions, session_id)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if session.state != DebuggerState.PAUSED:
        return {
            "status": "error",
            "error": f"Cannot print slice: debugger is {session.state.value}, not paused",
        }

    try:
        thread = get_active_thread(session.process)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    frame = thread.GetSelectedFrame()
    if not frame or not frame.IsValid():
        return {"status": "error", "error": "No active frame"}

    # Evaluate the expression to get the slice/vec value
    result = frame.EvaluateExpression(expression)
    if not result or result.GetError().Fail():
        error_msg = result.GetError().GetCString() if result else "Unknown error"
        return {
            "status": "error",
            "error": f"Failed to evaluate expression '{expression}': {error_msg}",
        }

    type_name = result.GetTypeName() or ""

    # Try to extract data pointer and length based on the type
    data_ptr_expr = None
    length_expr = None
    detected_length = None

    # Handle Vec<T>
    if "Vec<" in type_name or "vec::Vec<" in type_name:
        # Vec has buf.ptr.pointer and len fields
        data_ptr_expr = f"({expression}).as_ptr()"
        length_expr = f"({expression}).len()"
    # Handle &[T] or &mut [T] slices
    elif type_name.startswith("&[") or type_name.startswith("&mut ["):
        # Slice reference - use as_ptr() and len()
        data_ptr_expr = f"({expression}).as_ptr()"
        length_expr = f"({expression}).len()"
    # Handle Box<[T]>
    elif "Box<[" in type_name:
        data_ptr_expr = f"({expression}).as_ptr()"
        length_expr = f"({expression}).len()"
    # Handle fixed-size arrays [T; N]
    elif type_name.startswith("[") and ";" in type_name:
        # Fixed array - extract N from [T; N]
        try:
            # Parse the size from type like "[i32; 5]"
            size_part = type_name.split(";")[1].strip().rstrip("]").strip()
            detected_length = int(size_part)
            data_ptr_expr = f"&({expression})[0]"
        except (ValueError, IndexError):
            data_ptr_expr = f"&({expression})[0]"
    else:
        # Try to treat it as a pointer/array directly
        # This handles raw pointers and C-style arrays
        data_ptr_expr = expression

    # Get the length if we have a length expression
    if length_expr and detected_length is None:
        len_result = frame.EvaluateExpression(length_expr)
        if len_result and not len_result.GetError().Fail():
            try:
                detected_length = int(len_result.GetValue() or "0")
            except ValueError:
                pass

    # Use the provided count (required parameter)

    # If start is beyond the length, return empty
    if detected_length is not None and start >= detected_length:
        return {
            "output": "",
            "expression": expression,
            "start": start,
            "count": 0,
            "detected_length": detected_length,
            "detected_type": type_name,
            "pagination": {
                "total_chars": 0,
                "offset": 0,
                "limit": limit,
                "has_more": False,
            },
        }

    # Delegate to print_array
    array_result = await print_array(
        session_id=session_id,
        expression=data_ptr_expr,
        count=count,
        start=start if data_ptr_expr != expression else 0,  # Only apply start if we have data_ptr
        element_type=None,
        limit=limit,
        offset=offset,
    )

    # Enhance the result with slice-specific info
    if array_result.get("status") == "error":
        return array_result

    return {
        "output": array_result.get("output", ""),
        "expression": expression,
        "start": start,
        "count": count,
        "detected_length": detected_length,
        "detected_type": type_name,
        "data_ptr_expression": data_ptr_expr,
        "pagination": array_result.get("pagination", {}),
    }
