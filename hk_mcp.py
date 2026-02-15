#!/usr/bin/env python3
"""HarnessKit MCP Server — Model Context Protocol server for fuzzy file editing.

Exposes HarnessKit's edit capabilities as MCP tools that any compatible
AI agent can use. Runs over stdio using JSON-RPC 2.0.

Tools provided:
  - harnesskit_apply: Apply a fuzzy edit to a file
  - harnesskit_apply_batch: Apply multiple edits atomically
  - harnesskit_match: Find the best match without applying (dry run)

Usage:
  python hk_mcp.py          # stdio mode (default)
  python hk_mcp.py --sse    # SSE mode (HTTP, future)
"""

import json
import sys
from typing import Any, Optional

from hk import apply_edit, create_file, find_best_match, validate_syntax, result_to_dict, AmbiguousMatchError

# MCP Protocol version
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "harnesskit"
SERVER_VERSION = "0.3.0"

TOOLS = [
    {
        "name": "harnesskit_apply",
        "description": (
            "Apply a fuzzy edit to a file. Finds old_text in the file using "
            "4-strategy matching (exact → whitespace-normalized → fuzzy → "
            "line-fuzzy) and replaces it with new_text. Returns match type "
            "and confidence score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find (fuzzy matching supported)",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text",
                },
                "threshold": {
                    "type": "number",
                    "description": "Fuzzy match threshold 0-1 (default: 0.8)",
                    "default": 0.8,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, show what would change without applying",
                    "default": False,
                },
                "validate": {
                    "type": "boolean",
                    "description": "If true, validate syntax after applying (rollback on failure)",
                    "default": False,
                },
            },
            "required": ["file", "old_text", "new_text"],
        },
    },
    {
        "name": "harnesskit_apply_batch",
        "description": (
            "Apply multiple fuzzy edits to files. Each edit is applied "
            "independently. Returns an array of results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "Array of edit objects",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                        },
                        "required": ["file", "old_text", "new_text"],
                    },
                },
                "threshold": {
                    "type": "number",
                    "default": 0.8,
                },
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                },
            },
            "required": ["edits"],
        },
    },
    {
        "name": "harnesskit_match",
        "description": (
            "Find the best match for text in a file without modifying it. "
            "Returns match type, confidence, and the matched text. Useful "
            "for previewing what would be replaced."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to the file to search",
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find",
                },
                "threshold": {
                    "type": "number",
                    "default": 0.8,
                },
            },
            "required": ["file", "old_text"],
        },
    },
    {
        "name": "harnesskit_create",
        "description": (
            "Create a new file with the given content. Fails if file "
            "already exists unless force is true. Optionally validates syntax."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to the file to create",
                },
                "content": {
                    "type": "string",
                    "description": "File content",
                },
                "force": {
                    "type": "boolean",
                    "description": "Overwrite if file exists",
                    "default": False,
                },
                "validate": {
                    "type": "boolean",
                    "description": "Validate syntax before writing",
                    "default": False,
                },
            },
            "required": ["file", "content"],
        },
    },
    {
        "name": "harnesskit_validate",
        "description": (
            "Validate a file's syntax without modifying it. Supports Python, "
            "JSON, XML/HTML, YAML, and JavaScript/TypeScript."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to the file to validate",
                },
            },
            "required": ["file"],
        },
    },
]


def make_response(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def make_error(id: Any, code: int, message: str, data: Any = None) -> dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": err}


def handle_initialize(id: Any, params: dict) -> dict:
    return make_response(id, {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {},
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
        },
    })


def handle_tools_list(id: Any, params: dict) -> dict:
    return make_response(id, {"tools": TOOLS})


def handle_tool_call(id: Any, params: dict) -> dict:
    name = params.get("name", "")
    args = params.get("arguments", {})

    if name == "harnesskit_apply":
        result = apply_edit(
            file_path=args["file"],
            old_text=args["old_text"],
            new_text=args["new_text"],
            threshold=args.get("threshold", 0.8),
            dry_run=args.get("dry_run", False),
            validate=args.get("validate", False),
        )
        rd = result_to_dict(result)
        is_error = result.status in ("no_match", "error", "ambiguous", "validation_error")
        return make_response(id, {
            "content": [{"type": "text", "text": json.dumps(rd, indent=2)}],
            "isError": is_error,
        })

    elif name == "harnesskit_apply_batch":
        edits = args.get("edits", [])
        threshold = args.get("threshold", 0.8)
        dry_run = args.get("dry_run", False)
        results = []
        any_error = False
        for edit in edits:
            r = apply_edit(
                file_path=edit["file"],
                old_text=edit["old_text"],
                new_text=edit["new_text"],
                threshold=threshold,
                dry_run=dry_run,
            )
            rd = result_to_dict(r)
            results.append(rd)
            if r.status in ("no_match", "error", "ambiguous"):
                any_error = True
        return make_response(id, {
            "content": [{"type": "text", "text": json.dumps(results, indent=2)}],
            "isError": any_error,
        })

    elif name == "harnesskit_match":
        file_path = args["file"]
        old_text = args["old_text"]
        threshold = args.get("threshold", 0.8)
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except (FileNotFoundError, OSError) as e:
            return make_response(id, {
                "content": [{"type": "text", "text": json.dumps({"status": "error", "error": str(e)})}],
                "isError": True,
            })
        try:
            match = find_best_match(content, old_text, threshold)
        except AmbiguousMatchError as e:
            return make_response(id, {
                "content": [{"type": "text", "text": json.dumps({
                    "status": "ambiguous",
                    "count": len(e.matches),
                    "match_type": e.matches[0].match_type,
                    "confidence": e.matches[0].confidence,
                })}],
                "isError": True,
            })
        if match is None:
            return make_response(id, {
                "content": [{"type": "text", "text": json.dumps({"status": "no_match"})}],
                "isError": True,
            })
        return make_response(id, {
            "content": [{"type": "text", "text": json.dumps({
                "status": "found",
                "match_type": match.match_type,
                "confidence": match.confidence,
                "matched_text": match.matched_text,
                "start": match.start,
                "end": match.end,
            }, indent=2)}],
        })

    elif name == "harnesskit_create":
        result = create_file(
            file_path=args["file"],
            content=args["content"],
            force=args.get("force", False),
            validate=args.get("validate", False),
        )
        rd = result_to_dict(result)
        is_error = result.status in ("error", "validation_error")
        return make_response(id, {
            "content": [{"type": "text", "text": json.dumps(rd, indent=2)}],
            "isError": is_error,
        })

    elif name == "harnesskit_validate":
        file_path = args["file"]
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except (FileNotFoundError, OSError) as e:
            return make_response(id, {
                "content": [{"type": "text", "text": json.dumps({"status": "error", "error": str(e)})}],
                "isError": True,
            })
        valid, err = validate_syntax(file_path, content)
        result = {"status": "valid" if valid else "invalid", "file": file_path}
        if err:
            result["error"] = err
        return make_response(id, {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
            "isError": not valid,
        })

    else:
        return make_error(id, -32601, f"Unknown tool: {name}")


HANDLERS = {
    "initialize": handle_initialize,
    "notifications/initialized": None,  # notification, no response
    "tools/list": handle_tools_list,
    "tools/call": handle_tool_call,
}


def run_stdio():
    """Main stdio loop — read JSON-RPC messages, dispatch, respond."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp = make_error(None, -32700, "Parse error")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        method = msg.get("method", "")
        id = msg.get("id")
        params = msg.get("params", {})

        handler = HANDLERS.get(method)
        if handler is None:
            if id is not None and method not in HANDLERS:
                resp = make_error(id, -32601, f"Method not found: {method}")
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
            # notifications (no id) or known notification methods → no response
            continue

        resp = handler(id, params)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    run_stdio()
