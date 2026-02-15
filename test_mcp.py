"""Tests for hk_mcp.py — MCP server for HarnessKit."""

import json
import os
import subprocess
import sys
import tempfile

import pytest


def mcp_call(*messages):
    """Send JSON-RPC messages to MCP server, return parsed responses."""
    input_str = "\n".join(json.dumps(m) for m in messages) + "\n"
    proc = subprocess.run(
        [sys.executable, "hk_mcp.py"],
        input=input_str, capture_output=True, text=True,
        cwd=os.path.dirname(__file__),
    )
    lines = [l for l in proc.stdout.strip().split("\n") if l.strip()]
    return [json.loads(l) for l in lines]


def init_msg(id=1):
    return {"jsonrpc": "2.0", "id": id, "method": "initialize", "params": {}}


def tool_call(id, name, arguments):
    return {"jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": {"name": name, "arguments": arguments}}


class TestInitialize:
    def test_returns_server_info(self):
        [resp] = mcp_call(init_msg())
        assert resp["result"]["serverInfo"]["name"] == "harnesskit"
        assert resp["result"]["protocolVersion"] == "2024-11-05"

    def test_has_tools_capability(self):
        [resp] = mcp_call(init_msg())
        assert "tools" in resp["result"]["capabilities"]


class TestToolsList:
    def test_lists_three_tools(self):
        resps = mcp_call(init_msg(), {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = resps[1]["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {"harnesskit_apply", "harnesskit_apply_batch", "harnesskit_match", "harnesskit_create", "harnesskit_validate"}


class TestApply:
    def test_exact_match(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        resps = mcp_call(init_msg(), tool_call(2, "harnesskit_apply", {
            "file": str(f), "old_text": "hello world", "new_text": "goodbye world",
        }))
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "applied"
        assert result["match_type"] == "exact"
        assert f.read_text() == "goodbye world"

    def test_whitespace_match(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        resps = mcp_call(init_msg(), tool_call(2, "harnesskit_apply", {
            "file": str(f), "old_text": "hello  world", "new_text": "goodbye",
        }))
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "applied"
        assert result["match_type"] == "whitespace"

    def test_dry_run(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("original content")
        resps = mcp_call(init_msg(), tool_call(2, "harnesskit_apply", {
            "file": str(f), "old_text": "original content", "new_text": "changed",
            "dry_run": True,
        }))
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "applied"
        assert f.read_text() == "original content"  # unchanged

    def test_file_not_found(self):
        resps = mcp_call(init_msg(), tool_call(2, "harnesskit_apply", {
            "file": "/nonexistent/file.txt", "old_text": "x", "new_text": "y",
        }))
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "error"
        assert resps[1]["result"]["isError"] is True

    def test_no_match(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        resps = mcp_call(init_msg(), tool_call(2, "harnesskit_apply", {
            "file": str(f), "old_text": "completely different text that won't match anything",
            "new_text": "replacement",
        }))
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "no_match"
        assert resps[1]["result"]["isError"] is True


class TestBatch:
    def test_multiple_edits(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("foo bar")
        f2.write_text("baz qux")
        resps = mcp_call(init_msg(), tool_call(2, "harnesskit_apply_batch", {
            "edits": [
                {"file": str(f1), "old_text": "foo bar", "new_text": "changed1"},
                {"file": str(f2), "old_text": "baz qux", "new_text": "changed2"},
            ],
        }))
        results = json.loads(resps[1]["result"]["content"][0]["text"])
        assert len(results) == 2
        assert all(r["status"] == "applied" for r in results)
        assert f1.read_text() == "changed1"
        assert f2.read_text() == "changed2"


class TestMatch:
    def test_finds_match(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        resps = mcp_call(init_msg(), tool_call(2, "harnesskit_match", {
            "file": str(f), "old_text": "hello world",
        }))
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "found"
        assert result["match_type"] == "exact"
        assert result["confidence"] == 1.0

    def test_no_match(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        resps = mcp_call(init_msg(), tool_call(2, "harnesskit_match", {
            "file": str(f), "old_text": "zzzzzzzzzzzzzzzzz",
        }))
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "no_match"


class TestErrors:
    def test_unknown_tool(self):
        resps = mcp_call(init_msg(), tool_call(2, "nonexistent_tool", {}))
        assert "error" in resps[1]
        assert resps[1]["error"]["code"] == -32601

    def test_unknown_method(self):
        resps = mcp_call(init_msg(), {"jsonrpc": "2.0", "id": 2, "method": "fake/method", "params": {}})
        assert "error" in resps[1]

    def test_parse_error(self):
        proc = subprocess.run(
            [sys.executable, "hk_mcp.py"],
            input="not json\n", capture_output=True, text=True,
            cwd=os.path.dirname(__file__),
        )
        resp = json.loads(proc.stdout.strip())
        assert resp["error"]["code"] == -32700


class TestMCPCreate:
    def test_create_file(self, tmp_path):
        f = str(tmp_path / "new.py")
        resps = mcp_call(
            init_msg(),
            tool_call(2, "harnesskit_create", {"file": f, "content": "x = 1\n"}),
        )
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "created"
        assert open(f).read() == "x = 1\n"

    def test_create_existing_fails(self, tmp_path):
        f = str(tmp_path / "exist.py")
        open(f, 'w').write("old")
        resps = mcp_call(
            init_msg(),
            tool_call(2, "harnesskit_create", {"file": f, "content": "new"}),
        )
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "error"
        assert resps[1]["result"]["isError"]

    def test_create_with_validate(self, tmp_path):
        f = str(tmp_path / "bad.py")
        resps = mcp_call(
            init_msg(),
            tool_call(2, "harnesskit_create", {"file": f, "content": "def(\n", "validate": True}),
        )
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "validation_error"


class TestMCPValidate:
    def test_validate_valid(self, tmp_path):
        f = str(tmp_path / "good.py")
        open(f, 'w').write("x = 1\n")
        resps = mcp_call(
            init_msg(),
            tool_call(2, "harnesskit_validate", {"file": f}),
        )
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "valid"
        assert not resps[1]["result"]["isError"]

    def test_validate_invalid(self, tmp_path):
        f = str(tmp_path / "bad.json")
        open(f, 'w').write("{bad json")
        resps = mcp_call(
            init_msg(),
            tool_call(2, "harnesskit_validate", {"file": f}),
        )
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "invalid"
        assert resps[1]["result"]["isError"]


class TestMCPApplyWithValidate:
    def test_apply_validate_rollback(self, tmp_path):
        f = str(tmp_path / "test.py")
        open(f, 'w').write("x = 1\nprint(x)\n")
        resps = mcp_call(
            init_msg(),
            tool_call(2, "harnesskit_apply", {
                "file": f, "old_text": "x = 1", "new_text": "x = 1 +",
                "validate": True,
            }),
        )
        result = json.loads(resps[1]["result"]["content"][0]["text"])
        assert result["status"] == "validation_error"
        # File unchanged
        assert open(f).read() == "x = 1\nprint(x)\n"
