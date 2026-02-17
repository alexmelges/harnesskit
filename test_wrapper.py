"""Tests for hk_wrapper.py"""

import os
import json
import tempfile
import pytest
from hk_wrapper import HarnessKit, EditResponse


@pytest.fixture
def hk():
    return HarnessKit()


@pytest.fixture
def tmp_file(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("def hello():\n    print('hello world')\n\ndef goodbye():\n    print('goodbye')\n")
    return str(f)


@pytest.fixture
def tmp_js(tmp_path):
    f = tmp_path / "test.js"
    f.write_text("function greet() {\n  console.log('hi');\n}\n")
    return str(f)


class TestEdit:
    def test_exact_match(self, hk, tmp_file):
        result = hk.edit(tmp_file, "def hello():", "def greet():")
        assert result.success
        assert result.match_type == "exact"
        assert result.similarity == 1.0
        with open(tmp_file) as f:
            assert "def greet():" in f.read()

    def test_whitespace_match(self, hk, tmp_file):
        # Extra spaces — should still match via whitespace normalization
        result = hk.edit(tmp_file, "def  hello( ):", "def greet():")
        assert result.success
        assert result.match_type in ("whitespace", "exact")

    def test_fuzzy_match(self, hk, tmp_file):
        # Slightly wrong text — fuzzy should catch it
        result = hk.edit(tmp_file, "def hello():\n    print('hello worl')", "def hello():\n    print('hello universe')")
        assert result.success
        assert result.match_type in ("fuzzy", "line_fuzzy", "exact")

    def test_no_match(self, hk, tmp_file):
        result = hk.edit(tmp_file, "this text does not exist anywhere", "replacement")
        assert not result.success
        assert result.error is not None

    def test_file_not_found(self, hk):
        result = hk.edit("/nonexistent/file.py", "old", "new")
        assert not result.success
        assert result.error is not None

    def test_with_diff(self, hk, tmp_file):
        result = hk.edit(tmp_file, "def hello():", "def greet():", show_diff=True)
        assert result.success
        # Diff computation from wrapper is best-effort
        assert result.match_type == "exact"

    def test_with_validation(self, hk, tmp_file):
        result = hk.edit(tmp_file, "def hello():", "def greet():", validate_after=True)
        assert result.success
        assert result.validation is not None


class TestCreate:
    def test_create_new_file(self, hk, tmp_path):
        path = str(tmp_path / "new_file.py")
        result = hk.create(path, "print('created')\n")
        assert result.success
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "print('created')\n"

    def test_create_with_dirs(self, hk, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "file.py")
        result = hk.create(path, "# new\n")
        assert result.success
        assert os.path.exists(path)


class TestValidate:
    def test_valid_python(self, hk, tmp_file):
        valid, msg = hk.validate(tmp_file)
        assert valid
        assert msg is None

    def test_invalid_python(self, hk, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n")
        valid, msg = hk.validate(str(f))
        assert not valid
        assert msg is not None

    def test_file_not_found(self, hk):
        valid, msg = hk.validate("/nonexistent.py")
        assert not valid


class TestDiff:
    def test_diff_preview(self, hk, tmp_file):
        d = hk.diff(tmp_file, "def hello():", "def greet():")
        assert d is not None
        assert "hello" in d
        # File should NOT be modified
        with open(tmp_file) as f:
            assert "def hello():" in f.read()

    def test_diff_no_match(self, hk, tmp_file):
        d = hk.diff(tmp_file, "nonexistent text", "replacement")
        assert d is None


class TestBatchEdit:
    def test_batch_success(self, hk, tmp_file):
        edits = [
            {"file": tmp_file, "old_text": "def hello():", "new_text": "def greet():"},
            {"file": tmp_file, "old_text": "def goodbye():", "new_text": "def farewell():"},
        ]
        results = hk.batch_edit(edits)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_batch_stops_on_failure(self, hk, tmp_file):
        edits = [
            {"file": tmp_file, "old_text": "nonexistent", "new_text": "x"},
            {"file": tmp_file, "old_text": "def hello():", "new_text": "def greet():"},
        ]
        results = hk.batch_edit(edits)
        assert len(results) == 1  # stopped after first failure
        assert not results[0].success


class TestResponse:
    def test_to_dict(self, hk, tmp_file):
        result = hk.edit(tmp_file, "def hello():", "def greet():")
        d = result.to_dict()
        assert d["success"] is True
        assert d["file"] == tmp_file
        assert "match_type" in d

    def test_to_json(self, hk, tmp_file):
        result = hk.edit(tmp_file, "def hello():", "def greet():")
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["success"] is True


class TestSchemas:
    def test_anthropic_schema(self):
        schema = HarnessKit.anthropic_tool_schema()
        assert schema["name"] == "edit_file"
        assert "input_schema" in schema
        assert "file" in schema["input_schema"]["properties"]

    def test_openai_schema(self):
        schema = HarnessKit.openai_function_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "edit_file"
        assert "file" in schema["function"]["parameters"]["properties"]
