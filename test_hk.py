#!/usr/bin/env python3
"""Comprehensive tests for hk.py — HarnessKit fuzzy edit tool."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from hk import (
    AmbiguousMatchError,
    apply_edit,
    find_best_match,
    find_exact_matches,
    find_fuzzy_matches,
    find_line_fuzzy_matches,
    find_whitespace_matches,
    main,
    normalize_whitespace,
)


class TestNormalizeWhitespace(unittest.TestCase):
    def test_collapses_spaces(self):
        self.assertEqual(normalize_whitespace("a  b   c"), "a b c")

    def test_collapses_tabs_and_newlines(self):
        self.assertEqual(normalize_whitespace("a\t\nb"), "a b")

    def test_strips_edges(self):
        self.assertEqual(normalize_whitespace("  hello  "), "hello")


class TestExactMatch(unittest.TestCase):
    def test_single_match(self):
        matches = find_exact_matches("hello world", "world")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].start, 6)
        self.assertEqual(matches[0].end, 11)
        self.assertEqual(matches[0].confidence, 1.0)
        self.assertEqual(matches[0].match_type, "exact")

    def test_no_match(self):
        matches = find_exact_matches("hello world", "xyz")
        self.assertEqual(len(matches), 0)

    def test_multiple_matches(self):
        matches = find_exact_matches("abcabc", "abc")
        self.assertEqual(len(matches), 2)

    def test_multiline_exact(self):
        content = "line1\nline2\nline3\n"
        matches = find_exact_matches(content, "line2\nline3")
        self.assertEqual(len(matches), 1)


class TestWhitespaceMatch(unittest.TestCase):
    def test_extra_spaces(self):
        content = "def  hello( x,  y ):"
        matches = find_whitespace_matches(content, "def hello(x, y):")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].match_type, "whitespace")
        self.assertAlmostEqual(matches[0].confidence, 0.95)

    def test_tabs_vs_spaces(self):
        content = "if\tx > 0:"
        matches = find_whitespace_matches(content, "if x > 0:")
        self.assertEqual(len(matches), 1)

    def test_no_match(self):
        content = "def hello():"
        matches = find_whitespace_matches(content, "def goodbye():")
        self.assertEqual(len(matches), 0)


class TestFuzzyMatch(unittest.TestCase):
    def test_close_match(self):
        content = "def hello_world():\n    print('hello')\n"
        old = "def hello_world():\n    print('helo')\n"
        matches = find_fuzzy_matches(content, old, 0.8)
        self.assertEqual(len(matches), 1)
        self.assertGreaterEqual(matches[0].confidence, 0.8)
        self.assertEqual(matches[0].match_type, "fuzzy")

    def test_below_threshold(self):
        content = "completely different text"
        old = "nothing like it at all here xyz"
        matches = find_fuzzy_matches(content, old, 0.8)
        self.assertEqual(len(matches), 0)


class TestLineFuzzyMatch(unittest.TestCase):
    def test_line_match_with_minor_diffs(self):
        content = "def foo():\n    x = 1\n    y = 2\n    return x + y\n"
        old = "def foo():\n    x = 1\n    y = 3\n    return x + y\n"
        matches = find_line_fuzzy_matches(content, old, 0.8)
        self.assertEqual(len(matches), 1)
        self.assertGreaterEqual(matches[0].confidence, 0.8)
        self.assertEqual(matches[0].match_type, "line_fuzzy")

    def test_no_line_match(self):
        content = "aaa\nbbb\nccc\n"
        old = "xxx\nyyy\nzzz\n"
        matches = find_line_fuzzy_matches(content, old, 0.8)
        self.assertEqual(len(matches), 0)


class TestFindBestMatch(unittest.TestCase):
    def test_prefers_exact(self):
        content = "def hello(): pass"
        match = find_best_match(content, "hello()")
        self.assertEqual(match.match_type, "exact")
        self.assertEqual(match.confidence, 1.0)

    def test_falls_back_to_whitespace(self):
        content = "def  hello( ):"
        match = find_best_match(content, "def hello():")
        self.assertEqual(match.match_type, "whitespace")

    def test_falls_back_to_fuzzy(self):
        content = "def hello_world():\n    print('hello')\n"
        old = "def hello_world():\n    print('helo')\n"
        match = find_best_match(content, old, threshold=0.8)
        self.assertIsNotNone(match)
        self.assertIn(match.match_type, ("fuzzy", "line_fuzzy"))

    def test_no_match_returns_none(self):
        content = "abc"
        match = find_best_match(content, "completely unrelated long string xyz", threshold=0.8)
        self.assertIsNone(match)

    def test_ambiguous_raises(self):
        content = "foo bar\nfoo bar\n"
        with self.assertRaises(AmbiguousMatchError):
            find_best_match(content, "foo bar")


class TestApplyEdit(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _write(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def test_exact_apply(self):
        path = self._write("test.py", "def hello():\n    pass\n")
        result = apply_edit(path, "pass", "return 42")
        self.assertEqual(result.status, "applied")
        self.assertEqual(result.match_type, "exact")
        with open(path) as f:
            self.assertIn("return 42", f.read())

    def test_whitespace_apply(self):
        path = self._write("test.py", "def  hello( x,  y ):\n    pass\n")
        result = apply_edit(path, "def hello(x, y):", "def hello(a, b):")
        self.assertEqual(result.status, "applied")
        self.assertEqual(result.match_type, "whitespace")
        with open(path) as f:
            content = f.read()
            self.assertIn("def hello(a, b):", content)

    def test_fuzzy_apply(self):
        path = self._write("test.py", "def hello_world():\n    print('hello')\n")
        result = apply_edit(
            path,
            "def hello_world():\n    print('helo')\n",
            "def hello_world():\n    print('hello world')\n",
        )
        self.assertEqual(result.status, "applied")
        self.assertIn(result.match_type, ("fuzzy", "line_fuzzy"))
        with open(path) as f:
            self.assertIn("hello world", f.read())

    def test_missing_file(self):
        result = apply_edit("/nonexistent/path/file.py", "old", "new")
        self.assertEqual(result.status, "error")
        self.assertIn("not found", result.error.lower())

    def test_no_match(self):
        path = self._write("test.py", "hello world")
        result = apply_edit(path, "completely unrelated long string xyz", "new")
        self.assertEqual(result.status, "no_match")

    def test_ambiguous(self):
        path = self._write("test.py", "foo bar\nfoo bar\n")
        result = apply_edit(path, "foo bar", "baz")
        self.assertEqual(result.status, "ambiguous")

    def test_dry_run_does_not_modify(self):
        original = "def hello():\n    pass\n"
        path = self._write("test.py", original)
        result = apply_edit(path, "pass", "return 42", dry_run=True)
        self.assertEqual(result.status, "applied")
        with open(path) as f:
            self.assertEqual(f.read(), original)

    def test_threshold_parameter(self):
        path = self._write("test.py", "abcdef")
        # Very high threshold — a fuzzy match for slightly different text should fail
        result = apply_edit(path, "abcxyz", "new", threshold=0.99)
        self.assertIn(result.status, ("no_match", "error"))


class TestCLIMain(unittest.TestCase):
    """Test the main() function with argv simulation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _write(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def test_cli_file_old_new(self):
        path = self._write("test.py", "hello world\n")
        code = main(["apply", "--file", path, "--old", "hello", "--new", "goodbye"])
        self.assertEqual(code, 0)
        with open(path) as f:
            self.assertIn("goodbye", f.read())

    def test_cli_dry_run(self):
        path = self._write("test.py", "hello world\n")
        code = main(["apply", "--file", path, "--old", "hello", "--new", "goodbye", "--dry-run"])
        self.assertEqual(code, 0)
        with open(path) as f:
            self.assertIn("hello", f.read())

    def test_cli_edit_json_file(self):
        target = self._write("test.py", "hello world\n")
        edit_data = {"file": target, "old_text": "hello", "new_text": "goodbye"}
        edit_path = self._write("edit.json", json.dumps(edit_data))
        code = main(["apply", "--edit", edit_path])
        self.assertEqual(code, 0)
        with open(target) as f:
            self.assertIn("goodbye", f.read())

    def test_cli_threshold(self):
        path = self._write("test.py", "abcdef\n")
        code = main([
            "apply", "--file", path,
            "--old", "abcxyz", "--new", "new",
            "--threshold", "0.99",
        ])
        self.assertEqual(code, 1)

    def test_cli_exit_code_ambiguous(self):
        path = self._write("test.py", "foo bar\nfoo bar\n")
        code = main(["apply", "--file", path, "--old", "foo bar", "--new", "baz"])
        self.assertEqual(code, 2)

    def test_cli_exit_code_no_match(self):
        path = self._write("test.py", "hello world\n")
        code = main([
            "apply", "--file", path,
            "--old", "completely unrelated long string xyz",
            "--new", "new",
        ])
        self.assertEqual(code, 1)


class TestStdinJSON(unittest.TestCase):
    """Test --stdin mode via subprocess."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _write(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def test_stdin_single_edit(self):
        target = self._write("test.py", "hello world\n")
        edit_data = json.dumps({
            "file": target,
            "old_text": "hello",
            "new_text": "goodbye",
        })
        result = subprocess.run(
            [sys.executable, "hk.py", "apply", "--stdin"],
            input=edit_data,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertEqual(result.returncode, 0)
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "applied")
        with open(target) as f:
            self.assertIn("goodbye", f.read())

    def test_stdin_multi_edit(self):
        target_a = self._write("a.py", "aaa\n")
        target_b = self._write("b.py", "bbb\n")
        edit_data = json.dumps({
            "edits": [
                {"file": target_a, "old_text": "aaa", "new_text": "AAA"},
                {"file": target_b, "old_text": "bbb", "new_text": "BBB"},
            ]
        })
        result = subprocess.run(
            [sys.executable, "hk.py", "apply", "--stdin"],
            input=edit_data,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertEqual(result.returncode, 0)
        output = json.loads(result.stdout)
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), 2)
        with open(target_a) as f:
            self.assertEqual(f.read(), "AAA\n")
        with open(target_b) as f:
            self.assertEqual(f.read(), "BBB\n")


class TestMultiEdit(unittest.TestCase):
    """Test multi-edit via JSON file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _write(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def test_multi_edit_json_file(self):
        target_a = self._write("a.py", "aaa\n")
        target_b = self._write("b.py", "bbb\n")
        edit_data = {
            "edits": [
                {"file": target_a, "old_text": "aaa", "new_text": "AAA"},
                {"file": target_b, "old_text": "bbb", "new_text": "BBB"},
            ]
        }
        edit_path = self._write("edits.json", json.dumps(edit_data))
        code = main(["apply", "--edit", edit_path])
        self.assertEqual(code, 0)
        with open(target_a) as f:
            self.assertEqual(f.read(), "AAA\n")
        with open(target_b) as f:
            self.assertEqual(f.read(), "BBB\n")

    def test_multi_edit_partial_failure(self):
        """One edit succeeds, one fails — exit code reflects worst."""
        target = self._write("a.py", "hello\n")
        edit_data = {
            "edits": [
                {"file": target, "old_text": "hello", "new_text": "goodbye"},
                {"file": "/nonexistent/file.py", "old_text": "x", "new_text": "y"},
            ]
        }
        edit_path = self._write("edits.json", json.dumps(edit_data))
        code = main(["apply", "--edit", edit_path])
        self.assertEqual(code, 1)


class TestOutputFormat(unittest.TestCase):
    """Test that CLI output is valid JSON with expected fields."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _write(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def test_success_output_fields(self):
        target = self._write("test.py", "hello world\n")
        result = subprocess.run(
            [sys.executable, "hk.py", "apply", "--file", target,
             "--old", "hello", "--new", "goodbye"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "applied")
        self.assertIn("file", output)
        self.assertIn("match_type", output)
        self.assertIn("confidence", output)
        self.assertIn("matched_text", output)

    def test_error_output_fields(self):
        result = subprocess.run(
            [sys.executable, "hk.py", "apply", "--file", "/nonexistent/file.py",
             "--old", "x", "--new", "y"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "error")
        self.assertIn("error", output)


if __name__ == "__main__":
    unittest.main()
