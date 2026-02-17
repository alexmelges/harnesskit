#!/usr/bin/env python3
"""Comprehensive tests for hk.py — HarnessKit fuzzy edit tool."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hk

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
        self.original_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.original_cwd)
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
        self.original_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.original_cwd)
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


class TestXMLInput(unittest.TestCase):
    """Test XML edit input format."""

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

    def test_xml_single_edit_stdin(self):
        target = self._write("test.py", "def hello():\n    pass\n")
        xml_input = f'<edit file="{target}"><old>def hello():</old><new>def hello(name="world"):</new></edit>'
        result = subprocess.run(
            [sys.executable, "hk.py", "apply", "--stdin"],
            input=xml_input,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertEqual(result.returncode, 0)
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "applied")
        with open(target) as f:
            self.assertIn('def hello(name="world"):', f.read())

    def test_xml_multi_edit_stdin(self):
        target_a = self._write("a.py", "aaa\n")
        target_b = self._write("b.py", "bbb\n")
        xml_input = f'''<edits>
  <edit file="{target_a}"><old>aaa</old><new>AAA</new></edit>
  <edit file="{target_b}"><old>bbb</old><new>BBB</new></edit>
</edits>'''
        result = subprocess.run(
            [sys.executable, "hk.py", "apply", "--stdin"],
            input=xml_input,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertEqual(result.returncode, 0)
        with open(target_a) as f:
            self.assertEqual(f.read(), "AAA\n")
        with open(target_b) as f:
            self.assertEqual(f.read(), "BBB\n")

    def test_xml_edit_file(self):
        target = self._write("test.py", "old code\n")
        xml_content = f'<edit file="{target}"><old>old code</old><new>new code</new></edit>'
        edit_file = self._write("edits.xml", xml_content)
        code = main(["apply", "--edit", edit_file])
        self.assertEqual(code, 0)
        with open(target) as f:
            self.assertEqual(f.read(), "new code\n")

    def test_xml_multiline_edit(self):
        target = self._write("test.py", "def foo():\n    x = 1\n    return x\n")
        xml_input = f'''<edit file="{target}">
<old>
def foo():
    x = 1
    return x
</old>
<new>
def foo(multiplier=1):
    x = 1 * multiplier
    return x
</new>
</edit>'''
        result = subprocess.run(
            [sys.executable, "hk.py", "apply", "--stdin"],
            input=xml_input,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertEqual(result.returncode, 0)
        with open(target) as f:
            content = f.read()
            self.assertIn("multiplier", content)

    def test_xml_missing_file_attr(self):
        xml_input = '<edit><old>x</old><new>y</new></edit>'
        result = subprocess.run(
            [sys.executable, "hk.py", "apply", "--stdin"],
            input=xml_input,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertNotEqual(result.returncode, 0)

    def test_xml_path_attr_alias(self):
        """Test that 'path' attribute works as alias for 'file'."""
        target = self._write("test.py", "hello\n")
        xml_input = f'<edit path="{target}"><old>hello</old><new>goodbye</new></edit>'
        result = subprocess.run(
            [sys.executable, "hk.py", "apply", "--stdin"],
            input=xml_input,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertEqual(result.returncode, 0)
        with open(target) as f:
            self.assertEqual(f.read(), "goodbye\n")

    def test_xml_invalid_format(self):
        result = subprocess.run(
            [sys.executable, "hk.py", "apply", "--stdin"],
            input="<broken xml",
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertNotEqual(result.returncode, 0)


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


class TestValidateSyntax(unittest.TestCase):
    """Tests for syntax validation."""

    def test_valid_python(self):
        valid, err = hk.validate_syntax("test.py", "x = 1\nprint(x)\n")
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_invalid_python(self):
        valid, err = hk.validate_syntax("test.py", "def foo(\n  x = 1\n")
        self.assertFalse(valid)
        self.assertIn("syntax error", err.lower())

    def test_valid_json(self):
        valid, err = hk.validate_syntax("test.json", '{"key": "value"}')
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_invalid_json(self):
        valid, err = hk.validate_syntax("test.json", '{"key": }')
        self.assertFalse(valid)
        self.assertIn("JSON", err)

    def test_valid_xml(self):
        valid, err = hk.validate_syntax("test.xml", "<root><child/></root>")
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_invalid_xml(self):
        valid, err = hk.validate_syntax("test.xml", "<root><child></root>")
        self.assertFalse(valid)
        self.assertIn("parse error", err.lower())

    def test_valid_js(self):
        valid, err = hk.validate_syntax("test.js", "function foo() { return 1; }\n")
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_invalid_js_unmatched_brace(self):
        valid, err = hk.validate_syntax("test.js", "function foo() { return 1;\n")
        self.assertFalse(valid)
        self.assertIn("Unclosed", err)

    def test_invalid_js_unclosed_string(self):
        valid, err = hk.validate_syntax("test.js", 'var x = "hello\nvar y = 1;\n')
        self.assertFalse(valid)
        self.assertIn("string", err.lower())

    def test_js_template_literal_multiline(self):
        valid, err = hk.validate_syntax("test.js", "var x = `hello\nworld`;\n")
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_js_comments_ignored(self):
        valid, err = hk.validate_syntax("test.js", "// { unclosed\nvar x = 1;\n")
        self.assertTrue(valid)

    def test_unknown_extension(self):
        valid, err = hk.validate_syntax("test.txt", "anything goes")
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_typescript_validates(self):
        valid, err = hk.validate_syntax("test.tsx", "const App = () => { return (<div></div>); };\n")
        self.assertTrue(valid)


class TestValidateOnApply(unittest.TestCase):
    """Tests for --validate flag on apply."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_validate_pass(self):
        f = os.path.join(self.tmpdir, "test.py")
        with open(f, 'w') as fh:
            fh.write("x = 1\nprint(x)\n")
        result = hk.apply_edit(f, "x = 1", "x = 2", validate=True)
        self.assertEqual(result.status, "applied")
        self.assertTrue(result.validated)

    def test_validate_fail_rollback(self):
        f = os.path.join(self.tmpdir, "test.py")
        original = "x = 1\nprint(x)\n"
        with open(f, 'w') as fh:
            fh.write(original)
        # This edit introduces a syntax error
        result = hk.apply_edit(f, "x = 1", "x = 1 +", validate=True)
        self.assertEqual(result.status, "validation_error")
        self.assertIn("syntax error", result.error.lower())
        # File should be unchanged (rollback)
        with open(f) as fh:
            self.assertEqual(fh.read(), original)

    def test_validate_json_fail_rollback(self):
        f = os.path.join(self.tmpdir, "test.json")
        original = '{"key": "value"}'
        with open(f, 'w') as fh:
            fh.write(original)
        result = hk.apply_edit(f, '"value"', '"value",', validate=True)
        self.assertEqual(result.status, "validation_error")
        with open(f) as fh:
            self.assertEqual(fh.read(), original)


class TestDiffOutput(unittest.TestCase):
    """Tests for diff computation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_diff_present_on_apply(self):
        f = os.path.join(self.tmpdir, "test.py")
        with open(f, 'w') as fh:
            fh.write("x = 1\n")
        result = hk.apply_edit(f, "x = 1", "x = 2")
        self.assertIsNotNone(result.diff)
        self.assertIn("-x = 1", result.diff)
        self.assertIn("+x = 2", result.diff)

    def test_diff_in_result_dict(self):
        f = os.path.join(self.tmpdir, "test.py")
        with open(f, 'w') as fh:
            fh.write("x = 1\n")
        result = hk.apply_edit(f, "x = 1", "x = 2")
        d = hk.result_to_dict(result)
        self.assertIn("diff", d)
        self.assertIn("-x = 1", d["diff"])

    def test_diff_unified_format(self):
        f = os.path.join(self.tmpdir, "test.py")
        with open(f, 'w') as fh:
            fh.write("a = 1\nb = 2\nc = 3\n")
        result = hk.apply_edit(f, "b = 2", "b = 99")
        self.assertIn("---", result.diff)
        self.assertIn("+++", result.diff)
        self.assertIn("@@", result.diff)


class TestCreateFile(unittest.TestCase):
    """Tests for create file command."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_create_new_file(self):
        f = os.path.join(self.tmpdir, "new.py")
        result = hk.create_file(f, "x = 1\n")
        self.assertEqual(result.status, "created")
        with open(f) as fh:
            self.assertEqual(fh.read(), "x = 1\n")

    def test_create_existing_fails(self):
        f = os.path.join(self.tmpdir, "exist.py")
        with open(f, 'w') as fh:
            fh.write("old")
        result = hk.create_file(f, "new")
        self.assertEqual(result.status, "error")
        self.assertIn("already exists", result.error)
        with open(f) as fh:
            self.assertEqual(fh.read(), "old")

    def test_create_force_overwrite(self):
        f = os.path.join(self.tmpdir, "exist.py")
        with open(f, 'w') as fh:
            fh.write("old")
        result = hk.create_file(f, "new", force=True)
        self.assertEqual(result.status, "created")
        with open(f) as fh:
            self.assertEqual(fh.read(), "new")

    def test_create_with_validation_pass(self):
        f = os.path.join(self.tmpdir, "valid.py")
        result = hk.create_file(f, "x = 1\n", validate=True)
        self.assertEqual(result.status, "created")
        self.assertTrue(result.validated)

    def test_create_with_validation_fail(self):
        f = os.path.join(self.tmpdir, "invalid.py")
        result = hk.create_file(f, "def foo(\n", validate=True)
        self.assertEqual(result.status, "validation_error")
        self.assertFalse(os.path.exists(f))

    def test_create_nested_dirs(self):
        f = os.path.join(self.tmpdir, "a", "b", "c", "test.py")
        result = hk.create_file(f, "x = 1\n")
        self.assertEqual(result.status, "created")
        self.assertTrue(os.path.exists(f))


class TestValidateCommand(unittest.TestCase):
    """Tests for the validate CLI command."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_validate_valid_file(self):
        f = os.path.join(self.tmpdir, "test.py")
        with open(f, 'w') as fh:
            fh.write("x = 1\n")
        exit_code = hk.main(["validate", f])
        self.assertEqual(exit_code, 0)

    def test_validate_invalid_file(self):
        f = os.path.join(self.tmpdir, "test.py")
        with open(f, 'w') as fh:
            fh.write("def foo(\n")
        exit_code = hk.main(["validate", f])
        self.assertEqual(exit_code, 1)


class TestCRLFHandling(unittest.TestCase):
    """Test CRLF line ending preservation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_crlf_preserved(self):
        """CRLF files should maintain CRLF after edits."""
        f = os.path.join(self.tmpdir, "test.sh")
        with open(f, 'wb') as fh:
            fh.write(b"line1\r\nline2\r\nline3\r\n")
        result = hk.apply_edit(f, "line2", "replaced")
        self.assertEqual(result.status, "applied")
        with open(f, 'rb') as fh:
            content = fh.read()
        self.assertIn(b"\r\n", content)
        self.assertEqual(content, b"line1\r\nreplaced\r\nline3\r\n")

    def test_lf_not_converted(self):
        """LF files should stay LF."""
        f = os.path.join(self.tmpdir, "test.sh")
        with open(f, 'wb') as fh:
            fh.write(b"line1\nline2\nline3\n")
        result = hk.apply_edit(f, "line2", "replaced")
        self.assertEqual(result.status, "applied")
        with open(f, 'rb') as fh:
            content = fh.read()
        self.assertNotIn(b"\r\n", content)


class TestTabSpaceMapping(unittest.TestCase):
    """Test tabs-to-spaces and spaces-to-tabs indent mapping."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_spaces_to_tabs(self):
        """File uses tabs, edit uses spaces — should convert."""
        f = os.path.join(self.tmpdir, "test.py")
        with open(f, 'w') as fh:
            fh.write("class Foo:\n\tdef bar(self):\n\t\treturn 1\n")
        result = hk.apply_edit(f, "class Foo:\n    def bar(self):\n        return 1",
                                   "class Foo:\n    def bar(self):\n        return 2")
        self.assertEqual(result.status, "applied")
        with open(f, 'r') as fh:
            content = fh.read()
        self.assertIn("\treturn 2", content)


class TestCSSTokenMapping(unittest.TestCase):
    """Test hyphenated token mapping (CSS class renames)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_css_class_rename(self):
        """Stale CSS class names should be mapped to actual names."""
        f = os.path.join(self.tmpdir, "test.css")
        with open(f, 'w') as fh:
            fh.write(".my-card {\n    color: red;\n}\n")
        result = hk.apply_edit(f, ".card {\n    color: red;\n}",
                                   ".card {\n    color: blue;\n}")
        self.assertEqual(result.status, "applied")
        with open(f, 'r') as fh:
            content = fh.read()
        self.assertIn(".my-card", content)
        self.assertIn("blue", content)


class TestBenchmarkCommand(unittest.TestCase):
    """Test the benchmark subcommand."""

    def test_benchmark_runs(self):
        """Benchmark command should run and report results."""
        bench_dir = os.path.join(os.path.dirname(__file__), "benchmarks")
        if not os.path.isdir(bench_dir):
            self.skipTest("No benchmarks/ directory")
        result = hk.run_benchmark(bench_dir)
        self.assertIn("total", result)
        self.assertIn("passed", result)
        self.assertIn("pass_rate", result)
        self.assertGreater(result["total"], 0)
        self.assertEqual(result["pass_rate"], 100.0)

    def test_benchmark_cli_json(self):
        """Benchmark CLI should output JSON."""
        bench_dir = os.path.join(os.path.dirname(__file__), "benchmarks")
        if not os.path.isdir(bench_dir):
            self.skipTest("No benchmarks/ directory")
        exit_code = hk.main(["benchmark", "--json"])
        self.assertEqual(exit_code, 0)

    def test_benchmark_missing_dir(self):
        """Benchmark with missing dir should fail."""
        exit_code = hk.main(["benchmark", "--dir", "/nonexistent/path"])
        self.assertEqual(exit_code, 1)


class TestAtomicTransactions(unittest.TestCase):
    """Tests for --atomic flag on apply."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.tmpdir)

    def _write(self, name, content):
        path = os.path.join(self.tmpdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def test_atomic_all_succeed(self):
        """All edits succeed in atomic mode — files are modified."""
        a = self._write("a.py", "aaa\n")
        b = self._write("b.py", "bbb\n")
        edit_data = json.dumps({
            "edits": [
                {"file": a, "old_text": "aaa", "new_text": "AAA"},
                {"file": b, "old_text": "bbb", "new_text": "BBB"},
            ]
        })
        edit_path = self._write("edits.json", edit_data)
        code = main(["apply", "--edit", edit_path, "--atomic"])
        self.assertEqual(code, 0)
        with open(a) as f:
            self.assertEqual(f.read(), "AAA\n")
        with open(b) as f:
            self.assertEqual(f.read(), "BBB\n")

    def test_atomic_middle_fails_rollback(self):
        """Middle edit fails — ALL edits rolled back, exit code 3."""
        a = self._write("a.py", "aaa\n")
        b = self._write("b.py", "bbb\n")
        c = self._write("c.py", "ccc\n")
        edit_data = json.dumps({
            "edits": [
                {"file": a, "old_text": "aaa", "new_text": "AAA"},
                {"file": b, "old_text": "NOMATCH_xyz_longstring", "new_text": "BBB"},
                {"file": c, "old_text": "ccc", "new_text": "CCC"},
            ]
        })
        edit_path = self._write("edits.json", edit_data)
        code = main(["apply", "--edit", edit_path, "--atomic"])
        self.assertEqual(code, 3)
        # All files should be restored to original
        with open(a) as f:
            self.assertEqual(f.read(), "aaa\n")
        with open(b) as f:
            self.assertEqual(f.read(), "bbb\n")
        with open(c) as f:
            self.assertEqual(f.read(), "ccc\n")

    def test_atomic_first_fails_rollback(self):
        """First edit fails — nothing changed, exit code 3."""
        a = self._write("a.py", "aaa\n")
        b = self._write("b.py", "bbb\n")
        edit_data = json.dumps({
            "edits": [
                {"file": a, "old_text": "NOMATCH_xyz_longstring", "new_text": "AAA"},
                {"file": b, "old_text": "bbb", "new_text": "BBB"},
            ]
        })
        edit_path = self._write("edits.json", edit_data)
        code = main(["apply", "--edit", edit_path, "--atomic"])
        self.assertEqual(code, 3)
        with open(a) as f:
            self.assertEqual(f.read(), "aaa\n")
        with open(b) as f:
            self.assertEqual(f.read(), "bbb\n")

    def test_atomic_single_edit(self):
        """Single edit with --atomic works normally."""
        a = self._write("a.py", "hello\n")
        code = main(["apply", "--file", a, "--old", "hello", "--new", "goodbye", "--atomic"])
        self.assertEqual(code, 0)
        with open(a) as f:
            self.assertEqual(f.read(), "goodbye\n")

    def test_atomic_with_dry_run(self):
        """--atomic --dry-run doesn't modify files."""
        a = self._write("a.py", "hello\n")
        code = main(["apply", "--file", a, "--old", "hello", "--new", "goodbye",
                      "--atomic", "--dry-run"])
        self.assertEqual(code, 0)
        with open(a) as f:
            self.assertEqual(f.read(), "hello\n")

    def test_atomic_rollback_output_format(self):
        """Rolled-back atomic transaction outputs expected JSON."""
        a = self._write("a.py", "aaa\n")
        hk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hk.py")
        edit_data = json.dumps({
            "edits": [
                {"file": a, "old_text": "aaa", "new_text": "AAA"},
                {"file": "/nonexistent/file.py", "old_text": "x", "new_text": "y"},
            ]
        })
        result = subprocess.run(
            [sys.executable, hk_path, "apply", "--stdin", "--atomic"],
            input=edit_data,
            capture_output=True,
            text=True,
            cwd=self.tmpdir,
        )
        self.assertEqual(result.returncode, 3)
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "rolled_back")
        self.assertIn("failed_edit_index", output)
        self.assertIn("failed_edit", output)
        self.assertIn("edits_attempted", output)
        self.assertIn("edits_total", output)

    def test_atomic_creates_backups_on_success(self):
        """Successful atomic edits create backup entries with shared timestamp."""
        a = self._write("a.py", "aaa\n")
        b = self._write("b.py", "bbb\n")
        edit_data = json.dumps({
            "edits": [
                {"file": a, "old_text": "aaa", "new_text": "AAA"},
                {"file": b, "old_text": "bbb", "new_text": "BBB"},
            ]
        })
        edit_path = self._write("edits.json", edit_data)
        code = main(["apply", "--edit", edit_path, "--atomic"])
        self.assertEqual(code, 0)
        backups = hk._list_backups()
        self.assertEqual(len(backups), 2)
        # All backups should share the same timestamp
        timestamps = set(b["timestamp"] for b in backups)
        self.assertEqual(len(timestamps), 1)

    def test_atomic_no_backups_on_rollback(self):
        """Rolled-back atomic transaction doesn't create backups."""
        a = self._write("a.py", "aaa\n")
        edit_data = json.dumps({
            "edits": [
                {"file": a, "old_text": "aaa", "new_text": "AAA"},
                {"file": a, "old_text": "NOMATCH_xyz_longstring", "new_text": "BBB"},
            ]
        })
        edit_path = self._write("edits.json", edit_data)
        code = main(["apply", "--edit", edit_path, "--atomic"])
        self.assertEqual(code, 3)
        backups = hk._list_backups()
        self.assertEqual(len(backups), 0)


class TestUndoSystem(unittest.TestCase):
    """Tests for the undo subcommand."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.tmpdir)

    def _write(self, name, content):
        path = os.path.join(self.tmpdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def test_undo_single_edit(self):
        """Apply edit then undo — file restored to original."""
        path = self._write("test.py", "hello\n")
        code = main(["apply", "--file", path, "--old", "hello", "--new", "goodbye"])
        self.assertEqual(code, 0)
        with open(path) as f:
            self.assertEqual(f.read(), "goodbye\n")
        code = main(["undo"])
        self.assertEqual(code, 0)
        with open(path) as f:
            self.assertEqual(f.read(), "hello\n")

    def test_undo_list(self):
        """List shows available backups after an edit."""
        path = self._write("test.py", "hello\n")
        main(["apply", "--file", path, "--old", "hello", "--new", "goodbye"])
        backups = hk._list_backups()
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0]["operation"], "edit")

    def test_undo_list_empty(self):
        """List with no backups returns empty status."""
        code = main(["undo", "--list"])
        self.assertEqual(code, 0)

    def test_undo_clean(self):
        """Clean removes all backups."""
        path = self._write("test.py", "hello\n")
        main(["apply", "--file", path, "--old", "hello", "--new", "goodbye"])
        self.assertGreater(len(hk._list_backups()), 0)
        code = main(["undo", "--clean"])
        self.assertEqual(code, 0)
        self.assertEqual(len(hk._list_backups()), 0)

    def test_undo_auto_prune(self):
        """Auto-prune removes oldest backups when limit exceeded."""
        import time as time_mod
        original_max = hk.MAX_BACKUPS
        hk.MAX_BACKUPS = 3
        try:
            for i in range(5):
                p = self._write(f"f{i}.py", f"old{i}\n")
                hk._save_backup(p, "edit", content=f"old{i}\n".encode(),
                                timestamp=hk._make_timestamp())
                time_mod.sleep(0.01)
            backups = hk._list_backups()
            self.assertLessEqual(len(backups), 3)
        finally:
            hk.MAX_BACKUPS = original_max

    def test_undo_after_create(self):
        """Undo after create removes the newly created file."""
        path = os.path.join(self.tmpdir, "brand_new.py")
        code = main(["create", "--file", path, "--content", "x = 1\n"])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(path))
        code = main(["undo"])
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(path))

    def test_undo_after_create_force(self):
        """Undo after force-create restores original file content."""
        path = self._write("exist.py", "original\n")
        code = main(["create", "--file", path, "--content", "overwritten\n", "--force"])
        self.assertEqual(code, 0)
        with open(path) as f:
            self.assertEqual(f.read(), "overwritten\n")
        code = main(["undo"])
        self.assertEqual(code, 0)
        with open(path) as f:
            self.assertEqual(f.read(), "original\n")

    def test_undo_no_backups(self):
        """Undo with no backups returns error."""
        code = main(["undo"])
        self.assertEqual(code, 1)

    def test_undo_all_transaction(self):
        """Undo --all restores all files from the last atomic transaction."""
        a = self._write("a.py", "aaa\n")
        b = self._write("b.py", "bbb\n")
        edit_data = json.dumps({
            "edits": [
                {"file": a, "old_text": "aaa", "new_text": "AAA"},
                {"file": b, "old_text": "bbb", "new_text": "BBB"},
            ]
        })
        edit_path = self._write("edits.json", edit_data)
        code = main(["apply", "--edit", edit_path, "--atomic"])
        self.assertEqual(code, 0)
        with open(a) as f:
            self.assertEqual(f.read(), "AAA\n")
        with open(b) as f:
            self.assertEqual(f.read(), "BBB\n")
        code = main(["undo", "--all"])
        self.assertEqual(code, 0)
        with open(a) as f:
            self.assertEqual(f.read(), "aaa\n")
        with open(b) as f:
            self.assertEqual(f.read(), "bbb\n")


if __name__ == "__main__":
    unittest.main()
