# 🔧 HarnessKit

> **Fuzzy edit tool for LLM coding agents — never fail a `str_replace` again.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](#)

---

## The Problem

Every LLM coding agent has the same Achilles' heel: **edit application**.

When Claude, GPT, or any model tries to modify code, it generates an `old_text` → `new_text` pair. The tool then does an exact string match to find where to apply the change. And it fails. A lot.

- **Whitespace differences** — the model adds a space, drops a tab, or normalizes indentation
- **Minor hallucinations** — a variable name is slightly off, a comment is paraphrased
- **Format fragility** — diffs, patches, and line-number schemes all break in different ways

The result? Up to **50% edit failure rates** on non-native models. Every failed edit wastes a tool call, burns tokens on retries, and breaks agent flow.

## The Solution

HarnessKit (`hk`) is a drop-in edit tool that **fuzzy-matches** the old text before replacing it. It uses a 4-stage matching cascade:

1. **Exact match** — zero overhead when the model is precise
2. **Normalized whitespace** — catches the most common failure mode
3. **Sequence matching** — `difflib.SequenceMatcher` with configurable threshold (default 0.8)
4. **Line-by-line fuzzy** — finds the best contiguous block match for heavily drifted edits

Every edit returns a **confidence score** and **match type**, so your agent knows exactly how the edit was resolved.

## Quick Start

```bash
pip install harnesskit
```

Or just copy `hk.py` into your project — it's a single file, stdlib only.

### CLI Usage

```bash
# Direct arguments
hk apply --file app.py --old "def hello():\n    print('hi')" --new "def hello():\n    print('hello world')"

# JSON from stdin (perfect for tool_use integration)
echo '{"file": "app.py", "old_text": "def hello():", "new_text": "def greet():"}' | hk apply --stdin

# From a JSON file
hk apply --edit changes.json

# XML format (natural for Claude and other LLMs)
echo '<edit file="app.py"><old>def hello():</old><new>def greet():</new></edit>' | hk apply --stdin

# XML from file
hk apply --edit changes.xml

# Dry run — see what would change without writing
hk apply --file app.py --old "..." --new "..." --dry-run
```

### JSON Edit Format

```json
{
  "file": "path/to/file.py",
  "old_text": "def hello():\n    print('hi')",
  "new_text": "def hello():\n    print('hello world')"
}
```

Batch multiple edits:

```json
{
  "edits": [
    {"file": "a.py", "old_text": "...", "new_text": "..."},
    {"file": "b.py", "old_text": "...", "new_text": "..."}
  ]
}
```

### XML Edit Format

HarnessKit auto-detects XML input — ideal for LLMs that naturally output XML:

```xml
<edit file="path/to/file.py">
  <old>def hello():
    print('hi')</old>
  <new>def hello():
    print('hello world')</new>
</edit>
```

Batch multiple edits:

```xml
<edits>
  <edit file="a.py"><old>...</old><new>...</new></edit>
  <edit file="b.py"><old>...</old><new>...</new></edit>
</edits>
```

The `path` attribute works as an alias for `file`.

### Output

```json
{
  "status": "applied",
  "file": "app.py",
  "match_type": "fuzzy",
  "confidence": 0.92,
  "matched_text": "def hello():\n    print( 'hi' )"
}
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0`  | Edit applied successfully |
| `1`  | No match found |
| `2`  | Ambiguous — multiple matches |

## MCP Server

HarnessKit ships an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server for plug-and-play integration with any MCP-compatible agent.

### Quick Start

Add to your MCP client config (e.g. Claude Desktop, Cursor, etc.):

```json
{
  "mcpServers": {
    "harnesskit": {
      "command": "python3",
      "args": ["/path/to/hk_mcp.py"]
    }
  }
}
```

### Tools

| Tool | Description |
|------|-------------|
| `harnesskit_apply` | Apply a fuzzy edit to a file (supports `validate` param) |
| `harnesskit_apply_batch` | Apply multiple edits in one call |
| `harnesskit_match` | Preview the match without modifying (dry run) |
| `harnesskit_create` | Create a new file (with optional validation) |
| `harnesskit_validate` | Validate a file's syntax without modifying it |

Each tool returns the match type, confidence score, and matched text — giving the agent full visibility into how the edit was resolved.

### Example

```json
{
  "name": "harnesskit_apply",
  "arguments": {
    "file": "app.py",
    "old_text": "def hello():\n    print('hi')",
    "new_text": "def hello():\n    print('hello world')",
    "threshold": 0.8
  }
}
```

Response:
```json
{
  "status": "applied",
  "match_type": "whitespace",
  "confidence": 0.95
}
```

## Examples

The [`examples/`](examples/) directory contains complete, working integration examples:

| Example | Description |
|---|---|
| [`claude_agent.py`](examples/claude_agent.py) | Full Claude `tool_use` agent loop — give it a task and files, it edits them |
| [`openai_agent.py`](examples/openai_agent.py) | Same pattern with OpenAI function calling |
| [`batch_edit.py`](examples/batch_edit.py) | Apply edits from JSON or XML files (with `--dry-run` and `--validate`) |

```bash
# Claude agent: "add type hints to all functions in src/"
python examples/claude_agent.py --task "Add type hints" --files src/

# Batch edits from XML (natural LLM output)
echo '<edits><edit file="app.py"><old>def foo():</old><new>def foo() -> None:</new></edit></edits>' \
  | python examples/batch_edit.py --validate
```

## Integration

HarnessKit is designed to slot into any agent framework as the edit backend:

```python
import subprocess, json

def apply_edit(file, old_text, new_text):
    result = subprocess.run(
        ["hk", "apply", "--stdin"],
        input=json.dumps({"file": file, "old_text": old_text, "new_text": new_text}),
        capture_output=True, text=True
    )
    return json.loads(result.stdout)
```

Or import directly:

```python
from hk import apply_edit

result = apply_edit("app.py", old_text, new_text, threshold=0.8)
```

## Benchmarks

We tested HarnessKit against **50 realistic edit failure scenarios** across 6 categories — the kind that break `str_replace` and `apply_patch` in production agent workflows.

| Category | Cases | Pass Rate | Avg Confidence |
|---|---|---|---|
| **Whitespace Mismatch** (tabs/spaces, trailing, CRLF, vendor prefixes, commas) | 12 | **100%** | 0.950 |
| **Stale Context** (renames, decorators, type changes, docstrings, hooks) | 13 | **100%** | 0.937 |
| **Partial Match** (incomplete blocks, missing context, shell scripts) | 7 | **100%** | 1.000 |
| **Indentation Drift** (mixed tabs/spaces, YAML, Makefile, class methods) | 7 | **100%** | 0.950 |
| **Line Number Off** (shifted imports, functions, comments) | 4 | **100%** | 1.000 |
| **Encoding Issues** (Unicode, BOM, invisible chars, CRLF, smart quotes) | 5 | **100%** | 1.000 |
| **Total** | **50** | **100%** | **0.960** |

> **50/50 benchmarks passing.** Covers Python, TypeScript, Rust, Go, Java, Ruby, CSS, HTML, YAML, Makefile, Bash, and more.

Run the benchmarks yourself:

```bash
python3 benchmarks/run_benchmarks.py
```

## GitHub Action

Use HarnessKit in your CI/CD pipelines:

```yaml
- name: Apply fuzzy edit
  uses: alexmelges/harnesskit@v0.5
  with:
    edit-file: changes.json
    validate: true
    atomic: true
```

Or inline:

```yaml
- name: Apply edit
  uses: alexmelges/harnesskit@v0.5
  with:
    edit-json: '{"file": "src/config.py", "old_text": "DEBUG = True", "new_text": "DEBUG = False"}'
```

The action outputs `status`, `confidence`, and `match-type` for downstream steps.

## Pre-commit Hook

Add syntax validation to your pre-commit workflow:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/alexmelges/harnesskit
    rev: v0.5.0
    hooks:
      - id: harnesskit-validate
```

## How HarnessKit Compares

| Feature | HarnessKit | `str_replace` (Anthropic) | `apply_patch` (OpenAI) | SWE-bench harness |
|---|---|---|---|---|
| **Fuzzy matching** | ✅ 4-stage cascade | ❌ Exact only | ❌ Line numbers | ❌ N/A |
| **Dependencies** | Zero (stdlib) | N/A (built-in) | N/A (built-in) | Heavy |
| **Single file** | ✅ ~1300 LOC | N/A | N/A | ❌ |
| **Confidence score** | ✅ Per-edit | ❌ | ❌ | ❌ |
| **Syntax validation** | ✅ + auto-rollback | ❌ | ❌ | ❌ |
| **Undo/rollback** | ✅ Atomic | ❌ | ❌ | ❌ |
| **MCP server** | ✅ Built-in | ❌ | ❌ | ❌ |
| **Model-agnostic** | ✅ Any LLM | Claude only | GPT only | N/A |
| **XML + JSON input** | ✅ Auto-detect | ❌ | ❌ | ❌ |

**TL;DR:** HarnessKit is what `str_replace` and `apply_patch` should have been — tolerant of the imprecision that every LLM exhibits.

## Design Principles

- **Single file, stdlib only** — copy it, vendor it, pip install it. No dependency hell.
- **~1250 lines of Python** — still small enough to audit in one sitting
- **Graceful degradation** — exact match when possible, fuzzy only when needed
- **Transparent** — every result tells you *how* it matched and *how confident* it is
- **Model-agnostic** — works with any LLM that can produce old/new text pairs

## Post-Edit Validation

HarnessKit can verify that edits don't break your code's syntax — and **automatically rolls back** if they do. No other edit tool does this.

```bash
# Validate after applying — rollback on syntax error
hk apply --file app.py --old "x = 1" --new "x = 1 +" --validate
# → status: "validation_error", file unchanged

# Validate a file without editing
hk validate app.py
# → {"status": "valid", "file": "app.py"}
```

Supported languages (all stdlib, zero dependencies):

| Extension | Validator |
|-----------|-----------|
| `.py` | `compile()` — catches all Python syntax errors |
| `.json` | `json.loads()` — strict JSON validation |
| `.xml`, `.html`, `.htm` | `ElementTree` — XML/HTML parse check |
| `.yaml`, `.yml` | `yaml.safe_load()` (if PyYAML installed) |
| `.js`, `.ts`, `.jsx`, `.tsx` | Bracket/brace/paren balance + unclosed string detection |
| Other | Always passes (no false positives) |

## Diff Output

See exactly what changed with unified diff output:

```bash
# Show diff on stderr (JSON still goes to stdout)
hk apply --file app.py --old "x = 1" --new "x = 2" --diff

# Preview changes without writing
hk apply --file app.py --old "x = 1" --new "x = 2" --diff --dry-run
```

Diff is also included in the JSON output (`"diff"` field) for programmatic use.

## Create Files

Coding agents don't just edit — they create files too:

```bash
# Create a new file
hk create --file src/utils.py --content "def helper(): pass"

# Fail if file exists (safe default)
hk create --file src/utils.py --content "..."
# → error: "File already exists"

# Overwrite with --force
hk create --file src/utils.py --content "..." --force

# Validate syntax before creating
hk create --file src/utils.py --content "def(" --validate
# → validation_error, file NOT created

# Read content from stdin
echo 'print("hello")' | hk create --file hello.py --stdin
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | `0.8` | Minimum similarity score for fuzzy matching |
| `--dry-run` | `false` | Preview changes without writing to disk |
| `--validate` | `false` | Validate syntax after edit (rollback on failure) |
| `--diff` | `false` | Print unified diff to stderr |
| `--force` | `false` | Overwrite existing file (create command) |

## Human-Friendly Output

When running in a terminal, HarnessKit auto-detects TTY and shows colored, readable output:

```bash
$ hk apply --file app.py --old "x = 1" --new "x = 2"
✅ app.py
   Match: exact (100.0% confidence)

$ hk apply --file app.py --old "def foo" --new "def bar" --diff
✅ app.py
   Match: fuzzy (92.3% confidence)
   -def foo():
   +def bar():
```

Force JSON output (for scripts): `--format json`
Force human output (for pipes): `--format human`

## Development

```bash
git clone https://github.com/alexmelges/harnesskit.git
cd harnesskit
python3 -m pytest test_hk.py test_mcp.py test_wrapper.py -v  # 137 tests
```

## License

MIT — see [LICENSE](LICENSE).

---

**Built for the agents that build everything else.**
