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

We tested HarnessKit against **26 realistic edit failure scenarios** — the kind that break `str_replace` and `apply_patch` in production agent workflows.

| Category | Exact Match | HarnessKit | Recovery Rate |
|---|---|---|---|
| **Whitespace** (tabs/spaces, trailing, indentation, CRLF) | 0/8 | **8/8** | 100% |
| **Hallucinations** (typos, wrong quotes, missing types) | 0/7 | **7/7** | 100% |
| **Line Drift** (shifted context, extra decorators) | 2/3 | **3/3** | 100% |
| **Partial Matches** (subset of target) | 2/2 | **2/2** | — |
| **Real-World** (str_replace failures, docstring diffs) | 0/6 | **6/6** | 100% |
| **Total** | **4/26 (15%)** | **26/26 (100%)** | **100%** |

> **Exact match succeeds 15% of the time. HarnessKit succeeds 100% of the time.**
> 22 out of 22 failed edits recovered.

Run the benchmarks yourself:

```bash
python3 benchmarks/benchmark.py
```

## Design Principles

- **Single file, stdlib only** — copy it, vendor it, pip install it. No dependency hell.
- **419 lines of Python** — small enough to audit in one sitting
- **Graceful degradation** — exact match when possible, fuzzy only when needed
- **Transparent** — every result tells you *how* it matched and *how confident* it is
- **Model-agnostic** — works with any LLM that can produce old/new text pairs

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | `0.8` | Minimum similarity score for fuzzy matching |
| `--dry-run` | `false` | Preview changes without writing to disk |

## Development

```bash
git clone https://github.com/alexmelges/harnesskit.git
cd harnesskit
python3 test_hk.py  # 39 tests, stdlib unittest
```

## License

MIT — see [LICENSE](LICENSE).

---

**Built for the agents that build everything else.**
