# HarnessKit — Fuzzy Edit Tool for LLM Coding Agents

## Problem
LLM coding agents use edit tools (str_replace, apply_patch) that fail frequently because:
- Exact string matching breaks on whitespace differences
- Models hallucinate line numbers
- Diff formats are fragile across different LLMs
- ~50% failure rate on non-native models (per "The Harness Problem")

## Solution
A Python CLI tool (`hk`) that applies edits to files using fuzzy matching. It accepts a simple, model-agnostic edit format and robustly applies changes even when the LLM's output isn't pixel-perfect.

## Core Algorithm
1. Parse edit instruction (old_text → new_text for a given file)
2. Find best match for old_text in the file using fuzzy matching:
   - Try exact match first
   - Fall back to normalized whitespace match
   - Fall back to difflib.SequenceMatcher with configurable threshold (default 0.8)
   - Fall back to line-by-line fuzzy match (find best contiguous block)
3. Apply the replacement at the matched location
4. Report confidence score and what was matched

## CLI Interface
```
hk apply --file path/to/file --old "old text" --new "new text"
hk apply --edit edit_instruction.json
hk apply --stdin  (reads JSON from stdin — for tool_use integration)
```

## JSON Edit Format (stdin/file)
```json
{
  "file": "path/to/file.py",
  "old_text": "def hello():\n    print('hi')",
  "new_text": "def hello():\n    print('hello world')"
}
```

## Multiple edits:
```json
{"edits": [
  {"file": "a.py", "old_text": "...", "new_text": "..."},
  {"file": "b.py", "old_text": "...", "new_text": "..."}
]}
```

## Output (JSON)
```json
{
  "status": "applied",
  "file": "path/to/file.py",
  "match_type": "fuzzy",
  "confidence": 0.92,
  "matched_text": "def hello():\n    print( 'hi' )"
}
```

## Key Design Decisions
- Python stdlib only (no deps) — maximizes portability
- Single file implementation — easy to vendor/embed
- Exit codes: 0=applied, 1=no match found, 2=ambiguous (multiple matches)
- Confidence threshold configurable via --threshold (default 0.8)
- Dry-run mode via --dry-run (show what would change without applying)

## Files to Create
1. `hk.py` — single-file implementation (~300 lines)
2. `test_hk.py` — test suite with edge cases
3. `README.md` — usage docs
