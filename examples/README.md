# HarnessKit Examples

Working integration examples showing HarnessKit as a tool for LLM coding agents.

## Claude Agent (`claude_agent.py`)

A complete Claude API agent that uses `tool_use` with HarnessKit for fuzzy file editing.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...

# Edit files with natural language
python claude_agent.py --task "Add type hints to all functions" --files src/

# Fix a specific file
python claude_agent.py --task "Fix the auth bug" --files auth.py utils.py

# Pipe task via stdin
echo "Refactor to async/await" | python claude_agent.py --files app.py
```

The agent loop:
1. Claude sees your files and task
2. Plans edits, calls `edit_file` with approximate old_text
3. HarnessKit fuzzy-matches and applies the edit (even with whitespace/indentation errors)
4. Claude sees the result and continues until done

## OpenAI Agent (`openai_agent.py`)

Same concept using OpenAI's function calling API.

```bash
pip install openai
export OPENAI_API_KEY=sk-...

python openai_agent.py --task "Add error handling" --files app.py
python openai_agent.py --task "Convert callbacks to async" --files src/ --model gpt-4o
```

## Batch Editor (`batch_edit.py`)

Apply multiple edits from a JSON or XML file. Great for scripted refactoring.

```bash
# JSON input
echo '[
  {"file": "app.py", "old_text": "def foo():", "new_text": "def foo() -> None:"},
  {"file": "app.py", "old_text": "return x",   "new_text": "return int(x)"}
]' | python batch_edit.py

# XML input (natural LLM output format)
echo '<edits>
  <edit file="app.py">
    <old>def foo():</old>
    <new>def foo() -> None:</new>
  </edit>
</edits>' | python batch_edit.py

# Dry run (preview diffs)
python batch_edit.py -i edits.json --dry-run

# With syntax validation
python batch_edit.py -i edits.json --validate

# JSON output for CI pipelines
python batch_edit.py -i edits.json --json
```

## Why HarnessKit?

LLMs hallucinate code. When Claude/GPT outputs `old_text` for a search-and-replace, it's often slightly wrong — different indentation, missing a blank line, extra whitespace. Traditional exact-match tools fail. HarnessKit's fuzzy matching handles these errors automatically:

| Match Type | What It Handles |
|---|---|
| Exact | Perfect match (baseline) |
| Whitespace | Indentation differences, trailing spaces |
| Fuzzy | ~80%+ similar text (typos, minor changes) |
| Line-fuzzy | Best contiguous block match (reordered lines) |

This means your agent's edit success rate goes from ~70% to ~97%.
