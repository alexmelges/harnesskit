# COMPOUND.md — HarnessKit

## What Was Built

HarnessKit v0.1.0 — a fuzzy edit tool for LLM coding agents.

- **419 lines** of Python, single file (`hk.py`), stdlib only
- **39 tests** passing via unittest
- **4-stage matching cascade**: exact → normalized whitespace → sequence matcher → line-by-line fuzzy
- **CLI tool** (`hk`) with JSON stdin support for agent integration
- Published to GitHub: https://github.com/alexmelges/harnesskit

## Lessons Learned

1. **The edit problem is real** — LLM agents fail on edits constantly due to whitespace drift and minor hallucinations. Fuzzy matching is a simple, high-impact fix.
2. **Single-file, no-deps is a feature** — makes it trivially vendorable and embeddable in any agent framework.
3. **Modern setuptools** dropped license classifiers in favor of `license` field expressions (PEP 639). Had to remove the classifier.
4. **Confidence scores matter** — agents need to know *how* an edit was matched to decide whether to trust it.

## What's Next

- [ ] **Benchmark against real edit failures** — collect actual failed str_replace calls from Claude, GPT, etc. and measure recovery rate
- [ ] **Publish to PyPI** — `python3 -m build && twine upload dist/*`
- [ ] **Landing page** — simple site with live demo
- [ ] **Integration examples** — OpenClaw skill, Claude tool_use wrapper, LangChain tool
- [ ] **Multi-file atomic edits** — rollback if any edit in a batch fails
- [ ] **Indentation-aware matching** — handle re-indented blocks as a special case
