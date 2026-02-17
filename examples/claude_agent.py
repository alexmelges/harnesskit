#!/usr/bin/env python3
"""
HarnessKit + Claude Agent — Complete tool_use integration.

A working coding agent that uses Claude to plan edits and HarnessKit
to apply them with fuzzy matching. Handles the full tool_use cycle.

Requirements:
    pip install anthropic harnesskit

Usage:
    python claude_agent.py --task "Add type hints to all functions" --files src/
    python claude_agent.py --task "Fix the bug in auth.py" --files auth.py utils.py
    echo "Add error handling" | python claude_agent.py --files app.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Allow importing from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import anthropic
except ImportError:
    print("Error: pip install anthropic")
    sys.exit(1)

from hk_wrapper import HarnessKit, EditResponse

# ── Tool Definitions ──────────────────────────────────────────────

TOOLS = [
    {
        "name": "edit_file",
        "description": (
            "Edit a file by fuzzy-matching old_text and replacing with new_text. "
            "Tolerates whitespace differences, indentation mismatches, and minor "
            "inaccuracies in the old_text. Always provide enough context lines "
            "to uniquely identify the edit location."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find (fuzzy matched — include surrounding context lines)",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text",
                },
            },
            "required": ["file", "old_text", "new_text"],
        },
    },
    {
        "name": "create_file",
        "description": "Create a new file with the given content. Parent directories are created automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path for the new file",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content",
                },
            },
            "required": ["file", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path to read",
                },
            },
            "required": ["file"],
        },
    },
]

# ── Colors ────────────────────────────────────────────────────────

class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"


def log(icon: str, msg: str) -> None:
    print(f"{C.DIM}│{C.RESET} {icon} {msg}")


# ── Tool Execution ────────────────────────────────────────────────

hk = HarnessKit(threshold=0.8, validate_after=True)


def handle_tool(name: str, input: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "edit_file":
        result = hk.edit(
            file=input["file"],
            old_text=input["old_text"],
            new_text=input["new_text"],
        )
        if result.success:
            log(
                "✅",
                f"{C.GREEN}Edited{C.RESET} {C.BOLD}{result.file}{C.RESET} "
                f"({result.match_type}, {result.similarity:.0%} match)"
                + (f" — syntax {result.validation}" if result.validation else ""),
            )
        else:
            log("❌", f"{C.RED}Failed{C.RESET} {result.file}: {result.error}")
        return result.to_json()

    elif name == "create_file":
        result = hk.create(file=input["file"], content=input["content"])
        if result.success:
            log("📄", f"{C.GREEN}Created{C.RESET} {C.BOLD}{result.file}{C.RESET}")
        else:
            log("❌", f"{C.RED}Failed{C.RESET} {result.file}: {result.error}")
        return result.to_json()

    elif name == "read_file":
        try:
            with open(input["file"], "r") as f:
                content = f.read()
            log("📖", f"{C.DIM}Read{C.RESET} {input['file']} ({len(content)} chars)")
            return content
        except Exception as e:
            log("❌", f"{C.RED}Read failed{C.RESET}: {e}")
            return json.dumps({"error": str(e)})

    return json.dumps({"error": f"Unknown tool: {name}"})


# ── File Context ──────────────────────────────────────────────────

def gather_files(paths: list[str], max_chars: int = 100_000) -> str:
    """Read files/directories and format as context for the LLM."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("py", "js", "ts", "jsx", "tsx", "go", "rs", "rb", "java", "c", "cpp", "h"):
                files.extend(glob.glob(os.path.join(p, f"**/*.{ext}"), recursive=True))
        elif os.path.isfile(p):
            files.append(p)

    context_parts = []
    total = 0
    for f in sorted(set(files)):
        try:
            content = open(f).read()
        except Exception:
            continue
        if total + len(content) > max_chars:
            context_parts.append(f"\n... (truncated, {len(files) - len(context_parts)} files remaining)")
            break
        context_parts.append(f"── {f} ──\n{content}")
        total += len(content)

    return "\n\n".join(context_parts) if context_parts else "(no files found)"


# ── Agent Loop ────────────────────────────────────────────────────

def run_agent(
    task: str,
    file_context: str,
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 20,
) -> None:
    """Run the agent loop until the task is complete."""
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    system = (
        "You are a precise coding agent. You edit files using the edit_file tool, "
        "which supports fuzzy matching — your old_text doesn't need to be exact, "
        "but include enough surrounding lines for a unique match.\n\n"
        "Rules:\n"
        "- Read files before editing if you're unsure of the exact content\n"
        "- Make one logical change per edit_file call\n"
        "- After all edits, summarize what you changed\n"
    )

    messages = [
        {
            "role": "user",
            "content": f"## Task\n{task}\n\n## Current Files\n{file_context}",
        }
    ]

    print(f"\n{C.BOLD}{'─' * 60}{C.RESET}")
    print(f"{C.BOLD}🔧 HarnessKit Agent{C.RESET}")
    print(f"{C.DIM}Task:{C.RESET} {task}")
    print(f"{C.DIM}Model:{C.RESET} {model}")
    print(f"{C.BOLD}{'─' * 60}{C.RESET}")

    for turn in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        # Process response content blocks
        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n{C.CYAN}Claude:{C.RESET} {block.text}")
            elif block.type == "tool_use":
                log(
                    "🔨",
                    f"{C.MAGENTA}{block.name}{C.RESET}({C.DIM}{_summarize_input(block.name, block.input)}{C.RESET})",
                )
                result = handle_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        # If there were tool calls, send results back
        if tool_results:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # No tool calls = agent is done
            break

        if response.stop_reason == "end_turn":
            break

    print(f"\n{C.BOLD}{'─' * 60}{C.RESET}")
    print(f"{C.GREEN}✓ Done{C.RESET} ({turn + 1} turn{'s' if turn else ''})")
    print(f"{C.BOLD}{'─' * 60}{C.RESET}\n")


def _summarize_input(name: str, input: dict) -> str:
    """One-line summary of tool input for logging."""
    if name == "edit_file":
        lines = input.get("old_text", "").count("\n") + 1
        return f"{input.get('file', '?')}, {lines} lines"
    elif name == "create_file":
        return input.get("file", "?")
    elif name == "read_file":
        return input.get("file", "?")
    return str(input)[:80]


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HarnessKit + Claude coding agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  %(prog)s --task "Add type hints" --files src/\n'
            '  %(prog)s --task "Fix the auth bug" --files auth.py utils.py\n'
            '  echo "Refactor to async" | %(prog)s --files app.py\n'
        ),
    )
    parser.add_argument("--task", "-t", help="Coding task to perform")
    parser.add_argument("--files", "-f", nargs="+", default=["."], help="Files or directories to include as context")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum agent turns")
    args = parser.parse_args()

    task = args.task or sys.stdin.read().strip()
    if not task:
        parser.error("Provide --task or pipe task via stdin")

    file_context = gather_files(args.files)
    run_agent(task, file_context, model=args.model, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
