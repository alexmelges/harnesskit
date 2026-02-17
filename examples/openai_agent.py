#!/usr/bin/env python3
"""
HarnessKit + OpenAI Agent — Complete function calling integration.

A working coding agent that uses GPT-4o to plan edits and HarnessKit
to apply them with fuzzy matching. Handles the full function calling cycle.

Requirements:
    pip install openai harnesskit

Usage:
    python openai_agent.py --task "Add docstrings to all functions" --files src/
    python openai_agent.py --task "Convert to async/await" --files app.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openai import OpenAI
except ImportError:
    print("Error: pip install openai")
    sys.exit(1)

from hk_wrapper import HarnessKit

# ── Tool Definitions (OpenAI format) ─────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Edit a file by fuzzy-matching old_text and replacing with new_text. "
                "Tolerates whitespace differences and minor inaccuracies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Path to the file"},
                    "old_text": {"type": "string", "description": "Text to find (fuzzy matched)"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                },
                "required": ["file", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file with given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Path for new file"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["file", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Path to read"},
                },
                "required": ["file"],
            },
        },
    },
]

# ── Colors ────────────────────────────────────────────────────────

class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"


def log(icon: str, msg: str) -> None:
    print(f"{C.DIM}│{C.RESET} {icon} {msg}")


# ── Tool Execution ────────────────────────────────────────────────

hk = HarnessKit(threshold=0.8, validate_after=True)


def handle_tool(name: str, args: dict) -> str:
    if name == "edit_file":
        result = hk.edit(file=args["file"], old_text=args["old_text"], new_text=args["new_text"])
        status = "✅" if result.success else "❌"
        color = C.GREEN if result.success else C.RED
        detail = (
            f"{result.match_type}, {result.similarity:.0%}"
            if result.success
            else result.error
        )
        log(status, f"{color}Edit{C.RESET} {C.BOLD}{result.file}{C.RESET} ({detail})")
        return result.to_json()

    elif name == "create_file":
        result = hk.create(file=args["file"], content=args["content"])
        status = "📄" if result.success else "❌"
        log(status, f"{C.GREEN}Created{C.RESET} {args['file']}" if result.success else f"{C.RED}Failed{C.RESET}: {result.error}")
        return result.to_json()

    elif name == "read_file":
        try:
            content = open(args["file"]).read()
            log("📖", f"Read {args['file']} ({len(content)} chars)")
            return content
        except Exception as e:
            return json.dumps({"error": str(e)})

    return json.dumps({"error": f"Unknown tool: {name}"})


# ── File Context ──────────────────────────────────────────────────

def gather_files(paths: list[str], max_chars: int = 100_000) -> str:
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("py", "js", "ts", "jsx", "tsx", "go", "rs", "rb", "java"):
                files.extend(glob.glob(os.path.join(p, f"**/*.{ext}"), recursive=True))
        elif os.path.isfile(p):
            files.append(p)

    parts, total = [], 0
    for f in sorted(set(files)):
        try:
            content = open(f).read()
        except Exception:
            continue
        if total + len(content) > max_chars:
            break
        parts.append(f"── {f} ──\n{content}")
        total += len(content)
    return "\n\n".join(parts) or "(no files found)"


# ── Agent Loop ────────────────────────────────────────────────────

def run_agent(task: str, file_context: str, model: str = "gpt-4o", max_turns: int = 20) -> None:
    client = OpenAI()

    print(f"\n{C.BOLD}{'─' * 60}{C.RESET}")
    print(f"{C.BOLD}🔧 HarnessKit Agent (OpenAI){C.RESET}")
    print(f"{C.DIM}Task:{C.RESET} {task}")
    print(f"{C.DIM}Model:{C.RESET} {model}")
    print(f"{C.BOLD}{'─' * 60}{C.RESET}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise coding agent. Use edit_file (fuzzy matching) for edits. "
                "Include enough context lines in old_text for a unique match. "
                "Read files first if unsure of content. Summarize changes when done."
            ),
        },
        {"role": "user", "content": f"## Task\n{task}\n\n## Current Files\n{file_context}"},
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            max_tokens=4096,
        )

        msg = response.choices[0].message

        if msg.content:
            print(f"\n{C.CYAN}GPT:{C.RESET} {msg.content}")

        if not msg.tool_calls:
            break

        # Append assistant message with tool calls
        messages.append(msg)

        # Execute each tool call
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            log("🔨", f"{C.MAGENTA}{tc.function.name}{C.RESET}({C.DIM}{args.get('file', '?')}{C.RESET})")
            result = handle_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    print(f"\n{C.BOLD}{'─' * 60}{C.RESET}")
    print(f"{C.GREEN}✓ Done{C.RESET} ({turn + 1} turns)")
    print(f"{C.BOLD}{'─' * 60}{C.RESET}\n")


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HarnessKit + OpenAI coding agent")
    parser.add_argument("--task", "-t", help="Coding task")
    parser.add_argument("--files", "-f", nargs="+", default=["."], help="Files/dirs for context")
    parser.add_argument("--model", "-m", default="gpt-4o", help="OpenAI model")
    parser.add_argument("--max-turns", type=int, default=20)
    args = parser.parse_args()

    task = args.task or sys.stdin.read().strip()
    if not task:
        parser.error("Provide --task or pipe via stdin")

    run_agent(task, gather_files(args.files), model=args.model, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
