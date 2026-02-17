#!/usr/bin/env python3
"""
HarnessKit Batch Editor — Apply multiple edits from JSON or XML.

Reads a list of edits from stdin or file and applies them sequentially
using HarnessKit's fuzzy matching.

Requirements:
    pip install harnesskit

Usage (JSON):
    cat edits.json | python batch_edit.py
    python batch_edit.py -i edits.json

Usage (XML — natural for LLM output):
    cat edits.xml | python batch_edit.py
    python batch_edit.py -i edits.xml

JSON format:
    [
      {"file": "app.py", "old_text": "def foo():", "new_text": "def foo() -> None:"},
      {"file": "app.py", "old_text": "return x", "new_text": "return int(x)"}
    ]

XML format (auto-detected):
    <edits>
      <edit file="app.py">
        <old>def foo():</old>
        <new>def foo() -> None:</new>
      </edit>
    </edits>
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hk_wrapper import HarnessKit


class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"


def parse_json_edits(text: str) -> list[dict]:
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    return data


def parse_xml_edits(text: str) -> list[dict]:
    root = ET.fromstring(text)
    edits = []
    for el in root.iter("edit"):
        file = el.get("file") or el.get("path") or ""
        old = el.findtext("old") or el.findtext("old_text") or ""
        new = el.findtext("new") or el.findtext("new_text") or ""
        if file and old:
            edits.append({"file": file, "old_text": old, "new_text": new})
    return edits


def parse_edits(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("<"):
        return parse_xml_edits(text)
    return parse_json_edits(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply batch edits with HarnessKit")
    parser.add_argument("-i", "--input", help="Input file (JSON or XML). Default: stdin")
    parser.add_argument("-t", "--threshold", type=float, default=0.8, help="Fuzzy match threshold")
    parser.add_argument("--validate", action="store_true", help="Validate syntax after each edit")
    parser.add_argument("--dry-run", action="store_true", help="Show diffs without applying")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    args = parser.parse_args()

    if args.input:
        text = open(args.input).read()
    else:
        if sys.stdin.isatty():
            parser.error("Provide -i file or pipe edits via stdin")
        text = sys.stdin.read()

    edits = parse_edits(text)
    hk = HarnessKit(threshold=args.threshold, validate_after=args.validate)

    if not args.json_output:
        print(f"{C.BOLD}Applying {len(edits)} edit(s)...{C.RESET}\n")

    results = []
    ok = fail = 0

    for i, edit in enumerate(edits, 1):
        f = edit["file"]
        old = edit["old_text"]
        new = edit["new_text"]

        if args.dry_run:
            diff = hk.diff(f, old, new, threshold=args.threshold)
            if diff:
                if not args.json_output:
                    print(f"{C.GREEN}[{i}] Would edit {f}{C.RESET}")
                    print(diff)
                results.append({"file": f, "success": True, "diff": diff})
                ok += 1
            else:
                if not args.json_output:
                    print(f"{C.RED}[{i}] No match in {f}{C.RESET}")
                results.append({"file": f, "success": False, "error": "no match"})
                fail += 1
            continue

        result = hk.edit(file=f, old_text=old, new_text=new)
        results.append(result.to_dict())

        if result.success:
            ok += 1
            if not args.json_output:
                validation = f" (syntax: {result.validation})" if result.validation else ""
                print(
                    f"  {C.GREEN}✓{C.RESET} [{i}] {C.BOLD}{f}{C.RESET} — "
                    f"{result.match_type} ({result.similarity:.0%}){validation}"
                )
        else:
            fail += 1
            if not args.json_output:
                print(f"  {C.RED}✗{C.RESET} [{i}] {C.BOLD}{f}{C.RESET} — {result.error}")

    if args.json_output:
        print(json.dumps({"total": len(edits), "ok": ok, "failed": fail, "results": results}, indent=2))
    else:
        print(f"\n{C.BOLD}Results:{C.RESET} {C.GREEN}{ok} ok{C.RESET}, {C.RED}{fail} failed{C.RESET}")


if __name__ == "__main__":
    main()
