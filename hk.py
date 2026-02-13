#!/usr/bin/env python3
"""HarnessKit — Fuzzy edit tool for LLM coding agents.

A single-file CLI tool that applies edits to files using fuzzy matching.
Accepts a simple, model-agnostic edit format and robustly applies changes
even when the LLM's output isn't pixel-perfect.

Algorithm:
  1. Try exact match
  2. Fall back to normalized whitespace match
  3. Fall back to difflib.SequenceMatcher (configurable threshold)
  4. Fall back to line-by-line fuzzy match (best contiguous block)

Exit codes: 0=applied, 1=no match found, 2=ambiguous (multiple matches)
"""

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MatchResult:
    start: int
    end: int
    matched_text: str
    match_type: str  # "exact", "whitespace", "fuzzy", "line_fuzzy"
    confidence: float


@dataclass
class EditResult:
    status: str  # "applied", "no_match", "ambiguous", "error"
    file: str
    match_type: Optional[str] = None
    confidence: Optional[float] = None
    matched_text: Optional[str] = None
    error: Optional[str] = None


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip."""
    return re.sub(r'\s+', ' ', text).strip()


def find_exact_matches(content: str, old_text: str) -> List[MatchResult]:
    """Find all exact occurrences of old_text in content."""
    matches = []
    start = 0
    while True:
        idx = content.find(old_text, start)
        if idx == -1:
            break
        matches.append(MatchResult(
            start=idx,
            end=idx + len(old_text),
            matched_text=old_text,
            match_type="exact",
            confidence=1.0,
        ))
        start = idx + 1
    return matches


def _strip_whitespace_with_map(text: str) -> Tuple[str, List[int]]:
    """Strip all whitespace from text, returning (stripped, position_map).

    position_map[i] = index in original text of the i-th non-ws char.
    """
    chars: List[str] = []
    positions: List[int] = []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            positions.append(i)
    return ''.join(chars), positions


def find_whitespace_matches(content: str, old_text: str) -> List[MatchResult]:
    """Find matches where the only differences are whitespace.

    Strips all whitespace from both strings, finds substring matches
    in the stripped content, then maps positions back to the original.
    """
    stripped_old, _ = _strip_whitespace_with_map(old_text)
    if not stripped_old:
        return []

    stripped_content, content_pos_map = _strip_whitespace_with_map(content)

    matches = []
    start = 0
    while True:
        idx = stripped_content.find(stripped_old, start)
        if idx == -1:
            break
        # Map back to original positions
        orig_start = content_pos_map[idx]
        orig_end_char = content_pos_map[idx + len(stripped_old) - 1]
        orig_end = orig_end_char + 1
        matched = content[orig_start:orig_end]
        matches.append(MatchResult(
            start=orig_start,
            end=orig_end,
            matched_text=matched,
            match_type="whitespace",
            confidence=0.95,
        ))
        start = idx + 1
    return matches


def find_fuzzy_matches(
    content: str, old_text: str, threshold: float
) -> List[MatchResult]:
    """Find fuzzy matches using SequenceMatcher on sliding windows."""
    if not old_text.strip():
        return []

    old_len = len(old_text)
    best: List[MatchResult] = []
    best_ratio = threshold

    # Slide a window of varying sizes around the expected length
    min_window = max(1, int(old_len * 0.7))
    max_window = min(len(content), int(old_len * 1.3) + 1)

    for window_size in range(min_window, max_window + 1):
        for start in range(0, len(content) - window_size + 1):
            candidate = content[start:start + window_size]
            ratio = difflib.SequenceMatcher(
                None, old_text, candidate
            ).ratio()
            if ratio > best_ratio:
                best = [MatchResult(
                    start=start,
                    end=start + window_size,
                    matched_text=candidate,
                    match_type="fuzzy",
                    confidence=round(ratio, 4),
                )]
                best_ratio = ratio
            elif ratio == best_ratio and best:
                # Check overlap — only add if non-overlapping
                overlaps = any(
                    not (start >= m.end or start + window_size <= m.start)
                    for m in best
                )
                if not overlaps:
                    best.append(MatchResult(
                        start=start,
                        end=start + window_size,
                        matched_text=candidate,
                        match_type="fuzzy",
                        confidence=round(ratio, 4),
                    ))

    return best


def find_line_fuzzy_matches(
    content: str, old_text: str, threshold: float
) -> List[MatchResult]:
    """Find best contiguous block of lines matching old_text lines."""
    content_lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)

    if not old_lines or not content_lines:
        return []

    n_old = len(old_lines)
    best_score = threshold
    best_matches: List[Tuple[int, int, float]] = []  # (start_line, end_line, score)

    for start_line in range(0, len(content_lines) - n_old + 1):
        block = content_lines[start_line:start_line + n_old]
        # Compare line by line
        total = 0.0
        for ol, cl in zip(old_lines, block):
            total += difflib.SequenceMatcher(None, ol, cl).ratio()
        avg = total / n_old

        if avg > best_score:
            best_score = avg
            best_matches = [(start_line, start_line + n_old, avg)]
        elif avg == best_score and best_matches:
            best_matches.append((start_line, start_line + n_old, avg))

    results = []
    for start_line, end_line, score in best_matches:
        block = content_lines[start_line:end_line]
        matched_text = ''.join(block)
        # Calculate char offsets
        char_start = sum(len(l) for l in content_lines[:start_line])
        char_end = char_start + len(matched_text)
        results.append(MatchResult(
            start=char_start,
            end=char_end,
            matched_text=matched_text,
            match_type="line_fuzzy",
            confidence=round(score, 4),
        ))
    return results


def find_best_match(
    content: str, old_text: str, threshold: float = 0.8
) -> Optional[MatchResult]:
    """Find the best match for old_text in content, trying strategies in order.

    Returns None if no match meets the threshold, or if multiple ambiguous
    matches are found (raises AmbiguousMatchError).
    """
    # Strategy 1: Exact match
    matches = find_exact_matches(content, old_text)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AmbiguousMatchError(matches)

    # Strategy 2: Whitespace-normalized match
    matches = find_whitespace_matches(content, old_text)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AmbiguousMatchError(matches)

    # Strategy 3: SequenceMatcher fuzzy match
    matches = find_fuzzy_matches(content, old_text, threshold)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AmbiguousMatchError(matches)

    # Strategy 4: Line-by-line fuzzy match
    matches = find_line_fuzzy_matches(content, old_text, threshold)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AmbiguousMatchError(matches)

    return None


class AmbiguousMatchError(Exception):
    """Raised when multiple equally-good matches are found."""

    def __init__(self, matches: List[MatchResult]):
        self.matches = matches
        super().__init__(f"Found {len(matches)} ambiguous matches")


def apply_edit(
    file_path: str,
    old_text: str,
    new_text: str,
    threshold: float = 0.8,
    dry_run: bool = False,
) -> EditResult:
    """Apply a single edit to a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return EditResult(
            status="error",
            file=file_path,
            error=f"File not found: {file_path}",
        )
    except OSError as e:
        return EditResult(
            status="error",
            file=file_path,
            error=str(e),
        )

    try:
        match = find_best_match(content, old_text, threshold)
    except AmbiguousMatchError as e:
        return EditResult(
            status="ambiguous",
            file=file_path,
            match_type=e.matches[0].match_type,
            confidence=e.matches[0].confidence,
            error=f"Found {len(e.matches)} ambiguous matches",
        )

    if match is None:
        return EditResult(
            status="no_match",
            file=file_path,
            error="No match found above threshold",
        )

    new_content = content[:match.start] + new_text + content[match.end:]

    if not dry_run:
        with open(file_path, 'w') as f:
            f.write(new_content)

    return EditResult(
        status="applied",
        file=file_path,
        match_type=match.match_type,
        confidence=match.confidence,
        matched_text=match.matched_text,
    )


def result_to_dict(result: EditResult) -> dict:
    """Convert EditResult to JSON-serializable dict."""
    d = {"status": result.status, "file": result.file}
    if result.match_type is not None:
        d["match_type"] = result.match_type
    if result.confidence is not None:
        d["confidence"] = result.confidence
    if result.matched_text is not None:
        d["matched_text"] = result.matched_text
    if result.error is not None:
        d["error"] = result.error
    return d


def parse_edit_input(args) -> List[dict]:
    """Parse edit instructions from CLI args or stdin."""
    if args.stdin:
        data = json.load(sys.stdin)
        if "edits" in data:
            return data["edits"]
        return [data]

    if args.edit:
        with open(args.edit, 'r') as f:
            data = json.load(f)
        if "edits" in data:
            return data["edits"]
        return [data]

    if args.file and args.old is not None and args.new is not None:
        return [{"file": args.file, "old_text": args.old, "new_text": args.new}]

    raise ValueError(
        "Must provide --file/--old/--new, --edit <file>, or --stdin"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hk",
        description="HarnessKit — Fuzzy edit tool for LLM coding agents",
    )
    sub = parser.add_subparsers(dest="command")

    apply_parser = sub.add_parser("apply", help="Apply edit(s) to file(s)")
    apply_parser.add_argument("--file", help="Target file path")
    apply_parser.add_argument("--old", help="Text to find")
    apply_parser.add_argument("--new", help="Replacement text")
    apply_parser.add_argument("--edit", help="JSON edit instruction file")
    apply_parser.add_argument(
        "--stdin", action="store_true", help="Read JSON from stdin"
    )
    apply_parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Fuzzy match threshold (default: 0.8)",
    )
    apply_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without applying",
    )

    args = parser.parse_args(argv)

    if args.command != "apply":
        parser.print_help()
        return 1

    try:
        edits = parse_edit_input(args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return 1

    results = []
    exit_code = 0

    for edit in edits:
        file_path = edit.get("file", "")
        old_text = edit.get("old_text", "")
        new_text = edit.get("new_text", "")

        result = apply_edit(
            file_path, old_text, new_text,
            threshold=args.threshold,
            dry_run=args.dry_run,
        )
        results.append(result_to_dict(result))

        if result.status == "no_match" or result.status == "error":
            exit_code = max(exit_code, 1)
        elif result.status == "ambiguous":
            exit_code = max(exit_code, 2)

    if len(results) == 1:
        print(json.dumps(results[0], indent=2))
    else:
        print(json.dumps(results, indent=2))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
