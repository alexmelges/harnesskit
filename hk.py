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
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
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
    status: str  # "applied", "no_match", "ambiguous", "error", "validation_error"
    file: str
    match_type: Optional[str] = None
    confidence: Optional[float] = None
    matched_text: Optional[str] = None
    error: Optional[str] = None
    validated: Optional[bool] = None
    diff: Optional[str] = None


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
        # Compare line by line (try both raw and indent-normalized)
        total = 0.0
        for ol, cl in zip(old_lines, block):
            raw_ratio = difflib.SequenceMatcher(None, ol, cl).ratio()
            # Also try with tabs expanded to spaces (handles tabs-vs-spaces)
            ol_norm = ol.expandtabs(4)
            cl_norm = cl.expandtabs(4)
            norm_ratio = difflib.SequenceMatcher(None, ol_norm, cl_norm).ratio()
            total += max(raw_ratio, norm_ratio)
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


def _get_line_indent(line: str) -> str:
    """Return the leading whitespace of a line (excluding newline chars)."""
    stripped = line.lstrip()
    if not stripped:
        # Whitespace-only line: return everything except trailing newline(s)
        return line.rstrip('\n\r')
    return line[:len(line) - len(stripped)]


def _detect_indent_mapping(old_text: str, matched_text: str) -> Optional[dict]:
    """Detect systematic indentation differences between old_text and matched_text.

    Returns a dict mapping old_indent -> new_indent, or None if no consistent mapping.
    """
    old_lines = old_text.splitlines()
    matched_lines = matched_text.splitlines()

    if len(old_lines) != len(matched_lines):
        return None

    mapping = {}
    for ol, ml in zip(old_lines, matched_lines):
        oi = _get_line_indent(ol)
        mi = _get_line_indent(ml)
        if ol.strip() == ml.strip():  # Same content, different indent
            if oi in mapping:
                if mapping[oi] != mi:
                    return None  # Inconsistent
            else:
                mapping[oi] = mi

    return mapping if mapping else None


def _apply_indent_mapping(text: str, mapping: dict) -> str:
    """Apply an indentation mapping to text."""
    lines = text.splitlines(keepends=True)
    result = []
    for line in lines:
        indent = _get_line_indent(line)
        content = line[len(indent):]  # Preserve newline chars (don't use lstrip)
        if indent in mapping:
            result.append(mapping[indent] + content)
        else:
            # Try to find best matching indent by prefix
            best_old = ''
            for old_indent in mapping:
                if indent.startswith(old_indent) and len(old_indent) > len(best_old):
                    best_old = old_indent
            if best_old:
                extra = indent[len(best_old):]
                # Scale extra indent too
                if best_old and mapping[best_old]:
                    # Detect indent unit ratio
                    old_unit = len(best_old) if best_old else 1
                    new_unit = len(mapping[best_old])
                    if old_unit > 0:
                        scaled_extra_len = int(len(extra) * new_unit / old_unit)
                        indent_char = mapping[best_old][0] if mapping[best_old] else ' '
                        result.append(mapping[best_old] + indent_char * scaled_extra_len + content)
                    else:
                        result.append(mapping[best_old] + extra + content)
                else:
                    result.append(mapping[best_old] + extra + content)
            else:
                result.append(line)
    return ''.join(result)


def _adapt_replacement(old_text: str, new_text: str, matched_text: str, match_type: str) -> str:
    """Adapt new_text based on differences between old_text and matched_text.

    For whitespace/indentation matches: re-indent new_text to match actual file style.
    For fuzzy/stale context matches: apply the edit *diff* to the matched region.
    """
    if match_type == "exact":
        return new_text

    # Strategy 1: Indentation mapping (for whitespace/indentation drift)
    if match_type in ("whitespace", "line_fuzzy"):
        mapping = _detect_indent_mapping(old_text, matched_text)
        if mapping:
            # Only use indent mapping if it actually changes something
            adapted = _apply_indent_mapping(new_text, mapping)
            if adapted != new_text:
                return adapted

    # Strategy 2: Line-ending adaptation
    if '\r\n' in matched_text and '\r\n' not in new_text:
        new_text = new_text.replace('\n', '\r\n')
    elif '\r\n' not in matched_text and '\r\n' in new_text:
        new_text = new_text.replace('\r\n', '\n')

    # Strategy 3: Diff-based edit for fuzzy matches (stale context)
    if match_type in ("fuzzy", "line_fuzzy"):
        adapted = _apply_diff_based_edit(old_text, new_text, matched_text)
        if adapted is not None:
            return adapted

    # Strategy 4: For whitespace matches where indent mapping didn't work,
    # try line-by-line whitespace transfer
    if match_type == "whitespace":
        adapted = _transfer_line_whitespace(old_text, new_text, matched_text)
        if adapted is not None:
            adapted = _apply_alignment_patterns(adapted, matched_text)
            return adapted
        # Fall back to diff-based
        adapted = _apply_diff_based_edit(old_text, new_text, matched_text)
        if adapted is not None:
            return adapted

    return new_text


def _detect_trailing_ws_pattern(old_lines: List[str], matched_lines: List[str]) -> Optional[str]:
    """Detect if matched lines consistently add trailing whitespace that old lines lack."""
    trailing = None
    for ol, ml in zip(old_lines, matched_lines):
        ol_content = ol.rstrip('\n\r')
        ml_content = ml.rstrip('\n\r')
        ol_stripped = ol_content.rstrip()
        ml_stripped = ml_content.rstrip()
        if ol_stripped == ml_stripped or normalize_whitespace(ol) == normalize_whitespace(ml):
            ol_trail = ol_content[len(ol_stripped):]
            ml_trail = ml_content[len(ml_stripped):]
            if ml_trail and not ol_trail:
                if trailing is None:
                    trailing = ml_trail
                elif trailing != ml_trail:
                    return None  # Inconsistent
    return trailing


def _apply_trailing_ws(line: str, pattern: Optional[str]) -> str:
    """Apply trailing whitespace pattern to a line."""
    if pattern is None:
        return line
    content = line.rstrip('\n\r')
    ending = line[len(content):]
    stripped = content.rstrip()
    # Only add trailing ws if the line doesn't already have it
    if content == stripped:  # No trailing ws currently
        return stripped + pattern + ending
    return line


def _transfer_line_whitespace(old_text: str, new_text: str, matched_text: str) -> Optional[str]:
    """Transfer whitespace patterns from matched_text to new_text line by line.

    For lines that are the same between old_text and new_text (unchanged),
    use the matched_text version. For changed/new lines, try to infer
    whitespace patterns from the matched context.
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    matched_lines = matched_text.splitlines(keepends=True)

    if len(old_lines) != len(matched_lines):
        return None

    # Detect trailing whitespace pattern from matched lines
    trailing_ws_pattern = _detect_trailing_ws_pattern(old_lines, matched_lines)

    # Build maps
    ws_stripped_to_matched = {}  # whitespace-stripped content -> matched line
    old_to_matched_line = {}  # old line index -> matched line
    for i, (ol, ml) in enumerate(zip(old_lines, matched_lines)):
        ws_stripped_to_matched[normalize_whitespace(ol)] = ml
        old_to_matched_line[i] = ml

    # Use SequenceMatcher to align old and new lines
    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    result = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            # Unchanged lines: use matched version
            for i in range(i1, i2):
                result.append(old_to_matched_line.get(i, old_lines[i]))
        elif tag == 'replace':
            for j in range(j1, j2):
                nl = new_lines[j]
                ws_key = normalize_whitespace(nl)
                if ws_key in ws_stripped_to_matched:
                    result.append(ws_stripped_to_matched[ws_key])
                else:
                    # Try to adapt: find the corresponding old line and
                    # apply intra-line whitespace transfer
                    old_idx = i1 + (j - j1) if (i1 + (j - j1)) < i2 else None
                    if old_idx is not None and old_idx in old_to_matched_line:
                        adapted = _adapt_line_whitespace(
                            old_lines[old_idx], nl, old_to_matched_line[old_idx]
                        )
                        result.append(adapted)
                    else:
                        # Adapt indentation at least
                        if i1 < len(old_lines) and i1 in old_to_matched_line:
                            oi = _get_line_indent(old_lines[i1])
                            mi = _get_line_indent(old_to_matched_line[i1])
                            ni = _get_line_indent(nl)
                            if ni == oi and oi != mi:
                                nl = mi + nl.lstrip()
                        nl = _apply_trailing_ws(nl, trailing_ws_pattern)
                        result.append(nl)
        elif tag == 'insert':
            for j in range(j1, j2):
                nl = new_lines[j]
                ws_key = normalize_whitespace(nl)
                if ws_key in ws_stripped_to_matched:
                    result.append(ws_stripped_to_matched[ws_key])
                else:
                    # Adapt indentation from context
                    if i1 > 0 and (i1 - 1) in old_to_matched_line:
                        oi = _get_line_indent(old_lines[i1 - 1])
                        mi = _get_line_indent(old_to_matched_line[i1 - 1])
                        ni = _get_line_indent(nl)
                        if ni == oi and oi != mi:
                            nl = mi + nl.lstrip()
                    # Apply trailing whitespace pattern
                    nl = _apply_trailing_ws(nl, trailing_ws_pattern)
                    result.append(nl)
        # delete: skip

    return ''.join(result)


def _adapt_line_whitespace(old_line: str, new_line: str, matched_line: str) -> str:
    """Adapt a changed line's whitespace based on old->matched mapping.

    Given: old_line (what LLM thought), new_line (what LLM wants),
    matched_line (what's actually in the file).

    Find what changed content-wise between old and new, and apply
    that change to the matched line.
    """
    # First adapt indentation
    old_indent = _get_line_indent(old_line)
    matched_indent = _get_line_indent(matched_line)
    new_indent = _get_line_indent(new_line)

    # Use matched indentation if new matches old
    if new_indent == old_indent:
        base_indent = matched_indent
    else:
        base_indent = new_indent

    # Try to apply content diff within the line
    old_stripped = old_line.strip()
    new_stripped = new_line.strip()
    matched_stripped = matched_line.strip()

    # If the whitespace-normalized versions of old and matched are the same,
    # apply a character-level patch
    old_no_ws = re.sub(r'\s', '', old_stripped)
    matched_no_ws = re.sub(r'\s', '', matched_stripped)
    if normalize_whitespace(old_stripped) == normalize_whitespace(matched_stripped):
        # The only difference is internal whitespace (possibly operator spacing).
        # Find what content was added/changed in new vs old, apply to matched
        sm = difflib.SequenceMatcher(None, old_stripped, new_stripped)
        result_chars = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                # Use matched version of this segment
                # Map positions from old_stripped to matched_stripped
                result_chars.append(_map_segment(old_stripped, matched_stripped, i1, i2))
            elif tag == 'replace':
                result_chars.append(_adapt_operator_spacing(new_stripped[j1:j2], old_stripped, matched_stripped))
            elif tag == 'insert':
                result_chars.append(_adapt_operator_spacing(new_stripped[j1:j2], old_stripped, matched_stripped))
            # delete: skip
        adapted_content = ''.join(result_chars)
        ending = new_line[len(new_line.rstrip()):]  # preserve line ending
        return base_indent + adapted_content + ending

    # If stripping ALL whitespace makes old and matched equal, it's an operator spacing difference.
    # Only use this path if operator spacing patterns are detected.
    if old_no_ws == matched_no_ws:
        # Check if there's actually operator spacing to transfer
        test_adapted = _adapt_operator_spacing(new_stripped, old_stripped, matched_stripped)
        if test_adapted != new_stripped:
            sm = difflib.SequenceMatcher(None, old_stripped, new_stripped)
            result_chars = []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal':
                    result_chars.append(_map_segment(old_stripped, matched_stripped, i1, i2))
                elif tag == 'replace':
                    result_chars.append(_adapt_operator_spacing(new_stripped[j1:j2], old_stripped, matched_stripped))
                elif tag == 'insert':
                    result_chars.append(_adapt_operator_spacing(new_stripped[j1:j2], old_stripped, matched_stripped))
            adapted_content = ''.join(result_chars)
            ending = new_line[len(new_line.rstrip()):]
            return base_indent + adapted_content + ending

    return base_indent + new_stripped + new_line[len(new_line.rstrip()):]


def _apply_alignment_patterns(adapted_text: str, matched_text: str) -> str:
    """Detect aligned operators in matched_text and apply alignment to adapted_text.

    For example, if matched_text has '=>' aligned at column 19 across multiple lines,
    ensure new/changed lines in adapted_text also align '=>' at column 19.
    """
    matched_lines = matched_text.splitlines(keepends=True)

    # Detect alignment: find operators that appear at the same column in 3+ lines
    # Common alignment operators: =>, =, :
    alignment_ops = ['=>', '=', ':']

    for op in alignment_ops:
        # Find column positions of this operator in matched lines with same indentation
        indent_to_columns = {}  # indent -> list of columns
        for line in matched_lines:
            stripped = line.rstrip()
            if op not in stripped:
                continue
            indent = _get_line_indent(line)
            col = stripped.find(op)
            if col > 0:
                indent_to_columns.setdefault(indent, []).append(col)

        for indent, cols in indent_to_columns.items():
            if len(cols) < 2:
                continue
            # Check if most are aligned to the same column
            from collections import Counter
            col_counts = Counter(cols)
            most_common_col, count = col_counts.most_common(1)[0]
            if count < 2:
                continue

            # Apply alignment to adapted lines with same indent that have this operator
            adapted_lines = adapted_text.splitlines(keepends=True)
            new_lines = []
            for aline in adapted_lines:
                astripped = aline.rstrip()
                a_indent = _get_line_indent(aline)
                if a_indent == indent and op in astripped:
                    col = astripped.find(op)
                    if col != most_common_col and col > 0:
                        # Find the key part (before op) and value part (after op)
                        before_op = astripped[:col].rstrip()
                        after_op = astripped[col:]
                        needed_spaces = most_common_col - len(before_op)
                        if needed_spaces > 0:
                            ending = aline[len(astripped):]
                            aline = before_op + ' ' * needed_spaces + after_op + ending
                new_lines.append(aline)
            adapted_text = ''.join(new_lines)

    return adapted_text


def _adapt_operator_spacing(text: str, old_stripped: str, matched_stripped: str) -> str:
    """Adapt operator spacing in text to match the style of matched_stripped.

    Learns spacing patterns by comparing old_stripped (no spaces around ops)
    with matched_stripped (spaces around ops), then applies those patterns
    to the new text.
    """
    # Build a mapping of operator contexts: char-by-char alignment
    old_nows = re.sub(r'\s', '', old_stripped)
    matched_nows = re.sub(r'\s', '', matched_stripped)
    if old_nows != matched_nows:
        return text  # Content differs, can't learn spacing

    # Walk through matched_stripped and learn spacing around each operator char
    # by comparing with old_stripped (which has no/less spacing)
    # Operator groups: learn spacing for a group, apply to all members
    op_groups = {
        'arith': set('+-*/%'),
        'compare': set('=<>!'),
        'bitwise': set('&|^~'),
        'punct': set(',;:'),
    }
    char_to_group = {}
    for group, chars in op_groups.items():
        for c in chars:
            char_to_group[c] = group

    # Walk matched_stripped aligned with old_nows to learn spacing per group
    spacing_before_group = set()  # groups that have space before
    spacing_after_group = set()   # groups that have space after

    ni = 0
    for mc in range(len(matched_stripped)):
        if ni >= len(old_nows):
            break
        if matched_stripped[mc] == old_nows[ni]:
            ch = old_nows[ni]
            group = char_to_group.get(ch)
            if group:
                if mc > 0 and matched_stripped[mc-1] == ' ':
                    spacing_before_group.add(group)
                if mc + 1 < len(matched_stripped) and matched_stripped[mc+1] == ' ':
                    spacing_after_group.add(group)
            ni += 1

    if not spacing_before_group and not spacing_after_group:
        return text

    # Apply learned patterns to text
    result = []
    for i, ch in enumerate(text):
        group = char_to_group.get(ch)
        if group:
            before = group in spacing_before_group
            after = group in spacing_after_group
            if before and (not result or result[-1] != ' '):
                result.append(' ')
            result.append(ch)
            if after and i + 1 < len(text) and text[i+1] != ' ':
                result.append(' ')
        else:
            result.append(ch)
    return ''.join(result)


def _map_segment(old_str: str, matched_str: str, start: int, end: int) -> str:
    """Map a segment from old_str positions to the corresponding matched_str positions.

    Uses character-level alignment between old and matched strings.
    """
    sm = difflib.SequenceMatcher(None, old_str, matched_str)
    # Find where old[start:end] maps to in matched
    matched_start = None
    matched_end = None

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('equal', 'replace'):
            # Check overlap with [start, end)
            overlap_start = max(i1, start)
            overlap_end = min(i2, end)
            if overlap_start < overlap_end:
                # Map proportionally
                if i2 > i1:
                    ratio_start = (overlap_start - i1) / (i2 - i1)
                    ratio_end = (overlap_end - i1) / (i2 - i1)
                    ms = j1 + int(ratio_start * (j2 - j1))
                    me = j1 + int(ratio_end * (j2 - j1))
                else:
                    ms = j1
                    me = j2
                if matched_start is None:
                    matched_start = ms
                matched_end = me

    if matched_start is not None and matched_end is not None:
        return matched_str[matched_start:matched_end]
    return old_str[start:end]


def _apply_line_change(old_line: str, new_line: str, matched_line: str) -> str:
    """Apply the change from old_line->new_line onto matched_line.

    If old_line and matched_line differ (stale context), apply the
    character-level diff from old->new onto matched.
    """
    # If old matches matched, just use new
    if old_line.strip() == matched_line.strip():
        # Preserve matched indentation
        mi = _get_line_indent(matched_line)
        ni = _get_line_indent(new_line)
        oi = _get_line_indent(old_line)
        if ni == oi:
            return mi + new_line.lstrip()
        return new_line

    # old and matched differ: apply old->new diff onto matched
    old_s = old_line.rstrip('\n\r')
    new_s = new_line.rstrip('\n\r')
    matched_s = matched_line.rstrip('\n\r')
    ending = matched_line[len(matched_s):]

    sm_on = difflib.SequenceMatcher(None, old_s, new_s)
    # Align old to matched
    sm_om = difflib.SequenceMatcher(None, old_s, matched_s)

    # Build position map: old char index -> matched char index
    old_to_m_pos = {}
    for tag, i1, i2, j1, j2 in sm_om.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                old_to_m_pos[i1 + k] = j1 + k
        elif tag == 'replace':
            # Approximate mapping
            old_len = i2 - i1
            m_len = j2 - j1
            for k in range(old_len):
                mk = j1 + int(k * m_len / old_len) if old_len > 0 else j1
                old_to_m_pos[i1 + k] = mk

    # Apply old->new opcodes but using matched content for 'equal' parts
    result_chars = []
    for tag, i1, i2, j1, j2 in sm_on.get_opcodes():
        if tag == 'equal':
            # Use matched version of these characters
            m_start = old_to_m_pos.get(i1)
            m_end = old_to_m_pos.get(i2 - 1)
            if m_start is not None and m_end is not None:
                result_chars.append(matched_s[m_start:m_end + 1])
            else:
                result_chars.append(old_s[i1:i2])
        elif tag == 'replace':
            result_chars.append(new_s[j1:j2])
        elif tag == 'insert':
            result_chars.append(new_s[j1:j2])
        # delete: skip

    return ''.join(result_chars) + ending


def _build_token_mapping(old_text: str, matched_text: str) -> dict:
    """Build a mapping of tokens that differ between old_text and matched_text.

    This captures variable renames and other identifier changes so we can
    adapt new_text to use the actual file's identifiers.
    """
    old_lines = old_text.splitlines()
    matched_lines = matched_text.splitlines()

    # Align lines
    sm = difflib.SequenceMatcher(None, old_lines, matched_lines)
    token_map = {}

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace' and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                ol = old_lines[i1 + k].strip()
                ml = matched_lines[j1 + k].strip()
                if ol != ml:
                    # Find differing tokens
                    _extract_token_diffs(ol, ml, token_map)

    return token_map


def _extract_token_diffs(old_line: str, matched_line: str, mapping: dict):
    """Extract token-level differences between two lines."""
    # Use word-level tokenization (include hyphens for CSS-style identifiers)
    old_tokens = re.findall(r'[\w.\-]+|[^\w\s\-]+|\s+', old_line)
    matched_tokens = re.findall(r'[\w.\-]+|[^\w\s\-]+|\s+', matched_line)

    sm = difflib.SequenceMatcher(None, old_tokens, matched_tokens)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            old_chunk = ''.join(old_tokens[i1:i2]).strip()
            matched_chunk = ''.join(matched_tokens[j1:j2]).strip()
            if old_chunk and matched_chunk and old_chunk != matched_chunk:
                # Map identifier-like tokens (including hyphenated like CSS classes)
                if re.match(r'^[\w.\-]+$', old_chunk) and re.match(r'^[\w.\-]+$', matched_chunk):
                    mapping[old_chunk] = matched_chunk


def _apply_token_mapping(text: str, mapping: dict) -> str:
    """Apply token mapping to text, replacing old identifiers with actual ones."""
    if not mapping:
        return text
    # Sort by length (longest first) to avoid partial replacements
    for old_tok, new_tok in sorted(mapping.items(), key=lambda x: -len(x[0])):
        # Replace exact occurrences — use word boundary aware replacement
        # For dotted identifiers like self.db -> self.db_conn,
        # we need to match self.db but not when it's already self.db_conn
        # Check if it's not already replaced by checking what follows
        pattern = re.escape(old_tok) + r'(?!' + re.escape(new_tok[len(old_tok):]) + r')' if new_tok.startswith(old_tok) else re.escape(old_tok)
        # Ensure we don't match in the middle of a longer identifier
        # But allow matching at word/dot/hyphen boundaries
        pattern = r'(?<![\w\-])' + pattern + r'(?=[\W]|$)'
        text = re.sub(pattern, new_tok, text)
    return text


def _apply_diff_based_edit(old_text: str, new_text: str, matched_text: str) -> Optional[str]:
    """Apply the conceptual diff between old_text and new_text to matched_text.

    This handles stale context: old_text doesn't exactly match matched_text,
    but the structural edit (additions/removals) should still apply.
    """
    # Build token mapping for stale context adaptation
    token_map = _build_token_mapping(old_text, matched_text)

    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    matched_lines = matched_text.splitlines(keepends=True)

    if not old_lines or not new_lines or not matched_lines:
        return None

    # Ensure trailing newlines are consistent for difflib
    if old_lines and not old_lines[-1].endswith('\n'):
        old_lines[-1] += '\n'
    if new_lines and not new_lines[-1].endswith('\n'):
        new_lines[-1] += '\n'
    if matched_lines and not matched_lines[-1].endswith('\n'):
        matched_lines[-1] += '\n'

    # Use SequenceMatcher to get opcodes between old and new
    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = sm.get_opcodes()

    # Also align old_lines to matched_lines
    sm2 = difflib.SequenceMatcher(None, old_lines, matched_lines)
    old_to_matched = {}  # old line index -> matched line index
    for tag, i1, i2, j1, j2 in sm2.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                old_to_matched[i1 + k] = j1 + k
        elif tag == 'replace':
            for k in range(min(i2 - i1, j2 - j1)):
                old_to_matched[i1 + k] = j1 + k

    # Apply the old->new opcodes, but using matched_lines instead of old_lines
    result = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Use matched version of these lines (preserves actual file content)
            for i in range(i1, i2):
                if i in old_to_matched:
                    idx = old_to_matched[i]
                    if idx < len(matched_lines):
                        result.append(matched_lines[idx])
                        continue
                result.append(old_lines[i])
        elif tag == 'replace':
            # Lines changed: apply the change onto matched lines
            old_chunk = old_lines[i1:i2]
            new_chunk = new_lines[j1:j2]
            matched_chunk = []
            for i in range(i1, i2):
                if i in old_to_matched and old_to_matched[i] < len(matched_lines):
                    matched_chunk.append(matched_lines[old_to_matched[i]])
                else:
                    matched_chunk.append(old_lines[i])

            # For 1:1 replacements, adapt each line
            if len(old_chunk) == len(new_chunk) == len(matched_chunk):
                for oc, nc, mc in zip(old_chunk, new_chunk, matched_chunk):
                    adapted = _apply_line_change(oc, nc, mc)
                    adapted = _apply_token_mapping(adapted, token_map)
                    result.append(adapted)
            else:
                # Different line counts: use new lines with indent/token adaptation
                for j in range(j1, j2):
                    new_line = _apply_token_mapping(new_lines[j], token_map)
                    old_idx = i1 + (j - j1) if (i1 + (j - j1)) < i2 else i1
                    if old_idx in old_to_matched:
                        midx = old_to_matched[old_idx]
                        if midx < len(matched_lines):
                            old_indent = _get_line_indent(old_lines[old_idx])
                            matched_indent = _get_line_indent(matched_lines[midx])
                            new_indent = _get_line_indent(new_line)
                            if new_indent == old_indent and old_indent != matched_indent:
                                new_line = matched_indent + new_line.lstrip()
                    result.append(new_line)
        elif tag == 'insert':
            # New lines being added: adapt indentation and tokens
            for j in range(j1, j2):
                new_line = _apply_token_mapping(new_lines[j], token_map)
                # Use surrounding context for indentation hints
                if i1 > 0 and (i1 - 1) in old_to_matched:
                    prev_old_indent = _get_line_indent(old_lines[i1 - 1])
                    prev_matched_indent = _get_line_indent(
                        matched_lines[old_to_matched[i1 - 1]]
                    ) if old_to_matched[i1 - 1] < len(matched_lines) else prev_old_indent
                    new_indent = _get_line_indent(new_line)
                    if prev_old_indent != prev_matched_indent:
                        # There's an indent shift; try to apply it
                        if new_indent.startswith(prev_old_indent):
                            extra = new_indent[len(prev_old_indent):]
                            new_line = prev_matched_indent + extra + new_line.lstrip()
                result.append(new_line)
        elif tag == 'delete':
            # Lines removed — skip them
            pass

    result_text = ''.join(result)
    # Remove trailing newline we may have added
    if not matched_text.endswith('\n') and result_text.endswith('\n'):
        result_text = result_text[:-1]

    return result_text


def validate_syntax(file_path: str, content: str) -> Tuple[bool, Optional[str]]:
    """Validate syntax of content based on file extension.

    Returns (valid, error_message). If valid is True, error_message is None.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.py':
        try:
            compile(content, file_path, 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"Python syntax error: {e}"

    elif ext == '.json':
        try:
            json.loads(content)
            return True, None
        except (json.JSONDecodeError, ValueError) as e:
            return False, f"JSON syntax error: {e}"

    elif ext in ('.xml', '.html', '.htm'):
        try:
            ET.fromstring(content)
            return True, None
        except ET.ParseError as e:
            return False, f"XML/HTML parse error: {e}"

    elif ext in ('.yaml', '.yml'):
        try:
            import yaml
            yaml.safe_load(content)
            return True, None
        except ImportError:
            return True, None  # skip if PyYAML not available
        except Exception as e:
            return False, f"YAML syntax error: {e}"

    elif ext in ('.js', '.ts', '.jsx', '.tsx'):
        return _validate_js_syntax(content)

    # Generic fallback: always passes
    return True, None


def _validate_js_syntax(content: str) -> Tuple[bool, Optional[str]]:
    """Basic JS/TS syntax validation: bracket/brace/paren balance + unclosed strings."""
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    openers = set('([{')
    in_single_line_comment = False
    in_multi_line_comment = False
    in_string = None  # None, or the quote char (' " `)
    i = 0
    while i < len(content):
        ch = content[i]

        # Handle comments
        if in_single_line_comment:
            if ch == '\n':
                in_single_line_comment = False
            i += 1
            continue
        if in_multi_line_comment:
            if ch == '*' and i + 1 < len(content) and content[i + 1] == '/':
                in_multi_line_comment = False
                i += 2
                continue
            i += 1
            continue

        # Handle strings
        if in_string:
            if ch == '\\':
                i += 2  # skip escaped char
                continue
            if ch == in_string:
                in_string = None
            elif ch == '\n' and in_string != '`':
                return False, f"Unclosed string literal at position {i}"
            i += 1
            continue

        # Start of comment
        if ch == '/' and i + 1 < len(content):
            next_ch = content[i + 1]
            if next_ch == '/':
                in_single_line_comment = True
                i += 2
                continue
            elif next_ch == '*':
                in_multi_line_comment = True
                i += 2
                continue

        # Start of string
        if ch in ('"', "'", '`'):
            in_string = ch
            i += 1
            continue

        # Brackets
        if ch in openers:
            stack.append(ch)
        elif ch in pairs:
            if not stack:
                return False, f"Unmatched closing '{ch}' at position {i}"
            if stack[-1] != pairs[ch]:
                return False, f"Mismatched '{ch}' at position {i}, expected closing for '{stack[-1]}'"
            stack.pop()

        i += 1

    if in_string:
        return False, "Unclosed string literal at end of file"
    if stack:
        return False, f"Unclosed '{stack[-1]}' at end of file"
    return True, None


def _compute_diff(old_content: str, new_content: str, file_path: str) -> str:
    """Compute a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff_lines = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{os.path.basename(file_path)}",
        tofile=f"b/{os.path.basename(file_path)}",
    )
    return ''.join(diff_lines)


def apply_edit(
    file_path: str,
    old_text: str,
    new_text: str,
    threshold: float = 0.8,
    dry_run: bool = False,
    validate: bool = False,
) -> EditResult:
    """Apply a single edit to a file."""
    try:
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        content = raw_bytes.decode('utf-8', errors='replace')
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

    # Detect CRLF before normalizing
    use_crlf = b'\r\n' in raw_bytes

    # Normalize line endings for matching
    content_normalized = content.replace('\r\n', '\n')
    old_normalized = old_text.replace('\r\n', '\n')
    new_normalized = new_text.replace('\r\n', '\n')

    try:
        match = find_best_match(content_normalized, old_normalized, threshold)
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

    # Adapt replacement text based on match type
    adapted_new = _adapt_replacement(
        old_normalized, new_normalized, match.matched_text, match.match_type
    )

    new_content = content_normalized[:match.start] + adapted_new + content_normalized[match.end:]

    # Restore original line endings if file used CRLF
    if use_crlf:
        new_content = new_content.replace('\n', '\r\n')

    # Compute diff
    diff_text = _compute_diff(content, new_content, file_path)

    # Validate if requested
    if validate:
        valid, err = validate_syntax(file_path, new_content)
        if not valid:
            # Rollback: don't write, return validation error
            return EditResult(
                status="validation_error",
                file=file_path,
                match_type=match.match_type,
                confidence=match.confidence,
                matched_text=match.matched_text,
                error=err,
                diff=diff_text,
            )

    if not dry_run:
        if use_crlf:
            with open(file_path, 'wb') as f:
                f.write(new_content.encode('utf-8'))
        else:
            with open(file_path, 'w') as f:
                f.write(new_content)

    result = EditResult(
        status="applied",
        file=file_path,
        match_type=match.match_type,
        confidence=match.confidence,
        matched_text=match.matched_text,
        diff=diff_text,
    )
    if validate:
        result.validated = True
    return result


def create_file(
    file_path: str,
    content: str,
    force: bool = False,
    validate: bool = False,
) -> EditResult:
    """Create a new file with the given content.

    Fails if file already exists unless force=True.
    Optionally validates syntax before writing.
    """
    if os.path.exists(file_path) and not force:
        return EditResult(
            status="error",
            file=file_path,
            error=f"File already exists: {file_path} (use --force to overwrite)",
        )

    if validate:
        valid, err = validate_syntax(file_path, content)
        if not valid:
            return EditResult(
                status="validation_error",
                file=file_path,
                error=err,
            )

    # Create parent directories if needed
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, 'w') as f:
        f.write(content)

    result = EditResult(
        status="created",
        file=file_path,
    )
    if validate:
        result.validated = True
    return result


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
    if result.validated is not None:
        d["validated"] = result.validated
    if result.diff is not None:
        d["diff"] = result.diff
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
    apply_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate syntax after applying (rollback on failure)",
    )
    apply_parser.add_argument(
        "--diff",
        action="store_true",
        help="Print unified diff to stderr",
    )

    # Create command
    create_parser = sub.add_parser("create", help="Create a new file")
    create_parser.add_argument("--file", help="Target file path", required=True)
    create_parser.add_argument("--content", help="File content")
    create_parser.add_argument(
        "--stdin", action="store_true", help="Read content from stdin (or JSON with action=create)"
    )
    create_parser.add_argument(
        "--force", action="store_true", help="Overwrite if file exists"
    )
    create_parser.add_argument(
        "--validate", action="store_true", help="Validate syntax before writing"
    )

    # Validate command
    validate_parser = sub.add_parser("validate", help="Validate a file's syntax")
    validate_parser.add_argument("file", help="File to validate")

    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            with open(args.file, 'r') as f:
                content = f.read()
        except (FileNotFoundError, OSError) as e:
            print(json.dumps({"status": "error", "file": args.file, "error": str(e)}))
            return 1
        valid, err = validate_syntax(args.file, content)
        result = {"status": "valid" if valid else "invalid", "file": args.file}
        if err:
            result["error"] = err
        print(json.dumps(result, indent=2))
        return 0 if valid else 1

    if args.command == "create":
        if args.stdin:
            raw = sys.stdin.read()
            # Try JSON first
            try:
                data = json.loads(raw)
                content = data.get("content", raw)
                file_path = data.get("file", args.file)
            except (json.JSONDecodeError, ValueError):
                content = raw
                file_path = args.file
        elif args.content is not None:
            content = args.content
            file_path = args.file
        else:
            print(json.dumps({"status": "error", "error": "Must provide --content or --stdin"}))
            return 1

        result = create_file(
            file_path, content,
            force=args.force,
            validate=args.validate,
        )
        print(json.dumps(result_to_dict(result), indent=2))
        return 0 if result.status == "created" else 1

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
            validate=args.validate,
        )

        # Print diff to stderr if requested
        if args.diff and result.diff:
            print(result.diff, file=sys.stderr, end='')

        results.append(result_to_dict(result))

        if result.status in ("no_match", "error", "validation_error"):
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
