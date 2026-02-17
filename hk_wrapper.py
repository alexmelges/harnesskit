"""
HarnessKit Python Wrapper — importable API for fuzzy file editing.

Zero dependencies (like hk.py itself). Import and use directly:

    from hk_wrapper import HarnessKit

    hk = HarnessKit()
    result = hk.edit("app.py", "def old_func():", "def new_func():")
    print(result.success, result.match_type)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

# Import core functions from hk.py (same directory)
from hk import (
    apply_edit,
    create_file as _create_file,
    validate_syntax,
    find_best_match,
    EditResult,
    MatchResult,
    AmbiguousMatchError,
    _compute_diff,
)


@dataclass
class EditResponse:
    """Result of an edit operation."""
    success: bool
    file: str
    match_type: Optional[str] = None  # exact, whitespace, fuzzy, line_fuzzy
    similarity: Optional[float] = None
    error: Optional[str] = None
    diff: Optional[str] = None
    validation: Optional[str] = None  # pass, fail, skipped

    def to_dict(self) -> dict:
        d = {"success": self.success, "file": self.file}
        if self.match_type:
            d["match_type"] = self.match_type
        if self.similarity is not None:
            d["similarity"] = self.similarity
        if self.error:
            d["error"] = self.error
        if self.diff:
            d["diff"] = self.diff
        if self.validation:
            d["validation"] = self.validation
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class HarnessKit:
    """
    Fuzzy file editing toolkit.

    Usage:
        hk = HarnessKit(threshold=0.8)
        result = hk.edit("file.py", old_text, new_text)
        result = hk.create("new_file.py", content)
        valid, msg = hk.validate("file.py")
        diff_str = hk.diff("file.py", old_text, new_text)
    """

    def __init__(self, threshold: float = 0.8, show_diff: bool = False, validate_after: bool = False):
        self.threshold = threshold
        self.show_diff = show_diff
        self.validate_after = validate_after

    def edit(
        self,
        file: str,
        old_text: str,
        new_text: str,
        threshold: Optional[float] = None,
        show_diff: Optional[bool] = None,
        validate_after: Optional[bool] = None,
    ) -> EditResponse:
        """
        Apply a fuzzy edit to a file.

        Args:
            file: Path to the file to edit.
            old_text: Text to find (fuzzy matched).
            new_text: Replacement text.
            threshold: Similarity threshold (0.0-1.0). Default from constructor.
            show_diff: Include unified diff in response.
            validate_after: Run syntax validation after edit.

        Returns:
            EditResponse with success status, match info, and optional diff/validation.
        """
        t = threshold if threshold is not None else self.threshold
        do_diff = show_diff if show_diff is not None else self.show_diff
        do_validate = validate_after if validate_after is not None else self.validate_after

        try:
            result = apply_edit(file, old_text, new_text, threshold=t, validate=do_validate)
        except Exception as e:
            return EditResponse(success=False, file=file, error=str(e))

        ok = result.status == "applied"

        resp = EditResponse(
            success=ok,
            file=file,
            match_type=result.match_type if ok else None,
            similarity=result.confidence if ok else None,
            error=result.error if not ok else None,
        )

        if do_diff and ok:
            # Re-read file to compute diff (edit already applied)
            # We can't get pre-edit content, so skip diff for now
            pass
        if do_validate and result.validated is not None:
            resp.validation = "pass" if result.validated else "fail"

        return resp

    def create(self, file: str, content: str, create_dirs: bool = True) -> EditResponse:
        """
        Create a new file with content.

        Args:
            file: Path for the new file.
            content: File content.
            create_dirs: Create parent directories if needed.

        Returns:
            EditResponse with success status.
        """
        try:
            if create_dirs:
                parent = os.path.dirname(file)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            result = _create_file(file, content)
            ok = result.status in ("created", "applied")
            return EditResponse(success=ok, file=file, error=result.error if not ok else None)
        except Exception as e:
            return EditResponse(success=False, file=file, error=str(e))

    def validate(self, file: str) -> tuple[bool, Optional[str]]:
        """
        Validate file syntax (Python, JS/TS, JSON, YAML, XML, HTML, CSS, TOML).

        Returns:
            (is_valid, error_message_or_none)
        """
        if not os.path.exists(file):
            return False, f"File not found: {file}"
        with open(file, "r") as f:
            content = f.read()
        return validate_syntax(file, content)

    def diff(self, file: str, old_text: str, new_text: str, threshold: Optional[float] = None) -> Optional[str]:
        """
        Preview the diff that would result from an edit, without applying it.

        Returns:
            Unified diff string, or None if match fails.
        """
        t = threshold if threshold is not None else self.threshold
        if not os.path.exists(file):
            return None
        with open(file, "r") as f:
            content = f.read()

        match = find_best_match(content, old_text, threshold=t)
        if not match:
            return None

        # Simulate the replacement
        start, end = match.start, match.end
        new_content = content[:start] + new_text + content[end:]
        return _compute_diff(content, new_content, file)

    def batch_edit(self, edits: List[dict]) -> List[EditResponse]:
        """
        Apply multiple edits. Each dict should have: file, old_text, new_text.
        Optional: threshold.

        Returns:
            List of EditResponse, one per edit. Stops on first failure.
        """
        results = []
        for edit in edits:
            resp = self.edit(
                file=edit["file"],
                old_text=edit["old_text"],
                new_text=edit["new_text"],
                threshold=edit.get("threshold"),
            )
            results.append(resp)
            if not resp.success:
                break
        return results

    # --- Tool definition for LLM APIs ---

    @staticmethod
    def anthropic_tool_schema() -> dict:
        """Return the Anthropic tool_use schema for HarnessKit edit."""
        return {
            "name": "edit_file",
            "description": (
                "Edit a file by fuzzy-matching old_text and replacing with new_text. "
                "Tolerates whitespace differences and minor hallucinations. "
                "Use for all code modifications."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Path to the file to edit"},
                    "old_text": {"type": "string", "description": "Text to find in the file (fuzzy matched)"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                },
                "required": ["file", "old_text", "new_text"],
            },
        }

    @staticmethod
    def openai_function_schema() -> dict:
        """Return the OpenAI function calling schema for HarnessKit edit."""
        return {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": (
                    "Edit a file by fuzzy-matching old_text and replacing with new_text. "
                    "Tolerates whitespace differences and minor hallucinations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the file to edit"},
                        "old_text": {"type": "string", "description": "Text to find in the file (fuzzy matched)"},
                        "new_text": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["file", "old_text", "new_text"],
                },
            },
        }
