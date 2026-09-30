"""Controlled summary degradations for the signal check in issue #150."""

from __future__ import annotations

import re


def drop_phrase(summary: str, phrase: str) -> str:
    """Remove one qualifier phrase (and the comma or space before it); it must be present."""
    pattern = re.compile(r"[,;]?\s*" + re.escape(phrase), re.IGNORECASE)
    if not pattern.search(summary):
        raise ValueError(f"phrase not in summary: {phrase!r}")
    return pattern.sub("", summary, count=1)


def drop_sentences_with(summary: str, needle: str) -> str:
    """Remove every sentence that mentions the needle (for example a named entity)."""
    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    kept = [s for s in sentences if needle.lower() not in s.lower()]
    if len(kept) == len(sentences):
        raise ValueError(f"needle not in summary: {needle!r}")
    return " ".join(kept)
