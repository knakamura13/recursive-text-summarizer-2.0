"""A compatibility-folded view of a text, for finding words the text layer spelled differently.

A PDF text layer may spell "fi" or "ff" as one ligature character (U+FB01, U+FB00), which
case-insensitive matching does not treat as the letters. NFKC folds it, and the other
compatibility forms, to the plain characters. Folding changes lengths (the "fi" ligature
becomes two characters), so offsets in the folded view are not offsets in the source. The
canonical text must not change, so a match is found in the folded view and its offsets are
mapped back to the source.

Each character is folded alone, so every folded character belongs to exactly one source
character and the mapping stays exact.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass


def compat_fold(text: str) -> str:
    """`text` with each character folded to its NFKC form."""
    return "".join(unicodedata.normalize("NFKC", char) for char in text)


@dataclass(frozen=True)
class FoldedText:
    """`text` folded, with where each changed source character sits in it.

    The three lists run in step, one entry per source character that folding changed, in
    source order: its source offset, the offset where its folding begins in `text`, and
    that folding's length. Every other character folds to itself.
    """

    text: str
    source_at: list[int]
    folded_at: list[int]
    folded_len: list[int]

    @classmethod
    def of(cls, source: str) -> FoldedText:
        changed = {
            char: folded
            for char in set(source)
            if (folded := unicodedata.normalize("NFKC", char)) != char
        }
        if not changed:
            return cls(source, [], [], [])
        pattern = re.compile("[" + re.escape("".join(changed)) + "]")
        parts: list[str] = []
        source_at: list[int] = []
        folded_at: list[int] = []
        folded_len: list[int] = []
        taken = 0  # Source characters copied so far.
        growth = 0  # Folded length minus source length over what was copied.
        for match in pattern.finditer(source):
            start = match.start()
            folded = changed[match.group()]
            parts.append(source[taken:start])
            parts.append(folded)
            source_at.append(start)
            folded_at.append(start + growth)
            folded_len.append(len(folded))
            growth += len(folded) - 1
            taken = start + 1
        parts.append(source[taken:])
        return cls("".join(parts), source_at, folded_at, folded_len)

    def to_folded(self, offset: int) -> int:
        """Where the source character at `offset` begins in the folded text."""
        before = bisect.bisect_left(self.source_at, offset) - 1  # The last change before it.
        if before < 0:
            return offset
        after_change = self.folded_at[before] + self.folded_len[before]
        return after_change + (offset - self.source_at[before] - 1)

    def to_source(self, offset: int) -> int:
        """The offset of the source character that the folded character at `offset` came from."""
        change = bisect.bisect_right(self.folded_at, offset) - 1  # The last change at or before it.
        if change < 0:
            return offset
        after_change = self.folded_at[change] + self.folded_len[change]
        if offset < after_change:
            return self.source_at[change]
        return self.source_at[change] + 1 + (offset - after_change)

    def span_to_source(self, start: int, end: int) -> tuple[int, int]:
        """The source span of the characters that a non-empty folded span `[start, end)` came from."""
        return self.to_source(start), self.to_source(end - 1) + 1
