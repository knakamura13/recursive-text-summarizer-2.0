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
from array import array
from dataclasses import dataclass


def compat_fold(text: str) -> str:
    """`text` with each character folded to its NFKC form."""
    return "".join(unicodedata.normalize("NFKC", char) for char in text)


@dataclass(frozen=True)
class FoldedText:
    """`source` folded into `text`, with where each source character that folding resized sits.

    The three arrays run in step, one entry per source character whose folding is not one
    character long, in source order: its source offset, the offset where its folding begins
    in `text`, and that folding's length. A character that folds to one character, itself
    or another, keeps its offset and needs no entry.
    """

    source: str
    text: str
    source_at: array
    folded_at: array
    folded_len: array

    @classmethod
    def of(cls, source: str) -> FoldedText:
        changed = {
            char: folded
            for char in set(source)
            if (folded := unicodedata.normalize("NFKC", char)) != char
        }
        source_at, folded_at, folded_len = array("q"), array("q"), array("q")
        if not changed:
            return cls(source, source, source_at, folded_at, folded_len)
        resized = {char: len(folded) for char, folded in changed.items() if len(folded) != 1}
        if resized:
            pattern = re.compile("[" + re.escape("".join(resized)) + "]")
            growth = 0  # Folded length minus source length over the characters passed.
            for match in pattern.finditer(source):
                start, length = match.start(), resized[match.group()]
                source_at.append(start)
                folded_at.append(start + growth)
                folded_len.append(length)
                growth += length - 1
        text = source.translate({ord(char): folded for char, folded in changed.items()})
        return cls(source, text, source_at, folded_at, folded_len)

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

    def _inside_a_character(self, offset: int) -> bool:
        """True when the folded `offset` falls between the folded characters of one source character."""
        change = bisect.bisect_right(self.folded_at, offset) - 1
        return (
            change >= 0
            and self.folded_at[change] < offset < self.folded_at[change] + self.folded_len[change]
        )

    def _source_span(self, start: int, end: int) -> tuple[int, int] | None:
        """The source span of a non-empty folded span `[start, end)`, or None when it begins or
        ends inside one source character's folding (a title "2" is not half of "½")."""
        if self._inside_a_character(start) or self._inside_a_character(end):
            return None
        return self.to_source(start), self.to_source(end - 1) + 1

    def search(self, pattern: re.Pattern[str], start: int, end: int) -> tuple[int, int] | None:
        """The first match of `pattern`'s first group in the folded view of the source's
        `[start:end]` that does not cut a source character, as a source span."""
        for match in pattern.finditer(self.text, self.to_folded(start), self.to_folded(end)):
            span = self._source_span(*match.span(1))
            if span is not None:
                return span
        return None

    def match(self, pattern: re.Pattern[str], offset: int) -> tuple[int, int] | None:
        """`pattern`'s first group matched in the folded view at the source `offset`, as a source span."""
        match = pattern.match(self.text, self.to_folded(offset))
        return self._source_span(*match.span(1)) if match is not None else None
