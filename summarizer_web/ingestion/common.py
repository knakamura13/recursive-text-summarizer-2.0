"""Shared building blocks of an Import: the failure type, progress reporting,
structural text blocks, pages, and the result every format extractor returns."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from summarizer.ingestion import normalize_source_text
from summarizer_web.models.api import DocumentFormat, ImportPhase, Notice

ProgressUnit = Literal["pages", "bytes"]


class ImportFailure(Exception):
    """The Import cannot produce a Document. `message` is shown to the user."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ProgressCallback(Protocol):
    def __call__(
        self,
        phase: ImportPhase,
        done: int = 0,
        total: int | None = None,
        *,
        unit: ProgressUnit | None = None,
        message: str | None = None,
    ) -> None: ...


def ignore_progress(
    phase: ImportPhase,
    done: int = 0,
    total: int | None = None,
    *,
    unit: ProgressUnit | None = None,
    message: str | None = None,
) -> None:
    """Progress sink for callers that do not display progress."""


def _clean_table() -> dict[int, str | None]:
    table: dict[int, str | None] = {
        code: None for code in range(0x00, 0x20) if code not in (0x09, 0x0A, 0x0D)
    }
    table.update({code: None for code in range(0x7F, 0xA0)})
    # Page and paragraph separators of other conventions become line breaks.
    for code in (0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029):
        table[code] = "\n"
    # Byte-order marks and soft hyphens are invisible and break quote matching.
    table[0xFEFF] = None
    table[0x00AD] = None
    return table


_CLEAN_TABLE = _clean_table()
_ALNUM = re.compile(r"[^\W_]")


def clean_text(text: str) -> str:
    """Remove control characters, then apply the pipeline's normalization.

    The result is a fixed point of `normalize_source_text`, so joining cleaned
    pieces with blank lines never shifts offsets during canonicalization.
    """
    return normalize_source_text(text.translate(_CLEAN_TABLE))


def has_text(text: str) -> bool:
    """True when the text holds at least one letter or digit."""
    return _ALNUM.search(text) is not None


def collapse_spaces(text: str) -> str:
    """Collapse runs of whitespace inside each line; keep explicit line breaks."""
    lines = (" ".join(line.split()) for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


BlockKind = Literal["paragraph", "heading", "list_item", "table_row", "preformatted"]
_TIGHT_KINDS = frozenset({"list_item", "table_row"})


@dataclass(frozen=True)
class Block:
    """One structural unit of a document: a paragraph, heading, list item,
    table row, or preformatted section. `breaks_before` starts a new list or
    table right after another one."""

    kind: BlockKind
    text: str
    breaks_before: bool = False


def render_blocks(blocks: Iterable[Block]) -> str:
    """Join blocks with a blank line; the items of one list and the rows of
    one table are separated by a single line break."""
    parts: list[str] = []
    previous: BlockKind | None = None
    for block in blocks:
        text = block.text.strip("\n")
        if not text.strip():
            continue
        if parts:
            tight = (
                block.kind == previous and block.kind in _TIGHT_KINDS and not block.breaks_before
            )
            parts.append("\n" if tight else "\n\n")
        parts.append(text)
        previous = block.kind
    return "".join(parts)


@dataclass(frozen=True)
class PageText:
    """Text of one page (PDF page or image frame), already `clean_text`-ed.
    `ocr` is true when Tesseract recognized the page."""

    number: int
    text: str
    ocr: bool = False


@dataclass
class Extraction:
    """What an extractor produced: flowing `text` or a list of `pages`."""

    format: DocumentFormat
    text: str | None = None
    pages: list[PageText] | None = None
    encoding: str | None = None
    text_layer_pages: int | None = None
    notices: list[Notice] = field(default_factory=list)


def format_page_ranges(pages: Sequence[int], *, limit: int = 12) -> str:
    """Render sorted page numbers compactly: [1, 2, 3, 7] -> "1–3, 7"."""
    ranges: list[str] = []
    ordered = sorted(set(pages))
    index = 0
    while index < len(ordered):
        start = ordered[index]
        end = start
        while index + 1 < len(ordered) and ordered[index + 1] == end + 1:
            index += 1
            end = ordered[index]
        ranges.append(str(start) if start == end else f"{start}–{end}")
        index += 1
    if len(ranges) > limit:
        return ", ".join(ranges[:limit]) + f", and {len(ranges) - limit} more"
    return ", ".join(ranges)


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    word = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count:,} {word}"
