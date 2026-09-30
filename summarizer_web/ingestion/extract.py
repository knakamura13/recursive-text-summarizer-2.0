"""Import extraction: pick the extractor for a format, assemble canonical text,
its page map and its outline, and describe the result in an Import report.

Canonical text always comes from `summarizer.ingestion.ingest_text`. For paged
formats (PDF, images) the page texts are joined with the page markers
"--- Page N ---" (none before the first page). Every piece is already a fixed
point of the pipeline's normalization, so page offsets computed while joining
are exact code-point offsets into the final canonical text.

The outline never changes the canonical text. Extractors report headings as
hints, and each is placed by finding its title in the finished text, so its
offsets hold whatever the extractor did to whitespace.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from summarizer.ingestion import ingest_text
from summarizer.segmentation import detect_markdown_headings
from summarizer_web.config import EXTRACTION_VERSION
from summarizer_web.ingestion.common import (
    Extraction,
    ImportFailure,
    OutlineHint,
    ProgressCallback,
    clean_text,
    has_text,
    ignore_progress,
    plural,
)
from summarizer_web.ingestion.encoding import DecodedText, decode_text
from summarizer_web.ingestion.markup import declared_charset, html_to_text
from summarizer_web.ingestion.ocr import Tesseract, extract_image
from summarizer_web.ingestion.packages import extract_docx, extract_epub, extract_odt
from summarizer_web.ingestion.pdf import extract_pdf
from summarizer_web.ingestion.plain import rtf_to_plain, srt_to_text, vtt_to_text
from summarizer_web.models.api import DocumentFormat, ImportReport, Notice

_EMPTY_MESSAGES: dict[str, str] = {
    "txt": "The file contains no text.",
    "md": "The file contains no text.",
    "srt": "The subtitle file contains no cue text.",
    "vtt": "The subtitle file contains no cue text.",
    "html": "The HTML file has no visible text.",
    "rtf": "The document contains no text.",
    "docx": "The document contains no text.",
    "odt": "The document contains no text.",
    "epub": "The e-book contains no text.",
}
_PREVIEW_CHARS = 800
_WORD_COUNT_CHUNK = 1 << 20


def page_marker(number: int) -> str:
    return f"--- Page {number} ---"


@dataclass(frozen=True)
class PageSpan:
    """Where a page's text sits in the canonical text (end exclusive)."""

    page: int
    start: int
    end: int
    ocr: bool
    blank: bool

    def as_json(self) -> dict[str, int | bool]:
        return {
            "page": self.page,
            "start": self.start,
            "end": self.end,
            "ocr": self.ocr,
            "blank": self.blank,
        }


@dataclass(frozen=True)
class OutlineEntry:
    """A heading placed in the canonical text.

    `start` is where its title begins. Its section runs to `end` (exclusive):
    the start of the next heading at the same or a higher level, or the end
    of the text. Pages are 1-based and inclusive, and None without pages.
    """

    title: str
    level: int
    start: int
    end: int
    page_start: int | None
    page_end: int | None

    def as_json(self) -> dict[str, str | int | None]:
        return {
            "title": self.title,
            "level": self.level,
            "start": self.start,
            "end": self.end,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


@dataclass(frozen=True)
class ImportedText:
    text: str
    source_id: str
    pages: list[PageSpan] | None
    word_count: int
    outline: tuple[OutlineEntry, ...] = ()
    # Hints whose title was not found in the text.
    unplaced_headings: int = 0


def _title_patterns(title: str) -> tuple[re.Pattern[str], ...] | None:
    """Patterns for a title with any whitespace between its words, ignoring
    case, from most to least likely a heading: the title as a whole line,
    then at the start of a line, then anywhere. Each needs whole words, so
    "Intro" does not match inside "Introduction"."""
    words = clean_text(title).split()
    if not words:
        return None
    body = r"(?<!\w)" + r"\s+".join(re.escape(word) for word in words) + r"(?!\w)"
    return (
        re.compile(rf"^[ \t]*({body})[ \t]*$", re.IGNORECASE | re.MULTILINE),
        re.compile(rf"^[ \t]*({body})", re.IGNORECASE | re.MULTILINE),
        re.compile(f"({body})", re.IGNORECASE),
    )


def _find_title(
    patterns: tuple[re.Pattern[str], ...], text: str, start: int, end: int
) -> tuple[int, int] | None:
    """The title's first occurrence in `text[start:end]` under the most likely pattern."""
    for pattern in patterns:
        match = pattern.search(text, start, end)
        if match is not None:
            return match.span(1)
    return None


def anchor_outline(
    text: str, hints: Sequence[OutlineHint], pages: Sequence[PageSpan] | None
) -> tuple[tuple[OutlineEntry, ...], int]:
    """Place each hint in `text`; returns the outline and how many hints were not found.

    Hints are searched in order, each after the previous one's title, so a
    repeated title lands on its next occurrence. A hint with a page is
    searched within that page, after the previous title when that is on the
    page. Two hints found at one place keep the first.
    """
    by_page = {span.page: span for span in pages or () if not span.blank}
    placed: dict[int, tuple[str, int]] = {}
    cursor = 0
    unplaced = 0
    for hint in hints:
        patterns = _title_patterns(hint.title)
        found = None
        if patterns is not None:
            if hint.page is None:
                found = _find_title(patterns, text, cursor, len(text))
            elif (span := by_page.get(hint.page)) is not None:
                found = _find_title(
                    patterns, text, max(cursor, span.start), span.end
                ) or _find_title(patterns, text, span.start, span.end)
        if found is None or found[0] in placed:
            unplaced += 1
            continue
        start, stop = found
        placed[start] = (" ".join(text[start:stop].split()), hint.level)
        cursor = max(cursor, stop)

    text_pages = [span for span in pages or () if not span.blank]
    page_starts = [span.start for span in text_pages]

    def page_at(offset: int) -> int | None:
        index = bisect.bisect_right(page_starts, offset) - 1
        return text_pages[max(index, 0)].page if text_pages else None

    starts = sorted(placed)
    ends = [len(text)] * len(starts)
    open_sections: list[int] = []
    for index, start in enumerate(starts):
        level = placed[start][1]
        while open_sections and placed[starts[open_sections[-1]]][1] >= level:
            ends[open_sections.pop()] = start
        open_sections.append(index)
    outline = tuple(
        OutlineEntry(
            title=placed[start][0],
            level=placed[start][1],
            start=start,
            end=end,
            page_start=page_at(start),
            page_end=page_at(end - 1),
        )
        for start, end in zip(starts, ends, strict=True)
    )
    return outline, unplaced


def count_words(text: str) -> int:
    """Whitespace-separated words, counted in bounded chunks."""
    count = 0
    for start in range(0, len(text), _WORD_COUNT_CHUNK):
        chunk = text[start : start + _WORD_COUNT_CHUNK]
        count += len(chunk.split())
        if start and not text[start - 1].isspace() and not chunk[0].isspace():
            count -= 1  # A word straddling the boundary was counted twice.
    return count


def assemble(extraction: Extraction) -> ImportedText:
    if extraction.pages is None:
        text = clean_text(extraction.text or "")
        if not has_text(text):
            raise ImportFailure(_EMPTY_MESSAGES.get(extraction.format, "The file contains no text."))
        document = ingest_text(text)
        outline, unplaced = anchor_outline(document.text, extraction.outline_hints, None)
        return ImportedText(
            document.text, document.source_id, None, count_words(document.text), outline, unplaced
        )

    parts: list[str] = []
    spans: list[PageSpan] = []
    position = 0

    def append(piece: str) -> int:
        nonlocal position
        if parts:
            parts.append("\n\n")
            position += 2
        start = position
        parts.append(piece)
        position += len(piece)
        return start

    words = 0
    for page in extraction.pages:
        if spans:
            append(page_marker(page.number))
        if has_text(page.text):
            start = append(page.text)
            spans.append(PageSpan(page.number, start, position, page.ocr, False))
            words += count_words(page.text)
        else:
            spans.append(PageSpan(page.number, position, position, page.ocr, True))
    if words == 0:
        raise ImportFailure("No text was found in the file.")
    text = "".join(parts)
    document = ingest_text(text)
    if document.text != text:
        raise RuntimeError("assembled page text is not in canonical form; page offsets would shift")
    outline, unplaced = anchor_outline(document.text, extraction.outline_hints, spans)
    return ImportedText(document.text, document.source_id, spans, words, outline, unplaced)


def _decoding_notices(decoded: DecodedText) -> list[Notice]:
    notices: list[Notice] = []
    if decoded.detected:
        notices.append(
            Notice(
                code="encoding_detected",
                message=f"The file is not UTF-8; it was read as {decoded.encoding} (detected automatically).",
            )
        )
    if decoded.replaced:
        notices.append(
            Notice(
                code="decoding_replacements",
                severity="warning",
                message=(
                    f"{plural(decoded.replaced, 'character')} could not be decoded "
                    "and were replaced with \u201c\ufffd\u201d."
                ),
            )
        )
    return notices


def extract_document(
    path: Path,
    document_format: DocumentFormat,
    progress: ProgressCallback = ignore_progress,
    *,
    tesseract: Tesseract | None,
) -> Extraction:
    if document_format == "pdf":
        return extract_pdf(path, progress, tesseract)
    if document_format in ("png", "jpeg", "tiff"):
        return extract_image(path, document_format, progress, tesseract)
    progress("reading")
    if document_format == "epub":
        return Extraction("epub", text=extract_epub(path, progress))
    if document_format in ("docx", "odt"):
        progress("extracting")
        reader = extract_docx if document_format == "docx" else extract_odt
        return Extraction(document_format, text=reader(path))
    data = path.read_bytes()
    if document_format == "rtf":
        progress("extracting")
        return Extraction("rtf", text=rtf_to_plain(data))
    decoded = decode_text(data, declared=declared_charset(data) if document_format == "html" else None)
    del data
    notices = _decoding_notices(decoded)
    if document_format in ("srt", "vtt"):
        progress("extracting")
        convert = srt_to_text if document_format == "srt" else vtt_to_text
        text, cues = convert(decoded.text)
        if cues:
            notices.append(
                Notice(
                    code="subtitle_cues",
                    message=f"Kept the text of {plural(cues, 'cue')}; cue numbers and timestamps were removed.",
                )
            )
    elif document_format == "html":
        progress("extracting")
        text = html_to_text(decoded.text)
    else:
        text = decoded.text
    hints = (
        [OutlineHint(heading.title, heading.level) for heading in detect_markdown_headings(text)]
        if document_format in ("txt", "md")
        else []
    )
    return Extraction(document_format, text=text, encoding=decoded.encoding, notices=notices, outline_hints=hints)


def preview_text(text: str) -> str:
    """The opening of a text for the Import report, cut at a word boundary."""
    if len(text) <= _PREVIEW_CHARS:
        return text
    cut = text[:_PREVIEW_CHARS]
    space = cut.rfind(" ")
    if space > _PREVIEW_CHARS - 80:
        cut = cut[:space]
    return cut.rstrip() + "\u2026"


def build_report(
    extraction: Extraction, imported: ImportedText, *, duration_seconds: float
) -> ImportReport:
    pages = imported.pages or []
    notices = list(extraction.notices)
    if imported.unplaced_headings:
        notices.append(
            Notice(
                code="headings_unplaced",
                message=(
                    f"The outline leaves out {plural(imported.unplaced_headings, 'heading')} "
                    "that could not be found in the text."
                ),
            )
        )
    return ImportReport(
        detected_format=extraction.format,
        encoding=extraction.encoding,
        page_count=len(pages) if imported.pages is not None else None,
        text_layer_pages=extraction.text_layer_pages,
        ocr_pages=[span.page for span in pages if span.ocr],
        blank_pages=[span.page for span in pages if span.blank],
        char_count=len(imported.text),
        word_count=imported.word_count,
        heading_count=len(imported.outline),
        unplaced_headings=imported.unplaced_headings,
        notices=notices,
        preview=preview_text(imported.text),
        extraction_version=EXTRACTION_VERSION,
        duration_seconds=round(duration_seconds, 2),
    )
