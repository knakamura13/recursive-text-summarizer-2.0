"""Import extraction: pick the extractor for a format, assemble canonical text
and its page map, and describe the result in an Import report.

Canonical text always comes from `summarizer.ingestion.ingest_text`. For paged
formats (PDF, images) the page texts are joined with the page markers
"--- Page N ---" (none before the first page). Every piece is already a fixed
point of the pipeline's normalization, so page offsets computed while joining
are exact code-point offsets into the final canonical text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from summarizer.ingestion import ingest_text
from summarizer_web.config import EXTRACTION_VERSION
from summarizer_web.ingestion.common import (
    Extraction,
    ImportFailure,
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
class ImportedText:
    text: str
    source_id: str
    pages: list[PageSpan] | None
    word_count: int


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
        return ImportedText(document.text, document.source_id, None, count_words(document.text))

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
    return ImportedText(document.text, document.source_id, spans, words)


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
    return Extraction(document_format, text=text, encoding=decoded.encoding, notices=notices)


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
    return ImportReport(
        detected_format=extraction.format,
        encoding=extraction.encoding,
        page_count=len(pages) if imported.pages is not None else None,
        text_layer_pages=extraction.text_layer_pages,
        ocr_pages=[span.page for span in pages if span.ocr],
        blank_pages=[span.page for span in pages if span.blank],
        char_count=len(imported.text),
        word_count=imported.word_count,
        notices=extraction.notices,
        preview=preview_text(imported.text),
        extraction_version=EXTRACTION_VERSION,
        duration_seconds=round(duration_seconds, 2),
    )
