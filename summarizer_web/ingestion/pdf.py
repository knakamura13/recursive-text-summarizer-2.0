"""Bounded PDF text extraction for the application layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from summarizer.ingestion import ingest_text, normalize_source_text
from summarizer_web.config import (
    EXTRACTION_VERSION,
    MAX_PDF_PAGES,
    PDF_EXTRACTION_TIMEOUT_SECONDS,
)


class PdfImportError(ValueError):
    """Raised when a PDF cannot be imported."""


@dataclass(frozen=True)
class PdfExtractionResult:
    canonical_text: str
    source_id: str
    page_map: list[dict[str, int]]
    blank_pages: list[int]
    page_count: int
    extraction_version: str = EXTRACTION_VERSION


def _page_separator(page_number: int) -> str:
    return f"\n\n--- Page {page_number} ---\n\n"


def extract_pdf(path: Path) -> PdfExtractionResult:
    try:
        reader = PdfReader(str(path), strict=True)
    except PdfReadError as error:
        raise PdfImportError(f"malformed PDF: {error}") from error
    if reader.is_encrypted:
        raise PdfImportError("encrypted PDFs requiring passwords are not supported")
    page_count = len(reader.pages)
    if page_count == 0:
        raise PdfImportError("PDF has no pages")
    if page_count > MAX_PDF_PAGES:
        raise PdfImportError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit")

    parts: list[str] = []
    page_map: list[dict[str, int]] = []
    blank_pages: list[int] = []
    offset = 0
    for index, page in enumerate(reader.pages, start=1):
        separator = _page_separator(index) if index > 1 else ""
        if separator:
            parts.append(separator)
            offset += len(separator)
        start = offset
        try:
            text = page.extract_text() or ""
        except Exception as error:
            raise PdfImportError(f"failed to extract page {index}: {error}") from error
        normalized_page = normalize_source_text(text) if text.strip() else ""
        if not normalized_page:
            blank_pages.append(index)
        parts.append(normalized_page)
        end = offset + len(normalized_page)
        page_map.append({"page": index, "start": start, "end": end})
        offset = end

    canonical = "".join(parts)
    if not canonical.strip():
        raise PdfImportError("PDF has no extractable text")
    document = ingest_text(canonical)
    return PdfExtractionResult(
        canonical_text=document.text,
        source_id=document.source_id,
        page_map=page_map,
        blank_pages=blank_pages,
        page_count=page_count,
    )


def serialize_page_map(page_map: list[dict[str, int]]) -> str:
    return json.dumps(page_map)
