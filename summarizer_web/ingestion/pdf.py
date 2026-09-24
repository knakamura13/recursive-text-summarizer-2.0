"""PDF extraction: the text layer through pypdf, OCR where it is missing.

A page is OCR'd (rendered whole with pypdfium2, recognized by Tesseract) when
its text layer has fewer than SPARSE_PAGE_CHARS non-space characters. The OCR
text replaces the layer only when it is more than 1.5 times longer. Text stays
per page; the canonical text and its page map are assembled afterwards (see
extract.assemble).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader

from summarizer_web.config import MAX_IMPORT_PAGES, OCR_DPI
from summarizer_web.ingestion.common import (
    Extraction,
    ImportFailure,
    PageText,
    ProgressCallback,
    clean_text,
    format_page_ranges,
    has_text,
    plural,
)
from summarizer_web.ingestion.ocr import (
    TESSERACT_INSTALL_HINT,
    OcrError,
    OcrOutcome,
    Tesseract,
    ocr_problem_notices,
    recognize_pages,
)
from summarizer_web.models.api import Notice

# Pages larger than this (e.g. posters) are rendered below OCR_DPI.
_MAX_RENDER_PIXELS = 40_000_000

SPARSE_PAGE_CHARS = 1_000


def _visible_chars(text: str) -> int:
    return len(text) - sum(text.count(space) for space in (" ", "\n", "\t"))


def sparse_pages(layer: list[str]) -> list[int]:
    """Page numbers whose text layer is too thin to trust (see module doc)."""
    counts = [_visible_chars(text) for text in layer]
    return [
        number
        for number, (text, count) in enumerate(zip(layer, counts), start=1)
        if count < SPARSE_PAGE_CHARS or not has_text(text)
    ]


def prefer_ocr(layer_text: str, ocr_text: str) -> bool:
    """Use OCR when the layer is empty or OCR is more than 1.5 times longer."""
    if not has_text(ocr_text):
        return False
    if not has_text(layer_text):
        return True
    layer_chars, ocr_chars = _visible_chars(layer_text), _visible_chars(ocr_text)
    return ocr_chars > 1.5 * layer_chars


def _open_reader(path: Path) -> PdfReader:
    try:
        reader = PdfReader(str(path))
        encrypted = reader.is_encrypted
    except Exception as error:  # pypdf raises many exception types on damaged files.
        raise ImportFailure(f"The PDF could not be read: {error}") from error
    if encrypted:
        try:
            decrypted = bool(reader.decrypt(""))
        except Exception:
            decrypted = False
        if not decrypted:
            raise ImportFailure(
                "The PDF is password-protected. Remove the password and import it again."
            )
    return reader


def _page_count(reader: PdfReader) -> int:
    try:
        count = len(reader.pages)
    except Exception as error:
        raise ImportFailure(f"The PDF page list could not be read: {error}") from error
    if count == 0:
        raise ImportFailure("The PDF has no pages.")
    if count > MAX_IMPORT_PAGES:
        raise ImportFailure(
            f"The PDF has {count:,} pages; imports are limited to {MAX_IMPORT_PAGES:,} pages."
        )
    return count


def _pgm(bitmap: pdfium.PdfBitmap) -> bytes:
    width, height, stride = bitmap.width, bitmap.height, bitmap.stride
    pixels = bytes(bitmap.buffer)
    header = b"P5\n%d %d\n255\n" % (width, height)
    if stride == width:
        return header + pixels
    return header + b"".join(pixels[row * stride : row * stride + width] for row in range(height))


def _render(document: pdfium.PdfDocument, index: int) -> tuple[bytes, int]:
    """A grayscale PGM of the page at OCR_DPI (less for huge pages) and its DPI."""
    page = document[index]
    try:
        width, height = page.get_size()
        scale = OCR_DPI / 72
        pixels = width * height * scale * scale
        if pixels > _MAX_RENDER_PIXELS:
            scale *= math.sqrt(_MAX_RENDER_PIXELS / pixels)
        bitmap = page.render(scale=scale, grayscale=True)
        try:
            return _pgm(bitmap), max(1, round(scale * 72))
        finally:
            bitmap.close()
    finally:
        page.close()


def _recognize(
    path: Path, numbers: list[int], progress: ProgressCallback, tesseract: Tesseract
) -> dict[int, OcrOutcome]:
    try:
        document = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as error:
        failure = OcrError(f"the pages could not be rendered ({error})")
        return {number: OcrOutcome(error=failure) for number in numbers}
    try:

        def prepare(number: int) -> Callable[[], str]:
            try:
                image, dpi = _render(document, number - 1)
            except (pdfium.PdfiumError, ValueError, IndexError) as error:
                raise OcrError(f"the page could not be rendered ({error})") from error
            return lambda: tesseract.recognize_pixels(image, dpi=dpi)

        return recognize_pages(numbers, prepare, progress)
    finally:
        document.close()


def extract_pdf(path: Path, progress: ProgressCallback, tesseract: Tesseract | None) -> Extraction:
    progress("reading")
    reader = _open_reader(path)
    page_count = _page_count(reader)

    layer: list[str] = []
    unreadable: list[int] = []
    progress("extracting", 0, page_count, unit="pages")
    for number in range(1, page_count + 1):
        try:
            raw = reader.pages[number - 1].extract_text() or ""
        except Exception:  # A damaged page must not fail the whole Import.
            raw = ""
            unreadable.append(number)
        layer.append(clean_text(raw))
        progress("extracting", number, page_count, unit="pages")

    candidates = sparse_pages(layer)
    outcomes = _recognize(path, candidates, progress, tesseract) if candidates and tesseract else {}

    pages: list[PageText] = []
    for number, text in enumerate(layer, start=1):
        recognized = outcomes[number].text if number in outcomes else None
        if recognized is not None and prefer_ocr(text, recognized):
            pages.append(PageText(number, recognized, ocr=True))
        else:
            pages.append(PageText(number, text if has_text(text) else ""))

    if not any(has_text(page.text) for page in pages):
        if tesseract is None:
            raise ImportFailure(
                "The PDF has no text layer, and reading scanned pages needs Tesseract, "
                "which is not installed. " + TESSERACT_INSTALL_HINT
            )
        errors = [outcome.error for outcome in outcomes.values() if outcome.error is not None]
        if len(errors) == len(candidates):
            raise ImportFailure(f"The PDF has no text layer, and OCR failed: {errors[0].message}.")
        raise ImportFailure(
            f"No text was found in the PDF ({plural(page_count, 'page')}), including with OCR."
        )

    notices: list[Notice] = []
    if unreadable:
        notices.append(
            Notice(
                code="text_layer_unreadable",
                severity="warning",
                message=(
                    f"The text layer of {plural(len(unreadable), 'page')} "
                    f"({format_page_ranges(unreadable)}) could not be read."
                ),
            )
        )
    if candidates and tesseract is None:
        notices.append(
            Notice(
                code="ocr_unavailable",
                severity="warning",
                message=(
                    f"{plural(len(candidates), 'page')} ({format_page_ranges(candidates)}) "
                    "have little or no text layer; text in their images was not read because "
                    "OCR needs Tesseract, which is not installed. " + TESSERACT_INSTALL_HINT
                ),
            )
        )
    ocr_pages = [page.number for page in pages if page.ocr]
    if ocr_pages:
        notices.append(
            Notice(
                code="ocr_used",
                message=(
                    f"OCR recognized the text of {plural(len(ocr_pages), 'page')} "
                    f"({format_page_ranges(ocr_pages)}); recognition errors are possible."
                ),
            )
        )
    if tesseract is not None:
        notices.extend(ocr_problem_notices(outcomes, tesseract.timeout_seconds))
    blank = [page.number for page in pages if not has_text(page.text)]
    if blank:
        notices.append(
            Notice(
                code="blank_pages",
                message=f"No text on {plural(len(blank), 'page')}: {format_page_ranges(blank)}.",
            )
        )
    return Extraction(
        "pdf",
        pages=pages,
        text_layer_pages=sum(1 for page in pages if not page.ocr and has_text(page.text)),
        notices=notices,
    )
