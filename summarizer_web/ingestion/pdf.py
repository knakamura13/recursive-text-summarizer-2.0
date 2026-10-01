"""PDF extraction: the text layer through pypdf, OCR where it is missing.

A page is OCR'd (rendered whole with pypdfium2, recognized by Tesseract) when
its text layer has fewer than SPARSE_PAGE_CHARS non-space characters. The OCR
text replaces the layer only when it is more than 1.5 times longer. Text stays
per page; the canonical text and its page map are assembled afterwards (see
extract.assemble).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader

from summarizer_web.config import MAX_IMPORT_PAGES, OCR_DPI
from summarizer_web.ingestion.common import (
    Extraction,
    ImportFailure,
    OutlineHint,
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


_MAX_OUTLINE_DEPTH = 32


def _bookmark_hints(reader: PdfReader, page_count: int) -> list[OutlineHint]:
    """Bookmarks as outline hints in document order; never raises.

    Nesting depth is the level (top level is 1) and the destination page is
    1-based like `PageText.number`. A destination that cannot be resolved, or
    falls outside the document, leaves the hint without a page.
    """
    hints: list[OutlineHint] = []

    def walk(items: list, depth: int) -> None:
        if depth > _MAX_OUTLINE_DEPTH:
            return
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            title = str(getattr(item, "title", None) or "").strip()
            if not title:
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:
                page = None
            if page is not None and not 1 <= page <= page_count:
                page = None
            hints.append(OutlineHint(title, depth, page))

    try:
        walk(reader.outline, 1)
    except Exception:  # A malformed outline must not fail the Import.
        pass
    return hints


# Layout heading heuristic (PDFs without bookmarks). Each constant is a tunable.
_SIZE_STEP = 0.5  # Sizes are bucketed to half a point so rounding noise does not split a size.
_MIN_SIZE_RATIO = 1.1  # A line must be set at least 10% larger than body text to count as larger.
_MAX_HEADING_CHARS = 120  # A longer line is prose, not a title.
_MAX_HEADING_WORDS_LARGE = 20  # A title set larger than body may be a sentence-length phrase.
_MAX_HEADING_WORDS_BOLD = 12  # A bold line at body size must be short, or it is an emphasized sentence.
_MAX_SENTENCE_WORDS = 5  # A line ending in a period with more words than this is a sentence.
_MIN_LETTERS = 3  # Fewer letters than this is a label, a number or a formula.
_MIN_LETTER_RATIO = 0.5  # Mostly non-letters is a formula, a table row or a page number.
_MERGE_GAP_EM = 1.8  # Lines of one style closer than this many line sizes are one wrapped title.
_SAME_LINE_EM = 0.3  # Fragments whose baselines differ by less than this many sizes share a line.
_SPACE_GAP_EM = -0.25  # Between fragments of a line, a gap above this (in sizes, vs an estimated 0.5em per char) is a space.
_REPEAT_MIN_PAGES = 3  # Text at the same place on fewer pages than this is not a running header or footer.
_SIZE_CLUSTER_RATIO = 0.05  # Heading sizes within 5% of each other (OCR or rounding noise) are one level.
_MIN_CAPS_LETTERS = 2  # A single capital letter is not "all caps".
_SET_OFF_RATIO = 1.3  # A heading sits above more space than this times the page's body line spacing...
_BODY_GAP_MAX_EM = 3.0  # ...where body spacing ignores gaps over this many body sizes (paragraph/figure breaks).
_MIN_PAGE_GAPS = 3  # A page with fewer body line gaps than this borrows the document's typical spacing.
_POSITION_TOLERANCE = 6.0  # Points: text this close in y on another page is "the same place".
_MIN_BYLINE_RUN = 3  # This many same-style candidates in a row with no body text between are bylines or a contents list, not sections.
_NEAR_BODY_RATIO = 1.25  # Non-bold lines set less than this much larger than body are suspect (OCR size noise)...
_MAX_HEADING_WORDS_NEAR_BODY = 12  # ...so they must be short...
_FORMULA_CHARS = "=<>\u2264\u2265\u2260"  # ...and free of comparison operators, which mark formula lines.
_BOLD_WEIGHT = 600  # FontDescriptor /FontWeight at or above this is bold (600 = semibold).
_FORCE_BOLD_FLAG = 1 << 18  # FontDescriptor /Flags bit for ForceBold.
_BOLD_NAME = re.compile(
    r"bold|black|heavy|semibold|demi(?!-?light)|extrabold|[-,+]medi(?:um)?(?![a-z])",
    re.IGNORECASE,
)  # "-Medi" is the bold face of TeX's Nimbus/Times family.
_SENTENCE_END = re.compile(r"[.!?]$")


@dataclass
class _Line:
    page: int
    y: float
    size: float  # Dominant size by characters, bucketed.
    bold: bool
    text: str


@dataclass
class _Fragment:
    x: float
    y: float
    size: float
    bold: bool
    text: str


def _is_bold(font: object) -> bool:
    try:
        font = font.get_object() if hasattr(font, "get_object") else font  # type: ignore[union-attr]
        if not font:
            return False
        name = str(font.get("/BaseFont", ""))  # type: ignore[union-attr]
        if _BOLD_NAME.search(name):
            return True
        descriptor = font.get("/FontDescriptor")  # type: ignore[union-attr]
        descriptor = descriptor.get_object() if descriptor is not None else None
        if descriptor is None:
            return False
        if float(descriptor.get("/FontWeight", 0)) >= _BOLD_WEIGHT:
            return True
        return bool(int(descriptor.get("/Flags", 0)) & _FORCE_BOLD_FLAG)
    except Exception:
        return False


class _PageCollector:
    """A pypdf text visitor that records each shown string with its position, size and weight."""

    def __init__(self) -> None:
        self.fragments: list[_Fragment] = []
        self._bold_by_font: dict[int, bool] = {}

    def __call__(self, text, cm, tm, font_dict, font_size) -> None:
        try:
            if not text or not text.strip():
                return
            a = tm[0] * cm[0] + tm[1] * cm[2]
            b = tm[0] * cm[1] + tm[1] * cm[3]
            c = tm[2] * cm[0] + tm[3] * cm[2]
            d = tm[2] * cm[1] + tm[3] * cm[3]
            size = float(font_size) * math.sqrt(abs(a * d - b * c))
            if size <= 0:
                return
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            key = id(font_dict)
            bold = self._bold_by_font.get(key)
            if bold is None:
                bold = self._bold_by_font[key] = _is_bold(font_dict)
            self.fragments.append(_Fragment(x, y, size, bold, text))
        except Exception:
            return

    def lines(self, page: int) -> list[_Line]:
        groups: list[list[_Fragment]] = []
        for fragment in self.fragments:
            if groups:
                last = groups[-1][-1]
                if abs(fragment.y - last.y) < _SAME_LINE_EM * max(fragment.size, last.size):
                    groups[-1].append(fragment)
                    continue
            groups.append([fragment])
        lines = []
        for group in groups:
            parts: list[str] = []
            weight: dict[float, int] = {}
            bold_chars = letters = 0
            previous: _Fragment | None = None
            for fragment in group:
                text = fragment.text
                if previous is not None and not previous.text[-1:].isspace() and not text[:1].isspace():
                    end = previous.x + 0.5 * previous.size * len(previous.text)
                    if fragment.x - end > _SPACE_GAP_EM * fragment.size:
                        parts.append(" ")
                parts.append(text)
                count = sum(1 for ch in text if not ch.isspace())
                bucket = round(fragment.size / _SIZE_STEP) * _SIZE_STEP
                weight[bucket] = weight.get(bucket, 0) + count
                letters += count
                if fragment.bold:
                    bold_chars += count
                previous = fragment
            text = " ".join("".join(parts).split())
            if text and letters:
                size = max(weight.items(), key=lambda item: item[1])[0]
                lines.append(_Line(page, group[0].y, size, bold_chars == letters, text))
        return lines


def _collect_lines(reader: PdfReader, number: int) -> list[_Line]:
    return _collect_page(reader, number)[0]


def _collect_page(reader: PdfReader, number: int) -> tuple[list[_Line], str]:
    collector = _PageCollector()
    text = reader.pages[number - 1].extract_text(visitor_text=collector) or ""
    return collector.lines(number), text


def _reconcile_titles(hints: list[OutlineHint], texts: dict[int, str]) -> list[OutlineHint]:
    """Respell each title as its page's text layer has it, so it can be found in the canonical text.

    The visitor's fragment spacing can differ from the layer's ("A - B" vs "A-B"). The title's
    non-space characters are searched in the page text with optional whitespace between them.
    """
    result = []
    for hint in hints:
        text = texts.get(hint.page) if hint.page is not None else None
        if text:
            try:
                pattern = r"\s*".join(re.escape(ch) for ch in hint.title if not ch.isspace())
                match = re.search(pattern, text, re.IGNORECASE)
            except re.error:
                match = None
            if match is not None:
                hint = replace(hint, title=" ".join(match.group().split()))
        result.append(hint)
    return result


def _repeat_key(text: str, page: int) -> str:
    """Text for comparing lines across pages; a leading or trailing page number becomes its offset.

    A page number is the one number on a running header that moves in step with the page, so
    "Title 12" on page 5 and "Title 13" on page 6 share a key, while "Chapter 1" on page 10
    and "Chapter 2" on page 40 do not.
    """
    tokens = text.lower().split()
    for index in {0, len(tokens) - 1} if tokens else ():
        if tokens[index].isdigit():
            tokens[index] = f"<off:{int(tokens[index]) - page}>"
    return " ".join(tokens)


def _looks_like_heading_text(text: str, words_limit: int, formula_free: bool = False) -> bool:
    if len(text) > _MAX_HEADING_CHARS:
        return False
    if formula_free and any(ch in _FORMULA_CHARS for ch in text):
        return False
    words = text.split()
    if len(words) > words_limit:
        return False
    letters = sum(1 for ch in text if ch.isalpha())
    if letters < _MIN_LETTERS or letters < _MIN_LETTER_RATIO * len(text.replace(" ", "")):
        return False
    first = next((ch for ch in text if ch.isalpha()), "")
    if first.islower():  # A fragment continuing a paragraph or a sentence.
        return False
    if text[-1] in ",;:":
        return False
    return not (_SENTENCE_END.search(text) and len(words) > _MAX_SENTENCE_WORDS)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _is_caps(text: str) -> bool:
    return sum(1 for ch in text if ch.isalpha()) >= _MIN_CAPS_LETTERS and text.isupper()


def _cluster_sizes(sizes: set[float]) -> dict[float, float]:
    """Map each size to the largest size of its cluster; sizes within _SIZE_CLUSTER_RATIO chain together."""
    mapping: dict[float, float] = {}
    top = previous = None
    for size in sorted(sizes, reverse=True):
        if previous is None or size < previous * (1 - _SIZE_CLUSTER_RATIO):
            top = size
        mapping[size] = top  # type: ignore[assignment]
        previous = size
    return mapping


def _headings_from_lines(pages: dict[int, list[_Line]]) -> list[OutlineHint]:
    """Headings from laid-out lines: larger than body text, or bold on a line of their own."""
    chars: dict[float, int] = {}
    placed: dict[str, list[tuple[int, float]]] = {}
    for number, lines in pages.items():
        for line in lines:
            chars[line.size] = chars.get(line.size, 0) + len(line.text)
            placed.setdefault(_repeat_key(line.text, number), []).append((number, line.y))
    if not chars:
        return []
    body = max(chars.items(), key=lambda item: item[1])[0]

    def is_running(line: _Line) -> bool:
        """The same text (page numbers aside) at the same place on several pages: a header or footer."""
        near = {n for n, y in placed[_repeat_key(line.text, line.page)] if abs(y - line.y) <= _POSITION_TOLERANCE}
        return len(near) >= _REPEAT_MIN_PAGES

    # A running header does not count against a heading that repeats its text on the page.
    per_page: dict[int, dict[str, int]] = {}
    for number, lines in pages.items():
        counts = per_page[number] = {}
        for line in lines:
            if not is_running(line):
                counts[line.text] = counts.get(line.text, 0) + 1

    gaps_all = [
        gap
        for lines in pages.values()
        for gap in _body_gaps(lines, body)
    ]
    document_gap = _median(gaps_all) if gaps_all else 0.0

    found: list[tuple[int, _Line, str, int, int]] = []  # page, first line, title, first and last line index
    for number in sorted(pages):
        lines = pages[number]
        page_gaps = _body_gaps(lines, body)
        typical = _median(page_gaps) if len(page_gaps) >= _MIN_PAGE_GAPS else document_gap
        run: list[_Line] = []
        run_start = 0
        run_set_off = False

        def flush() -> None:
            if not run or not run_set_off:
                return
            title = " ".join(line.text for line in run)
            first = run[0]
            larger = first.size >= body * _MIN_SIZE_RATIO
            near_body = larger and not first.bold and first.size < body * _NEAR_BODY_RATIO
            limit = _MAX_HEADING_WORDS_LARGE if larger else _MAX_HEADING_WORDS_BOLD
            if near_body:
                limit = _MAX_HEADING_WORDS_NEAR_BODY
            if _looks_like_heading_text(title, limit, near_body) and per_page[number].get(title, 0) < 2:
                found.append((number, first, title, run_start, run_start + len(run) - 1))

        for index, line in enumerate(lines):
            larger = line.size >= body * _MIN_SIZE_RATIO
            candidate = (larger or (line.bold and line.size >= body)) and any(
                ch.isalpha() for ch in line.text
            )
            candidate = candidate and not is_running(line) and per_page[number][line.text] < 2
            if candidate and run:
                last = run[-1]
                if (
                    last.size == line.size
                    and last.bold == line.bold
                    and 0 < last.y - line.y <= _MERGE_GAP_EM * line.size
                ):
                    run.append(line)
                    continue
            flush()
            run = [line] if candidate else []
            if candidate:
                run_start = index
                above = lines[index - 1].y - line.y if index else None
                run_set_off = above is None or above <= 0 or above > _SET_OFF_RATIO * typical

        flush()

    found = _without_runs(found, pages, body)
    styles = {
        (line.size, line.bold, _is_caps(title)): None for _, line, title, _, _ in found
    }
    clusters = _cluster_sizes({size for size, _, _ in styles})
    order = sorted(
        {(clusters[size], bold, caps) for size, bold, caps in styles},
        key=lambda style: (-style[0], not style[1], not style[2]),
    )
    rank = {style: level for level, style in enumerate(order, start=1)}
    return [
        OutlineHint(
            title,
            rank[(clusters[line.size], line.bold, _is_caps(title))],
            number,
        )
        for number, line, title, _, _ in found
    ]


def _without_runs(
    found: list[tuple[int, _Line, str, int, int]], pages: dict[int, list[_Line]], body: float
) -> list[tuple[int, _Line, str, int, int]]:
    """Drop runs of _MIN_BYLINE_RUN same-style headings on a page with no body text between them."""
    drop: set[int] = set()
    start = 0
    for end in range(1, len(found) + 1):
        if end < len(found):
            number, line, _, first, _ = found[end]
            pnumber, pline, _, _, plast = found[end - 1]
            between = pages[number][plast + 1 : first] if number == pnumber else None
            if (
                between is not None
                and (line.size, line.bold) == (pline.size, pline.bold)
                and not any(b.size == body and not b.bold for b in between)
            ):
                continue
        if end - start >= _MIN_BYLINE_RUN:
            drop.update(range(start, end))
        start = end
    return [item for index, item in enumerate(found) if index not in drop]


def _body_gaps(lines: list[_Line], body: float) -> list[float]:
    """Baseline distances between consecutive body-size lines: the page's normal line spacing."""
    return [
        gap
        for above, below in zip(lines, lines[1:])
        if above.size == body
        and below.size == body
        and 0 < (gap := above.y - below.y) <= _BODY_GAP_MAX_EM * body
    ]


def detect_layout_headings(reader: PdfReader) -> list[OutlineHint]:
    """Headings found from text layout alone, for PDFs without bookmarks; never raises."""
    try:
        pages: dict[int, list[_Line]] = {}
        texts: dict[int, str] = {}
        for number in range(1, _page_count(reader) + 1):
            try:
                pages[number], texts[number] = _collect_page(reader, number)
            except Exception:
                continue
        return _reconcile_titles(_headings_from_lines(pages), texts)
    except Exception:
        return []


def extract_pdf(path: Path, progress: ProgressCallback, tesseract: Tesseract | None) -> Extraction:
    progress("reading")
    reader = _open_reader(path)
    page_count = _page_count(reader)

    hints = _bookmark_hints(reader, page_count)
    layout: dict[int, list[_Line]] = {}
    layer: list[str] = []
    unreadable: list[int] = []
    progress("extracting", 0, page_count, unit="pages")
    for number in range(1, page_count + 1):
        try:
            # Bookmark-free PDFs collect layout in this same walk, for the heading heuristic.
            collector = None if hints else _PageCollector()
            raw = reader.pages[number - 1].extract_text(visitor_text=collector) or ""
            if collector is not None:
                try:
                    layout[number] = collector.lines(number)
                except Exception:
                    pass
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
    if not hints:
        ocr_numbers = {page.number for page in pages if page.ocr}
        try:
            hints = _reconcile_titles(
                _headings_from_lines(
                    {number: lines for number, lines in layout.items() if number not in ocr_numbers}
                ),
                {number: text for number, text in enumerate(layer, start=1)},
            )
        except Exception:
            hints = []
    return Extraction(
        "pdf",
        pages=pages,
        text_layer_pages=sum(1 for page in pages if not page.ocr and has_text(page.text)),
        notices=notices,
        outline_hints=hints,
    )
