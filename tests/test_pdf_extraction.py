from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from summarizer_web.ingestion.common import ImportFailure, clean_text, ignore_progress
from summarizer_web.ingestion.extract import assemble
from summarizer_web.ingestion.ocr import TESSERACT_INSTALL_HINT
from summarizer_web.ingestion.pdf import detect_layout_headings, extract_pdf

FIXTURES = Path(__file__).parent / "fixtures" / "import"


def test_pdf_without_text_fails_without_tesseract():
    with pytest.raises(ImportFailure, match="Tesseract") as raised:
        extract_pdf(FIXTURES / "blank.pdf", ignore_progress, None)
    assert TESSERACT_INSTALL_HINT in raised.value.message


def test_sparse_caption_page_is_retained_with_ocr_unavailable_notice():
    extraction = extract_pdf(FIXTURES / "layered.pdf", ignore_progress, None)
    assert any(page.text for page in extraction.pages)
    assert any(notice.code == "ocr_unavailable" for notice in extraction.notices)


def _pdf_with_text(tmp_path: Path, pages: list[str]) -> tuple[Path, PdfWriter]:
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
    return tmp_path / "book.pdf", writer


def _save(path: Path, writer: PdfWriter) -> Path:
    with path.open("wb") as handle:
        writer.write(handle)
    return path


_BODIES = [
    "Overview of the book. Words on the first page go here.",
    "Chapter Alpha. Alpha content sits on the second page.",
    "Section Beta. Beta content sits on the third page.",
    "Chapter Gamma. Gamma content sits on the fourth page.",
]


def test_bookmarks_become_nested_hints_with_pages_and_anchor(tmp_path: Path):
    path, writer = _pdf_with_text(tmp_path, _BODIES)
    writer.add_outline_item("Overview", 0)
    alpha = writer.add_outline_item("Chapter Alpha", 1)
    writer.add_outline_item("Section Beta", 2, parent=alpha)
    writer.add_outline_item("Chapter Gamma", 3)
    extraction = extract_pdf(_save(path, writer), ignore_progress, None)

    assert [(h.title, h.level, h.page) for h in extraction.outline_hints] == [
        ("Overview", 1, 1),
        ("Chapter Alpha", 1, 2),
        ("Section Beta", 2, 3),
        ("Chapter Gamma", 1, 4),
    ]
    imported = assemble(extraction)
    assert [
        (e.title, e.level, imported.text[e.start : e.end].split(".")[0], e.page_start, e.page_end)
        for e in imported.outline
    ] == [
        ("Overview", 1, "Overview of the book", 1, 1),
        ("Chapter Alpha", 1, "Chapter Alpha", 2, 3),
        ("Section Beta", 2, "Section Beta", 3, 3),
        ("Chapter Gamma", 1, "Chapter Gamma", 4, 4),
    ]


def test_pdf_without_bookmarks_has_no_hints(tmp_path: Path):
    path, writer = _pdf_with_text(tmp_path, _BODIES)
    extraction = extract_pdf(_save(path, writer), ignore_progress, None)
    assert extraction.outline_hints == []
    assert assemble(extraction).outline == ()


def test_unresolvable_bookmark_destination_keeps_title_and_level(tmp_path: Path):
    path, writer = _pdf_with_text(tmp_path, _BODIES)
    writer.add_outline_item("Chapter Alpha", 1)
    broken = writer.add_outline_item("Chapter Gamma", 3)
    broken = broken.get_object()
    del broken["/A"]
    broken[NameObject("/Dest")] = NameObject("/no-such-destination")
    extraction = extract_pdf(_save(path, writer), ignore_progress, None)

    assert [(h.title, h.level, h.page) for h in extraction.outline_hints] == [
        ("Chapter Alpha", 1, 2),
        ("Chapter Gamma", 1, None),
    ]
    assert [e.title for e in assemble(extraction).outline] == ["Chapter Alpha", "Chapter Gamma"]


def _layout_pdf(tmp_path: Path, pages: list[list[tuple[str, int, float]]]) -> tuple[Path, PdfWriter]:
    """Pages of (text, size, y) lines; Helvetica-Bold when the text starts with '*', a Medi face with '~'."""
    writer = PdfWriter()
    fonts = {}
    for name, base in (("F1", "Helvetica"), ("F2", "Helvetica-Bold"), ("F3", "NimbusRomNo9L-Medi")):
        fonts[NameObject(f"/{name}")] = writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject(f"/{base}"),
                }
            )
        )
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        ops = []
        for text, size, y in lines:
            font = {"*": "F2", "~": "F3"}.get(text[:1], "F1")
            ops.append(f"BT /{font} {size} Tf 72 {y} Td ({text.lstrip('*~')}) Tj ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(ops).encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject(fonts)})
    return tmp_path / "layout.pdf", writer


def _body(y: float) -> list[tuple[str, int, float]]:
    return [
        (f"This is an ordinary sentence of body text number {i} that runs on for a while.", 11, y - 14 * i)
        for i in range(6)
    ]


def _hints(path: Path) -> list[tuple[str, int, int | None]]:
    extraction = extract_pdf(path, ignore_progress, None)
    return [(h.title, h.level, h.page) for h in extraction.outline_hints]


def test_layout_headings_get_levels_from_size_and_bold(tmp_path: Path):
    pages = [
        [("Big Title", 24, 740), ("Introduction", 16, 700), *_body(670)],
        [("*Bold Subsection", 11, 740), *_body(720), ("Results", 16, 600), *_body(570)],
    ]
    path, writer = _layout_pdf(tmp_path, pages)
    assert _hints(_save(path, writer)) == [
        ("Big Title", 1, 1),
        ("Introduction", 2, 1),
        ("Bold Subsection", 3, 2),
        ("Results", 2, 2),
    ]


def test_body_text_alone_has_no_layout_headings(tmp_path: Path):
    path, writer = _layout_pdf(tmp_path, [_body(700), _body(700)])
    assert _hints(_save(path, writer)) == []


def test_running_header_and_page_numbers_are_not_headings(tmp_path: Path):
    pages = [
        [("Annual Report", 16, 760), ("*" + str(n), 11, 40), *_body(700)] for n in range(1, 5)
    ]
    pages[1].insert(0, ("Findings", 16, 740))
    path, writer = _layout_pdf(tmp_path, pages)
    assert _hints(_save(path, writer)) == [("Findings", 1, 2)]


def test_bold_sentence_and_continuation_fragment_are_not_headings(tmp_path: Path):
    long_bold = "*This bold sentence is emphasized text that goes on well beyond any plausible title length."
    pages = [[(long_bold, 11, 740), ("and so on, continuing a sentence", 16, 720), *_body(690)]]
    path, writer = _layout_pdf(tmp_path, pages)
    assert _hints(_save(path, writer)) == []


def test_bookmarks_take_precedence_over_layout_headings(tmp_path: Path):
    pages = [[("Introduction", 16, 740), *_body(700)], [("Methods", 16, 740), *_body(700)]]
    path, writer = _layout_pdf(tmp_path, pages)
    writer.add_outline_item("Only Bookmark", 1)
    assert _hints(_save(path, writer)) == [("Only Bookmark", 1, 2)]


def test_layout_heading_detection_can_be_forced_and_does_not_change_text(tmp_path: Path):
    pages = [[("Introduction", 16, 740), *_body(700)], [("Methods", 16, 740), *_body(700)]]
    path, writer = _layout_pdf(tmp_path, pages)
    writer.add_outline_item("Only Bookmark", 1)
    _save(path, writer)
    assert [(h.title, h.level) for h in detect_layout_headings(PdfReader(path))] == [
        ("Introduction", 1),
        ("Methods", 1),
    ]
    extraction = extract_pdf(path, ignore_progress, None)
    reader = PdfReader(path)
    assert [page.text for page in extraction.pages] == [
        clean_text(p.extract_text() or "") for p in reader.pages
    ]


def test_medi_font_counts_as_bold(tmp_path: Path):
    path, writer = _layout_pdf(tmp_path, [[("~2.1 Sampling", 11, 740), *_body(710)]])
    assert _hints(_save(path, writer)) == [("2.1 Sampling", 1, 1)]


def test_larger_line_inside_a_paragraph_is_not_a_heading(tmp_path: Path):
    lines = _body(700)
    lines[3] = (lines[3][0], 14, lines[3][2])  # OCR-style size noise, normal line spacing
    path, writer = _layout_pdf(tmp_path, [lines])
    assert _hints(_save(path, writer)) == []


def test_heading_needs_more_space_above_than_body_lines(tmp_path: Path):
    body = _body(700)
    lines = [*body[:3], ("Spaced Heading", 14, 642), *[(t, s, y - 36) for t, s, y in body[3:]]]
    path, writer = _layout_pdf(tmp_path, [lines])
    assert _hints(_save(path, writer)) == [("Spaced Heading", 1, 1)]


def test_text_repeated_on_one_page_is_not_a_heading(tmp_path: Path):
    lines = [("Panel A", 14, 740), ("Panel A", 14, 500), *_body(700)]
    path, writer = _layout_pdf(tmp_path, [lines])
    assert _hints(_save(path, writer)) == []


def test_text_at_the_same_place_on_a_few_pages_is_a_running_header(tmp_path: Path):
    pages = [[*_body(700)] for _ in range(20)]
    for number in (2, 9, 15):
        pages[number].insert(0, ("Figure Panel Title", 14, 740))
    pages[5].insert(0, ("Real Section", 14, 740))
    path, writer = _layout_pdf(tmp_path, pages)
    assert _hints(_save(path, writer)) == [("Real Section", 1, 6)]


def test_levels_rank_style_keys_and_are_consecutive(tmp_path: Path):
    pages = [
        [("PART ONE", 20, 740), *_body(700)],
        [("Section Title", 20.4, 740), *_body(700)],  # within 5% of 20: same size bucket
        [("*Bold Sub", 11, 740), *_body(700)],
        [("Another Part", 15, 740), ("LOUD PART", 15, 640), *_body(600)],
    ]
    path, writer = _layout_pdf(tmp_path, pages)
    assert _hints(_save(path, writer)) == [
        ("PART ONE", 1, 1),
        ("Section Title", 2, 2),
        ("Bold Sub", 5, 3),
        ("Another Part", 4, 4),
        ("LOUD PART", 3, 4),
    ]


def test_heading_is_kept_beside_a_running_header_with_the_page_number(tmp_path: Path):
    pages = []
    for number in range(1, 9):
        lines = [(f"Chapter Notes {number}", 10, 770), *_body(700)]
        pages.append(lines)
    pages[3] += [("Chapter Notes", 14, 600)]
    pages[3] = [pages[3][0], ("Chapter Notes", 14, 740), *pages[3][1:-1]]
    path, writer = _layout_pdf(tmp_path, pages)
    assert _hints(_save(path, writer)) == [("Chapter Notes", 1, 4)]


def test_heading_repeating_a_header_text_on_its_page_is_kept(tmp_path: Path):
    pages = [[("Notes", 10, 770), *_body(700)] for _ in range(8)]
    pages[2].insert(1, ("Notes", 14, 740))
    path, writer = _layout_pdf(tmp_path, pages)
    assert _hints(_save(path, writer)) == [("Notes", 1, 3)]


def test_run_of_same_style_lines_without_body_text_is_dropped(tmp_path: Path):
    title_page = [("Paper Title", 24, 740), ("Ada Lovelace", 14, 690), ("Alan Turing", 14, 660),
                  ("Grace Hopper", 14, 630), *_body(560)]
    path, writer = _layout_pdf(tmp_path, [title_page, [("Methods", 14, 740), *_body(700)]])
    assert _hints(_save(path, writer)) == [("Paper Title", 1, 1), ("Methods", 2, 2)]


def test_same_style_headings_with_body_between_are_kept(tmp_path: Path):
    body = _body(700)
    lines = [("First", 14, 740), *body[:3], ("Second", 14, 640), *[(t, s, y - 36) for t, s, y in body[3:]]]
    lines += [("Third", 14, 500)]
    path, writer = _layout_pdf(tmp_path, [lines])
    assert [h.title for h in extract_pdf(_save(path, writer), ignore_progress, None).outline_hints] == [
        "First", "Second", "Third"
    ]


def test_slightly_larger_formula_or_long_line_is_not_a_heading(tmp_path: Path):
    formula = ("x = y + z and more", 13, 740)
    prose = ("one two three four five six seven eight nine ten eleven twelve thirteen", 13, 700)
    path, writer = _layout_pdf(tmp_path, [[formula, prose, ("Short Title", 13, 660), *_body(620)]])
    assert _hints(_save(path, writer)) == [("Short Title", 1, 1)]
