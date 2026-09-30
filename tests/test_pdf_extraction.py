from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from summarizer_web.ingestion.common import ImportFailure, ignore_progress
from summarizer_web.ingestion.extract import assemble
from summarizer_web.ingestion.ocr import TESSERACT_INSTALL_HINT
from summarizer_web.ingestion.pdf import extract_pdf

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
