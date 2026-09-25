from pathlib import Path

import pytest

from summarizer_web.ingestion import pdf
from summarizer_web.ingestion.common import ignore_progress
from summarizer_web.ingestion.detect import detect_format
from summarizer_web.ingestion.extract import assemble, extract_document
from summarizer_web.ingestion.ocr import OcrOutcome, Tesseract

FIXTURES = Path(__file__).parent / "fixtures" / "import"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("plain.txt", "txt"),
        ("notes.md", "md"),
        ("captions.srt", "srt"),
        ("captions.vtt", "vtt"),
        ("article.html", "html"),
        ("report.rtf", "rtf"),
        ("report.docx", "docx"),
        ("report.odt", "odt"),
        ("stories.epub", "epub"),
        ("layered.pdf", "pdf"),
        ("encrypted-aes.pdf", "pdf"),
        ("blank.pdf", "pdf"),
        ("scanned.pdf", "pdf"),
        ("password.pdf", "pdf"),
        ("notice.png", "png"),
        ("notice.jpg", "jpeg"),
        ("scan-2pages.tiff", "tiff"),
    ],
)
def test_import_fixtures_are_detected_by_content(filename: str, expected: str) -> None:
    detection = detect_format(FIXTURES / filename, filename)
    assert detection.format == expected


def test_container_and_binary_signatures_override_misleading_extension(tmp_path: Path) -> None:
    pdf_as_text = tmp_path / "report.txt"
    pdf_as_text.write_bytes((FIXTURES / "layered.pdf").read_bytes())
    image_as_pdf = tmp_path / "notice.pdf"
    image_as_pdf.write_bytes((FIXTURES / "notice.png").read_bytes())

    assert detect_format(pdf_as_text, pdf_as_text.name).format == "pdf"
    assert detect_format(image_as_pdf, image_as_pdf.name).format == "png"


def test_text_format_fixtures_extract_to_canonical_text() -> None:
    tesseract = Tesseract.find()
    for filename, expected_format in (
        ("plain.txt", "txt"),
        ("notes.md", "md"),
        ("captions.srt", "srt"),
        ("captions.vtt", "vtt"),
        ("article.html", "html"),
        ("report.rtf", "rtf"),
        ("report.docx", "docx"),
        ("report.odt", "odt"),
        ("stories.epub", "epub"),
        ("layered.pdf", "pdf"),
    ):
        extraction = extract_document(
            FIXTURES / filename, expected_format, ignore_progress, tesseract=tesseract
        )
        imported = assemble(extraction)
        assert imported.text.strip()
        assert imported.source_id


@pytest.mark.parametrize(
    ("filename", "document_format"),
    [("notice.png", "png"), ("notice.jpg", "jpeg"), ("scan-2pages.tiff", "tiff"), ("scanned.pdf", "pdf")],
)
def test_ocr_fixtures_extract_when_tesseract_is_available(filename: str, document_format: str) -> None:
    tesseract = Tesseract.find()
    if tesseract is None:
        pytest.skip("Tesseract is not installed; OCR fixture extraction requires it")
    extraction = extract_document(
        FIXTURES / filename, document_format, ignore_progress, tesseract=tesseract
    )
    imported = assemble(extraction)
    assert imported.text.strip()
    assert imported.pages is not None


def test_caption_sized_pdf_text_layer_is_ocr_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = "caption text " * 25  # 275 non-space characters, below the 1,000-char floor.
    reader = type("Reader", (), {"pages": [type("Page", (), {"extract_text": lambda self: layer})()]})()
    calls: list[list[int]] = []

    def recognize(_path, numbers, _progress, _tesseract):
        calls.append(numbers)
        return {1: OcrOutcome(text="recognized caption and additional details " * 20)}

    monkeypatch.setattr(pdf, "_open_reader", lambda _path: reader)
    monkeypatch.setattr(pdf, "_recognize", recognize)
    extraction = pdf.extract_pdf(FIXTURES / "layered.pdf", ignore_progress, Tesseract("unused"))

    assert calls == [[1]]
    assert extraction.pages[0].ocr is True
    assert extraction.pages[0].text.startswith("recognized caption")


def test_ocr_replaces_layer_only_when_over_one_and_a_half_times_longer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = "x" * 300
    reader = type("Reader", (), {"pages": [type("Page", (), {"extract_text": lambda self: layer})()]})()
    monkeypatch.setattr(pdf, "_open_reader", lambda _path: reader)
    monkeypatch.setattr(
        pdf,
        "_recognize",
        lambda *_args: {1: OcrOutcome(text="y" * 450)},
    )

    extraction = pdf.extract_pdf(FIXTURES / "layered.pdf", ignore_progress, Tesseract("unused"))

    assert extraction.pages[0].text == layer
    assert extraction.pages[0].ocr is False


def test_layer_at_one_thousand_characters_is_not_sparse() -> None:
    assert pdf.sparse_pages(["a" * 999, "b" * 1_000]) == [1]
