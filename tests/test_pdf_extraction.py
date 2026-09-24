from pathlib import Path

import pytest

from summarizer_web.ingestion.common import ImportFailure, ignore_progress
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
