from pathlib import Path

import pytest

from summarizer_web.ingestion.pdf import PdfImportError, extract_pdf


def test_extract_pdf_rejects_empty_text(tmp_path: Path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF")
    with pytest.raises(PdfImportError):
        extract_pdf(pdf_path)
