import re
from pathlib import Path

import pytest

from summarizer.ingestion import ingest_text
from summarizer.segmentation import SegmentationConfig, segment_document
from summarizer.tokenization import ConservativeUtf8TokenCounter


FIXTURES = Path(__file__).parent / "fixtures"
REQUIRED_FIXTURES = [
    "article.txt",
    "report.txt",
    "transcript.txt",
    "structured.md",
    "narrative.txt",
]


@pytest.mark.parametrize("name", REQUIRED_FIXTURES)
def test_representative_fixture_is_nonempty_utf8(name: str) -> None:
    content = (FIXTURES / name).read_text(encoding="utf-8")

    assert len(content.split()) >= 80


def test_fixture_corpus_contains_fenced_code_block() -> None:
    fence_pattern = re.compile(r"^\s*(```+|~~~+)", re.MULTILINE)
    has_fenced_code = any(
        fence_pattern.search((FIXTURES / name).read_text(encoding="utf-8"))
        for name in REQUIRED_FIXTURES
    )
    assert has_fenced_code, (
        "At least one fixture in the representative corpus must contain a fenced code block"
    )


@pytest.mark.parametrize("name", REQUIRED_FIXTURES)
def test_representative_fixture_can_be_segmented(name: str) -> None:
    raw_text = (FIXTURES / name).read_text(encoding="utf-8")
    doc = ingest_text(raw_text)
    counter = ConservativeUtf8TokenCounter()
    config = SegmentationConfig(max_tokens=200)
    segments = segment_document(doc, counter, config)

    assert len(segments) > 0
    reconstructed = "".join(
        doc.text[s.core_start : s.core_end] for s in segments
    )
    assert reconstructed == doc.text
