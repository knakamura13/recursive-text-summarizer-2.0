from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "name",
    [
        "article.txt",
        "report.txt",
        "transcript.txt",
        "structured.md",
        "narrative.txt",
    ],
)
def test_representative_fixture_is_nonempty_utf8(name: str) -> None:
    content = (FIXTURES / name).read_text(encoding="utf-8")

    assert len(content.split()) >= 80
