from pathlib import Path

from summarizer.segmentation import detect_markdown_headings
from summarizer_web.ingestion.extract import assemble, extract_document


def summary(text: str) -> list[tuple[int, str, str]]:
    return [(h.level, h.title, text[h.start : h.end]) for h in detect_markdown_headings(text)]


def test_atx_levels_one_to_six() -> None:
    text = "\n\n".join(f"{'#' * n} Level {n}\n\nBody." for n in range(1, 7))
    assert [(h.level, h.title) for h in detect_markdown_headings(text)] == [
        (n, f"Level {n}") for n in range(1, 7)
    ]


def test_seven_hashes_is_not_a_heading() -> None:
    assert detect_markdown_headings("####### Too deep\n\nBody.") == []


def test_setext_levels_one_and_two() -> None:
    text = "Main Title\n==========\n\nBody.\n\nSub Title\n---\n\nMore body."
    assert summary(text) == [
        (1, "Main Title", "Main Title\n=========="),
        (2, "Sub Title", "Sub Title\n---"),
    ]


def test_closing_hash_markers_are_stripped() -> None:
    assert [h.title for h in detect_markdown_headings("## Title ##\n\nBody.\n\n# C# #\n\n### Plain###")] == [
        "Title",
        "C#",
        "Plain###",
    ]


def test_headings_inside_fences_are_ignored() -> None:
    text = "# Real\n\n```\n# not a heading\nFake\n----\n```\n\n~~~\n## nor this\n~~~\n\n## Also Real\n"
    assert [h.title for h in detect_markdown_headings(text)] == ["Real", "Also Real"]


def test_text_without_headings_yields_nothing() -> None:
    assert detect_markdown_headings("Just a paragraph.\n\nAnother one, with #hashtag.") == []
    assert detect_markdown_headings("") == []


def test_offsets_exclude_trailing_blank_lines() -> None:
    text = "# One\n\n\nBody.\n"
    [heading] = detect_markdown_headings(text)
    assert (heading.start, heading.end) == (0, 5)


def test_markdown_import_yields_outline_and_keeps_canonical_text(tmp_path: Path) -> None:
    body = "# Guide\n\nIntro text.\n\n## Setup ##\n\nSteps.\n\n```\n# comment\n```\n\nAfter Setup\n-----\n\nEnd."
    md, txt = tmp_path / "a.md", tmp_path / "a.txt"
    md.write_text(body, encoding="utf-8")
    txt.write_text(body, encoding="utf-8")
    md_import = assemble(extract_document(md, "md", tesseract=None))
    txt_import = assemble(extract_document(txt, "txt", tesseract=None))

    assert [(e.title, e.level, md_import.text[e.start : e.end].split("\n")[0]) for e in md_import.outline] == [
        ("Guide", 1, "Guide"),
        ("Setup", 2, "Setup ##"),
        ("After Setup", 2, "After Setup"),
    ]
    # Sections start at the title text, as for every other format.
    assert md_import.text == body
    assert (md_import.text, md_import.source_id) == (txt_import.text, txt_import.source_id)
