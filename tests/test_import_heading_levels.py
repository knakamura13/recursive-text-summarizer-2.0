import zipfile
from pathlib import Path

from summarizer_web.ingestion.common import Extraction
from summarizer_web.ingestion.extract import extract_document

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def read(path: Path, document_format: str) -> Extraction:
    return extract_document(path, document_format, tesseract=None)  # type: ignore[arg-type]


def pairs(extraction: Extraction) -> list[tuple[str, int]]:
    return [(hint.title, hint.level) for hint in extraction.outline_hints]


def test_html_headings_keep_their_levels_in_order(tmp_path) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        "<h1>Top</h1><p>a</p><h3>Deep</h3><p>b</p><h2>Middle</h2><h6>Six</h6>", encoding="utf-8"
    )
    extraction = read(path, "html")
    assert pairs(extraction) == [("Top", 1), ("Deep", 3), ("Middle", 2), ("Six", 6)]


def test_html_without_headings_has_no_hints(tmp_path) -> None:
    path = tmp_path / "page.html"
    path.write_text("<p>one</p><ul><li>two</li></ul>", encoding="utf-8")
    assert read(path, "html").outline_hints == []


def _epub(path: Path, chapters: dict[str, str], spine: list[str], nav: str) -> None:
    items = "".join(
        f'<item id="{name}" href="{name}.xhtml" media-type="application/xhtml+xml"/>' for name in chapters
    )
    refs = "".join(f'<itemref idref="{name}"/>' for name in spine)
    opf = (
        '<package xmlns="http://www.idpf.org/2007/opf"><manifest>'
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
        f"{items}</manifest><spine>{refs}</spine></package>"
    )
    container = (
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
        '<rootfile full-path="OEBPS/content.opf"/></rootfiles></container>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/nav.xhtml", nav)
        for name, body in chapters.items():
            archive.writestr(f"OEBPS/{name}.xhtml", f"<html><body>{body}</body></html>")


def test_epub_hints_follow_spine_order_and_ignore_the_navigation_document(tmp_path) -> None:
    path = tmp_path / "book.epub"
    _epub(
        path,
        {"b": "<h1>Second</h1><h2>Second A</h2><p>x</p>", "a": "<h1>First</h1><p>y</p>"},
        ["a", "b"],
        "<h1>Contents</h1><ol><li>Nav only</li></ol>",
    )
    assert pairs(read(path, "epub")) == [("First", 1), ("Second", 1), ("Second A", 2)]


def _docx(path: Path, body: str, styles: str | None) -> None:
    document = f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
        if styles is not None:
            archive.writestr("word/styles.xml", f'<w:styles xmlns:w="{W}">{styles}</w:styles>')


def _p(text: str, style: str | None = None, outline: int | None = None) -> str:
    props = ""
    if style:
        props += f'<w:pStyle w:val="{style}"/>'
    if outline is not None:
        props += f'<w:outlineLvl w:val="{outline}"/>'
    return f"<w:p><w:pPr>{props}</w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>"


def _style(style_id: str, name: str, based_on: str | None = None, outline: int | None = None) -> str:
    based = f'<w:basedOn w:val="{based_on}"/>' if based_on else ""
    lvl = f'<w:pPr><w:outlineLvl w:val="{outline}"/></w:pPr>' if outline is not None else ""
    return f'<w:style w:type="paragraph" w:styleId="{style_id}"><w:name w:val="{name}"/>{based}{lvl}</w:style>'


def test_docx_levels_come_from_styles_outline_levels_title_and_subtitle(tmp_path) -> None:
    path = tmp_path / "doc.docx"
    styles = (
        _style("Title", "Title")
        + _style("Subtitle", "Subtitle")
        + _style("Heading1", "heading 1")
        + _style("Heading2", "heading 2")
        + _style("Custom", "Chapter Head", outline=2)
        + _style("Derived", "Derived Head", based_on="Heading2")
    )
    body = (
        _p("Book", "Title")
        + _p("Tagline", "Subtitle")
        + _p("Part", "Heading1")
        + _p("Body text")
        + _p("Sub", "Heading2")
        + _p("Custom one", "Custom")
        + _p("Inherited", "Derived")
        + _p("Direct", outline=0)
        + _p("Not a heading", outline=9)
    )
    _docx(path, body, styles)
    assert pairs(read(path, "docx")) == [
        ("Book", 1),
        ("Tagline", 2),
        ("Part", 1),
        ("Sub", 2),
        ("Custom one", 3),
        ("Inherited", 2),
        ("Direct", 1),
    ]


def test_docx_without_headings_has_no_hints(tmp_path) -> None:
    path = tmp_path / "doc.docx"
    _docx(path, _p("just text"), None)
    assert read(path, "docx").outline_hints == []


def test_odt_levels_come_from_the_outline_level(tmp_path) -> None:
    path = tmp_path / "doc.odt"
    content = (
        '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text>'
        '<text:h text:outline-level="1">One</text:h><text:p>body</text:p>'
        '<text:h text:outline-level="3">Three</text:h><text:h text:outline-level="2">Two</text:h>'
        "</office:text></office:body></office:document-content>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content.xml", content)
    assert pairs(read(path, "odt")) == [("One", 1), ("Three", 3), ("Two", 2)]


def test_docx_body_text_outline_level_beats_a_heading_style_on_the_paragraph_or_the_style(tmp_path) -> None:
    path = tmp_path / "doc.docx"
    styles = _style("Heading1", "heading 1") + _style("Quiet", "Heading Quiet", outline=9)
    body = (
        _p("Real", "Heading1")
        + _p("Demoted paragraph", "Heading1", outline=9)
        + _p("Demoted style", "Quiet")
        + _p("Promoted again", "Quiet", outline=1)
    )
    _docx(path, body, styles)
    extraction = read(path, "docx")
    assert pairs(extraction) == [("Real", 1), ("Promoted again", 2)]
    assert "Demoted paragraph" in extraction.text and "Demoted style" in extraction.text


def test_odt_levels_deeper_than_nine_are_kept(tmp_path) -> None:
    path = tmp_path / "doc.odt"
    content = (
        '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text>'
        '<text:h text:outline-level="12">Deep</text:h>'
        "</office:text></office:body></office:document-content>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content.xml", content)
    assert pairs(read(path, "odt")) == [("Deep", 12)]


def test_docx_style_based_on_a_heading_id_inherits_its_level_despite_its_name(tmp_path) -> None:
    path = tmp_path / "doc.docx"
    styles = (
        _style("Heading2", "Überschrift 2")
        + _style("Derived", "Derived", based_on="Heading2")
        + _style("FromMissing", "From missing", based_on="Subtitle")
        + _style("QuietBase", "Quiet base", outline=9).replace('w:styleId="QuietBase"', 'w:styleId="Heading3"')
        + _style("Quiet", "Quiet", based_on="Heading3")
    )
    body = _p("Inherited", "Derived") + _p("Missing base", "FromMissing") + _p("Body", "Quiet")
    _docx(path, body, styles)
    extraction = read(path, "docx")
    assert pairs(extraction) == [("Inherited", 2), ("Missing base", 2)]
    assert "Body" in extraction.text
