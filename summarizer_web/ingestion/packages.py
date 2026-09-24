"""DOCX, ODT, and EPUB: ZIP packages of XML read with the standard library.

Headings, paragraphs, list items, and table rows become text blocks. Tracked
deletions, field codes, notes, and comments are left out. EPUB content
documents are read in spine order.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

from summarizer_web.ingestion.common import (
    Block,
    ImportFailure,
    ProgressCallback,
    collapse_spaces,
    render_blocks,
)
from summarizer_web.ingestion.encoding import decode_text
from summarizer_web.ingestion.markup import declared_charset, html_to_blocks

_MAX_MEMBER_BYTES = 256 * 1024 * 1024


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@contextmanager
def _package(path: Path, label: str) -> Iterator[zipfile.ZipFile]:
    try:
        with zipfile.ZipFile(path) as archive:
            yield archive
    except (zipfile.BadZipFile, zipfile.LargeZipFile, zlib.error, EOFError) as error:
        raise ImportFailure(f"The {label} file is damaged and could not be read ({error}).") from error
    except NotImplementedError as error:
        raise ImportFailure(f"The {label} file uses an unsupported compression method.") from error
    except RuntimeError as error:
        # zipfile raises RuntimeError for password-protected members.
        raise ImportFailure(f"The {label} file is encrypted and cannot be imported.") from error


def _read_member(archive: zipfile.ZipFile, name: str, label: str) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError as error:
        raise ImportFailure(f"The {label} file is incomplete: {name} is missing.") from error
    too_large = ImportFailure(f"The {label} file is too large to import: {name} expands beyond 256 MiB.")
    if info.file_size > _MAX_MEMBER_BYTES:
        raise too_large
    with archive.open(info) as handle:
        data = handle.read(_MAX_MEMBER_BYTES + 1)
    if len(data) > _MAX_MEMBER_BYTES:
        raise too_large
    return data


def _parse_xml(data: bytes, name: str, label: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as error:
        raise ImportFailure(f"The {label} file is damaged: {name} is not valid XML ({error}).") from error


# --- DOCX ---------------------------------------------------------------------

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_W_VAL = f"{_W}val"
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
_DOCX_SKIPPED = frozenset(
    {
        f"{_W}pPr", f"{_W}rPr", f"{_W}del", f"{_W}moveFrom", f"{_W}instrText",
        f"{_W}delText", f"{_W}fldChar", f"{_W}footnoteReference", f"{_W}endnoteReference",
        f"{_W}commentReference", f"{_W}sdtPr", _MC_FALLBACK,
    }
)
_DOCX_WRAPPERS = frozenset({f"{_W}customXml", f"{_W}ins", f"{_W}moveTo", f"{_W}smartTag"})


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _docx_styles(archive: zipfile.ZipFile) -> tuple[frozenset[str], frozenset[str]]:
    """Paragraph style ids that are headings and those that are list styles."""
    if "word/styles.xml" not in archive.namelist():
        return frozenset(), frozenset()
    root = _parse_xml(_read_member(archive, "word/styles.xml", "DOCX"), "word/styles.xml", "DOCX")
    headings: set[str] = set()
    lists: set[str] = set()
    for style in root.iter(f"{_W}style"):
        if style.get(f"{_W}type") != "paragraph":
            continue
        style_id = style.get(f"{_W}styleId") or ""
        name_element = style.find(f"{_W}name")
        name = (name_element.get(_W_VAL) or "").lower() if name_element is not None else ""
        properties = style.find(f"{_W}pPr")
        outline = properties.find(f"{_W}outlineLvl") if properties is not None else None
        if (
            name.startswith("heading")
            or name in ("title", "subtitle")
            or (outline is not None and _int(outline.get(_W_VAL), 9) < 9)
        ):
            headings.add(style_id)
        if properties is not None and properties.find(f"{_W}numPr") is not None:
            lists.add(style_id)
    return frozenset(headings), frozenset(lists)


def _docx_text(element: ElementTree.Element) -> str:
    parts: list[str] = []

    def walk(node: ElementTree.Element) -> None:
        for child in node:
            tag = child.tag
            if tag == f"{_W}t":
                parts.append(child.text or "")
            elif tag == f"{_W}tab":
                parts.append(" ")
            elif tag in (f"{_W}br", f"{_W}cr"):
                parts.append("\n")
            elif tag == f"{_W}noBreakHyphen":
                parts.append("-")
            elif tag not in _DOCX_SKIPPED:
                walk(child)

    walk(element)
    return collapse_spaces("".join(parts))


def _docx_paragraph(
    paragraph: ElementTree.Element,
    headings: frozenset[str],
    list_styles: frozenset[str],
) -> Block | None:
    text = _docx_text(paragraph)
    if not text:
        return None
    properties = paragraph.find(f"{_W}pPr")
    if properties is None:
        return Block("paragraph", text)
    style = properties.find(f"{_W}pStyle")
    style_id = (style.get(_W_VAL) or "") if style is not None else ""
    lowered = style_id.lower()
    outline = properties.find(f"{_W}outlineLvl")
    if (
        style_id in headings
        or lowered.startswith("heading")
        or lowered in ("title", "subtitle")
        or (outline is not None and _int(outline.get(_W_VAL), 9) < 9)
    ):
        return Block("heading", text)
    numbering = properties.find(f"{_W}numPr")
    if numbering is not None:
        number_id = numbering.find(f"{_W}numId")
        if number_id is not None and number_id.get(_W_VAL) == "0":
            return Block("paragraph", text)
        level_element = numbering.find(f"{_W}ilvl")
        level = _int(level_element.get(_W_VAL), 0) if level_element is not None else 0
        return Block("list_item", "  " * level + "- " + text)
    if style_id in list_styles:
        return Block("list_item", "- " + text)
    return Block("paragraph", text)


def _docx_rows(table: ElementTree.Element) -> Iterator[ElementTree.Element]:
    for child in table:
        if child.tag == f"{_W}tr":
            yield child
        elif child.tag in _DOCX_WRAPPERS or child.tag == f"{_W}sdt":
            content = child.find(f"{_W}sdtContent") if child.tag == f"{_W}sdt" else child
            if content is not None:
                yield from _docx_rows(content)


def _docx_cells(row: ElementTree.Element) -> Iterator[ElementTree.Element]:
    for child in row:
        if child.tag == f"{_W}tc":
            yield child
        elif child.tag == f"{_W}sdt":
            content = child.find(f"{_W}sdtContent")
            if content is not None:
                yield from _docx_cells(content)


def _docx_blocks(
    container: ElementTree.Element,
    headings: frozenset[str],
    list_styles: frozenset[str],
    blocks: list[Block],
) -> None:
    for child in container:
        tag = child.tag
        if tag == f"{_W}p":
            block = _docx_paragraph(child, headings, list_styles)
            if block is not None:
                blocks.append(block)
        elif tag == f"{_W}tbl":
            for row in _docx_rows(child):
                cells = [
                    " ".join(filter(None, (_docx_text(p) for p in cell.iter(f"{_W}p"))))
                    for cell in _docx_cells(row)
                ]
                if any(cells):
                    blocks.append(Block("table_row", " | ".join(cells)))
        elif tag == f"{_W}sdt":
            content = child.find(f"{_W}sdtContent")
            if content is not None:
                _docx_blocks(content, headings, list_styles, blocks)
        elif tag in _DOCX_WRAPPERS:
            _docx_blocks(child, headings, list_styles, blocks)


def extract_docx(path: Path) -> str:
    with _package(path, "DOCX") as archive:
        document = _parse_xml(
            _read_member(archive, "word/document.xml", "DOCX"), "word/document.xml", "DOCX"
        )
        headings, list_styles = _docx_styles(archive)
    body = document.find(f"{_W}body")
    if body is None:
        raise ImportFailure("The DOCX file has no document body.")
    blocks: list[Block] = []
    _docx_blocks(body, headings, list_styles, blocks)
    return render_blocks(blocks)


# --- ODT ----------------------------------------------------------------------

_TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
_TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
_OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
_ODT_SKIPPED_INLINE = frozenset(
    {f"{_TEXT}note", f"{_OFFICE}annotation", f"{_OFFICE}annotation-end", f"{_TEXT}tracked-changes"}
)
_ODT_SECTIONS = frozenset(
    {
        f"{_TEXT}section", f"{_TEXT}index-body", f"{_TEXT}table-of-content",
        f"{_TEXT}illustration-index", f"{_TEXT}table-index", f"{_TEXT}object-index",
        f"{_TEXT}user-index", f"{_TEXT}alphabetical-index", f"{_TEXT}bibliography",
    }
)
_ODT_ROW_GROUPS = frozenset(
    {f"{_TABLE}table-header-rows", f"{_TABLE}table-rows", f"{_TABLE}table-row-group"}
)
# ODF collapses whitespace in character data; text:s, text:tab, and
# text:line-break carry the significant spacing.
_ODF_WHITESPACE = re.compile(r"[ \t\r\n]+")


def _odt_inline(element: ElementTree.Element) -> str:
    parts: list[str] = []

    def walk(node: ElementTree.Element) -> None:
        if node.text:
            parts.append(_ODF_WHITESPACE.sub(" ", node.text))
        for child in node:
            tag = child.tag
            if tag in (f"{_TEXT}s", f"{_TEXT}tab"):
                parts.append(" ")
            elif tag == f"{_TEXT}line-break":
                parts.append("\n")
            elif tag not in _ODT_SKIPPED_INLINE:
                walk(child)
            if child.tail:
                parts.append(_ODF_WHITESPACE.sub(" ", child.tail))

    walk(element)
    return collapse_spaces("".join(parts))


def _odt_list(element: ElementTree.Element, depth: int, blocks: list[Block]) -> None:
    for item in element:
        if item.tag not in (f"{_TEXT}list-item", f"{_TEXT}list-header"):
            continue
        first = True
        for child in item:
            if child.tag in (f"{_TEXT}p", f"{_TEXT}h"):
                text = _odt_inline(child)
                if text:
                    marker = "  " * depth + ("- " if first else "  ")
                    blocks.append(Block("list_item", marker + text))
                    first = False
            elif child.tag == f"{_TEXT}list":
                _odt_list(child, depth + 1, blocks)


def _odt_rows(table: ElementTree.Element) -> Iterator[ElementTree.Element]:
    for child in table:
        if child.tag == f"{_TABLE}table-row":
            yield child
        elif child.tag in _ODT_ROW_GROUPS:
            yield from _odt_rows(child)


def _odt_blocks(container: ElementTree.Element, blocks: list[Block]) -> None:
    for child in container:
        tag = child.tag
        if tag == f"{_TEXT}h":
            text = _odt_inline(child)
            if text:
                blocks.append(Block("heading", text))
        elif tag == f"{_TEXT}p":
            text = _odt_inline(child)
            if text:
                blocks.append(Block("paragraph", text))
        elif tag == f"{_TEXT}list":
            _odt_list(child, 0, blocks)
        elif tag == f"{_TABLE}table":
            for row in _odt_rows(child):
                cells = [
                    " ".join(filter(None, (_odt_inline(p) for p in cell.iter() if p.tag in (f"{_TEXT}p", f"{_TEXT}h"))))
                    for cell in row
                    if cell.tag == f"{_TABLE}table-cell"
                ]
                if any(cells):
                    blocks.append(Block("table_row", " | ".join(cells)))
        elif tag in _ODT_SECTIONS:
            _odt_blocks(child, blocks)


def extract_odt(path: Path) -> str:
    with _package(path, "ODT") as archive:
        root = _parse_xml(_read_member(archive, "content.xml", "ODT"), "content.xml", "ODT")
    body = root.find(f"{_OFFICE}body/{_OFFICE}text")
    if body is None:
        raise ImportFailure("The ODT file has no text body.")
    blocks: list[Block] = []
    _odt_blocks(body, blocks)
    return render_blocks(blocks)


# --- EPUB ---------------------------------------------------------------------

_CONTENT_TYPES = frozenset({"application/xhtml+xml", "text/html"})
# Font obfuscation, not DRM: the text itself stays readable.
_FONT_OBFUSCATION = frozenset(
    {"http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC"}
)


def _epub_encrypted(archive: zipfile.ZipFile) -> set[str]:
    if "META-INF/encryption.xml" not in archive.namelist():
        return set()
    root = _parse_xml(
        _read_member(archive, "META-INF/encryption.xml", "EPUB"), "META-INF/encryption.xml", "EPUB"
    )
    encrypted: set[str] = set()
    for data in root.iter():
        if _local(data.tag) != "EncryptedData":
            continue
        method = next((e for e in data.iter() if _local(e.tag) == "EncryptionMethod"), None)
        if method is not None and method.get("Algorithm") in _FONT_OBFUSCATION:
            continue
        for reference in data.iter():
            if _local(reference.tag) == "CipherReference" and reference.get("URI"):
                encrypted.add(posixpath.normpath(unquote(reference.get("URI", ""))))
    return encrypted


def _epub_spine(archive: zipfile.ZipFile) -> list[str]:
    container = _parse_xml(
        _read_member(archive, "META-INF/container.xml", "EPUB"), "META-INF/container.xml", "EPUB"
    )
    rootfile = next(
        (e for e in container.iter() if _local(e.tag) == "rootfile" and e.get("full-path")), None
    )
    if rootfile is None:
        raise ImportFailure("The EPUB file is incomplete: it names no package document.")
    package_path = unquote(rootfile.get("full-path", ""))
    package = _parse_xml(_read_member(archive, package_path, "EPUB"), package_path, "EPUB")
    base = posixpath.dirname(package_path)
    manifest = {
        item.get("id"): item
        for item in package.iter()
        if _local(item.tag) == "item" and item.get("id")
    }
    linear: list[str] = []
    auxiliary: list[str] = []
    for reference in package.iter():
        if _local(reference.tag) != "itemref":
            continue
        item = manifest.get(reference.get("idref"))
        if item is None or item.get("media-type") not in _CONTENT_TYPES:
            continue
        if "nav" in (item.get("properties") or "").split():
            continue
        href = unquote((item.get("href") or "").split("#", 1)[0])
        if not href:
            continue
        member = posixpath.normpath(posixpath.join(base, href))
        (auxiliary if reference.get("linear") == "no" else linear).append(member)
    return linear or auxiliary


def extract_epub(path: Path, progress: ProgressCallback) -> str:
    blocks: list[Block] = []
    with _package(path, "EPUB") as archive:
        spine = _epub_spine(archive)
        if not spine:
            raise ImportFailure("The EPUB file lists no readable chapters.")
        if _epub_encrypted(archive) & set(spine):
            raise ImportFailure("This EPUB is protected with DRM and cannot be imported.")
        progress("extracting", 0, len(spine))
        for index, member in enumerate(spine, start=1):
            data = _read_member(archive, member, "EPUB")
            blocks.extend(html_to_blocks(decode_text(data, declared=declared_charset(data)).text))
            progress("extracting", index, len(spine))
    return render_blocks(blocks)
