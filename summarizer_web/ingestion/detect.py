"""Upload format detection (D3).

Magic bytes and container manifests decide binary formats regardless of the
filename. For text, a known extension picks the flavour (plain, Markdown,
subtitles, HTML); without one, the content decides.
"""

from __future__ import annotations

import codecs
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

from charset_normalizer import from_bytes

from summarizer_web.ingestion.encoding import bom_encoding, utf16_without_bom
from summarizer_web.models.api import DocumentFormat

SUPPORTED_EXTENSIONS = (
    ".txt", ".md", ".srt", ".vtt", ".pdf", ".docx", ".odt", ".rtf", ".html", ".htm",
    ".epub", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
)

_TEXT_EXTENSIONS: dict[str, DocumentFormat] = {
    ".txt": "txt",
    ".text": "txt",
    ".md": "md",
    ".markdown": "md",
    ".srt": "srt",
    ".vtt": "vtt",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
}

SNIFF_BYTES = 64 * 1024
_MAX_CONTROL_RATIO = 0.01

_OFFICE_UNSUPPORTED = (
    "Legacy Microsoft Office files (.doc, .xls, .ppt) are not supported. "
    "Save the file as .docx or PDF and import it again."
)
_SIGNATURES: tuple[tuple[bytes, int, str], ...] = (
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", 0, _OFFICE_UNSUPPORTED),
    (b"GIF87a", 0, "GIF images are not supported. Convert the image to PNG."),
    (b"GIF89a", 0, "GIF images are not supported. Convert the image to PNG."),
    (b"WEBP", 8, "WebP images are not supported. Convert the image to PNG or JPEG."),
    (b"ftypheic", 4, "HEIC photos are not supported. Export the photo as JPEG."),
    (b"ftypmif1", 4, "HEIF images are not supported. Export the image as JPEG."),
    (b"ftypavif", 4, "AVIF images are not supported. Convert the image to PNG."),
    (b"BM", 0, "BMP images are not supported. Convert the image to PNG."),
    (b"\x1f\x8b", 0, "Compressed archives are not supported. Extract the document first."),
    (b"7z\xbc\xaf\x27\x1c", 0, "Compressed archives are not supported. Extract the document first."),
    (b"Rar!\x1a\x07", 0, "Compressed archives are not supported. Extract the document first."),
    (b"ID3", 0, "Audio files are not supported. Import a transcript (.srt, .vtt, or .txt)."),
    (b"OggS", 0, "Audio files are not supported. Import a transcript (.srt, .vtt, or .txt)."),
    (b"fLaC", 0, "Audio files are not supported. Import a transcript (.srt, .vtt, or .txt)."),
    (b"\x1a\x45\xdf\xa3", 0, "Video files are not supported. Import a transcript (.srt, .vtt, or .txt)."),
    (b"ftyp", 4, "Audio and video files are not supported. Import a transcript (.srt, .vtt, or .txt)."),
)

_SRT_START = re.compile(r"\A\d+[ \t]*\r?\n[ \t]*\d{1,3}:\d{2}:\d{2}[,.]\d{1,3}[ \t]*-->")
_HTML_START = re.compile(
    r"\A(?:<\?xml[^>]*>\s*)?(?:<!--.*?-->\s*)*(?:<!doctype\s+html|<html[\s>]|<head[\s>]|<body[\s>])",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Detection:
    """The detected format, or None with the reason the file is unsupported."""

    format: DocumentFormat | None
    reason: str | None = None


def detect_format(path: Path, filename: str) -> Detection:
    with path.open("rb") as handle:
        head = handle.read(SNIFF_BYTES)
        complete = not handle.read(1)

    if head.lstrip()[:5] == b"%PDF-":
        return Detection("pdf")
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return Detection("png")
    if head.startswith(b"\xff\xd8\xff"):
        return Detection("jpeg")
    if head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return Detection("tiff")
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return _detect_package(path)
    if head.removeprefix(codecs.BOM_UTF8).lstrip().startswith(b"{\\rtf"):
        return Detection("rtf")

    text = _sniff_text(head, complete)
    if text is None:
        # PDF readers accept a header anywhere in the first KiB of binary data.
        if b"%PDF-" in head[:1024]:
            return Detection("pdf")
        for signature, offset, reason in _SIGNATURES:
            if head[offset : offset + len(signature)] == signature:
                return Detection(None, reason)
        return Detection(None, "The file is not text, and its format is not supported.")
    suffix = Path(filename).suffix.lower()
    if suffix in _TEXT_EXTENSIONS:
        return Detection(_TEXT_EXTENSIONS[suffix])
    content = text.lstrip("\ufeff \t\r\n")
    if content.startswith("WEBVTT"):
        return Detection("vtt")
    if _SRT_START.match(content):
        return Detection("srt")
    if _HTML_START.match(content):
        return Detection("html")
    return Detection("txt")


def _detect_package(path: Path) -> Detection:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            mimetype = ""
            if "mimetype" in names and archive.getinfo("mimetype").file_size <= 256:
                mimetype = archive.read("mimetype").decode("ascii", errors="replace").strip()
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, RuntimeError, ValueError):
        return Detection(None, "The file looks like a ZIP package but could not be read.")
    if mimetype == "application/vnd.oasis.opendocument.text":
        return Detection("odt")
    if mimetype == "application/epub+zip":
        return Detection("epub")
    if mimetype.startswith("application/vnd.oasis.opendocument."):
        return Detection(
            None,
            "OpenDocument spreadsheets, presentations, and drawings are not supported. "
            "Export the file as .odt, .docx, or PDF.",
        )
    if "word/document.xml" in names:
        return Detection("docx")
    if "xl/workbook.xml" in names:
        return Detection(None, "Excel workbooks are not supported. Export the sheet as PDF.")
    if "ppt/presentation.xml" in names:
        return Detection(None, "PowerPoint presentations are not supported. Export them as PDF.")
    if "META-INF/container.xml" in names:
        return Detection("epub")
    return Detection(None, "ZIP archives are not supported. Extract the document first.")


def _utf8_prefix(head: bytes, complete: bool) -> str | None:
    try:
        return head.decode("utf-8")
    except UnicodeDecodeError as error:
        # The sniffed window may end inside a multi-byte sequence.
        if not complete and error.start >= len(head) - 3:
            try:
                return head[: error.start].decode("utf-8")
            except UnicodeDecodeError:
                return None
        return None


def _sniff_text(head: bytes, complete: bool) -> str | None:
    marked = bom_encoding(head)
    if marked is not None:
        codec, skip = marked
        width = 4 if codec.startswith("utf-32") else 2 if codec.startswith("utf-16") else 1
        body = head[skip:]
        text = body[: len(body) - len(body) % width].decode(codec, errors="replace")
    elif (wide := utf16_without_bom(head)) is not None:
        text = head[: len(head) - len(head) % 2].decode(wide, errors="replace")
    elif b"\x00" in head:
        return None
    else:
        text = _utf8_prefix(head, complete)
        if text is None:
            best = from_bytes(head).best()
            if best is None:
                return None
            text = str(best)
    return text if _mostly_printable(text) else None


def _mostly_printable(text: str) -> bool:
    if not text:
        return False
    controls = sum(
        1 for char in text if char not in "\t\n\r\f\v" and unicodedata.category(char) == "Cc"
    )
    return controls <= _MAX_CONTROL_RATIO * len(text)
