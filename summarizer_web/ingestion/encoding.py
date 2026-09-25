"""Text decoding for imported files: byte-order marks first, then strict
UTF-8, then UTF-16 without a BOM, then statistical detection."""

from __future__ import annotations

import codecs
from dataclasses import dataclass

from charset_normalizer import from_bytes

from summarizer_web.ingestion.common import ImportFailure

_BOMS = (
    # UTF-32 LE starts with the UTF-16 LE mark, so it is checked first.
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)

# Single-byte code pages for Latin scripts. Statistical detection often picks
# a Central European or Baltic page for Western text; when windows-1252 reads
# the bytes about as plausibly, it is the likelier source.
_LATIN_SINGLE_BYTE = frozenset(
    {
        "cp1250", "cp1252", "cp1254", "cp1257", "cp775", "cp850", "cp852", "cp858",
        "cp437", "latin_1", "iso8859_2", "iso8859_3", "iso8859_4", "iso8859_9",
        "iso8859_10", "iso8859_13", "iso8859_14", "iso8859_15", "iso8859_16",
        "mac_latin2", "mac_roman", "mac_iceland", "mac_turkish",
    }
)
_CP1252_CHAOS_MARGIN = 0.05
_CP1252_COHERENCE_MARGIN = 0.05

_DISPLAY_NAMES = {
    "utf_8": "utf-8",
    "cp1252": "windows-1252",
    "cp1250": "windows-1250",
    "cp1251": "windows-1251",
    "cp1253": "windows-1253",
    "cp1254": "windows-1254",
    "cp1255": "windows-1255",
    "cp1256": "windows-1256",
    "cp1257": "windows-1257",
    "iso8859_1": "iso-8859-1",
}


@dataclass(frozen=True)
class DecodedText:
    text: str
    encoding: str
    detected: bool = False
    replaced: int = 0


def encoding_label(codec: str) -> str:
    """Human-facing name of a Python codec, e.g. cp1252 -> windows-1252."""
    normalized = codecs.lookup(codec).name.replace("-", "_")
    return _DISPLAY_NAMES.get(normalized, normalized.replace("_", "-"))


def bom_encoding(data: bytes) -> tuple[str, int] | None:
    for bom, codec in _BOMS:
        if data.startswith(bom):
            return codec, len(bom)
    return None


def utf16_without_bom(data: bytes) -> str | None:
    """Recognize UTF-16 text without a BOM by its zero high bytes."""
    sample = data[:4096]
    half = len(sample) // 2
    if half < 2:
        return None
    even_zeros = sample[0::2].count(0)
    odd_zeros = sample[1::2].count(0)
    if odd_zeros > 0.4 * half and even_zeros < 0.05 * half:
        return "utf-16-le"
    if even_zeros > 0.4 * half and odd_zeros < 0.05 * half:
        return "utf-16-be"
    return None


def _decode_counting(data: bytes, codec: str) -> tuple[str, int]:
    try:
        return data.decode(codec), 0
    except UnicodeDecodeError:
        text = data.decode(codec, errors="replace")
        return text, text.count("\ufffd") - data.decode(codec, errors="ignore").count("\ufffd")


def _prefer_cp1252(data: bytes, results) -> str | None:
    best = results.best()
    if best is None or best.encoding not in _LATIN_SINGLE_BYTE:
        return None
    try:
        western = data.decode("cp1252")
    except UnicodeDecodeError:
        return None
    if best.encoding == "cp1252" or str(best) == western:
        return "cp1252"
    for match in results:
        if match.encoding == "cp1252":
            if (
                match.chaos <= best.chaos + _CP1252_CHAOS_MARGIN
                and match.coherence >= best.coherence - _CP1252_COHERENCE_MARGIN
            ):
                return "cp1252"
            return None
    return None


def decode_text(data: bytes, *, declared: str | None = None) -> DecodedText:
    """Decode file bytes. `declared` is an encoding named by the file itself
    (an HTML meta charset or XML declaration); it is tried after UTF-8."""
    marked = bom_encoding(data)
    if marked is not None:
        codec, skip = marked
        text, replaced = _decode_counting(data[skip:], codec)
        return DecodedText(text, encoding_label(codec), replaced=replaced)
    try:
        return DecodedText(data.decode("utf-8"), "utf-8")
    except UnicodeDecodeError:
        pass
    wide = utf16_without_bom(data)
    if wide is not None:
        text, replaced = _decode_counting(data[: len(data) - len(data) % 2], wide)
        return DecodedText(text, encoding_label(wide), detected=True, replaced=replaced)
    if declared:
        try:
            codecs.lookup(declared)
            return DecodedText(data.decode(declared), encoding_label(declared))
        except (LookupError, UnicodeDecodeError):
            pass
    results = from_bytes(data)
    codec = _prefer_cp1252(data, results)
    if codec is None:
        best = results.best()
        if best is None:
            raise ImportFailure(
                "The text encoding could not be determined. Save the file as UTF-8 and import it again."
            )
        codec = best.encoding
    text, replaced = _decode_counting(data, codec)
    return DecodedText(text, encoding_label(codec), detected=True, replaced=replaced)
