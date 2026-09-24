"""Text-based formats: plain text and Markdown, SRT and WebVTT subtitles, RTF."""

from __future__ import annotations

import codecs
import html
import re

from striprtf.striprtf import rtf_to_text

_SRT_TIMING = re.compile(
    r"^\d{1,3}:\d{2}:\d{2}[,.]\d{1,3}[ \t]*-->[ \t]*\d{1,3}:\d{2}:\d{2}[,.]\d{1,3}"
)
_VTT_TIMING = re.compile(r"^(?:\d+:)?\d{2}:\d{2}\.\d{3}[ \t]+-->[ \t]+(?:\d+:)?\d{2}:\d{2}\.\d{3}")
_VTT_SKIPPED_BLOCK = re.compile(r"^(?:NOTE|STYLE|REGION)(?:[ \t]|$)")
_VOICE = re.compile(r"^\s*<v(?:\.[^\s>]*)?[ \t]+([^>]+)>")
_TAG = re.compile(r"<[^>\n]*>")
_ASS_OVERRIDE = re.compile(r"\{\\[^}]*\}")
_ANSI_CODEPAGE = re.compile(r"\\ansicpg(\d+)")


def _blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip():
            current.append(line.strip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _cue_line(line: str) -> str:
    voice = _VOICE.match(line)
    cleaned = html.unescape(_TAG.sub("", _ASS_OVERRIDE.sub("", line)))
    cleaned = " ".join(cleaned.split())
    if voice and cleaned:
        return f"{voice.group(1).strip()}: {cleaned}"
    return cleaned


def _join_cues(cues: list[list[str]]) -> tuple[str, int]:
    """One line per cue. Rolling captions repeat the previous line; repeats
    are dropped so each spoken line appears once."""
    lines: list[str] = []
    previous = ""
    kept = 0
    for cue in cues:
        parts = []
        for raw in cue:
            line = _cue_line(raw)
            if line and line != previous:
                parts.append(line)
                previous = line
        if parts:
            lines.append(" ".join(parts))
            kept += 1
    return "\n".join(lines), kept


def srt_to_text(text: str) -> tuple[str, int]:
    """Cue text of an SRT file without indices or timestamps, and the number
    of cues kept."""
    cues: list[list[str]] = []
    for block in _blocks(text):
        timing = next((i for i, line in enumerate(block[:3]) if _SRT_TIMING.match(line)), None)
        if timing is not None:
            cues.append(block[timing + 1 :])
        else:
            cues.append([line for line in block if not line.isdigit()])
    return _join_cues(cues)


def vtt_to_text(text: str) -> tuple[str, int]:
    """Cue text of a WebVTT file without the header, identifiers, timings, cue
    settings, NOTE, STYLE, or REGION blocks, and the number of cues kept."""
    cues: list[list[str]] = []
    for index, block in enumerate(_blocks(text.removeprefix("\ufeff"))):
        if index == 0 and block[0].startswith("WEBVTT"):
            continue
        if _VTT_SKIPPED_BLOCK.match(block[0]):
            continue
        timing = next((i for i, line in enumerate(block[:2]) if _VTT_TIMING.match(line)), None)
        if timing is not None:
            cues.append(block[timing + 1 :])
    return _join_cues(cues)


def rtf_to_plain(data: bytes) -> str:
    """Plain text of an RTF document; escaped characters use the document's
    ANSI code page."""
    head = data[:8192].decode("latin-1")
    match = _ANSI_CODEPAGE.search(head)
    codepage = f"cp{match.group(1)}" if match else "cp1252"
    try:
        codecs.lookup(codepage)
    except LookupError:
        codepage = "cp1252"
    try:
        source = data.decode("ascii")
    except UnicodeDecodeError:
        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError:
            source = data.decode(codepage, errors="replace")
    return rtf_to_text(source, encoding=codepage, errors="replace")
