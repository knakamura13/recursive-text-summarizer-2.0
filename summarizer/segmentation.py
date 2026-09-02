"""Structure-aware source segmentation models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class BoundaryKind(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    SENTENCE = "sentence"
    HARD = "hard"


@dataclass(frozen=True)
class StructuralBlock:
    start: int
    end: int
    boundary_kind: BoundaryKind

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("structural block range must be nonempty and ordered")


@dataclass(frozen=True)
class SegmentationConfig:
    max_tokens: int
    overlap_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")


@dataclass(frozen=True)
class SourceSegment:
    segment_id: str
    source_id: str
    order: int
    text: str
    core_start: int
    core_end: int
    context_start: int
    context_end: int
    core_token_count: int
    token_count: int
    leading_overlap_tokens: int
    trailing_overlap_tokens: int
    boundary_kind: BoundaryKind

    def __post_init__(self) -> None:
        for field_name in ("segment_id", "source_id", "text"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty")
        if self.order < 0:
            raise ValueError("order cannot be negative")
        for field_name in (
            "core_token_count",
            "token_count",
            "leading_overlap_tokens",
            "trailing_overlap_tokens",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.core_start < 0 or self.core_end <= self.core_start:
            raise ValueError("core range must be nonempty and ordered")
        if self.context_start < 0 or self.context_end <= self.context_start:
            raise ValueError("context range must be nonempty and ordered")
        if self.context_start > self.core_start or self.context_end < self.core_end:
            raise ValueError("context range must contain core range")
        if len(self.text) != self.context_end - self.context_start:
            raise ValueError("text length must match context range")


_ATX_HEADING = re.compile(r" {0,3}#{1,6}(?:[ \t]+|$)")
_SETEXT_UNDERLINE = re.compile(r" {0,3}(?:=+|-+)[ \t]*$")
_LIST_ITEM = re.compile(r" {0,3}(?:[-+*][ \t]+|\d+[.)][ \t]+)")


@dataclass(frozen=True)
class _LineSpan:
    start: int
    end: int
    content: str


def _line_spans(text: str) -> list[_LineSpan]:
    return [
        _LineSpan(match.start(), match.end(), match.group().removesuffix("\n"))
        for match in re.finditer(r"[^\n]*(?:\n|$)", text)
        if match.end() > match.start()
    ]


def _is_setext_heading(lines: list[_LineSpan], index: int) -> bool:
    return (
        bool(lines[index].content)
        and index + 1 < len(lines)
        and _SETEXT_UNDERLINE.fullmatch(lines[index + 1].content) is not None
    )


def _consume_blank_lines(lines: list[_LineSpan], index: int) -> int:
    while index < len(lines) and not lines[index].content:
        index += 1
    return index


def detect_structural_blocks(text: str) -> list[StructuralBlock]:
    """Return contiguous heading, paragraph, and list ranges."""
    lines = _line_spans(text)
    blocks: list[StructuralBlock] = []
    index = 0
    while index < len(lines):
        start_index = index
        content = lines[index].content
        if _ATX_HEADING.match(content):
            kind = BoundaryKind.HEADING
            index += 1
        elif _is_setext_heading(lines, index):
            kind = BoundaryKind.HEADING
            index += 2
        elif _LIST_ITEM.match(content):
            kind = BoundaryKind.LIST
            index += 1
            while index < len(lines) and lines[index].content:
                if _ATX_HEADING.match(lines[index].content) or _is_setext_heading(
                    lines, index
                ):
                    break
                index += 1
        else:
            kind = BoundaryKind.PARAGRAPH
            index += 1
            while index < len(lines) and lines[index].content:
                if (
                    _ATX_HEADING.match(lines[index].content)
                    or _LIST_ITEM.match(lines[index].content)
                    or _is_setext_heading(lines, index)
                ):
                    break
                index += 1
        index = _consume_blank_lines(lines, index)
        blocks.append(
            StructuralBlock(
                start=lines[start_index].start,
                end=lines[index - 1].end,
                boundary_kind=kind,
            )
        )
    return blocks
