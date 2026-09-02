"""Structure-aware source segmentation models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
