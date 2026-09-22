from dataclasses import FrozenInstanceError

import pytest

from summarizer.segmentation import (
    BoundaryKind,
    SegmentationConfig,
    SourceSegment,
    StructuralBlock,
)


def test_structural_block_captures_nonempty_source_range() -> None:
    block = StructuralBlock(start=3, end=9, boundary_kind=BoundaryKind.PARAGRAPH)

    assert block.start == 3
    assert block.end == 9
    assert block.boundary_kind is BoundaryKind.PARAGRAPH


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 1), (2, 2), (3, 2)],
)
def test_structural_block_rejects_invalid_ranges(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="range"):
        StructuralBlock(
            start=start,
            end=end,
            boundary_kind=BoundaryKind.PARAGRAPH,
        )


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_segmentation_config_requires_positive_budget(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        SegmentationConfig(max_tokens=max_tokens)


def test_segmentation_config_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        SegmentationConfig(max_tokens=10, overlap_tokens=-1)


def test_source_segment_records_core_and_context_provenance() -> None:
    segment = SourceSegment(
        segment_id="S000002",
        source_id="source-hash",
        order=1,
        text="prior core",
        core_start=6,
        core_end=10,
        context_start=0,
        context_end=10,
        core_token_count=4,
        token_count=10,
        leading_overlap_tokens=6,
        trailing_overlap_tokens=0,
        boundary_kind=BoundaryKind.SENTENCE,
    )

    assert segment.core_start == 6
    assert segment.context_start == 0
    assert segment.leading_overlap_tokens == 6

    with pytest.raises(FrozenInstanceError):
        segment.order = 2  # type: ignore[misc]


def valid_segment_values() -> dict[str, object]:
    return {
        "segment_id": "S000001",
        "source_id": "source-hash",
        "order": 0,
        "text": "content",
        "core_start": 0,
        "core_end": 7,
        "context_start": 0,
        "context_end": 7,
        "core_token_count": 7,
        "token_count": 7,
        "leading_overlap_tokens": 0,
        "trailing_overlap_tokens": 0,
        "boundary_kind": BoundaryKind.HARD,
    }


@pytest.mark.parametrize("field", ["segment_id", "source_id", "text"])
def test_source_segment_rejects_empty_identifiers_and_text(field: str) -> None:
    values = valid_segment_values()
    values[field] = ""

    with pytest.raises(ValueError, match=field):
        SourceSegment(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("order", -1),
        ("core_token_count", -1),
        ("token_count", -1),
        ("leading_overlap_tokens", -1),
        ("trailing_overlap_tokens", -1),
    ],
)
def test_source_segment_rejects_negative_metadata(field: str, value: int) -> None:
    values = valid_segment_values()
    values[field] = value

    with pytest.raises(ValueError, match=field):
        SourceSegment(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("core_start", -1),
        ("core_end", 0),
        ("context_start", -1),
        ("context_end", 0),
    ],
)
def test_source_segment_rejects_invalid_or_empty_ranges(
    field: str,
    value: int,
) -> None:
    values = valid_segment_values()
    values[field] = value

    with pytest.raises(ValueError, match="range"):
        SourceSegment(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [("context_start", 1), ("context_end", 6)],
)
def test_context_range_must_contain_core_range(field: str, value: int) -> None:
    values = valid_segment_values()
    values[field] = value

    with pytest.raises(ValueError, match="contain"):
        SourceSegment(**values)  # type: ignore[arg-type]


def test_segment_text_length_must_match_context_range() -> None:
    values = valid_segment_values()
    values["text"] = "short"

    with pytest.raises(ValueError, match="text"):
        SourceSegment(**values)  # type: ignore[arg-type]
