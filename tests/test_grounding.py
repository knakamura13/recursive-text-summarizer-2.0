from dataclasses import dataclass

import pytest

from summarizer.budget import BudgetError
from summarizer.grounding import GroundingPolicy, select_source_passages, serialize_source_passage
from summarizer.summaries import SummaryNode


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


SOURCE = {
    "S000001": "ordinary source",
    "S000002": "quoted source",
    "S000003": "uncertain source",
    "S000004": "contradictory source",
}


def child() -> SummaryNode:
    return SummaryNode.model_validate(
        {
            "summary": "A generated child.",
            "content_units": [
                {
                    "text": "An ordinary claim.",
                    "kind": "claim",
                    "evidence": [{"segment_id": "S000001", "quote": None}],
                    "qualification": None,
                    "uncertain": False,
                },
                {
                    "text": "An uncertain claim.",
                    "kind": "claim",
                    "evidence": [{"segment_id": "S000003", "quote": None}],
                    "qualification": "The source hedges this.",
                    "uncertain": True,
                },
            ],
            "entities": [],
            "qualifications": [],
            "contradictions": [
                {
                    "text": "A source disagrees.",
                    "evidence": [{"segment_id": "S000004", "quote": None}],
                }
            ],
            "quotations": [{"segment_id": "S000002", "quote": "quoted"}],
            "provenance": ["S000001", "S000002", "S000003", "S000004"],
            "level": 1,
        }
    )


def test_select_source_passages_rejects_empty_children() -> None:
    with pytest.raises(ValueError, match="source grounding requires at least one child"):
        select_source_passages(
            (),
            source=SOURCE,
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=1000),
        )


def test_select_source_passages_rejects_unknown_segment() -> None:
    child_unknown = SummaryNode.model_validate(
        {
            "summary": "Unknown citation.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": ["S999999"],
            "level": 0,
        }
    )
    with pytest.raises(ValueError, match="source text is missing for segment S999999"):
        select_source_passages(
            (child_unknown,),
            source=SOURCE,
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=1000),
        )


def test_select_source_passages_at_exact_cost_boundary() -> None:
    from summarizer.grounding import SourcePassage, serialize_source_passage

    passage = SourcePassage("S000001", "ordinary source")
    cost = CharacterCounter().count(serialize_source_passage(passage))
    child_node = SummaryNode.model_validate(
        {
            "summary": "Exact boundary.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": ["S000001"],
            "level": 0,
        }
    )
    # At exact boundary (max_tokens == cost), passage should be accepted.
    selection = select_source_passages(
        (child_node,),
        source={"S000001": "ordinary source"},
        counter=CharacterCounter(),
        policy=GroundingPolicy(max_tokens=cost),
    )
    assert selection.selected_ids == ("S000001",)


def test_prioritizes_ambiguous_evidence_before_other_retained_claims() -> None:
    selection = select_source_passages(
        (child(),),
        source=SOURCE,
        counter=CharacterCounter(),
        policy=GroundingPolicy(max_tokens=10_000),
    )

    assert selection.selected_ids == (
        "S000004",
        "S000003",
        "S000002",
        "S000001",
    )


def test_refuses_to_drop_a_mandatory_ambiguous_source() -> None:
    with pytest.raises(ValueError, match="mandatory"):
        select_source_passages(
            (child(),),
            source=SOURCE,
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=1),
        )


def test_counts_the_complete_source_section_inside_the_grounding_reserve() -> None:
    with pytest.raises(ValueError, match="mandatory"):
        select_source_passages(
            (child(),),
            source=SOURCE,
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=99),
            selection_cost=lambda _passages: 100,
        )


def test_counts_separators_between_selected_source_blocks() -> None:
    def section_cost(passages: tuple[object, ...]) -> int:
        return 10 * len(passages) + max(len(passages) - 1, 0)

    with pytest.raises(ValueError, match="mandatory"):
        select_source_passages(
            (child(),),
            source=SOURCE,
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=20),
            selection_cost=section_cost,
        )


def test_empty_selection_reports_reserve_and_smallest_candidate() -> None:
    source = {
        "S000001": "x" * 30,
        "S000002": "y" * 20,
    }
    children = (
        SummaryNode.model_validate(
            {
                "summary": "Two sources.",
                "content_units": [],
                "entities": [],
                "qualifications": [],
                "contradictions": [],
                "quotations": [],
                "provenance": ["S000001", "S000002"],
                "level": 0,
            }
        ),
    )

    with pytest.raises(
        BudgetError,
        match=(
            r"grounding reserve of 10 tokens cannot hold source passage "
            r"S000002 costing 54 tokens"
        ),
    ):
        select_source_passages(
            children,
            source=source,
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=10),
        )


def _quoting_child(segment_id: str, quote: str) -> SummaryNode:
    return SummaryNode.model_validate(
        {
            "summary": "A child quoting its source.",
            "content_units": [
                {
                    "text": "A claim.",
                    "kind": "claim",
                    "evidence": [{"segment_id": segment_id, "quote": quote}],
                    "qualification": None,
                    "uncertain": False,
                }
            ],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [segment_id],
            "level": 0,
        }
    )


LONG_CORE = " ".join(f"word{index}" for index in range(600)) + " the cited sentence " + " ".join(
    f"tail{index}" for index in range(600)
)


def test_excerpt_keeps_the_cited_quote_when_no_whole_passage_fits() -> None:
    selection = select_source_passages(
        (_quoting_child("S000001", "the cited sentence"),),
        source={"S000001": LONG_CORE},
        counter=CharacterCounter(),
        policy=GroundingPolicy(max_tokens=1_500),
        allow_excerpts=True,
    )

    (passage,) = selection.passages
    assert selection.selected_ids == ("S000001",)
    assert "the cited sentence" in passage.text
    assert passage.text in LONG_CORE
    assert CharacterCounter().count(serialize_source_passage(passage)) <= 1_500
    assert not passage.text.startswith(" ") and not passage.text.endswith(" ")


def test_whole_passages_are_used_when_they_fit_even_with_excerpts_allowed() -> None:
    selection = select_source_passages(
        (_quoting_child("S000001", "the cited sentence"),),
        source={"S000001": LONG_CORE},
        counter=CharacterCounter(),
        policy=GroundingPolicy(max_tokens=100_000),
        allow_excerpts=True,
    )

    assert selection.passages[0].text == LONG_CORE


def test_excerpts_are_off_unless_allowed() -> None:
    with pytest.raises(BudgetError, match="cannot hold source passage S000001"):
        select_source_passages(
            (_quoting_child("S000001", "the cited sentence"),),
            source={"S000001": LONG_CORE},
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=1_500),
        )


def test_excerpt_still_fails_when_even_the_shortest_does_not_fit() -> None:
    with pytest.raises(BudgetError, match="cannot hold source passage S000001"):
        select_source_passages(
            (_quoting_child("S000001", "the cited sentence"),),
            source={"S000001": LONG_CORE},
            counter=CharacterCounter(),
            policy=GroundingPolicy(max_tokens=200),
            allow_excerpts=True,
        )
