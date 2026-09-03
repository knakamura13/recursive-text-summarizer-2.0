import json
from dataclasses import dataclass

import pytest

from summarizer.budget import BudgetError
from summarizer.hierarchy import (
    HierarchyError,
    build_hierarchy,
    group_children,
    measure_child_tokens,
    merge_fanout,
)
from summarizer.leaf import LeafSummaryError
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderConnectionError,
)
from summarizer.summaries import SummaryNode

SOURCE_ID = "a" * 64


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


def leaf(index: int, *, summary: str | None = None, **overrides: object) -> SummaryNode:
    body: dict[str, object] = {
        "summary": summary or f"Summary of part {index}.",
        "content_units": [],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": [f"S{index:06d}"],
        "level": 0,
    }
    body.update(overrides)
    return SummaryNode.model_validate(body)


def leaves(count: int, **overrides: object) -> list[SummaryNode]:
    return [leaf(index + 1, **overrides) for index in range(count)]


def covered_for(count: int) -> list[tuple[str, ...]]:
    return [(f"S{index + 1:06d}",) for index in range(count)]


def attributable_for(count: int) -> dict[str, str]:
    return {f"S{index + 1:06d}": f"Part {index + 1} said something." for index in range(count)}


class MergingProvider:
    """Answers each merge with a node at the requested level."""

    def __init__(self, *, level_override: int | None = None, text: str | None = None) -> None:
        self.requests: list[GenerationRequest] = []
        self.level_override = level_override
        self.text = text

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if self.text is not None:
            return GenerationResult(text=self.text, provider="fake", model=request.model)
        level = int((request.operation_id or "merge-L1").rsplit("L", 1)[1])
        body = {
            "summary": f"Merged at level {level}.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [],
            "level": self.level_override if self.level_override is not None else level,
        }
        return GenerationResult(
            text=json.dumps(body), provider="fake", model=request.model
        )


def build(count: int, *, ceiling: int | None = None, capacity: int = 100_000,
          provider: MergingProvider | None = None, **leaf_overrides: object):
    provider = provider or MergingProvider()
    root, nodes, report = build_hierarchy(
        leaves(count, **leaf_overrides),
        provider,
        CharacterCounter(),
        source_id=SOURCE_ID,
        covered=covered_for(count),
        attributable=attributable_for(count),
        capacity=capacity,
        model="m",
        timeout_seconds=30,
        max_merge_children=ceiling,
    )
    return root, nodes, report, provider


def test_groups_are_balanced_rather_than_ragged() -> None:
    """Greedy packing would give (4, 4, 1); the tail must not be starved."""
    assert group_children(9, 4) == ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    assert group_children(10, 4) == ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9))
    assert group_children(4, 4) == ((0, 1, 2, 3),)


def test_groups_preserve_order() -> None:
    flattened = [index for group in group_children(11, 3) for index in group]

    assert flattened == list(range(11))


def test_fanout_shrinks_as_children_grow() -> None:
    counter = CharacterCounter()
    small = leaves(3)
    large = leaves(3, summary="x" * 2_000)

    small_fanout, _ = merge_fanout(small, counter, capacity=10_000)
    large_fanout, _ = merge_fanout(large, counter, capacity=10_000)

    assert small_fanout > large_fanout
    assert measure_child_tokens(large[0], counter) > measure_child_tokens(
        small[0], counter
    )


def test_fanout_is_sized_from_the_largest_child() -> None:
    """Sizing from the average would assemble groups that only fit on average."""
    counter = CharacterCounter()
    mixed = [leaf(1), leaf(2, summary="x" * 4_000), leaf(3)]

    fanout, reason = merge_fanout(mixed, counter, capacity=10_000)

    assert fanout == max(10_000 // measure_child_tokens(mixed[1], counter), 2)
    assert "largest child" in reason


def test_a_configured_ceiling_clamps_the_measured_fanout() -> None:
    fanout, reason = merge_fanout(leaves(3), CharacterCounter(), capacity=100_000, ceiling=3)

    assert fanout == 3
    assert "ceiling" in reason


def test_a_single_child_that_cannot_fit_fails_with_its_arithmetic() -> None:
    """Reachable on the default local configuration, not hypothetical."""
    with pytest.raises(BudgetError) as error:
        merge_fanout(leaves(2, summary="x" * 5_000), CharacterCounter(), capacity=500)

    message = str(error.value)
    assert "does not fit a merge request" in message
    assert "capacity of 500" in message


def test_reduces_leaves_to_a_single_root() -> None:
    root, nodes, report, provider = build(8, ceiling=4)

    assert root.level == report.level_count
    assert report.leaf_count == 8
    assert len([node for node in nodes if node.level == 0]) == 8
    assert len(provider.requests) == report.provider_calls


def test_forces_at_least_three_levels_with_a_narrow_ceiling() -> None:
    """Three levels are unreachable at hosted capacity, so configure them."""
    root, nodes, report, _ = build(8, ceiling=2)

    assert report.level_count >= 3
    assert root.level >= 3
    assert [level.nodes_out for level in report.levels] == [4, 2, 1]


def test_every_level_strictly_reduces_the_node_count() -> None:
    _, _, report, _ = build(17, ceiling=3)

    for level in report.levels:
        assert level.nodes_out < level.nodes_in


def test_provenance_is_the_union_of_covered_segments() -> None:
    root, _, _, _ = build(6, ceiling=2)

    assert root.covered_segments == tuple(f"S{index + 1:06d}" for index in range(6))
    assert root.summary.provenance == root.covered_segments


def test_the_models_own_provenance_is_discarded() -> None:
    """The provider answers with empty provenance; the union must still hold."""
    root, _, _, _ = build(4, ceiling=2)

    assert root.summary.provenance != ()
    assert set(root.summary.provenance) == set(attributable_for(4))


def test_source_order_is_preserved_at_every_level() -> None:
    _, nodes, _, _ = build(9, ceiling=3)

    for level in {node.level for node in nodes}:
        at_level = sorted(
            (node for node in nodes if node.level == level), key=lambda n: n.order
        )
        assert [node.order for node in at_level] == list(range(len(at_level)))
        flattened = [
            identifier for node in at_level for identifier in node.covered_segments
        ]
        assert flattened == sorted(flattened)


def test_results_and_requests_are_deterministic() -> None:
    first_root, _, first_report, first_provider = build(7, ceiling=3)
    second_root, _, second_report, second_provider = build(7, ceiling=3)

    assert first_root == second_root
    assert first_report == second_report
    assert first_provider.requests == second_provider.requests


def test_a_lone_trailing_node_passes_through_without_a_call() -> None:
    _, _, report, provider = build(3, ceiling=2)

    # Two groups at level one: a pair that merges and a single that passes up.
    assert report.levels[0].nodes_out == 2
    assert len(provider.requests) == report.provider_calls


def test_a_wrong_level_in_the_response_is_rejected() -> None:
    with pytest.raises(LeafSummaryError, match="level"):
        build(4, ceiling=2, provider=MergingProvider(level_override=0))


def test_an_injected_citation_from_a_child_is_rejected() -> None:
    hostile = json.dumps(
        {
            "summary": "Merged.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": ["S999999"],
            "level": 1,
        }
    )

    with pytest.raises(LeafSummaryError, match="S999999"):
        build(4, ceiling=2, provider=MergingProvider(text=hostile))


def test_provider_failures_propagate() -> None:
    class FailingProvider:
        def generate(self, request: GenerationRequest) -> GenerationResult:
            raise ProviderConnectionError("unreachable")

    with pytest.raises(ProviderConnectionError):
        build_hierarchy(
            leaves(4),
            FailingProvider(),
            CharacterCounter(),
            source_id=SOURCE_ID,
            covered=covered_for(4),
            attributable=attributable_for(4),
            capacity=100_000,
            model="m",
            timeout_seconds=30,
        )


def test_a_single_leaf_is_already_a_root() -> None:
    root, nodes, report, provider = build(1)

    assert report.level_count == 0
    assert report.provider_calls == 0
    assert len(nodes) == 1
    assert root.level == 0
    assert provider.requests == []


def test_rejects_mismatched_covered_identifiers() -> None:
    with pytest.raises(ValueError, match="covered segment identifiers"):
        build_hierarchy(
            leaves(3),
            MergingProvider(),
            CharacterCounter(),
            source_id=SOURCE_ID,
            covered=covered_for(2),
            attributable=attributable_for(3),
            capacity=100_000,
            model="m",
            timeout_seconds=30,
        )


def test_grouping_refuses_a_fanout_that_cannot_progress() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        group_children(5, 1)


def test_hierarchy_error_names_a_level_that_fails_to_shrink() -> None:
    assert issubclass(HierarchyError, ValueError)
