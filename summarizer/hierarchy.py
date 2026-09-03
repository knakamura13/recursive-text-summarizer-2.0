from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence

from summarizer.budget import BudgetError
from summarizer.merge import build_merge_request, parse_merged_summary, serialize_child
from summarizer.providers.base import ModelProvider
from summarizer.summaries import SummaryNode
from summarizer.tokenization import TokenCounter

# Cost of the two markers wrapping each child in a merge request. Measured
# rather than guessed, and deliberately generous: a 16-hex-character digest
# tokenizes poorly, so a marker pair runs to roughly this much.
_CHILD_FENCE_TOKENS = 48


class HierarchyError(ValueError):
    """The summary tree could not be reduced to a single root."""


@dataclass(frozen=True)
class TreeNode:
    """One node of the summary tree, with the structure `SummaryNode` lacks.

    `SummaryNode` carries no identifier, no children, and no order, so the
    tree lives here. Order is explicit rather than implied by list position,
    which keeps ordering deterministic if merges later run concurrently.
    """

    node_id: str
    level: int
    order: int
    summary: SummaryNode
    children: tuple[str, ...]
    covered_segments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id must not be blank")
        if self.level < 0:
            raise ValueError("level must not be negative")
        if self.order < 0:
            raise ValueError("order must not be negative")
        if not self.covered_segments:
            raise ValueError("a node must cover at least one segment")


@dataclass(frozen=True)
class LevelReport:
    """What happened at one level of reduction."""

    level: int
    nodes_in: int
    nodes_out: int
    fanout: int
    reason: str


@dataclass(frozen=True)
class HierarchyReport:
    """The shape of the tree, and why it has that shape."""

    leaf_count: int
    level_count: int
    provider_calls: int
    levels: tuple[LevelReport, ...] = field(default_factory=tuple)


def measure_child_tokens(node: SummaryNode, counter: TokenCounter) -> int:
    """Measure what one child costs inside a merge request."""
    return counter.count(serialize_child(node)) + _CHILD_FENCE_TOKENS


def merge_fanout(
    children: Sequence[SummaryNode],
    counter: TokenCounter,
    *,
    capacity: int,
    ceiling: int | None = None,
) -> tuple[int, str]:
    """Derive how many children fit one merge request, and say why.

    Sized from the largest child rather than the average, so that a group is
    never assembled that only fits on average. Raises when a single child
    cannot fit at all, which is reachable on the default local configuration
    rather than hypothetical.
    """
    if not children:
        raise ValueError("fanout requires at least one child")

    costs = [measure_child_tokens(child, counter) for child in children]
    largest = max(costs)
    if largest > capacity:
        worst = costs.index(largest)
        raise BudgetError(
            f"a single summary does not fit a merge request: child {worst} "
            f"costs {largest} tokens against a capacity of {capacity}, "
            f"including {_CHILD_FENCE_TOKENS} tokens of delimiters"
        )

    measured = capacity // largest
    if ceiling is not None and ceiling < measured:
        return max(ceiling, 2), (
            f"the configured ceiling of {ceiling} is below the measured "
            f"fanout of {measured}"
        )
    return max(measured, 2), (
        f"a largest child of {largest} tokens fits {measured} times in a "
        f"capacity of {capacity}"
    )


def group_children(count: int, fanout: int) -> tuple[tuple[int, ...], ...]:
    """Split `count` children into balanced, order-preserving groups.

    Balanced rather than greedily packed: greedy filling leaves a ragged final
    group, which compresses the end of a document less than the beginning -
    the uneven compression a balanced hierarchy is meant to avoid.
    """
    if count <= 0:
        raise ValueError("cannot group zero children")
    if fanout < 2:
        raise ValueError("fanout must be at least 2 to make progress")

    groups = -(-count // fanout)
    base, remainder = divmod(count, groups)
    indices = []
    start = 0
    for position in range(groups):
        size = base + (1 if position < remainder else 0)
        indices.append(tuple(range(start, start + size)))
        start += size
    return tuple(indices)


def build_hierarchy(
    leaves: Sequence[SummaryNode],
    provider: ModelProvider,
    counter: TokenCounter,
    *,
    source_id: str,
    covered: Sequence[Sequence[str]],
    attributable: Mapping[str, str],
    capacity: int,
    model: str,
    timeout_seconds: float,
    max_merge_children: int | None = None,
) -> tuple[TreeNode, tuple[TreeNode, ...], HierarchyReport]:
    """Reduce ordered leaves to a single root through as many levels as needed.

    `covered` gives the segment identifiers each leaf covers, in the same
    order as `leaves`, and `attributable` maps each identifier to the text a
    quotation from it may be drawn from - injected rather than held in module
    state, so a run carries its own material and nothing leaks between runs. Every level must strictly reduce the node count, which
    is the only property that makes termination provable rather than hoped
    for.
    """
    if not leaves:
        raise ValueError("a hierarchy requires at least one leaf")
    if len(covered) != len(leaves):
        raise ValueError("each leaf needs its covered segment identifiers")

    all_nodes: list[TreeNode] = []
    current = [
        TreeNode(
            node_id=f"L0N{index + 1:04d}",
            level=0,
            order=index,
            summary=leaf,
            children=(),
            covered_segments=tuple(segments),
        )
        for index, (leaf, segments) in enumerate(zip(leaves, covered))
    ]
    all_nodes.extend(current)

    levels: list[LevelReport] = []
    provider_calls = 0
    level = 0

    while len(current) > 1:
        level += 1
        fanout, reason = merge_fanout(
            [node.summary for node in current],
            counter,
            capacity=capacity,
            ceiling=max_merge_children,
        )
        groups = group_children(len(current), fanout)

        produced: list[TreeNode] = []
        for order, indices in enumerate(groups):
            members = [current[index] for index in indices]
            if len(members) == 1:
                # Pass a lone node upward without a call; the count still
                # falls because other groups merged.
                only = members[0]
                produced.append(
                    TreeNode(
                        node_id=f"L{level}N{order + 1:04d}",
                        level=level,
                        order=order,
                        summary=only.summary,
                        children=(only.node_id,),
                        covered_segments=only.covered_segments,
                    )
                )
                continue

            node = _merge_group(
                members,
                provider,
                attributable=attributable,
                level=level,
                order=order,
                source_id=source_id,
                model=model,
                timeout_seconds=timeout_seconds,
            )
            provider_calls += 1
            produced.append(node)

        if len(produced) >= len(current):
            raise HierarchyError(
                f"level {level} did not reduce the tree: {len(current)} nodes "
                f"produced {len(produced)} with a fanout of {fanout}"
            )

        levels.append(
            LevelReport(
                level=level,
                nodes_in=len(current),
                nodes_out=len(produced),
                fanout=fanout,
                reason=reason,
            )
        )
        all_nodes.extend(produced)
        current = produced

    report = HierarchyReport(
        leaf_count=len(leaves),
        level_count=level,
        provider_calls=provider_calls,
        levels=tuple(levels),
    )
    return current[0], tuple(all_nodes), report


def _merge_group(
    members: Sequence[TreeNode],
    provider: ModelProvider,
    *,
    attributable: Mapping[str, str],
    level: int,
    order: int,
    source_id: str,
    model: str,
    timeout_seconds: float,
) -> TreeNode:
    node_id = f"L{level}N{order + 1:04d}"
    covered = tuple(
        identifier
        for member in members
        for identifier in member.covered_segments
    )
    legal = {
        identifier: attributable.get(identifier, "")
        for member in members
        for identifier in member.covered_segments
    }

    request = build_merge_request(
        [member.summary for member in members],
        level=level,
        source_id=source_id,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    result = provider.generate(request)
    summary = parse_merged_summary(
        result.text,
        legal=legal,
        subject=node_id,
        level=level,
        provenance=covered,
    )
    return TreeNode(
        node_id=node_id,
        level=level,
        order=order,
        summary=summary,
        children=tuple(member.node_id for member in members),
        covered_segments=covered,
    )
