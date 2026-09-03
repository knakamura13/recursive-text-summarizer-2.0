from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence

from summarizer.budget import BudgetError
from summarizer.merge import (
    build_merge_request,
    child_fence_tokens,
    measure_merge_overhead,
    parse_merged_summary,
    serialize_child,
)
from summarizer.providers.base import ModelProvider
from summarizer.summaries import SummaryNode
from summarizer.tokenization import TokenCounter



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
    """Measure what one child costs inside a merge request, delimiters included."""
    return counter.count(serialize_child(node)) + child_fence_tokens(counter)


def merge_fanout(
    children: Sequence[SummaryNode],
    counter: TokenCounter,
    *,
    capacity: int,
    ceiling: int | None = None,
) -> tuple[int, str]:
    """Derive how many children fit one merge request, and say why.

    Sized from the largest child rather than the average, so a group is never
    assembled that only fits on average.

    Raises `BudgetError` when the capacity cannot hold a *pair*, rather than
    returning two anyway: a merge of one is not a merge, and reporting two
    would assemble a request of twice the size it was sized against. That is
    reachable on the default local configuration, and on a provider that
    truncates an oversized prompt silently it would be undetectable content
    loss rather than an error.
    """
    if not children:
        raise ValueError("fanout requires at least one child")

    if ceiling is not None and ceiling < 2:
        raise ValueError("max_merge_children must be at least 2 to make progress")

    costs = [measure_child_tokens(child, counter) for child in children]
    largest = max(costs)
    measured = capacity // largest if largest else len(children)

    # A merge level cannot be narrower than a pair, so a capacity that admits
    # only one child admits no merge at all. Reporting a fanout of 2 here -
    # which an earlier revision did - assembles a request of twice the size it
    # was sized against, and on a provider that truncates silently that is
    # undetectable content loss rather than an error.
    if measured < 2:
        worst = costs.index(largest)
        raise BudgetError(
            f"a merge request cannot hold two summaries: child {worst} costs "
            f"{largest} tokens and a pair costs {2 * largest} against a "
            f"capacity of {capacity}"
        )

    if ceiling is not None and ceiling < measured:
        return ceiling, (
            f"the configured ceiling of {ceiling} is below the measured "
            f"fanout of {measured}"
        )
    return measured, (
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
    usable_tokens: int,
    model: str,
    timeout_seconds: float,
    max_merge_children: int | None = None,
) -> tuple[TreeNode, tuple[TreeNode, ...], HierarchyReport]:
    """Reduce ordered leaves to a single root through as many levels as needed.

    `covered` gives the segment identifiers each leaf covers, in the same
    order as `leaves`, and `attributable` maps each identifier to the text a
    quotation from it may be drawn from - injected rather than held in module
    state, so a run carries its own material and nothing leaks between runs.

    `usable_tokens` is the whole input budget for a request. The merge
    instructions, schema, and outer delimiters are subtracted here rather than
    by the caller, because the budget calculator measures a *leaf* request and
    a caller passing that figure straight through would under-reserve.

    Termination rests on a fanout of at least two, which `merge_fanout`
    guarantees by refusing a capacity that cannot hold a pair. The node count
    is additionally asserted to fall at each level, as a defensive invariant
    rather than the proof.
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
        overhead = measure_merge_overhead(counter, level=level)
        child_capacity = usable_tokens - overhead
        if child_capacity <= 0:
            raise BudgetError(
                f"no room for children in a merge request at level {level}: "
                f"{usable_tokens} usable tokens are consumed by {overhead} "
                f"tokens of instructions, schema, and delimiters"
            )
        fanout, reason = merge_fanout(
            [node.summary for node in current],
            counter,
            capacity=child_capacity,
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
                        # Restamped: the merged path asserts that a node's
                        # summary reports its own level, and this path is fed
                        # into the next level's payload, so a stale value
                        # would show the model children at mixed levels.
                        summary=only.summary.model_copy(update={"level": level}),
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
    # A union in document order: deduplicated, first occurrence wins. Three
    # documents call this a union, and a caller supplying overlapping coverage
    # would otherwise store duplicates for issue #8 to narrow.
    covered = tuple(
        dict.fromkeys(
            identifier
            for member in members
            for identifier in member.covered_segments
        )
    )
    missing = sorted(
        identifier for identifier in covered if identifier not in attributable
    )
    if missing:
        raise ValueError(
            "attributable text is missing for segments "
            f"{', '.join(missing)}"
        )
    legal = {identifier: attributable[identifier] for identifier in covered}

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
