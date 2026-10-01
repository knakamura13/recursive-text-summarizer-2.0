from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import Lock

from summarizer.budget import (
    BudgetError,
    RequestBudget,
    RequestLimits,
    measure_request_tokens,
)
from summarizer.cache import CacheDescriptor
from summarizer.grounding import (
    GroundingPolicy,
    GroundingSelection,
    select_source_passages,
)
from summarizer.leaf import derive_provenance, validate_provenance
from summarizer.merge import (
    MERGE_PROMPT_VERSION,
    build_merge_request,
    child_fence_tokens,
    measure_merge_overhead,
    parse_merged_summary,
    serialize_child,
    serialize_source_passage_block,
)
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelProvider,
)
from summarizer.runtime.items import ObservedItem, generate_observed, tree_node_id
from summarizer.runtime.observers import RuntimeObserver, StageName, get_observer
from summarizer.scheduler import BoundedScheduler, ScheduledResult, ScheduledWork
from summarizer.sections import SectionTree
from summarizer.segmentation import CacheCoordinator
from summarizer.summaries import LEAF_SCHEMA_VERSION, SummaryNode
from summarizer.tokenization import TokenCounter


class HierarchyError(ValueError):
    """The summary tree could not be reduced to a single root."""


@dataclass(frozen=True)
class MergeGrounding:
    """Selection metadata retained for an executed merge."""

    selection: GroundingSelection
    reserve_tokens: int | None
    request_capacity_tokens: int | None

    def __post_init__(self) -> None:
        if (self.reserve_tokens is None) == (
            self.request_capacity_tokens is None
        ):
            raise ValueError("merge grounding needs one budget mode")
        if self.reserve_tokens is not None and self.reserve_tokens <= 0:
            raise ValueError("grounding reserve must be positive")
        if (
            self.request_capacity_tokens is not None
            and self.request_capacity_tokens <= 0
        ):
            raise ValueError("grounding request capacity must be positive")


@dataclass(frozen=True)
class _PreparedMerge:
    """Frozen, independently executable work for one non-singleton merge."""

    node_id: str
    level: int
    order: int
    children: tuple[str, ...]
    covered_segments: tuple[str, ...]
    request: GenerationRequest
    legal: Mapping[str, str]
    quotation_sources: Mapping[str, str]
    preserved_provenance: tuple[str, ...]
    grounding: MergeGrounding
    descriptor: CacheDescriptor | None
    section_id: str | None = None

    def parse(self, text: str) -> SummaryNode:
        return parse_merged_summary(
            text,
            legal=self.legal,
            quotation_sources=self.quotation_sources,
            preserved_provenance=self.preserved_provenance,
            source_order=self.covered_segments,
            subject=self.node_id,
            level=self.level,
        )

    def decode(self, payload: object) -> SummaryNode:
        return self.parse(json.dumps(payload))

    def validate(self, payload: object) -> object:
        return self.decode(payload).model_dump(mode="json")

    def node(self, payload: object) -> TreeNode:
        return TreeNode(
            node_id=self.node_id,
            level=self.level,
            order=self.order,
            summary=self.decode(payload),
            children=self.children,
            covered_segments=self.covered_segments,
            grounding=self.grounding,
            section_id=self.section_id,
        )


DEFAULT_GROUNDING_POLICY = GroundingPolicy(max_tokens=1_024)


@dataclass(frozen=True)
class TreeNode:
    """One node of the summary tree, with the structure `SummaryNode` lacks.

    `SummaryNode` carries no identifier, no children, and no order, so the
    tree lives here. Order is explicit rather than implied by list position,
    which keeps ordering deterministic if merges later run concurrently.

    `section_id` names the section whose reduction built the node in a
    section-mode run, and is None otherwise.
    """

    node_id: str
    level: int
    order: int
    summary: SummaryNode
    children: tuple[str, ...]
    covered_segments: tuple[str, ...]
    grounding: MergeGrounding | None = None
    section_id: str | None = None

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


def measure_child_tokens(
    node: SummaryNode, counter: TokenCounter, heading: str | None = None
) -> int:
    """Measure what one child costs inside a merge request, delimiters included."""
    return counter.count(serialize_child(node, heading)) + child_fence_tokens(counter)


def plan_merge_request(
    limits: RequestLimits,
    *,
    level: int,
    provider_schema_reserve: int = 0,
    evidence: int = 0,
    section_headings: bool = False,
) -> RequestBudget:
    """Budget one merge request at `level`, or raise `RequestBudgetError`.

    Shared by the hierarchy and by preflight, so both refuse the same
    configurations with the same arithmetic.
    """
    return limits.plan(
        "merge",
        overhead=measure_merge_overhead(
            limits.counter,
            level=level,
            provider_schema_reserve=provider_schema_reserve,
            section_headings=section_headings,
        ),
        evidence=evidence,
    )


def merge_fanout(
    children: Sequence[SummaryNode],
    counter: TokenCounter,
    *,
    budget: RequestBudget,
    ceiling: int | None = None,
    headings: Sequence[str | None] | None = None,
) -> tuple[int, str]:
    """Derive how many children fit one merge request, and say why.

    Sized from the largest child rather than the average, so a group is never
    assembled that only fits on average.

    Raises `RequestBudgetError` when the budget cannot hold a *pair*, rather
    than returning two anyway: a merge of one is not a merge, and reporting
    two would assemble a request of twice the size it was sized against. On a
    provider that truncates an oversized prompt silently that would be
    undetectable content loss rather than an error.
    """
    if not children:
        raise ValueError("fanout requires at least one child")

    if ceiling is not None and ceiling < 2:
        raise ValueError("max_merge_children must be at least 2 to make progress")

    costs = [
        measure_child_tokens(
            child, counter, headings[index] if headings is not None else None
        )
        for index, child in enumerate(children)
    ]
    largest = max(costs)
    budget.require_merge_pair(largest)
    capacity = budget.input_capacity
    measured = capacity // largest if largest else len(children)

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


def _prepare_leaf_nodes(
    leaves: Sequence[SummaryNode],
    covered: Sequence[Sequence[str]],
    attributable: Mapping[str, str],
    section_ids: Sequence[str | None] | None = None,
) -> list[TreeNode]:
    """Validate each leaf against its own segments and wrap it as a level-0 node."""
    if len(covered) != len(leaves):
        raise ValueError("each leaf needs its covered segment identifiers")
    nodes: list[TreeNode] = []
    for index, (leaf, identifiers) in enumerate(zip(leaves, covered)):
        local_covered = tuple(identifiers)
        missing = [
            identifier for identifier in local_covered if identifier not in attributable
        ]
        if missing:
            raise ValueError(
                "attributable text is missing for segments " + ", ".join(missing)
            )
        legal = {identifier: attributable[identifier] for identifier in local_covered}
        subject = tree_node_id(0, index)
        validate_provenance(leaf, legal=legal, subject=subject)
        prepared = leaf.model_copy(
            update={"provenance": derive_provenance(leaf, source_order=local_covered)}
        )
        nodes.append(
            TreeNode(
                node_id=subject,
                level=0,
                order=index,
                summary=prepared,
                children=(),
                covered_segments=local_covered,
                section_id=section_ids[index] if section_ids is not None else None,
            )
        )
    return nodes


def _aligned(node: TreeNode, level: int) -> TreeNode:
    """Show a node to a merge at `level`, as the model sees its children.

    A flat reduction has every child at the previous level already. Section
    reduction mixes heights: a leaf may sit beside a node that took several
    levels to build. The prompt reports every child at the level just below the
    merge, so the model never sees children at mixed levels; the recorded node
    keeps its own.
    """
    if node.summary.level == level:
        return node
    return replace(node, summary=node.summary.model_copy(update={"level": level}))


class _Reducer:
    """Reduce ordered nodes to one through as many levels as needed.

    One reducer serves a whole run. Shared state - the merge-call count, the
    tree's nodes, the level reports, the manifest's planned merge ids and the
    next node order at each level - lives here so that several reductions (one
    per section) number their nodes and plan their work as one run.
    """

    def __init__(
        self,
        provider: ModelProvider,
        counter: TokenCounter,
        *,
        source_id: str,
        attributable: Mapping[str, str],
        limits: RequestLimits,
        model: str,
        timeout_seconds: float,
        max_merge_children: int | None,
        grounding_policy: GroundingPolicy | None,
        coordinator: CacheCoordinator | None,
        provider_schema_reserve: int,
        runtime: RuntimeObserver,
        section_headings: bool = False,
    ) -> None:
        self.calls = _CountingProvider(provider)
        self.counter = counter
        self.source_id = source_id
        self.attributable = attributable
        self.limits = limits
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_merge_children = max_merge_children
        self.configured_policy = (
            grounding_policy
            if grounding_policy is not None
            else DEFAULT_GROUNDING_POLICY
        )
        self.adaptive_grounding = grounding_policy is None
        self.coordinator = coordinator
        self.provider_schema_reserve = provider_schema_reserve
        self.runtime = runtime
        self.section_headings = section_headings
        # Heading data for a node that finishes a section, by node id.
        self.headings: dict[str, str] = {}
        self.all_nodes: list[TreeNode] = []
        self.levels: list[LevelReport] = []
        self._next_order: dict[int, int] = {}
        self._planned_merge_ids: list[str] = []
        if coordinator is not None and coordinator.session is not None:
            plan = coordinator.session.manifest.work_ids
            first_merge = next(
                (
                    index
                    for index, work_id in enumerate(plan)
                    if work_id.startswith("L")
                ),
                len(plan),
            )
            self._merge_prefix = plan[:first_merge]
        else:
            self._merge_prefix = ()

    def reduce(
        self, current: list[TreeNode], *, section_id: str | None = None
    ) -> TreeNode:
        """Merge `current`, in order, until one node remains, and return it.

        A merge group never holds a node from outside `current`. Nodes are
        numbered by level across the run: `order` continues from the last node
        built at that level, so reductions of different sections never collide
        and a flat run keeps `L{level}N0001` onwards.
        """
        runtime = self.runtime
        counter = self.counter
        coordinator = self.coordinator
        adaptive_grounding = self.adaptive_grounding
        configured_policy = self.configured_policy
        while len(current) > 1:
            runtime.raise_if_stopped("during merging")
            level = max(node.level for node in current) + 1
            budget = plan_merge_request(
                self.limits,
                level=level,
                provider_schema_reserve=self.provider_schema_reserve,
                evidence=0 if adaptive_grounding else configured_policy.max_tokens,
                section_headings=self.section_headings,
            )
            pool = [_aligned(node, level - 1) for node in current]
            headings = [self.headings.get(node.node_id) for node in current]
            base = self._next_order.get(level, 0)
            # An adaptive default has no fixed source reserve, so a fanout sized
            # purely from children can leave no room for a group's mandatory
            # source evidence. Retry with a narrower fanout when that happens;
            # a fanout that already cannot drop below a pair propagates the
            # failure, matching the fixed-policy path's fail-closed behavior.
            ceiling = self.max_merge_children
            while True:
                fanout, reason = merge_fanout(
                    [node.summary for node in pool],
                    counter,
                    budget=budget,
                    ceiling=ceiling,
                    headings=headings,
                )
                groups = group_children(len(current), fanout)
                prepared: list[_PreparedMerge] = []
                passthrough: dict[int, TreeNode] = {}
                try:
                    for position, indices in enumerate(groups):
                        members = [pool[index] for index in indices]
                        if len(members) == 1:
                            # Pass a lone node upward without a call; the count
                            # still falls because other groups merged.
                            only = members[0]
                            passthrough[position] = TreeNode(
                                node_id=tree_node_id(level, base + position),
                                level=level,
                                order=base + position,
                                # Restamped: the merged path asserts that a
                                # node's summary reports its own level, and this
                                # path is fed into the next level's payload, so
                                # a stale value would show the model children
                                # at mixed levels.
                                summary=only.summary.model_copy(
                                    update={"level": level}
                                ),
                                children=(only.node_id,),
                                covered_segments=only.covered_segments,
                                section_id=section_id,
                            )
                            continue
                        prepared.append(
                            _prepare_merge(
                                members,
                                attributable=self.attributable,
                                level=level,
                                order=base + position,
                                source_id=self.source_id,
                                model=self.model,
                                timeout_seconds=self.timeout_seconds,
                                counter=counter,
                                budget=budget,
                                grounding_policy=(
                                    GroundingPolicy(max_tokens=budget.request_capacity)
                                    if adaptive_grounding
                                    else configured_policy
                                ),
                                configured_grounding_policy=configured_policy,
                                adaptive_grounding=adaptive_grounding,
                                coordinator=coordinator,
                                provider_schema_reserve=self.provider_schema_reserve,
                                # A narrower fanout grounds on whole passages;
                                # excerpts only when no narrower fanout exists.
                                allow_excerpts=adaptive_grounding and fanout <= 2,
                                child_headings=[headings[index] for index in indices],
                                section_headings=self.section_headings,
                                section_id=section_id,
                            )
                        )
                except BudgetError:
                    if not adaptive_grounding or fanout <= 2:
                        raise
                    ceiling = fanout - 1
                    continue
                break

            by_position = {item.order - base: item for item in prepared}
            level_items = {
                position: ObservedItem(
                    kind="passthrough" if position in passthrough else "merge",
                    work_id=tree_node_id(level, base + position),
                    stage=StageName.MERGING,
                    level=level,
                    order=base + position,
                    total=len(groups),
                    child_ids=tuple(current[index].node_id for index in indices),
                    covered_segment_ids=(
                        passthrough[position].covered_segments
                        if position in passthrough
                        else by_position[position].covered_segments
                    ),
                )
                for position, indices in enumerate(groups)
            }
            for position in range(len(groups)):
                runtime.emit_item(level_items[position].event("planned"))
            for position, node in sorted(passthrough.items()):
                runtime.emit_item(
                    level_items[position].event(
                        "completed", summary=node.summary.model_dump(mode="json")
                    )
                )

            # Every request, descriptor, and legal grounding scope is frozen before
            # a sibling can call the provider. The manifest therefore witnesses the
            # entire level before its first externally visible side effect.
            if coordinator is not None and coordinator.session is not None:
                self._planned_merge_ids.extend(item.node_id for item in prepared)
                coordinator.session.ensure_work_prefix(
                    (*self._merge_prefix, *self._planned_merge_ids)
                )
                descriptors = {item.node_id: item.descriptor for item in prepared}
                assert all(
                    descriptor is not None for descriptor in descriptors.values()
                )
                values = coordinator.reusable_batch(
                    work_ids=tuple(item.node_id for item in prepared),
                    descriptors=descriptors,
                    validators={item.node_id: item.validate for item in prepared},
                )
                for item in prepared:
                    if item.node_id in values:
                        runtime.emit_item(
                            level_items[item.order - base].event(
                                "reused", summary=values[item.node_id]
                            )
                        )
                scheduled = BoundedScheduler(
                    max_in_flight=coordinator.max_in_flight,
                    cache=coordinator.store,
                    should_stop=runtime.should_stop,
                    on_complete=_completion_reporter(
                        runtime,
                        {
                            item.node_id: level_items[item.order - base]
                            for item in prepared
                        },
                    ),
                ).run(
                    tuple(
                        ScheduledWork(
                            descriptor=item.descriptor,
                            operation=lambda item=item, observed=level_items[
                                item.order - base
                            ]: _execute_prepared_merge(
                                item, self.calls, observed, runtime
                            ),
                            validate=item.validate,
                        )
                        for item in prepared
                        if item.node_id not in values
                    ),
                    coordinator.session,
                )
                values.update(
                    {result.work_id: result.payload for result in scheduled}
                )
            else:
                values = {}
                for item in prepared:
                    runtime.raise_if_stopped("during merging")
                    observed = level_items[item.order - base]
                    values[item.node_id] = _execute_prepared_merge(
                        item, self.calls, observed, runtime
                    )
                    runtime.emit_item(
                        observed.event("completed", summary=values[item.node_id])
                    )

            produced = []
            for position in range(len(groups)):
                if position in passthrough:
                    produced.append(passthrough[position])
                else:
                    item = by_position[position]
                    produced.append(item.node(values[item.node_id]))

            if len(produced) >= len(current):
                raise HierarchyError(
                    f"level {level} did not reduce the tree: {len(current)} nodes "
                    f"produced {len(produced)} with a fanout of {fanout}"
                )

            self.levels.append(
                LevelReport(
                    level=level,
                    nodes_in=len(current),
                    nodes_out=len(produced),
                    fanout=fanout,
                    reason=reason,
                )
            )
            self._next_order[level] = base + len(groups)
            self.all_nodes.extend(produced)
            current = produced
        return current[0]


def build_hierarchy(
    leaves: Sequence[SummaryNode],
    provider: ModelProvider,
    counter: TokenCounter,
    *,
    source_id: str,
    covered: Sequence[Sequence[str]],
    attributable: Mapping[str, str],
    limits: RequestLimits,
    model: str,
    timeout_seconds: float,
    max_merge_children: int | None = None,
    grounding_policy: GroundingPolicy | None = None,
    coordinator: CacheCoordinator | None = None,
    provider_schema_reserve: int = 0,
    observer: RuntimeObserver | None = None,
) -> tuple[TreeNode, tuple[TreeNode, ...], HierarchyReport]:
    """Reduce ordered leaves to a single root through as many levels as needed.

    `covered` gives the segment identifiers each leaf covers, in the same
    order as `leaves`, and `attributable` maps each identifier to the text a
    quotation from it may be drawn from - injected rather than held in module
    state, so a run carries its own material and nothing leaks between runs.

    `limits` supplies the run's context window, output allowance, margin and
    correction headroom. Each level plans its own merge budget from them, with
    the merge instructions, schema, delimiters and any fixed grounding reserve
    measured here, because the strategy decision measured a *leaf* request.

    Termination rests on a fanout of at least two, which `merge_fanout`
    guarantees by refusing a capacity that cannot hold a pair. The node count
    is additionally asserted to fall at each level, as a defensive invariant
    rather than the proof.

    Each level reports to `observer` once its groups are frozen: `planned` for
    every node of the level, with its children and covered segments, then
    `completed` for each passthrough at once and `reused` or `active`,
    `retrying`, and `completed` for each merge. A merge response that fails
    validation is re-asked with the validator's reason, up to twice, before
    `ItemFailedError` names the node. A stop request is honoured between
    merges.
    """
    if not leaves:
        raise ValueError("a hierarchy requires at least one leaf")
    if len(covered) != len(leaves):
        raise ValueError("each leaf needs its covered segment identifiers")
    if provider_schema_reserve < 0:
        raise ValueError("provider schema reserve must not be negative")
    reducer = _Reducer(
        provider,
        counter,
        source_id=source_id,
        attributable=attributable,
        limits=limits,
        model=model,
        timeout_seconds=timeout_seconds,
        max_merge_children=max_merge_children,
        grounding_policy=grounding_policy,
        coordinator=coordinator,
        provider_schema_reserve=provider_schema_reserve,
        runtime=get_observer(observer),
    )
    current = _prepare_leaf_nodes(leaves, covered, attributable)
    reducer.all_nodes.extend(current)
    root = reducer.reduce(current)
    report = HierarchyReport(
        leaf_count=len(leaves),
        level_count=root.level,
        provider_calls=reducer.calls.count,
        levels=tuple(reducer.levels),
    )
    return root, tuple(reducer.all_nodes), report


@dataclass(frozen=True)
class SectionHierarchy:
    """A summary tree reduced section by section.

    `section_nodes` maps each section id to the id of the node that summarizes
    it. A section with nothing to summarize has no entry, and a section whose
    only input was a single node is summarized by that node itself. Each
    `TreeNode.section_id` names the section whose reduction built the node;
    the nodes that merge top-level sections belong to none.
    """

    root: TreeNode
    nodes: tuple[TreeNode, ...]
    report: HierarchyReport
    section_nodes: Mapping[str, str]


def build_section_hierarchy(
    leaves: Sequence[SummaryNode],
    provider: ModelProvider,
    counter: TokenCounter,
    *,
    tree: SectionTree,
    leaf_sections: Sequence[str],
    source_id: str,
    covered: Sequence[Sequence[str]],
    attributable: Mapping[str, str],
    limits: RequestLimits,
    model: str,
    timeout_seconds: float,
    max_merge_children: int | None = None,
    grounding_policy: GroundingPolicy | None = None,
    coordinator: CacheCoordinator | None = None,
    provider_schema_reserve: int = 0,
    observer: RuntimeObserver | None = None,
) -> SectionHierarchy:
    """Reduce leaves bottom-up through the section tree; no merge crosses a section.

    A section's node reduces its own leaves, in order, followed by each child
    section's node in document order - a section's own text precedes its
    children's, so that is document order. The reduction uses the same fanout,
    budgets and grounding as `build_hierarchy`, so a section with many inputs
    takes several levels inside the section. A section with a single input
    adds no node. The document root reduces the top-level sections' nodes.

    `leaf_sections` gives each leaf's section id, in the order of `leaves`.
    """
    if not leaves:
        raise ValueError("a hierarchy requires at least one leaf")
    if len(leaf_sections) != len(leaves):
        raise ValueError("each leaf needs its section identifier")
    if provider_schema_reserve < 0:
        raise ValueError("provider schema reserve must not be negative")
    known = {node.id: node for node in tree.nodes}
    unknown = sorted(set(leaf_sections) - set(known))
    if unknown:
        raise ValueError("leaves name unknown sections: " + ", ".join(unknown))
    reducer = _Reducer(
        provider,
        counter,
        source_id=source_id,
        attributable=attributable,
        limits=limits,
        model=model,
        timeout_seconds=timeout_seconds,
        max_merge_children=max_merge_children,
        grounding_policy=grounding_policy,
        coordinator=coordinator,
        provider_schema_reserve=provider_schema_reserve,
        runtime=get_observer(observer),
        section_headings=True,
    )
    leaf_nodes = _prepare_leaf_nodes(leaves, covered, attributable, leaf_sections)
    reducer.all_nodes.extend(leaf_nodes)
    own_leaves: dict[str, list[TreeNode]] = {}
    for node in leaf_nodes:
        assert node.section_id is not None
        own_leaves.setdefault(node.section_id, []).append(node)

    section_nodes: dict[str, str] = {}

    def summarize(section_id: str) -> TreeNode | None:
        section = known[section_id]
        inputs = list(own_leaves.get(section_id, ()))
        for child_id in section.child_ids:
            child = summarize(child_id)
            if child is not None:
                inputs.append(child)
        if not inputs:
            return None
        node = reducer.reduce(inputs, section_id=section_id)
        section_nodes[section_id] = node.node_id
        # The finished section's own heading names its node from here on, even
        # when the node is the child's, which the section adopted unchanged.
        if section.heading is not None:
            reducer.headings[node.node_id] = section.heading
        else:
            reducer.headings.pop(node.node_id, None)
        return node

    tops = [node for root in tree.roots if (node := summarize(root.id)) is not None]
    root = reducer.reduce(tops)
    report = HierarchyReport(
        leaf_count=len(leaves),
        level_count=root.level,
        provider_calls=reducer.calls.count,
        levels=tuple(reducer.levels),
    )
    return SectionHierarchy(
        root=root,
        nodes=tuple(reducer.all_nodes),
        report=report,
        section_nodes=section_nodes,
    )


class _CountingProvider:
    """Count every merge call, re-asks included, across scheduler threads."""

    def __init__(self, delegate: ModelProvider) -> None:
        self._delegate = delegate
        self._lock = Lock()
        self.count = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        with self._lock:
            self.count += 1
        return self._delegate.generate(request)


def _completion_reporter(
    observer: RuntimeObserver, items: Mapping[str, ObservedItem]
) -> Callable[[ScheduledResult], None]:
    def report(result: ScheduledResult) -> None:
        observer.emit_item(
            items[result.work_id].event("completed", summary=result.payload)
        )

    return report


def _prepare_merge(
    members: Sequence[TreeNode],
    *,
    attributable: Mapping[str, str],
    level: int,
    order: int,
    source_id: str,
    model: str,
    timeout_seconds: float,
    counter: TokenCounter,
    budget: RequestBudget,
    grounding_policy: GroundingPolicy,
    configured_grounding_policy: GroundingPolicy,
    coordinator: CacheCoordinator | None,
    adaptive_grounding: bool,
    provider_schema_reserve: int,
    allow_excerpts: bool = False,
    child_headings: Sequence[str | None] | None = None,
    section_headings: bool = False,
    section_id: str | None = None,
) -> _PreparedMerge:
    node_id = tree_node_id(level, order)
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

    def selection_cost(passages: tuple[SourcePassage, ...]) -> int:
        if adaptive_grounding:
            candidate = replace(
                build_merge_request(
                    [member.summary for member in members],
                    passages=passages,
                    level=level,
                    source_id=source_id,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    max_output_tokens=budget.output_allowance_tokens,
                    child_headings=child_headings,
                    section_headings=section_headings,
                ),
                audit_work_id=node_id,
            )
            return measure_request_tokens(
                candidate,
                counter,
                provider_schema_reserve=provider_schema_reserve,
            )
        return counter.count(
            "\n".join(
                serialize_source_passage_block(
                    passage,
                    source_id=source_id,
                    level=level,
                    ordinal=ordinal,
                )
                for ordinal, passage in enumerate(passages)
            )
        )

    selection = select_source_passages(
        [member.summary for member in members],
        source=legal,
        counter=counter,
        policy=grounding_policy,
        selection_cost=selection_cost,
        allow_excerpts=allow_excerpts,
    )
    grounded = {passage.segment_id: passage.text for passage in selection.passages}
    preserved_provenance = tuple(
        dict.fromkeys(
            identifier
            for member in members
            for identifier in member.summary.provenance
            if identifier not in grounded
        )
    )
    request = build_merge_request(
        [member.summary for member in members],
        passages=selection.passages,
        level=level,
        source_id=source_id,
        model=model,
        timeout_seconds=timeout_seconds,
        max_output_tokens=budget.output_allowance_tokens,
        child_headings=child_headings,
        section_headings=section_headings,
    )
    request = replace(request, audit_work_id=node_id)
    budget.require_request(
        measure_request_tokens(
            request,
            counter,
            provider_schema_reserve=provider_schema_reserve,
        )
    )
    descriptor = None
    if coordinator is not None and coordinator.session is not None:
        descriptor = coordinator.descriptor_for(
            stage="merge",
            work_id=node_id,
            prompt_version=MERGE_PROMPT_VERSION,
            schema_version=LEAF_SCHEMA_VERSION,
            input_value={
                "child_node_ids": [member.node_id for member in members],
                "covered_segment_ids": list(covered),
                "grounding_source_ids": [
                    passage.segment_id for passage in selection.passages
                ],
                "preserved_ungrounded_provenance": list(preserved_provenance),
                "instructions": request.instructions,
                "input_text": request.input_text,
                "schema": request.response_schema,
                "max_output_tokens": request.max_output_tokens,
                "grounding_max_tokens": configured_grounding_policy.max_tokens,
                "effective_grounding_max_tokens": grounding_policy.max_tokens,
                "usable_tokens": budget.request_capacity,
            },
            behavior={
                "grounding": {"max_tokens": configured_grounding_policy.max_tokens},
                "grounding_policy": "grounding/1",
            },
        )
    return _PreparedMerge(
        node_id=node_id,
        level=level,
        order=order,
        children=tuple(member.node_id for member in members),
        covered_segments=covered,
        request=request,
        legal=legal,
        quotation_sources=grounded,
        preserved_provenance=preserved_provenance,
        grounding=MergeGrounding(
            selection=selection,
            reserve_tokens=(
                None if adaptive_grounding else grounding_policy.max_tokens
            ),
            request_capacity_tokens=(
                budget.request_capacity if adaptive_grounding else None
            ),
        ),
        descriptor=descriptor,
        section_id=section_id,
    )


def _execute_prepared_merge(
    prepared: _PreparedMerge,
    provider: ModelProvider,
    item: ObservedItem,
    observer: RuntimeObserver,
) -> object:
    """Make the merge call, re-asking a response that fails validation."""
    return generate_observed(
        item,
        provider,
        prepared.request,
        lambda result: prepared.parse(result.text).model_dump(mode="json"),
        observer=observer,
    )
