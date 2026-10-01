"""Orchestrate canonical source ingestion through final editorial output."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from threading import Lock

from summarizer.budget import (
    BudgetError,
    BudgetReport,
    RequestLimits,
    measure_overhead,
    plan_request,
    resolve_context_window,
    select_strategy,
)
from summarizer.cache import CacheStore
from summarizer.checkpoint import CheckpointStore, RunPlan
from summarizer.config import AppConfig, CacheConfig, ReliabilityConfig, StrategyConfig
from summarizer.direct import DIRECT_NODE_ID, summarize_direct, whole_document_segment
from summarizer.finalization import (
    FinalizationResult,
    SectionPublication,
    _finalize_summary,
    attach_section_records,
    finalize_sections,
    publish_final_output,
)
from summarizer.editorial import SectionScope, plan_editorial_request
from summarizer.hierarchy import (
    TreeNode,
    build_hierarchy,
    build_section_hierarchy,
    plan_merge_request,
)
from summarizer.ingestion import SourceDocument
from summarizer.leaf import measure_heading_overhead, summarize_segments
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelProvider,
    ProviderResponseError,
    ProviderRetriesExhaustedError,
)
from summarizer.reliability import ReliabilityTracker
from summarizer.sections import (
    SectionOutline,
    SectionTree,
    build_section_tree,
    own_text_spans,
)
from summarizer.segmentation import (
    CacheCoordinator,
    SegmentationConfig,
    SourceSegment,
    cached_segment_document,
)
from summarizer.summaries import MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES
from summarizer.tokenization import TokenCounter
from summarizer.runtime.observers import (
    RuntimeObserver,
    SegmentInfo,
    StageEvent,
    StageName,
    get_observer,
)
from summarizer.verification import VerificationConfig, VerificationRuntime


@dataclass(frozen=True)
class PipelineConfig:
    target_words: int = 300
    segmentation: SegmentationConfig | None = None
    max_merge_children: int | None = None
    include_citations: bool = False
    audit_path: Path | None = None
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    verification_runtime: VerificationRuntime | None = None
    cache: CacheConfig = field(default_factory=CacheConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    # The outline for section mode; None leaves the mode off. With it, each
    # section is segmented and reduced on its own and no merge crosses a
    # section boundary. A section mode run never takes the direct path, even
    # for a document that fits one request: every section needs its own node.
    sections: SectionOutline | None = None
    # Closed warning codes the caller adds to the audit, for a decision made
    # before the run (the web app notes that section mode found no outline).
    audit_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.target_words <= 0:
            raise ValueError("target_words must be positive")
        if self.verification_runtime is not None and not self.verification.enabled:
            raise ValueError("verification runtime requires enabled verification")


@dataclass(frozen=True)
class PipelineResult:
    final: FinalizationResult
    strategy: BudgetReport
    root: TreeNode
    nodes: tuple[TreeNode, ...]
    # Section mode only. `sections` is the tree the run summarized by, and
    # `section_nodes` maps a section id to the id of the node that summarizes
    # it. A section with no text and no summarized child has no entry.
    # `TreeNode.section_id` gives the reverse mapping.
    sections: SectionTree | None = None
    section_nodes: Mapping[str, str] = field(default_factory=dict)
    # Section mode only: the prose written and verified for each section that
    # has a node, by section id, checked against that section's own source.
    # It is extra output: `final` stays the root editorial and its verification.
    section_publications: Mapping[str, SectionPublication] = field(default_factory=dict)


# Bump when section prose changes in a way a resumed run must not reuse.
SECTION_PROSE_VERSION = 2
_DEFAULT_PIPELINE_CONFIG = PipelineConfig()
_DEFAULT_SEGMENT_CAPACITY_DIVISOR = 4


@dataclass(frozen=True)
class _RecordedGeneration:
    work_id: str | None
    sequence: int
    result: GenerationResult


class _RecordingProvider:
    """Capture completed logical calls without exposing request prompts to audit."""

    def __init__(
        self,
        delegate: ModelProvider,
        coordinator: CacheCoordinator | None = None,
        reliability_tracker: ReliabilityTracker | None = None,
    ) -> None:
        self._delegate = delegate
        self.cache_coordinator = coordinator
        self._reliability_tracker = reliability_tracker
        self._generations: list[_RecordedGeneration] = []
        self._next_sequence: dict[str | None, int] = {}
        self._lock = Lock()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        work_id = request.audit_work_id or request.operation_id
        with self._lock:
            sequence = self._next_sequence.get(work_id, 0)
            self._next_sequence[work_id] = sequence + 1
        try:
            result = self._delegate.generate(request)
        except ProviderRetriesExhaustedError as error:
            if self._reliability_tracker is not None:
                self._reliability_tracker.record_retry_exhaustion(
                    request,
                    attempt_count=error.attempts,
                    retry_attempts=error.retry_attempts,
                )
            raise
        except ProviderResponseError:
            # An incomplete or malformed answer is re-asked like one that fails
            # validation, so the call still counts as one of the work item's
            # attempts. It carries no transient retry category.
            if self._reliability_tracker is not None:
                self._reliability_tracker.record_retry_exhaustion(
                    request, attempt_count=1, retry_attempts=()
                )
            raise
        with self._lock:
            self._generations.append(_RecordedGeneration(work_id, sequence, result))
        if self._reliability_tracker is not None:
            self._reliability_tracker.record_generation(request, result)
        return result

    @property
    def generations(self) -> tuple[GenerationResult, ...]:
        with self._lock:
            records = tuple(self._generations)
        if self._reliability_tracker is None:
            return tuple(record.result for record in records)
        positions = {
            work_id: index
            for index, work_id in enumerate(
                self._reliability_tracker.manifest_work_order()
            )
        }
        ordered = sorted(
            enumerate(records),
            key=lambda item: (
                positions.get(item[1].work_id, len(positions)),
                item[1].sequence,
                item[0],
            ),
        )
        return tuple(record.result for _, record in ordered)


def _hierarchical_capacity(
    report: BudgetReport,
    counter: TokenCounter,
    app: AppConfig,
    strategy: StrategyConfig,
    segmentation: SegmentationConfig,
) -> int:
    """Re-measure overlap-bearing leaves instead of trusting direct capacity."""
    if segmentation.overlap_tokens == 0:
        return report.usable_input_capacity
    window = resolve_context_window(
        provider=app.provider, model=app.model, explicit=strategy.context_window
    )
    return plan_request(
        "leaf",
        window=window,
        overhead=measure_overhead(
            counter,
            with_overlap=True,
            provider_schema_reserve=_provider_schema_reserve(app),
        ),
        output_allowance=report.reserved_output_tokens,
        correction_headroom=report.correction_headroom_tokens,
        config=strategy,
    ).input_capacity


def leaf_segmentation(
    report: BudgetReport,
    counter: TokenCounter,
    *,
    app: AppConfig,
    strategy: StrategyConfig,
    requested: SegmentationConfig | None,
) -> SegmentationConfig:
    """Return the segmentation a hierarchical run gives its leaves.

    Unless one is requested, each segment may take a quarter of the usable
    input capacity. Either way it must fit a leaf request once overlap context
    is measured, or `BudgetError` is raised before any segment is produced.
    """
    segmentation = requested or SegmentationConfig(
        max_tokens=max(
            1, report.usable_input_capacity // _DEFAULT_SEGMENT_CAPACITY_DIVISOR
        )
    )
    if segmentation.max_tokens > _hierarchical_capacity(
        report, counter, app, strategy, segmentation
    ):
        raise BudgetError(
            "segmentation max_tokens exceeds the safely measured leaf capacity"
        )
    return segmentation


def plan_run_requests(
    report: BudgetReport,
    counter: TokenCounter,
    *,
    strategy: StrategyConfig,
    source_id: str,
    target_words: int,
) -> RequestLimits:
    """Return the run's request limits, refusing an infeasible editorial.

    The editorial request is budgeted before any model call, so an
    unreachable target fails with its arithmetic instead of after every leaf
    has been generated. Preflight calls this too, so both refuse the same
    configurations.
    """
    limits = RequestLimits.from_report(report, counter, strategy)
    plan_editorial_request(limits, source_id=source_id, target_words=target_words)
    return limits


def plan_section_requests(
    limits: RequestLimits,
    *,
    source_id: str,
    tree: SectionTree,
    target_words: int,
) -> None:
    """Refuse an infeasible section editorial before any model call.

    Each section's editorial is budgeted at its own target, as the root's is.
    A heading-only section asks for no editorial and is skipped.
    """
    for section in tree.nodes:
        if tree.is_heading_only(section.id, target_words):
            continue
        plan_editorial_request(
            limits,
            source_id=source_id,
            target_words=tree.target_words(section.id, target_words),
            section=SectionScope(section.id, section.heading),
        )


def _provider_schema_reserve(app: AppConfig) -> int:
    return MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES if app.provider == "ollama" else 0


def _segment_infos(segments: Sequence[SourceSegment]) -> tuple[SegmentInfo, ...]:
    return tuple(
        SegmentInfo(
            segment_id=segment.segment_id,
            order=segment.order,
            start=segment.context_start,
            end=segment.context_end,
            core_start=segment.core_start,
            core_end=segment.core_end,
            token_count=segment.token_count,
        )
        for segment in segments
    )


def run_pipeline(
    document: SourceDocument,
    provider: ModelProvider,
    counter: TokenCounter,
    *,
    app: AppConfig,
    strategy: StrategyConfig,
    config: PipelineConfig = _DEFAULT_PIPELINE_CONFIG,
    observer: RuntimeObserver | None = None,
) -> PipelineResult:
    runtime_observer = get_observer(observer)
    runtime_observer.emit(StageEvent(StageName.PREPARING, "active"))
    report = select_strategy(
        document, counter, provider=app.provider, model=app.model, config=strategy
    )
    limits = plan_run_requests(
        report,
        counter,
        strategy=strategy,
        source_id=document.source_id,
        target_words=config.target_words,
    )
    runtime_observer.emit(
        StageEvent(StageName.PREPARING, "completed", detail=report.strategy)
    )
    runtime_observer.raise_if_stopped("before pipeline execution")
    section_tree = section_tree_for(document, config)
    if section_tree is not None:
        plan_section_requests(
            limits,
            source_id=document.source_id,
            tree=section_tree,
            target_words=config.target_words,
        )
    if not config.cache.enabled:
        return _run_pipeline(
            document,
            provider,
            counter,
            app=app,
            strategy=strategy,
            config=config,
            report=report,
            limits=limits,
            coordinator=None,
            observer=runtime_observer,
            section_tree=section_tree,
        )
    if not config.reliability.run_id:
        raise ValueError("enabled cache requires a reliability run_id")
    seed = (
        ("D000001",)
        if report.strategy == "direct" and section_tree is None
        else ("segmentation",)
    )
    effective_segmentation = config.segmentation or SegmentationConfig(
        max_tokens=report.usable_input_capacity
    )
    verification_runtime_descriptor: dict[str, object] | None = None
    if config.verification.enabled:
        runtime = config.verification_runtime
        if runtime is None:
            verification_runtime_descriptor = {
                "provider": app.provider,
                "model": app.model,
                "counter_identity": counter.identity,
                "counter_exact": counter.exact,
                "context_window_tokens": report.context_window_tokens,
                "timeout_seconds": app.timeout_seconds,
            }
        else:
            verification_runtime_descriptor = {
                "provider": runtime.provider_identity,
                "model": runtime.model,
                "counter_identity": runtime.counter.identity,
                "counter_exact": runtime.counter.exact,
                "context_window_tokens": runtime.context_window_tokens,
                "timeout_seconds": runtime.timeout_seconds,
            }
    descriptor_fields: dict[str, object] = {
        "app": {
            "provider": app.provider,
            "model": app.model,
            "timeout_seconds": app.timeout_seconds,
        },
        "counter": {"identity": counter.identity, "exact": counter.exact},
        "strategy": asdict(strategy),
        "budget": asdict(report),
        "segmentation": asdict(effective_segmentation),
        "pipeline": {
            "target_words": config.target_words,
            "max_merge_children": config.max_merge_children,
            "include_citations": config.include_citations,
            "verification": asdict(config.verification),
            "max_in_flight": config.reliability.max_in_flight,
        },
        "verification_runtime": verification_runtime_descriptor,
    }
    if section_tree is not None:
        # Absent when the mode is off, so a run without sections keeps its key.
        descriptor_fields["sections"] = [
            [node.id, node.heading, node.level, node.start, node.end, node.parent_id]
            for node in section_tree.nodes
        ]
        descriptor_fields["section_prose"] = SECTION_PROSE_VERSION
    descriptor = hashlib.sha256(
        json.dumps(
            descriptor_fields,
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    plan = RunPlan(
        run_id=config.reliability.run_id,
        descriptor_sha256=descriptor,
        source_sha256=document.source_id,
        work_ids=seed,
    )
    with CheckpointStore(config.cache.root).open(
        plan, resume=config.reliability.run_mode == "resume"
    ) as session:
        reliability_tracker = ReliabilityTracker(
            lambda: session.manifest.work_ids,
            resumed=config.reliability.run_mode == "resume",
        )
        coordinator = CacheCoordinator(
            store=CacheStore(config.cache.root),
            source_id=document.source_id,
            provider=app.provider,
            model=app.model,
            ollama_host=app.ollama_host,
            counter_identity=counter.identity,
            counter_exact=counter.exact,
            context_window_tokens=report.context_window_tokens,
            behavior={
                "strategy_config": {
                    "strategy": strategy.strategy,
                    "context_window": report.context_window_tokens,
                    "max_direct_tokens": strategy.max_direct_tokens,
                    "max_output_tokens": strategy.max_output_tokens,
                    "safety_margin_tokens": strategy.safety_margin_tokens,
                },
                "budget": {
                    "context_window": report.context_window_tokens,
                    "max_output_tokens": strategy.max_output_tokens,
                    "safety_margin_tokens": strategy.safety_margin_tokens,
                    "safety_margin_fraction": strategy.safety_margin_fraction,
                },
            },
            session=session,
            allow_unreferenced_cache=config.reliability.run_mode == "new",
            max_in_flight=config.reliability.max_in_flight,
            reliability_tracker=reliability_tracker,
        )
        return _run_pipeline(
            document,
            provider,
            counter,
            app=app,
            strategy=strategy,
            config=config,
            report=report,
            limits=limits,
            coordinator=coordinator,
            observer=runtime_observer,
            section_tree=section_tree,
        )


def section_tree_for(
    document: SourceDocument, config: PipelineConfig
) -> SectionTree | None:
    """Return the section tree section mode summarizes by, or None when it is off."""
    if config.sections is None:
        return None
    return build_section_tree(
        document.text,
        config.sections.headings,
        target_words=config.target_words,
        pages=config.sections.pages,
    )


def _run_pipeline(
    document: SourceDocument,
    provider: ModelProvider,
    counter: TokenCounter,
    *,
    app: AppConfig,
    strategy: StrategyConfig,
    config: PipelineConfig,
    report: BudgetReport,
    limits: RequestLimits,
    coordinator: CacheCoordinator | None,
    observer: RuntimeObserver,
    section_tree: SectionTree | None = None,
) -> PipelineResult:
    """Execute direct or hierarchical stages, then final editorial writing.

    With a `section_tree` every section is summarized through the hierarchical
    path, whatever the strategy report says, so the strategy recorded for the
    run is hierarchical. The final editorial and verification then run over the
    root as they do without sections.
    """
    reliability_tracker = (
        coordinator.reliability_tracker if coordinator is not None else None
    )
    recording = _RecordingProvider(provider, coordinator, reliability_tracker)
    section_nodes: Mapping[str, str] = {}
    own_text_nodes: Mapping[str, str] = {}
    own_text_reductions: tuple[TreeNode, ...] = ()
    strategy_name = (
        "hierarchical" if section_tree is not None else report.strategy
    )
    if strategy_name == "direct":
        observer.emit(StageEvent(StageName.SEGMENTING, "skipped"))
        segment = whole_document_segment(document, counter)
        observer.emit_segments(_segment_infos((segment,)))
        observer.emit(StageEvent(StageName.SUMMARIZING, "active", total=1))
        summary = summarize_direct(
            document,
            recording,
            counter,
            model=app.model,
            timeout_seconds=app.timeout_seconds,
            max_output_tokens=strategy.max_output_tokens,
            coordinator=coordinator,
            observer=observer,
        )
        root = TreeNode(
            node_id=DIRECT_NODE_ID,
            level=0,
            order=0,
            summary=summary,
            children=(),
            covered_segments=(segment.segment_id,),
        )
        observer.emit(StageEvent(StageName.SUMMARIZING, "completed", completed=1, total=1))
        observer.emit(StageEvent(StageName.MERGING, "skipped"))
        nodes = (root,)
        segments = (segment,)
    else:
        observer.emit(StageEvent(StageName.SEGMENTING, "active"))
        segmentation = leaf_segmentation(
            report,
            counter,
            app=app,
            strategy=strategy,
            requested=config.segmentation,
        )
        spans = None
        leaf_headings: dict[str, str] = {}
        if section_tree is not None:
            own = own_text_spans(document.text, section_tree)
            spans = tuple((start, end) for _, start, end in own)
            if any(
                section_tree.get(section_id).heading is not None
                for section_id, _, _ in own
            ) and segmentation.max_tokens + measure_heading_overhead(
                counter
            ) > _hierarchical_capacity(report, counter, app, strategy, segmentation):
                raise BudgetError(
                    "segmentation max_tokens leaves no room for a section heading "
                    "in a leaf request"
                )
        segments = tuple(
            cached_segment_document(
                document, counter, segmentation, coordinator=coordinator, spans=spans
            )
        )
        leaf_sections: list[str] = []
        if section_tree is not None:
            for segment in segments:
                section_id = next(
                    id_
                    for id_, start, end in own
                    if start <= segment.core_start < end
                )
                leaf_sections.append(section_id)
                heading = section_tree.get(section_id).heading
                if heading is not None:
                    leaf_headings[segment.segment_id] = heading
        if len(segments) > 1:
            # Refuse a merge budget with no input room before any leaf call.
            plan_merge_request(
                limits,
                level=1,
                provider_schema_reserve=_provider_schema_reserve(app),
                section_headings=section_tree is not None,
            )
        observer.emit_segments(_segment_infos(segments))
        observer.emit(
            StageEvent(
                StageName.SEGMENTING,
                "completed",
                completed=len(segments),
                total=len(segments),
            )
        )
        observer.raise_if_stopped("during segmentation")
        observer.emit(
            StageEvent(StageName.SUMMARIZING, "active", total=len(segments))
        )
        if coordinator is not None and coordinator.session is not None:
            coordinator.session.ensure_work_prefix(
                ("segmentation", *(segment.segment_id for segment in segments))
            )
        leaves = summarize_segments(
            segments,
            recording,
            model=app.model,
            timeout_seconds=app.timeout_seconds,
            max_output_tokens=strategy.max_output_tokens,
            coordinator=coordinator,
            observer=observer,
            section_headings=leaf_headings or None,
        )
        observer.emit(
            StageEvent(
                StageName.SUMMARIZING,
                "completed",
                completed=len(leaves),
                total=len(segments),
            )
        )
        observer.raise_if_stopped("during summarization")
        if len(leaves) > 1:
            observer.emit(StageEvent(StageName.MERGING, "active"))
        else:
            observer.emit(StageEvent(StageName.MERGING, "skipped"))
        hierarchy_arguments = dict(
            source_id=document.source_id,
            covered=[(segment.segment_id,) for segment in segments],
            attributable={
                segment.segment_id: document.text[segment.core_start : segment.core_end]
                for segment in segments
            },
            limits=limits,
            model=app.model,
            timeout_seconds=app.timeout_seconds,
            max_merge_children=config.max_merge_children,
            coordinator=coordinator,
            provider_schema_reserve=_provider_schema_reserve(app),
            observer=observer,
        )
        if section_tree is None:
            root, nodes, _ = build_hierarchy(
                leaves, recording, counter, **hierarchy_arguments
            )
        else:
            by_section = build_section_hierarchy(
                leaves,
                recording,
                counter,
                tree=section_tree,
                leaf_sections=leaf_sections,
                **hierarchy_arguments,
            )
            root, nodes = by_section.root, by_section.nodes
            section_nodes = by_section.section_nodes
            own_text_nodes = by_section.own_text_nodes
            own_text_reductions = by_section.own_text_reductions
        if len(leaves) > 1:
            merged = sum(1 for node in nodes if node.level > 0)
            observer.emit(
                StageEvent(
                    StageName.MERGING, "completed", completed=merged, total=merged
                )
            )

    observer.raise_if_stopped("before finalization")
    completed_before_editorial = tuple(recording.generations)
    verifier_runtime = config.verification_runtime
    if config.verification.enabled and verifier_runtime is None:
        verifier_runtime = VerificationRuntime(
            provider=recording,
            counter=counter,
            model=app.model,
            timeout_seconds=app.timeout_seconds,
            context_window_tokens=report.context_window_tokens,
            provider_identity=app.provider,
        )
    elif verifier_runtime is not None and reliability_tracker is not None:
        verifier_runtime = replace(
            verifier_runtime,
            provider=_RecordingProvider(
                verifier_runtime.provider,
                reliability_tracker=reliability_tracker,
            ),
        )
    verification_coordinator = None
    if config.verification.enabled and coordinator is not None:
        assert verifier_runtime is not None
        verification_coordinator = CacheCoordinator(
            store=coordinator.store,
            source_id=coordinator.source_id,
            provider=verifier_runtime.provider_identity,
            model=verifier_runtime.model,
            ollama_host=coordinator.ollama_host,
            counter_identity=verifier_runtime.counter.identity,
            counter_exact=verifier_runtime.counter.exact,
            context_window_tokens=verifier_runtime.context_window_tokens,
            behavior={},
            session=coordinator.session,
            allow_unreferenced_cache=coordinator.allow_unreferenced_cache,
            max_in_flight=coordinator.max_in_flight,
            reliability_tracker=reliability_tracker,
        )
    final = _finalize_summary(
        root.summary,
        recording,
        source_id=document.source_id,
        model=app.model,
        timeout_seconds=app.timeout_seconds,
        target_words=config.target_words,
        strategy=strategy_name,
        segments=segments,
        nodes=nodes,
        root_node_id=root.node_id,
        request_limits=limits,
        include_citations=config.include_citations,
        audit_configuration={
            "app": asdict(app),
            "strategy": asdict(strategy),
            "pipeline": {
                "target_words": config.target_words,
                "max_merge_children": config.max_merge_children,
                "include_citations": config.include_citations,
            },
            "verification": asdict(config.verification),
            "budget": asdict(report),
        },
        audit_path=config.audit_path,
        generations=completed_before_editorial,
        warnings=config.audit_warnings,
        counter=counter,
        source_cores={
            segment.segment_id: document.text[segment.core_start : segment.core_end]
            for segment in segments
        },
        verification=config.verification,
        verification_runtime=verifier_runtime,
        verification_context_window_tokens=report.context_window_tokens,
        verification_coordinator=verification_coordinator,
        reliability_tracker=reliability_tracker,
        observer=observer,
        materialize_audit=not (
            config.audit_path is not None
            and coordinator is not None
            and coordinator.session is not None
        ),
    )
    section_publications: Mapping[str, SectionPublication] = {}
    if section_tree is not None and section_nodes:
        observer.raise_if_stopped("before section prose")
        section_publications = finalize_sections(
            section_tree,
            own_text_nodes,
            (*nodes, *own_text_reductions),
            recording,
            source_id=document.source_id,
            model=app.model,
            timeout_seconds=app.timeout_seconds,
            target_words=config.target_words,
            segments=segments,
            source_cores={
                segment.segment_id: document.text[segment.core_start : segment.core_end]
                for segment in segments
            },
            pages=config.sections.pages if config.sections is not None else None,
            request_limits=limits,
            counter=counter,
            verification=config.verification,
            verification_runtime=verifier_runtime,
            verification_context_window_tokens=report.context_window_tokens,
            verification_coordinator=verification_coordinator,
            observer=observer,
        )
    if section_tree is not None:
        session_publishes = (
            config.audit_path is not None
            and coordinator is not None
            and coordinator.session is not None
        )
        final = attach_section_records(
            final,
            section_tree,
            section_nodes,
            section_publications,
            segments,
            rewrite_path=None if session_publishes else config.audit_path,
        )
    observer.raise_if_stopped("before publication")
    observer.emit(StageEvent(StageName.PUBLISHING, "active"))
    if (
        config.audit_path is not None
        and coordinator is not None
        and coordinator.session is not None
    ):
        publish_final_output(
            final,
            summary_path=app.output_path,
            audit_path=config.audit_path,
            session=coordinator.session,
        )
    observer.emit(StageEvent(StageName.PUBLISHING, "completed"))
    return PipelineResult(
        final=final,
        strategy=report,
        root=root,
        nodes=nodes,
        sections=section_tree,
        section_nodes=section_nodes,
        section_publications=section_publications,
    )
