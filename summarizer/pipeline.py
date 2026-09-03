"""Library-only orchestration from canonical source to final editorial output."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from summarizer.budget import (
    BudgetError,
    BudgetReport,
    measure_overhead,
    resolve_context_window,
    select_strategy,
    usable_input_capacity,
)
from summarizer.config import AppConfig, StrategyConfig
from summarizer.direct import summarize_direct, whole_document_segment
from summarizer.finalization import FinalizationResult, finalize_summary
from summarizer.hierarchy import TreeNode, build_hierarchy
from summarizer.leaf import summarize_segments
from summarizer.providers.base import GenerationRequest, GenerationResult, ModelProvider
from summarizer.segmentation import SegmentationConfig, segment_document
from summarizer.ingestion import SourceDocument
from summarizer.tokenization import TokenCounter


@dataclass(frozen=True)
class PipelineConfig:
    target_words: int = 300
    segmentation: SegmentationConfig | None = None
    max_merge_children: int | None = None
    include_citations: bool = False
    audit_path: Path | None = None

    def __post_init__(self) -> None:
        if self.target_words <= 0:
            raise ValueError("target_words must be positive")


@dataclass(frozen=True)
class PipelineResult:
    final: FinalizationResult
    strategy: BudgetReport
    root: TreeNode
    nodes: tuple[TreeNode, ...]


class _RecordingProvider:
    """Capture completed logical calls without exposing request prompts to audit."""

    def __init__(self, delegate: ModelProvider) -> None:
        self._delegate = delegate
        self.generations: list[GenerationResult] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        result = self._delegate.generate(request)
        self.generations.append(result)
        return result


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
    return usable_input_capacity(
        window=window,
        overhead=measure_overhead(counter, with_overlap=True),
        config=strategy,
    )


def run_pipeline(
    document: SourceDocument,
    provider: ModelProvider,
    counter: TokenCounter,
    *,
    app: AppConfig,
    strategy: StrategyConfig,
    config: PipelineConfig = PipelineConfig(),
) -> PipelineResult:
    """Execute direct or hierarchical library stages, then final editorial writing.

    This is deliberately not wired into the command line. The workflow has a
    complete library seam now while #12 remains responsible for replacing the
    transitional legacy CLI path.
    """
    report = select_strategy(
        document,
        counter,
        provider=app.provider,
        model=app.model,
        config=strategy,
    )
    recording = _RecordingProvider(provider)
    if report.strategy == "direct":
        segment = whole_document_segment(document, counter)
        summary = summarize_direct(
            document, recording, counter, model=app.model, timeout_seconds=app.timeout_seconds
        )
        root = TreeNode(
            node_id="L0N0001",
            level=0,
            order=0,
            summary=summary,
            children=(),
            covered_segments=(segment.segment_id,),
        )
        nodes = (root,)
        segments = (segment,)
    else:
        requested_segmentation = config.segmentation or SegmentationConfig(
            max_tokens=report.usable_input_capacity
        )
        capacity = _hierarchical_capacity(
            report, counter, app, strategy, requested_segmentation
        )
        if requested_segmentation.max_tokens > capacity:
            raise BudgetError(
                "segmentation max_tokens exceeds the safely measured leaf capacity"
            )
        segments = tuple(segment_document(document, counter, requested_segmentation))
        leaves = summarize_segments(
            segments, recording, model=app.model, timeout_seconds=app.timeout_seconds
        )
        root, nodes, _ = build_hierarchy(
            leaves,
            recording,
            counter,
            source_id=document.source_id,
            covered=[(segment.segment_id,) for segment in segments],
            attributable={
                segment.segment_id: document.text[segment.core_start:segment.core_end]
                for segment in segments
            },
            usable_tokens=capacity,
            model=app.model,
            timeout_seconds=app.timeout_seconds,
            max_merge_children=config.max_merge_children,
        )

    completed_before_editorial = tuple(recording.generations)
    final = finalize_summary(
        root.summary,
        recording,
        source_id=document.source_id,
        model=app.model,
        timeout_seconds=app.timeout_seconds,
        target_words=config.target_words,
        strategy=report.strategy,
        segments=segments,
        nodes=nodes,
        root_node_id=root.node_id,
        include_citations=config.include_citations,
        audit_configuration={
            "app": asdict(app),
            "strategy": asdict(strategy),
            "pipeline": asdict(config),
            "budget": asdict(report),
        },
        audit_path=config.audit_path,
        generations=completed_before_editorial,
    )
    return PipelineResult(final=final, strategy=report, root=root, nodes=nodes)
