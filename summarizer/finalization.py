"""Compose final writing, optional citations, and optional audit output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from summarizer.audit import (
    AuditArtifact,
    Citation,
    build_audit_artifact,
    render_citations,
    resolve_citations,
    write_audit,
)
from summarizer.editorial import write_editorial
from summarizer.hierarchy import TreeNode
from summarizer.providers.base import GenerationResult, ModelProvider
from summarizer.segmentation import SourceSegment
from summarizer.summaries import SummaryNode


@dataclass(frozen=True)
class FinalizationResult:
    text: str
    citations: tuple[Citation, ...]
    audit: AuditArtifact | None


def finalize_summary(
    root: SummaryNode,
    provider: ModelProvider,
    *,
    source_id: str,
    model: str,
    timeout_seconds: float,
    target_words: int,
    strategy: str,
    segments: Sequence[SourceSegment],
    nodes: Sequence[TreeNode],
    root_node_id: str,
    include_citations: bool = False,
    audit_configuration: Mapping[str, object] | None = None,
    audit_path: Path | None = None,
    generations: Sequence[GenerationResult] = (),
    warnings: Sequence[str] = (),
    failures: Sequence[str] = (),
) -> FinalizationResult:
    """Run the final editor and materialize optional safe output views.

    The root's own validated provenance determines citations. The writer does
    not choose or invent source identifiers, so output formatting cannot leave
    a citation dangling from the recorded source metadata.
    """
    editorial = write_editorial(
        root,
        provider,
        source_id=source_id,
        model=model,
        timeout_seconds=timeout_seconds,
        target_words=target_words,
    )
    citations = resolve_citations(
        root.provenance, source_id=source_id, segments=segments
    )
    text = render_citations(editorial.text, citations) if include_citations else editorial.text

    artifact = None
    if audit_path is not None:
        artifact = build_audit_artifact(
            source_id=source_id,
            strategy=strategy,
            model=model,
            configuration=audit_configuration or {},
            segments=segments,
            nodes=nodes,
            root_node_id=root_node_id,
            citations=citations,
            generations=(*generations, editorial.generation),
            warnings=warnings,
            failures=failures,
        )
        write_audit(audit_path, artifact)
    return FinalizationResult(text=text, citations=citations, audit=artifact)
