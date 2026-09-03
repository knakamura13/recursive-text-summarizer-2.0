"""Versioned, secret-safe audit artifacts for completed summary runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from summarizer.hierarchy import TreeNode
from summarizer.providers.base import GenerationResult
from summarizer.safety import redact_text, safe_json_value
from summarizer.segmentation import SourceSegment


AUDIT_SCHEMA_VERSION = "audit/1"


class AuditError(ValueError):
    """An audit record is incomplete, inconsistent, or cannot be written."""


class _AuditRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AuditSegment(_AuditRecord):
    segment_id: str
    source_id: str
    order: int
    core_start: int
    core_end: int
    context_start: int
    context_end: int
    core_token_count: int
    token_count: int
    leading_overlap_tokens: int
    trailing_overlap_tokens: int
    boundary_kind: str


class AuditCitation(_AuditRecord):
    segment_id: str
    source_id: str
    order: int


class AuditUsage(_AuditRecord):
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    finish_status: str | None


class AuditEvidence(_AuditRecord):
    """A traceable evidence edge without copying a possibly sensitive quote."""

    segment_id: str
    quoted: bool


class AuditContentUnit(_AuditRecord):
    kind: str
    evidence: tuple[AuditEvidence, ...]
    qualified: bool
    uncertain: bool


class AuditAnnotation(_AuditRecord):
    evidence: tuple[AuditEvidence, ...]


class AuditSummary(_AuditRecord):
    """The structural/evidence view of a summary, never reader-facing text.

    A summary, entity, content-unit, or quotation string can reproduce a
    source credential. The audit needs their links and classifications, not a
    second durable copy of the generated prose, so this intentionally stores
    no free-form source-derived text.
    """

    level: int
    provenance: tuple[str, ...]
    content_units: tuple[AuditContentUnit, ...]
    qualification_evidence: tuple[AuditAnnotation, ...]
    contradiction_evidence: tuple[AuditAnnotation, ...]
    quotation_evidence: tuple[AuditEvidence, ...]


class AuditNode(_AuditRecord):
    node_id: str
    level: int
    order: int
    children: tuple[str, ...]
    covered_segments: tuple[str, ...]
    summary: AuditSummary


class AuditArtifact(_AuditRecord):
    schema_version: Literal["audit/1"]
    source_id: str
    strategy: str
    model: str
    configuration: dict[str, Any]
    source_segments: tuple[AuditSegment, ...]
    tree_nodes: tuple[AuditNode, ...]
    root_node_id: str
    citations: tuple[AuditCitation, ...]
    usage: tuple[AuditUsage, ...]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]

    @field_validator("source_id", "strategy", "model", "root_node_id")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _links_resolve(self) -> "AuditArtifact":
        segments = {segment.segment_id: segment for segment in self.source_segments}
        if len(segments) != len(self.source_segments):
            raise ValueError("source segment identifiers must be unique")
        if tuple(segment.segment_id for segment in self.source_segments) != tuple(
            segment.segment_id
            for segment in sorted(self.source_segments, key=lambda item: item.order)
        ):
            raise ValueError("source segments must be in source order")
        if any(segment.source_id != self.source_id for segment in self.source_segments):
            raise ValueError("every source segment must belong to the audit source")

        nodes = {node.node_id: node for node in self.tree_nodes}
        if len(nodes) != len(self.tree_nodes):
            raise ValueError("tree node identifiers must be unique")
        root = nodes.get(self.root_node_id)
        if root is None:
            raise ValueError("root_node_id must resolve to a tree node")

        for node in self.tree_nodes:
            unknown_covered = set(node.covered_segments) - set(segments)
            if unknown_covered:
                raise ValueError("tree coverage must resolve to source segments")
            for child_id in node.children:
                child = nodes.get(child_id)
                if child is None:
                    raise ValueError("tree child must resolve to a tree node")
                if child.level >= node.level:
                    raise ValueError("tree children must be at a lower level")
            references = _summary_references(node.summary)
            if not references <= set(segments):
                raise ValueError("summary evidence must resolve to source segments")

        citation_ids = tuple(citation.segment_id for citation in self.citations)
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("citation identifiers must be unique")
        expected_order = tuple(
            segment.segment_id
            for segment in self.source_segments
            if segment.segment_id in citation_ids
        )
        if citation_ids != expected_order:
            raise ValueError("citations must be unique and in source order")
        for citation in self.citations:
            segment = segments.get(citation.segment_id)
            if segment is None or (
                citation.source_id != segment.source_id or citation.order != segment.order
            ):
                raise ValueError("citation mappings must resolve to segment metadata")
        return self


@dataclass(frozen=True)
class Citation:
    segment_id: str
    source_id: str
    order: int


def _summary_references(node: AuditSummary) -> set[str]:
    identifiers = set(node.provenance)
    identifiers.update(
        evidence.segment_id for unit in node.content_units for evidence in unit.evidence
    )
    identifiers.update(
        evidence.segment_id
        for annotation in (
            *node.qualification_evidence,
            *node.contradiction_evidence,
        )
        for evidence in annotation.evidence
    )
    identifiers.update(evidence.segment_id for evidence in node.quotation_evidence)
    return identifiers


def resolve_citations(
    provenance: Sequence[str], *, source_id: str, segments: Sequence[SourceSegment]
) -> tuple[Citation, ...]:
    by_id = {segment.segment_id: segment for segment in segments}
    if len(by_id) != len(segments):
        raise AuditError("source segment identifiers must be unique")
    if any(segment.source_id != source_id for segment in segments):
        raise AuditError("every source segment must belong to the supplied source")
    unknown = set(provenance) - set(by_id)
    if unknown:
        raise AuditError("citation provenance contains an unknown source segment")
    cited = set(provenance)
    return tuple(
        Citation(segment.segment_id, segment.source_id, segment.order)
        for segment in sorted(segments, key=lambda item: item.order)
        if segment.segment_id in cited
    )


def render_citations(text: str, citations: Sequence[Citation]) -> str:
    """Return a readable opt-in source list; the default caller skips this."""
    if not citations:
        return text
    identifiers = ", ".join(citation.segment_id for citation in citations)
    return f"{text}\n\nSources: {identifiers}"


def _audit_segment(segment: SourceSegment) -> AuditSegment:
    return AuditSegment(
        segment_id=segment.segment_id,
        source_id=segment.source_id,
        order=segment.order,
        core_start=segment.core_start,
        core_end=segment.core_end,
        context_start=segment.context_start,
        context_end=segment.context_end,
        core_token_count=segment.core_token_count,
        token_count=segment.token_count,
        leading_overlap_tokens=segment.leading_overlap_tokens,
        trailing_overlap_tokens=segment.trailing_overlap_tokens,
        boundary_kind=segment.boundary_kind.value,
    )


def _audit_node(node: TreeNode) -> AuditNode:
    summary = node.summary
    audit_summary = AuditSummary(
        level=summary.level,
        provenance=summary.provenance,
        content_units=tuple(
            AuditContentUnit(
                kind=unit.kind.value,
                evidence=tuple(
                    AuditEvidence(
                        segment_id=evidence.segment_id,
                        quoted=evidence.quote is not None,
                    )
                    for evidence in unit.evidence
                ),
                qualified=unit.qualification is not None,
                uncertain=unit.uncertain,
            )
            for unit in summary.content_units
        ),
        qualification_evidence=tuple(
            AuditAnnotation(
                evidence=tuple(
                    AuditEvidence(
                        segment_id=evidence.segment_id,
                        quoted=evidence.quote is not None,
                    )
                    for evidence in annotation.evidence
                )
            )
            for annotation in summary.qualifications
        ),
        contradiction_evidence=tuple(
            AuditAnnotation(
                evidence=tuple(
                    AuditEvidence(
                        segment_id=evidence.segment_id,
                        quoted=evidence.quote is not None,
                    )
                    for evidence in annotation.evidence
                )
            )
            for annotation in summary.contradictions
        ),
        quotation_evidence=tuple(
            AuditEvidence(
                segment_id=evidence.segment_id,
                quoted=evidence.quote is not None,
            )
            for evidence in summary.quotations
        ),
    )
    return AuditNode(
        node_id=node.node_id,
        level=node.level,
        order=node.order,
        children=node.children,
        covered_segments=node.covered_segments,
        summary=audit_summary,
    )


def _usage(generation: GenerationResult) -> AuditUsage:
    return AuditUsage(
        provider=redact_text(generation.provider),
        model=redact_text(generation.model),
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        finish_status=(
            redact_text(generation.finish_status)
            if generation.finish_status is not None
            else None
        ),
    )


def build_audit_artifact(
    *,
    source_id: str,
    strategy: str,
    model: str,
    configuration: Mapping[str, object],
    segments: Sequence[SourceSegment],
    nodes: Sequence[TreeNode],
    root_node_id: str,
    citations: Sequence[Citation],
    generations: Sequence[GenerationResult] = (),
    warnings: Sequence[str] = (),
    failures: Sequence[str] = (),
) -> AuditArtifact:
    """Build a validated artifact without retaining source text or request data."""
    return AuditArtifact(
        schema_version=AUDIT_SCHEMA_VERSION,
        source_id=source_id,
        strategy=strategy,
        model=redact_text(model),
        configuration=safe_json_value(configuration),
        source_segments=tuple(_audit_segment(segment) for segment in segments),
        tree_nodes=tuple(_audit_node(node) for node in nodes),
        root_node_id=root_node_id,
        citations=tuple(
            AuditCitation(
                segment_id=citation.segment_id,
                source_id=citation.source_id,
                order=citation.order,
            )
            for citation in citations
        ),
        usage=tuple(_usage(generation) for generation in generations),
        warnings=tuple(redact_text(warning) for warning in warnings),
        failures=tuple(redact_text(failure) for failure in failures),
    )


def serialize_audit(artifact: AuditArtifact) -> bytes:
    """Serialize canonically and prove the written representation is valid."""
    encoded = json.dumps(
        artifact.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    try:
        AuditArtifact.model_validate_json(encoded)
    except ValueError as error:
        raise AuditError("audit serialization failed validation") from error
    return encoded


def write_audit(path: Path, artifact: AuditArtifact) -> None:
    """Atomically replace an artifact only after canonical validation succeeds."""
    payload = serialize_audit(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)
