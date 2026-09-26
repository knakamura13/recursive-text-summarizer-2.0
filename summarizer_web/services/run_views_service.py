"""Read models for a Run: node tree, node detail, segments, final summary, exports.

Inputs: ``node_projections`` rows (written by the worker from item events),
``run_segments`` rows (written from ``on_segments``), and the Run directory:
``summary.txt`` (the published text, followed by ``"\\n\\nSources: S000001,
..."`` when citations are on) and ``audit.json``.

Evidence offsets: a segment id resolves from ``run_segments``, then from
audit.json ``source_segments``, which also lists the verification passages
that ``run_segments`` lacks (numbered after the last leaf segment, with a
``parent_segment_id``); ids are unique across both and describe the same
ranges. Offsets the audit records are used as is; otherwise a quote is
located inside its segment's core range (then its whole context range) in
the canonical text, and a missing quote points at the core range with
``quote_found`` false. Pages always come from the revision's page map for the
resolved absolute range, with the rule the tree labels use.

audit.json ``publication`` (written by summarizer/finalization.py):
``{kind, sentences: [{index, paragraph, start, end, text, verdict,
evidence: [{segment_id, quote, start, end}]}], removed_sentences: [{text,
verdict, reason}]}``. Legacy audits lack it: sentences are rebuilt from the
text with the verifier's own splitter, and verdicts plus evidence segments
come from the last verification pass when its spans match the published text.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Literal

from summarizer.verification import split_draft_spans
from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database
from summarizer_web.errors import ApiError
from summarizer_web.models.api import (
    EvidenceRef,
    FinalSummaryResponse,
    NodeAnnotation,
    NodeContentUnit,
    NodeDetailResponse,
    NodeTreeResponse,
    Notice,
    RemovedSentence,
    RunConfig,
    SegmentListResponse,
    SegmentRef,
    SummaryCitation,
    SummarySentence,
)
from summarizer_web.services.documents_service import attachment_disposition, canonical_text
from summarizer_web.worker.projections import (
    PageMap,
    max_event_id,
    node_duration_seconds,
    tree_items,
)

ExportFormat = Literal["txt", "md", "json"]

_ACTIVE_STATES = frozenset({"queued", "running", "stopping"})
_NODE_KINDS = frozenset({"leaf", "merge", "passthrough"})
_NODE_STATES = frozenset({"pending", "active", "completed", "failed"})
_SENTENCE_VERDICTS = frozenset({"supported", "not_meaningfully_verifiable", "unchecked"})
_PUBLICATION_KINDS = frozenset({"editorial", "verified_subset", "content_unit_fallback"})
_SOURCES_TRAILER = re.compile(r"\n\nSources: [DS]\d{6}(?:, [DS]\d{6})*\s*\Z")
_SEGMENT_IDS = re.compile(r"[DS]\d{6}")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
_UNSAFE_FILENAME = re.compile(r'[\x00-\x1f\x7f/\\:*?"<>|]+')

# Readable messages for every code audit.json can carry in `warnings` and in
# `verification.{warning,limitation,failure}_codes`.
_NOTICE_TEXT: dict[str, tuple[Literal["info", "warning", "error"], str]] = {
    "verified_sentence_subset": (
        "warning",
        "Some sentences of the written draft could not be verified and were removed; "
        "the remaining sentences passed verification.",
    ),
    "verified_content_unit_fallback": (
        "warning",
        "The written draft could not be verified, so this summary was assembled from "
        "individually verified content units.",
    ),
    "verified_content_unit_fallback_failed": (
        "error",
        "Neither the written draft nor the verified content units produced a summary "
        "that passed verification.",
    ),
    "invalid_anchors_omitted": (
        "info",
        "Some claims the verifier extracted did not match the summary text and were skipped.",
    ),
    "invalid_evidence_quotes_downgraded": (
        "info",
        "Some verifier quotes did not match the source, so they were not counted as support.",
    ),
    "conflicting_evidence": ("warning", "The verifier found conflicting evidence for some claims."),
    "inconsistent_meaningfulness": (
        "info",
        "The verifier disagreed about whether some claims can be checked against the source.",
    ),
    "insufficient_support": (
        "warning",
        "Some claims were not sufficiently supported by the source.",
    ),
    "material_contradiction": ("warning", "The source contradicts part of the summary."),
    "repair_disabled": (
        "info",
        "Repair passes were off, so unsupported sentences were not rewritten.",
    ),
    "repair_conflicting_assessment": (
        "info",
        "Some sentences were not repaired because the verifier's assessments of them conflicted.",
    ),
    "repair_not_eligible": ("info", "Some sentences could not be repaired automatically."),
    "repair_capacity_failed": ("warning", "A repair request did not fit the context window."),
    "repair_provider_failed": ("warning", "The model failed during a repair pass."),
    "repair_failed": ("warning", "A repair pass returned unusable output."),
    "repair_reverification_failed": (
        "warning",
        "A repaired sentence did not pass verification again.",
    ),
    "decomposition_capacity_failed": (
        "warning",
        "The summary was too long to split into claims within the verification budget.",
    ),
    "decomposition_provider_failed": (
        "warning",
        "The model failed while splitting the summary into claims.",
    ),
    "decomposition_failed": (
        "warning",
        "The model's breakdown of the summary into claims was unusable.",
    ),
    "anchor_failed": ("warning", "The verifier could not match its claims to the summary text."),
    "evidence_capacity_failed": (
        "warning",
        "Evidence for a claim did not fit the verification budget.",
    ),
    "classification_capacity_failed": (
        "warning",
        "A claim check did not fit the context window.",
    ),
    "classification_provider_failed": (
        "warning",
        "The model failed while checking claims against the source.",
    ),
    "classification_failed": ("warning", "The model's claim check was unusable."),
    "escalation_classification_failed": (
        "warning",
        "An extra evidence check was unusable, so that claim kept its first result.",
    ),
}


@dataclass(frozen=True)
class _Segment:
    segment_id: str
    order: int
    start: int
    end: int
    core_start: int
    core_end: int


@dataclass(frozen=True)
class SummaryExport:
    content: bytes
    media_type: str
    filename: str

    @property
    def content_disposition(self) -> str:
        return attachment_disposition(self.filename)


class _RunContext:
    """One Run's stored artifacts, each loaded on first use."""

    def __init__(self, run: sqlite3.Row) -> None:
        self.run = run
        self.run_id: str = run["run_id"]
        self.run_dir = load_paths().runs / self.run_id

    @classmethod
    def load(cls, run_id: str) -> _RunContext:
        row = get_database().fetchone(
            """
            SELECT r.run_id, r.state, r.config_json, r.revision_id, d.title AS document_title
            FROM runs r LEFT JOIN documents d ON d.document_id = r.document_id
            WHERE r.run_id = ?
            """,
            (run_id,),
        )
        if row is None:
            raise ApiError(404, "run_not_found", "This Run does not exist.")
        return cls(row)

    @cached_property
    def config(self) -> RunConfig:
        try:
            stored = json.loads(self.run["config_json"])
        except (TypeError, ValueError):
            stored = None
        if not isinstance(stored, dict):
            return RunConfig.model_construct()
        return RunConfig.model_construct(
            **{key: value for key, value in stored.items() if key in RunConfig.model_fields}
        )

    @cached_property
    def audit(self) -> dict[str, Any] | None:
        path = self.run_dir / "audit.json"
        try:
            stat = path.stat()
        except OSError:
            return None
        return _parsed_audit(str(path), stat.st_mtime_ns, stat.st_size)

    @cached_property
    def revision(self) -> sqlite3.Row | None:
        return get_database().fetchone(
            "SELECT * FROM source_revisions WHERE revision_id = ?", (self.run["revision_id"],)
        )

    @cached_property
    def text(self) -> str | None:
        if self.revision is None:
            return None
        try:
            return canonical_text(self.revision["canonical_path"])
        except OSError:
            return None

    @cached_property
    def _page_map(self) -> PageMap:
        try:
            return PageMap.from_json(self.revision["page_map_json"] if self.revision else None)
        except (TypeError, ValueError):
            return PageMap()

    def page_span(self, start: int, end: int) -> tuple[int | None, int | None]:
        """First and last page overlapping [start, end), the rule tree labels use."""
        pages = self._page_map.pages_for(start, end)
        return pages if pages is not None else (None, None)

    @cached_property
    def _stored_segments(self) -> list[_Segment]:
        rows = get_database().fetchall(
            """
            SELECT segment_id, order_index, start_offset, end_offset, core_start, core_end
            FROM run_segments WHERE run_id = ? ORDER BY order_index
            """,
            (self.run_id,),
        )
        return [
            _Segment(
                row["segment_id"],
                row["order_index"],
                row["start_offset"],
                row["end_offset"],
                row["core_start"],
                row["core_end"],
            )
            for row in rows
        ]

    @cached_property
    def _audit_segments(self) -> list[_Segment]:
        entries = self.audit.get("source_segments") if self.audit else None
        segments = []
        for entry in entries if isinstance(entries, list) else ():
            if not isinstance(entry, dict):
                continue
            values = [entry.get(key) for key in ("core_start", "core_end", "context_start", "context_end")]
            if not isinstance(entry.get("segment_id"), str) or not all(
                type(value) is int for value in values
            ):
                continue
            core_start, core_end, context_start, context_end = values
            order = entry.get("order")
            segments.append(
                _Segment(
                    entry["segment_id"],
                    order if type(order) is int else len(segments),
                    context_start,
                    context_end,
                    core_start,
                    core_end,
                )
            )
        return segments

    @cached_property
    def _stored_index(self) -> dict[str, _Segment]:
        return {segment.segment_id: segment for segment in self._stored_segments}

    @cached_property
    def _audit_index(self) -> dict[str, _Segment]:
        return {segment.segment_id: segment for segment in self._audit_segments}

    def segment(self, segment_id: str) -> _Segment | None:
        """A segment by id; audit.json also knows verification sub-passages.

        Ids are unique across both sources and describe the same ranges, so
        audit.json is parsed only for ids that run_segments lacks.
        """
        return self._stored_index.get(segment_id) or self._audit_index.get(segment_id)

    def pipeline_segments(self) -> list[_Segment]:
        """The Run's leaf segments; legacy Runs fall back to the audit's tree coverage."""
        if self._stored_segments:
            return self._stored_segments
        covered = {
            segment_id
            for node in (self.audit or {}).get("tree_nodes") or ()
            if isinstance(node, dict)
            for segment_id in node.get("covered_segments") or ()
        }
        return sorted(
            (segment for segment in self._audit_segments if segment.segment_id in covered),
            key=lambda segment: segment.order,
        )

    def segment_ref(self, segment: _Segment) -> SegmentRef:
        page_start, page_end = self.page_span(segment.core_start, segment.core_end)
        return SegmentRef(
            segment_id=segment.segment_id,
            order=segment.order,
            start=segment.start,
            end=segment.end,
            core_start=segment.core_start,
            core_end=segment.core_end,
            page_start=page_start,
            page_end=page_end,
        )

    def evidence(
        self,
        segment_id: str,
        quote: str | None,
        *,
        start: object = None,
        end: object = None,
    ) -> EvidenceRef:
        quote = quote.strip() if isinstance(quote, str) and quote.strip() else None
        segment = self.segment(segment_id)
        located: tuple[int, int] | None = None
        if type(start) is int and type(end) is int and 0 <= start < end and (
            self.text is None or end <= len(self.text)
        ):
            located = (start, end)
        elif quote is not None and segment is not None and self.text is not None:
            located = _find_quote(self.text, quote, segment)
        if located is None and segment is None:
            return EvidenceRef(segment_id=segment_id, quote=quote)
        found = located is not None
        begin, finish = located if located is not None else (segment.core_start, segment.core_end)
        page_start, page_end = self.page_span(begin, finish)
        return EvidenceRef(
            segment_id=segment_id,
            quote=quote,
            quote_found=found,
            start=begin,
            end=finish,
            page_start=page_start,
            page_end=page_end,
        )


def get_node_tree(run_id: str) -> NodeTreeResponse:
    _RunContext.load(run_id)
    # A snapshot at the cursor: rows committed after it reach the client
    # through the stream (`after=cursor`) instead of being skipped.
    cursor = max_event_id(run_id)
    return NodeTreeResponse(nodes=tree_items(run_id, through_event_id=cursor), cursor=cursor)


def get_segments(run_id: str) -> SegmentListResponse:
    context = _RunContext.load(run_id)
    return SegmentListResponse(
        segments=[context.segment_ref(segment) for segment in context.pipeline_segments()]
    )


def get_node_detail(run_id: str, node_id: str) -> NodeDetailResponse:
    context = _RunContext.load(run_id)
    row = get_database().fetchone(
        "SELECT * FROM node_projections WHERE run_id = ? AND node_id = ?", (run_id, node_id)
    )
    if row is None:
        raise ApiError(404, "node_not_found", f"Node {node_id} is not part of this Run.")
    detail = _json_value(row["detail_json"])
    detail = detail if isinstance(detail, dict) else {}

    content_units = [
        NodeContentUnit(
            text=unit["text"],
            kind=unit["kind"] if isinstance(unit.get("kind"), str) else "other",
            uncertain=unit.get("uncertain") is True,
            qualification=unit["qualification"] if isinstance(unit.get("qualification"), str) else None,
            evidence=_evidence_list(context, unit.get("evidence")),
        )
        for unit in _dicts(_json_value(row["content_units_json"]))
        if isinstance(unit.get("text"), str)
    ]
    annotations = [
        NodeAnnotation(
            kind=kind,
            text=item["text"],
            evidence=_evidence_list(context, item.get("evidence")),
        )
        for key, kind in (("qualifications", "qualification"), ("contradictions", "contradiction"))
        for item in _dicts(detail.get(key))
        if isinstance(item.get("text"), str)
    ]
    if row["content_units_json"] is None and not detail:
        # Legacy projections kept only the flattened content-unit evidence.
        legacy = {
            (item.get("segment_id"), item.get("quote")): item
            for item in _dicts(_json_value(row["evidence_refs_json"]))
        }
        quotations = _evidence_list(context, list(legacy.values()))
    else:
        quotations = _evidence_list(context, detail.get("quotations"))

    covered_ids = [
        value
        for value in _json_value(row["covered_segment_ids_json"]) or ()
        if isinstance(value, str)
    ]
    covered_segments = [
        context.segment_ref(segment)
        for segment in map(context.segment, covered_ids)
        if segment is not None
    ]
    child_ids = _json_value(row["child_ids_json"])
    if not isinstance(child_ids, list):
        child_ids = [
            child["node_id"]
            for child in get_database().fetchall(
                "SELECT node_id FROM node_projections WHERE run_id = ? AND parent_id = ? "
                "ORDER BY order_index",
                (run_id, node_id),
            )
        ]
    kind = row["kind"] if row["kind"] in _NODE_KINDS else ("merge" if row["level"] > 0 else "leaf")
    state = row["state"] if row["state"] in _NODE_STATES else (
        "completed" if row["summary_text"] is not None else "pending"
    )
    return NodeDetailResponse(
        node_id=row["node_id"],
        parent_id=row["parent_id"],
        level=row["level"],
        order=row["order_index"],
        kind=kind,
        label=row["label"],
        state=state,
        summary_text=row["summary_text"],
        content_units=content_units,
        annotations=annotations,
        entities=[value for value in detail.get("entities") or () if isinstance(value, str)],
        quotations=quotations,
        covered_segments=covered_segments,
        child_ids=[value for value in child_ids if isinstance(value, str)],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        duration_seconds=node_duration_seconds(row),
        error=row["error"],
    )


def get_final_summary(run_id: str) -> FinalSummaryResponse:
    return _final_summary(_RunContext.load(run_id))


def export_summary(run_id: str, export_format: ExportFormat) -> SummaryExport:
    """The published summary as a file: txt verbatim, md with footnoted evidence, json = audit."""
    context = _RunContext.load(run_id)
    title = _filename_title(context.run["document_title"])
    if export_format == "json":
        audit_path = context.run_dir / "audit.json"
        if context.run["state"] in _ACTIVE_STATES or not audit_path.is_file():
            raise _export_unavailable("The audit for this Run is not available yet.")
        return SummaryExport(audit_path.read_bytes(), "application/json", f"{title}-summary.json")
    summary = _final_summary(context)
    if not summary.available:
        raise _export_unavailable("This Run has no published summary to export.")
    if export_format == "txt":
        content = (context.run_dir / "summary.txt").read_bytes()
        return SummaryExport(content, "text/plain; charset=utf-8", f"{title}-summary.txt")
    markdown = _markdown(context.run["document_title"] or "Summary", summary)
    return SummaryExport(
        markdown.encode("utf-8"), "text/markdown; charset=utf-8", f"{title}-summary.md"
    )


def _final_summary(context: _RunContext) -> FinalSummaryResponse:
    config = context.config
    target = config.target_words if type(config.target_words) is int else None
    audit = context.audit
    verification = audit.get("verification") if audit else None
    verification = verification if isinstance(verification, dict) else {}
    summary_path = context.run_dir / "summary.txt"
    state = context.run["state"]

    if state != "completed" or not summary_path.is_file():
        failed = verification.get("failed") is True
        if state in _ACTIVE_STATES:
            verification_state = "in_progress" if config.verify is not False else "not_run"
        else:
            verification_state = "failed" if failed else "not_run"
        return FinalSummaryResponse(
            available=False,
            target_words=target,
            verification_state=verification_state,
            notices=_audit_notices(audit) if failed else [],
        )

    raw = summary_path.read_text(encoding="utf-8")
    text = _SOURCES_TRAILER.sub("", raw).rstrip()
    publication = audit.get("publication") if audit else None
    publication = publication if isinstance(publication, dict) else None
    if publication is not None:
        sentences = _published_sentences(context, publication)
        removed = [
            RemovedSentence(
                text=item["text"],
                verdict=item["verdict"] if isinstance(item.get("verdict"), str) else "unverified",
                reason=item["reason"] if isinstance(item.get("reason"), str) else None,
            )
            for item in _dicts(publication.get("removed_sentences"))
            if isinstance(item.get("text"), str)
        ]
    else:
        sentences = _legacy_sentences(context, text, verification)
        removed = []

    warnings = [code for code in (audit or {}).get("warnings") or () if isinstance(code, str)]
    kind = publication.get("kind") if publication is not None else None
    if kind not in _PUBLICATION_KINDS:
        kind = (
            "content_unit_fallback"
            if "verified_content_unit_fallback" in warnings
            else "verified_subset"
            if "verified_sentence_subset" in warnings
            else "editorial"
        )
    if audit is not None and "enabled" in verification:
        verified = verification["enabled"] is True
    else:
        verified = _legacy_verification_state(context) == "completed"

    word_count = len(text.split())
    short = target is not None and word_count < target
    notices = _audit_notices(audit)
    if short:
        notices.insert(
            _leading_publication_notices(notices),
            Notice(
                code="short_of_target",
                message=f"This summary has {word_count} words, below the target of {target}.",
                severity="info" if kind == "editorial" else "warning",
            ),
        )
    if not verified:
        notices.append(
            Notice(
                code="verification_off",
                message="Verification was off for this Run, so its sentences are unchecked.",
                severity="info",
            )
        )
    return FinalSummaryResponse(
        available=True,
        text=text,
        sentences=sentences,
        removed_sentences=removed,
        citations=_citations(context, raw),
        word_count=word_count,
        target_words=target,
        short_of_target=short,
        notices=notices,
        verification_state="completed" if verified else "not_run",
        publication=kind,
    )


def _published_sentences(context: _RunContext, publication: dict[str, Any]) -> list[SummarySentence]:
    sentences = []
    for item in _dicts(publication.get("sentences")):
        if not isinstance(item.get("text"), str):
            continue
        index, paragraph = item.get("index"), item.get("paragraph")
        verdict = item.get("verdict")
        sentences.append(
            SummarySentence(
                index=index if type(index) is int else len(sentences),
                paragraph=paragraph if type(paragraph) is int else 0,
                text=item["text"],
                verdict=verdict if verdict in _SENTENCE_VERDICTS else "unchecked",
                evidence=[
                    context.evidence(
                        evidence["segment_id"],
                        evidence.get("quote"),
                        start=evidence.get("start"),
                        end=evidence.get("end"),
                    )
                    for evidence in _dicts(item.get("evidence"))
                    if isinstance(evidence.get("segment_id"), str)
                ],
            )
        )
    return sentences


def _legacy_sentences(
    context: _RunContext, text: str, verification: dict[str, Any]
) -> list[SummarySentence]:
    if not text.strip():
        return []
    support = _last_pass_support(verification, text)
    paragraph_starts = [0, *(match.end() for match in _PARAGRAPH_BREAK.finditer(text))]
    sentences = []
    for span in split_draft_spans(text, pass_index=1):
        sentence = span.text.strip()
        if not sentence:
            continue
        start = span.start + len(span.text) - len(span.text.lstrip())
        verdict, segment_ids = support.get((span.start, span.end), ("unchecked", ()))
        sentences.append(
            SummarySentence(
                index=len(sentences),
                paragraph=bisect_right(paragraph_starts, start) - 1,
                text=sentence,
                verdict=verdict,
                evidence=[context.evidence(segment_id, None) for segment_id in segment_ids],
            )
        )
    return sentences


def _last_pass_support(
    verification: dict[str, Any], text: str
) -> dict[tuple[int, int], tuple[str, tuple[str, ...]]]:
    """Verdict and supporting segments per span of the pass that checked `text`."""
    passes = verification.get("passes")
    if verification.get("enabled") is not True or verification.get("failed") or not passes:
        return {}
    last = passes[-1] if isinstance(passes[-1], dict) else {}
    claims: dict[str, list[str]] = {}
    for claim in _dicts(last.get("claims")):
        claims.setdefault(claim.get("span_id"), []).append(claim.get("claim_id"))
    assessments = {item.get("claim_id"): item for item in _dicts(last.get("assessments"))}
    support = {}
    for span in _dicts(last.get("spans")):
        start, end = span.get("start"), span.get("end")
        if not (type(start) is int and type(end) is int and 0 <= start < end <= len(text)):
            continue
        if hashlib.sha256(text[start:end].encode("utf-8")).hexdigest() != span.get("content_hash"):
            continue
        span_assessments = [assessments.get(claim_id) for claim_id in claims.get(span.get("span_id"), ())]
        if not span_assessments or None in span_assessments:
            continue
        verdicts = {item.get("verdict") for item in span_assessments}
        if verdicts == {"supported"}:
            verdict = "supported"
        elif verdicts <= {"supported", "not_meaningfully_verifiable"}:
            verdict = "not_meaningfully_verifiable"
        else:
            continue
        segment_ids = dict.fromkeys(
            segment_id
            for item in span_assessments
            if item.get("verdict") == "supported"
            for finding in _dicts(item.get("findings"))
            if finding.get("verdict") == "supported"
            for segment_id in finding.get("evidence_ids") or ()
            if isinstance(segment_id, str)
        )
        support[(start, end)] = (verdict, tuple(segment_ids))
    return support


def _citations(context: _RunContext, raw_summary: str) -> list[SummaryCitation]:
    entries = (context.audit or {}).get("citations")
    if isinstance(entries, list):
        segment_ids = [item.get("segment_id") for item in _dicts(entries)]
    else:
        match = _SOURCES_TRAILER.search(raw_summary)
        segment_ids = _SEGMENT_IDS.findall(match.group()) if match else []
    citations = []
    for segment_id in dict.fromkeys(value for value in segment_ids if isinstance(value, str)):
        segment = context.segment(segment_id)
        page_start, page_end = (
            context.page_span(segment.core_start, segment.core_end) if segment else (None, None)
        )
        citations.append(
            SummaryCitation(
                citation_id=str(len(citations) + 1),
                segment_id=segment_id,
                start=segment.core_start if segment else None,
                end=segment.core_end if segment else None,
                page_start=page_start,
                page_end=page_end,
            )
        )
    return citations


def _audit_notices(audit: dict[str, Any] | None) -> list[Notice]:
    if audit is None:
        return []
    verification = audit.get("verification") if isinstance(audit.get("verification"), dict) else {}
    failure_codes = set(verification.get("failure_codes") or ())
    codes = dict.fromkeys(
        code
        for group in (
            audit.get("warnings"),
            verification.get("warning_codes"),
            verification.get("limitation_codes"),
            verification.get("failure_codes"),
        )
        for code in group or ()
        if isinstance(code, str)
    )
    notices = []
    for code in codes:
        severity, message = _NOTICE_TEXT.get(
            code, ("info", f"Verification reported: {code.replace('_', ' ')}.")
        )
        notices.append(
            Notice(code=code, message=message, severity="error" if code in failure_codes else severity)
        )
    return notices


def _leading_publication_notices(notices: list[Notice]) -> int:
    publication_codes = {"verified_sentence_subset", "verified_content_unit_fallback"}
    return sum(1 for notice in notices if notice.code in publication_codes)


def _legacy_verification_state(context: _RunContext) -> str:
    try:
        stored = json.loads((context.run_dir / "verification.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = None
    if isinstance(stored, dict) and stored.get("state") in ("completed", "not_run"):
        return stored["state"]
    return "completed" if context.config.verify is not False else "not_run"


def _markdown(title: str, summary: FinalSummaryResponse) -> str:
    footnotes: dict[tuple[object, ...], tuple[int, str]] = {}
    paragraphs: dict[int, list[str]] = {}
    for sentence in summary.sentences:
        markers = []
        for evidence in sentence.evidence:
            key = (evidence.segment_id, evidence.start, evidence.end, evidence.quote)
            if key not in footnotes:
                footnotes[key] = (len(footnotes) + 1, _footnote_text(evidence))
            markers.append(f"[^{footnotes[key][0]}]")
        text = " ".join(sentence.text.split())
        paragraphs.setdefault(sentence.paragraph, []).append(text + "".join(markers))
    body = (
        "\n\n".join(" ".join(parts) for _, parts in sorted(paragraphs.items()))
        if paragraphs
        else summary.text or ""
    )
    lines = [f"# {title.strip()}", "", body]
    if footnotes:
        lines.append("")
        lines.extend(f"[^{number}]: {text}" for number, text in footnotes.values())
    elif summary.citations:
        lines.extend(["", "Sources:", ""])
        lines.extend(
            f"- {_location(citation.segment_id, citation.page_start, citation.page_end)}"
            for citation in summary.citations
        )
    return "\n".join(lines) + "\n"


def _footnote_text(evidence: EvidenceRef) -> str:
    location = _location(evidence.segment_id, evidence.page_start, evidence.page_end)
    if not evidence.quote:
        return location
    return f"{location}: “{' '.join(evidence.quote.split())}”"


def _location(segment_id: str, page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return _passage_name(segment_id)
    if page_end is None or page_end == page_start:
        return f"p. {page_start}"
    return f"pp. {page_start}–{page_end}"


def _passage_name(segment_id: str) -> str:
    """S000012 reads as Passage 12; the direct strategy's D000001 as Whole document."""
    match = re.fullmatch(r"([DS])0*(\d+)", segment_id)
    if match is None:
        return segment_id
    return "Whole document" if match.group(1) == "D" else f"Passage {match.group(2)}"


def _filename_title(title: str | None) -> str:
    cleaned = " ".join(_UNSAFE_FILENAME.sub(" ", title or "").split()).strip(" .-")
    return cleaned[:100].rstrip(" .-") or "summary"


def _export_unavailable(message: str) -> ApiError:
    return ApiError(404, "export_unavailable", message)


def _find_quote(text: str, quote: str, segment: _Segment) -> tuple[int, int] | None:
    for low, high in ((segment.core_start, segment.core_end), (segment.start, segment.end)):
        index = text.find(quote, low, high)
        if index >= 0:
            return index, index + len(quote)
    return None


def _evidence_list(context: _RunContext, items: object) -> list[EvidenceRef]:
    return [
        context.evidence(item["segment_id"], item.get("quote"))
        for item in _dicts(items)
        if isinstance(item.get("segment_id"), str)
    ]


@lru_cache(maxsize=8)
def _parsed_audit(path: str, mtime_ns: int, size: int) -> dict[str, Any] | None:
    """Parse audit.json once per file version; callers must not mutate the result."""
    del mtime_ns, size  # cache key only
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _json_value(raw: object) -> Any:
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _dicts(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
