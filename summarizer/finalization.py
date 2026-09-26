"""Compose the final summary: editorial writing, verified publication, and audit.

With verification enabled, exactly one of two verified texts is published:

1. `editorial`: the editorial draft passes verification, repairs included.
2. `verified_subset`: otherwise failing sentences are dropped from the first
   verification pass and the passing remainder is published without a second
   full verification (warning `verified_sentence_subset`).

Nothing is published unless at least one sentence passes that first check;
otherwise a failure audit is written and `FinalizationVerificationError` is
raised. The audit's `publication` records the kind, each published sentence
with its supporting quotations, and every removed sentence with its verdict.
"""

from __future__ import annotations

import hashlib
import re
import threading
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from summarizer.audit import (
    _atomic_replace,
    PUBLICATION_WARNINGS,
    AuditArtifact,
    AuditPublication,
    AuditPublishedSentence,
    AuditRemovedSentence,
    AuditSentenceEvidence,
    Citation,
    PublicationKind,
    build_audit_artifact,
    citation_provenance_for_summary,
    render_citations,
    resolve_citations,
    serialize_audit,
    write_audit,
)
from summarizer.checkpoint import (
    CheckpointSession,
    PublicationState,
    RunManifest,
)
from summarizer.compression import (
    compress_to_target,
    retain_sentences_with_missing_literals,
    word_count,
    _above_ceiling,
    _in_band,
    _under_floor,
)
from summarizer.editorial import write_editorial
from summarizer.grounding import SourcePassage, serialize_source_passage
from summarizer.hierarchy import TreeNode
from summarizer.ingestion import SourceDocument
from summarizer.providers.base import GenerationResult, ModelProvider
from summarizer.reliability import ReliabilityTracker
from summarizer.runtime.observers import (
    RuntimeObserver,
    StageEvent,
    StageName,
    get_observer,
)
from summarizer.safety import redact_text
from summarizer.segmentation import (
    BoundaryKind,
    CacheCoordinator,
    SegmentationConfig,
    SourceSegment,
    segment_document,
)
from summarizer.summaries import SummaryNode
from summarizer.text import default_sentence_tokenizer
from summarizer.tokenization import TokenCounter
from summarizer.verification import (
    BatchFinding,
    Claim,
    ClaimAssessment,
    DraftSpan,
    ClaimVerdict,
    EvidenceBundle,
    SourceLexicalIndex,
    VerificationConfig,
    VerificationPassResult,
    VerificationProgress,
    VerificationResult,
    VerificationRuntime,
    build_source_lexical_index,
    claim_drop_blocked_by_omitted_required,
    claim_kept_for_lenient_literals,
    split_draft_spans,
    verify_and_repair,
    verify_draft_once,
)

# Evidence passages are never split below this many tokens.
_MIN_PASSAGE_TOKENS = 32
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")
_PUBLISHABLE_VERDICTS = frozenset(
    {ClaimVerdict.SUPPORTED, ClaimVerdict.NOT_MEANINGFULLY_VERIFIABLE}
)
_VERDICT_SEVERITY = {
    ClaimVerdict.NOT_MEANINGFULLY_VERIFIABLE: 1,
    ClaimVerdict.INSUFFICIENTLY_SUPPORTED: 2,
    ClaimVerdict.CONTRADICTED: 3,
}
_REMOVAL_REASONS = {
    ClaimVerdict.CONTRADICTED: (
        "The source contradicts this sentence.",
        'The source contradicts "{anchor}".',
    ),
    ClaimVerdict.INSUFFICIENTLY_SUPPORTED: (
        "The source does not support this sentence.",
        'The source does not support "{anchor}".',
    ),
    ClaimVerdict.NOT_MEANINGFULLY_VERIFIABLE: (
        "This sentence cannot be checked against the source.",
        '"{anchor}" cannot be checked against the source.',
    ),
}
_UNVERIFIED_REASON = "Verification could not check this sentence."
_COMPLETED_DETAIL = {
    "editorial": "Editorial draft verified",
    "content_unit_fallback": "Published verified content units",
}

_DEFAULT_VERIFICATION_CONFIG = VerificationConfig()
_PUBLICATION_LOCKS_GUARD = threading.Lock()
_PUBLICATION_LOCKS: weakref.WeakValueDictionary[tuple[str, str], threading.RLock] = (
    weakref.WeakValueDictionary()
)


@dataclass(frozen=True)
class FinalizationResult:
    text: str
    citations: tuple[Citation, ...]
    audit: AuditArtifact | None


class FinalizationVerificationError(RuntimeError):
    """Verification closed without a safe reader-facing final summary."""


def _all_claims_supported(result: VerificationPassResult) -> bool:
    return (
        not result.failed
        and bool(result.assessments)
        and all(
            assessment.verdict is ClaimVerdict.SUPPORTED
            for assessment in result.assessments
        )
    )


def _original_sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in default_sentence_tokenizer(text):
        index = text.find(sentence, cursor)
        if index < 0:
            return []
        spans.append((index, index + len(sentence), sentence.strip()))
        cursor = index + len(sentence)
    return spans


def _supported_fragment_text(result: VerificationPassResult, unit_text: str) -> str | None:
    """Return original sentences whose claims are all supported.

    A fully supported unit is returned unchanged. A contradicted claim rejects
    the unit. Mixed support keeps an original sentence only when every claim
    in that sentence is supported. Supported anchors are not joined into a new
    sentence.
    """
    if result.failed or not result.assessments:
        return None
    if any(
        assessment.verdict is ClaimVerdict.CONTRADICTED
        for assessment in result.assessments
    ):
        return None
    if _all_claims_supported(result):
        return unit_text.strip()
    verdicts = {
        assessment.claim_id: assessment.verdict for assessment in result.assessments
    }
    sentences = _original_sentence_spans(unit_text)
    if not sentences:
        return None
    located: list[tuple[int, ClaimVerdict]] = []
    cursor = 0
    for claim in result.claims:
        if claim.is_fallback:
            continue
        verdict = verdicts.get(claim.claim_id)
        if verdict is None:
            return None
        index = unit_text.find(claim.anchor, cursor)
        if index < 0:
            index = unit_text.find(claim.anchor)
        if index < 0:
            return None
        located.append((index, verdict))
        cursor = index + len(claim.anchor)
    kept: list[str] = []
    for start, end, sentence in sentences:
        in_sentence = [verdict for index, verdict in located if start <= index < end]
        if not in_sentence:
            continue
        if any(verdict is not ClaimVerdict.SUPPORTED for verdict in in_sentence):
            continue
        if sentence:
            kept.append(sentence)
    if not kept:
        return None
    reduced = " ".join(kept)
    if reduced == unit_text.strip():
        return None
    return reduced


def _assembled_addition_supported(
    result: VerificationPassResult,
    prior_texts: Sequence[str],
) -> bool:
    """Accept a new sentence when only an already kept sentence's fragment flips.

    A new claim, a fallback claim, or a contradiction still rejects the addition.
    """
    if _all_claims_supported(result):
        return True
    claims = getattr(result, "claims", ())
    spans = getattr(result, "spans", ())
    if result.failed or not result.assessments or not claims or not spans:
        return False
    if any(
        assessment.verdict is ClaimVerdict.CONTRADICTED
        for assessment in result.assessments
    ):
        return False
    claims_by_id = {claim.claim_id: claim for claim in claims}
    span_text = {span.span_id: span.text.strip() for span in spans}
    prior = {text.strip() for text in prior_texts}
    for assessment in result.assessments:
        if assessment.verdict is ClaimVerdict.SUPPORTED:
            continue
        claim = claims_by_id.get(assessment.claim_id)
        if (
            claim is None
            or claim.is_fallback
            or span_text.get(claim.span_id, "") not in prior
        ):
            return False
    return True


def _effective_claim_verdict(
    claim: Claim,
    verdict: ClaimVerdict,
    bundles_by_claim: Mapping[str, EvidenceBundle],
    source_index: SourceLexicalIndex,
    *,
    strict_numbers: bool = False,
    strict_names: bool = False,
) -> ClaimVerdict:
    bundle = bundles_by_claim.get(claim.claim_id)
    if bundle is not None and claim_drop_blocked_by_omitted_required(
        claim, bundle, source_index, verdict
    ):
        return ClaimVerdict.SUPPORTED
    if bundle is not None and claim_kept_for_lenient_literals(
        claim,
        bundle,
        verdict,
        strict_numbers=strict_numbers,
        strict_names=strict_names,
    ):
        return ClaimVerdict.SUPPORTED
    return verdict


def _citation_for_promoted_claim(
    assessment: ClaimAssessment,
    bundle: EvidenceBundle | None,
) -> BatchFinding | None:
    """A source quote that lets a kept negative verdict be published as supported."""
    if bundle is None:
        return None
    examined = set(bundle.selection.examined_ids)
    quote: tuple[str, str] | None = None
    for finding in assessment.findings:
        for segment_id, text in zip(finding.evidence_ids, finding.exact_quotes, strict=True):
            if segment_id in examined and text.strip() and not text.startswith("[redacted:"):
                quote = (segment_id, text)
                break
        if quote is not None:
            break
    if quote is None:
        for passage in bundle.passages:
            if passage.segment_id in examined and passage.text.strip():
                quote = (passage.segment_id, passage.text)
                break
    if quote is None:
        return None
    return BatchFinding(
        claim_id=assessment.claim_id,
        verdict=ClaimVerdict.SUPPORTED,
        evidence_ids=(quote[0],),
        exact_quotes=(quote[1],),
    )


def _claim_can_publish(
    claim_id: str,
    *,
    claims_by_id: Mapping[str, Claim],
    verdicts: Mapping[str, ClaimVerdict],
    assessments_by_id: Mapping[str, ClaimAssessment],
    bundles_by_claim: Mapping[str, EvidenceBundle],
    source_index: SourceLexicalIndex,
    strict_numbers: bool,
    strict_names: bool,
) -> bool:
    claim = claims_by_id.get(claim_id)
    verdict = verdicts.get(claim_id)
    assessment = assessments_by_id.get(claim_id)
    if claim is None or verdict is None or assessment is None:
        return False
    effective = _effective_claim_verdict(
        claim,
        verdict,
        bundles_by_claim,
        source_index,
        strict_numbers=strict_numbers,
        strict_names=strict_names,
    )
    if effective is not ClaimVerdict.SUPPORTED:
        return False
    if verdict is ClaimVerdict.SUPPORTED:
        return True
    return (
        _citation_for_promoted_claim(assessment, bundles_by_claim.get(claim_id))
        is not None
    )


def _passing_sentence_text(
    result: VerificationResult,
    *,
    source_index: SourceLexicalIndex,
    strict_numbers: bool = False,
    strict_names: bool = False,
) -> str | None:
    """Keep sentences whose every claim passed, and drop the rest.

    Kept sentences stay in order, and a paragraph break survives wherever the
    dropped text crossed one. A contract failure has no sentence verdicts to
    trust, so nothing is kept. `None` also means nothing was dropped, which
    stops a failed draft from being checked again unchanged.
    """
    if not result.failed or not result.pass_results:
        return None
    passed = result.pass_results[-1]
    if passed.failed or not passed.spans or not passed.claims or not passed.assessments:
        return None
    verdicts = {
        assessment.claim_id: assessment.verdict for assessment in passed.assessments
    }
    claims_by_id = {claim.claim_id: claim for claim in passed.claims}
    assessments_by_id = {
        assessment.claim_id: assessment for assessment in passed.assessments
    }
    bundles_by_claim = {bundle.selection.claim_id: bundle for bundle in passed.bundles}
    claims_by_span: dict[str, list[str]] = {}
    for claim in passed.claims:
        claims_by_span.setdefault(claim.span_id, []).append(claim.claim_id)
    publish_kwargs = {
        "claims_by_id": claims_by_id,
        "verdicts": verdicts,
        "assessments_by_id": assessments_by_id,
        "bundles_by_claim": bundles_by_claim,
        "source_index": source_index,
        "strict_numbers": strict_numbers,
        "strict_names": strict_names,
    }
    pieces: list[str] = []
    kept = dropped = 0
    paragraph_break = False
    for span in passed.spans:
        text = span.text.strip()
        if not text:
            continue
        claim_ids = claims_by_span.get(span.span_id, [])
        if claim_ids and all(
            _claim_can_publish(claim_id, **publish_kwargs) for claim_id in claim_ids
        ):
            if pieces:
                pieces.append("\n\n" if paragraph_break else " ")
            pieces.append(text)
            kept += 1
            paragraph_break = bool(
                _PARAGRAPH_BREAK.search(span.text[len(span.text.rstrip()) :])
            )
        else:
            dropped += 1
            paragraph_break = paragraph_break or bool(_PARAGRAPH_BREAK.search(span.text))
    if not kept or not dropped:
        return None
    return "".join(pieces)


def _renumber_pass_ordinals(
    spans: tuple[DraftSpan, ...],
    claims: tuple[Claim, ...],
) -> tuple[tuple[DraftSpan, ...], tuple[Claim, ...]]:
    """Audit validation requires contiguous span and per-span claim ordinals."""
    if not spans or not isinstance(spans[0], DraftSpan):
        return spans, claims
    renumbered_spans = tuple(
        replace(span, ordinal=index) for index, span in enumerate(spans, start=1)
    )
    claims_by_span: dict[str, list[Claim]] = {}
    for claim in claims:
        claims_by_span.setdefault(claim.span_id, []).append(claim)
    renumbered_claims: list[Claim] = []
    for span in renumbered_spans:
        for claim_index, claim in enumerate(
            claims_by_span.get(span.span_id, ()), start=1
        ):
            renumbered_claims.append(replace(claim, ordinal=claim_index))
    return renumbered_spans, tuple(renumbered_claims)


def _subset_from_first_pass(
    result: VerificationResult,
    *,
    source_index: SourceLexicalIndex,
    strict_numbers: bool = False,
    strict_names: bool = False,
) -> VerificationResult | None:
    """Publish passing sentences from the first failed verification pass."""
    reduced = _passing_sentence_text(
        result,
        source_index=source_index,
        strict_numbers=strict_numbers,
        strict_names=strict_names,
    )
    if reduced is None:
        return None
    passed = result.pass_results[-1]
    verdicts = {
        assessment.claim_id: assessment.verdict for assessment in passed.assessments
    }
    claims_by_id = {claim.claim_id: claim for claim in passed.claims}
    assessments_by_id = {
        assessment.claim_id: assessment for assessment in passed.assessments
    }
    bundles_by_claim = {bundle.selection.claim_id: bundle for bundle in passed.bundles}
    claims_by_span: dict[str, list[str]] = {}
    for claim in passed.claims:
        claims_by_span.setdefault(claim.span_id, []).append(claim.claim_id)
    publish_kwargs = {
        "claims_by_id": claims_by_id,
        "verdicts": verdicts,
        "assessments_by_id": assessments_by_id,
        "bundles_by_claim": bundles_by_claim,
        "source_index": source_index,
        "strict_numbers": strict_numbers,
        "strict_names": strict_names,
    }

    def span_passes(span) -> bool:
        text = span.text.strip()
        if not text:
            return False
        claim_ids = claims_by_span.get(span.span_id, [])
        if not claim_ids:
            return False
        return all(_claim_can_publish(claim_id, **publish_kwargs) for claim_id in claim_ids)

    kept_spans = tuple(span for span in passed.spans if span_passes(span))
    kept_span_ids = {span.span_id for span in kept_spans}
    kept_claims = tuple(
        claim for claim in passed.claims if claim.span_id in kept_span_ids
    )
    kept_claim_ids = {claim.claim_id for claim in kept_claims}

    def published_assessment(assessment: ClaimAssessment) -> ClaimAssessment:
        if assessment.verdict is ClaimVerdict.SUPPORTED:
            return assessment
        finding = _citation_for_promoted_claim(
            assessment, bundles_by_claim.get(assessment.claim_id)
        )
        return replace(assessment, verdict=ClaimVerdict.SUPPORTED, findings=(finding,))

    kept_assessments = tuple(
        published_assessment(assessment)
        for assessment in passed.assessments
        if assessment.claim_id in kept_claim_ids
    )
    kept_bundles = tuple(
        bundle for bundle in passed.bundles if bundle.selection.claim_id in kept_claim_ids
    )
    kept_spans, kept_claims = _renumber_pass_ordinals(kept_spans, kept_claims)
    filtered_pass = VerificationPassResult(
        spans=kept_spans,
        claims=kept_claims,
        assessments=kept_assessments,
        selections=tuple(bundle.selection for bundle in kept_bundles),
        bundles=kept_bundles,
        generations=getattr(passed, "generations", ()),
        phase_generations=getattr(passed, "phase_generations", ()),
        diagnostic_codes=getattr(passed, "diagnostic_codes", ()),
    )
    return VerificationResult(
        text=reduced,
        passes=(kept_assessments,),
        selections=(filtered_pass.selections,),
        repairs=getattr(result, "repairs", ()),
        generations=getattr(result, "generations", ()),
        diagnostic_codes=(
            *getattr(result, "diagnostic_codes", ()),
            "verified_sentence_subset",
        ),
        exhausted=getattr(result, "exhausted", False),
        failed=False,
        pass_results=(filtered_pass,),
        phase_generations=getattr(result, "phase_generations", ()),
        warning_codes=(
            *getattr(result, "warning_codes", ()),
            "verified_sentence_subset",
        ),
    )


def _compression_source_text(
    root: SummaryNode,
    *,
    target_words: int,
    source_cores: Mapping[str, str],
    segments: Sequence[SourceSegment],
) -> str | None:
    """Return text to compress before editorial, or None when no pass is needed."""
    summary_words = word_count(root.summary)
    if _in_band(summary_words, target_words):
        return None
    if _above_ceiling(summary_words, target_words):
        return root.summary
    if _under_floor(summary_words, target_words):
        ordered = sorted(segments, key=lambda segment: segment.order)
        parts = [
            source_cores[segment.segment_id]
            for segment in ordered
            if segment.segment_id in source_cores
        ]
        if not parts:
            return None
        return "\n\n".join(parts)
    return None


def _prepare_root_for_editorial(
    root: SummaryNode,
    provider: ModelProvider,
    *,
    source_id: str,
    model: str,
    timeout_seconds: float,
    target_words: int,
    source_cores: Mapping[str, str],
    segments: Sequence[SourceSegment],
    strict_numbers: bool = False,
    strict_names: bool = False,
) -> tuple[SummaryNode, tuple[GenerationResult, ...]]:
    source_text = _compression_source_text(
        root,
        target_words=target_words,
        source_cores=source_cores,
        segments=segments,
    )
    if source_text is None:
        return root, ()
    coordinator = getattr(provider, "cache_coordinator", None)
    if coordinator is not None and not isinstance(coordinator, CacheCoordinator):
        raise TypeError("cache_coordinator must be a CacheCoordinator")
    compressed = compress_to_target(
        source_text,
        provider,
        source_id=source_id,
        model=model,
        timeout_seconds=timeout_seconds,
        target_words=target_words,
        coordinator=coordinator,
        strict_numbers=strict_numbers,
        strict_names=strict_names,
    )
    return root.model_copy(update={"summary": compressed.text}), compressed.generations


def _verified_content_unit_draft(
    root: SummaryNode,
    *,
    source_id: str,
    source_index: SourceLexicalIndex,
    runtime: VerificationRuntime,
    config: VerificationConfig,
    target_words: int,
    progress: VerificationProgress | None = None,
) -> tuple[str, tuple[GenerationResult, ...]]:
    """Select independently supported root assertions up to target words.

    Every root content unit is a candidate, in order, until the selection
    reaches `target_words`, so later units can fill the target when early
    ones fail. Each unit is examined once and costs at most three single
    verification passes: the unit alone, its supported-sentence reduction
    when only part of it passes, and the assembled draft with it added.
    Verification work is therefore at most 3 x len(root.content_units)
    passes, each over at most target_words plus one unit of text, and the
    root's content units are themselves bounded by its merge's output budget.
    This keeps the cost linear in the root rather than growing without limit
    while still never publishing a unit that failed.
    """
    selected: list[str] = []
    selected_words = 0
    generations: list[GenerationResult] = []

    def _verify(text: str) -> VerificationPassResult:
        result = verify_draft_once(
            text,
            source_id=source_id,
            source_index=source_index,
            runtime=runtime,
            config=config,
            pass_index=1,
            terminalize_errors=True,
            progress=progress,
        )
        generations.extend(result.generations)
        return result

    for unit in root.content_units:
        if unit.uncertain:
            continue
        result = _verify(unit.text)
        if result.failed:
            continue
        candidate = _supported_fragment_text(result, unit.text)
        if candidate is None:
            continue
        candidate_words = len(candidate.split())
        if selected_words + candidate_words > target_words:
            continue
        if candidate != unit.text.strip() and not _all_claims_supported(_verify(candidate)):
            continue
        if selected and not _assembled_addition_supported(
            _verify(" ".join((*selected, candidate))),
            selected,
        ):
            continue
        selected.append(candidate)
        selected_words += candidate_words
        if selected_words >= target_words:
            break
    return " ".join(selected), tuple(generations)


@dataclass(frozen=True)
class _EvidencePassages:
    """The verifier's source passages and where each lies in the document."""

    ids: tuple[str, ...]
    texts: Mapping[str, str]
    starts: Mapping[str, int]
    segments: tuple[SourceSegment, ...]
    parents: Mapping[str, str]


def _passage_start(segment: SourceSegment, text: str) -> int | None:
    """Locate a source-core or context passage of `segment` in the document."""
    core = segment.text[
        segment.core_start - segment.context_start : segment.core_end - segment.context_start
    ]
    if text == core:
        return segment.core_start
    if text == segment.text:
        return segment.context_start
    return None


def _fitting_ranges(
    text: str,
    *,
    source_id: str,
    counter: TokenCounter,
    max_tokens: int,
    fits: Callable[[str], bool],
) -> list[tuple[int, int, BoundaryKind]]:
    """Cover `text` with contiguous ranges whose serialized passages fit."""
    ranges: list[tuple[int, int, BoundaryKind]] = []
    for piece in segment_document(
        SourceDocument(text=text, source_id=source_id),
        counter,
        SegmentationConfig(max_tokens=max_tokens),
    ):
        start, end = piece.core_start, piece.core_end
        # JSON escaping can outgrow the raw token budget, so only a piece that
        # still does not fit is split again, with half the budget, down to
        # the minimum passage size.
        if fits(text[start:end]) or max_tokens // 2 < _MIN_PASSAGE_TOKENS:
            ranges.append((start, end, piece.boundary_kind))
            continue
        ranges.extend(
            (start + inner_start, start + inner_end, kind)
            for inner_start, inner_end, kind in _fitting_ranges(
                text[start:end],
                source_id=source_id,
                counter=counter,
                max_tokens=max_tokens // 2,
                fits=fits,
            )
        )
    return ranges


def _evidence_passages(
    *,
    root: SummaryNode,
    segments: Sequence[SourceSegment],
    source_cores: Mapping[str, str],
    counter: TokenCounter,
    config: VerificationConfig,
) -> _EvidencePassages:
    """Resolve the root's source passages, splitting any too large to cite.

    A claim's evidence bundle holds whole passages up to
    `config.evidence_tokens`, so a source core larger than that could never
    be selected. Such a core is split, like any document, into consecutive
    passages of about half the evidence budget (at least two fit a bundle),
    numbered after the last leaf segment (`S000087`, ... after `S000086`;
    `S000001`, ... for the direct strategy's `D000001`) and recorded with
    their parent segment and document offsets. Splitting only partitions
    text that was already evidence, so each bundle stays within the budget
    and the passage count stays proportional to the source length.
    """

    def fits(passage_id: str, text: str) -> bool:
        passage = serialize_source_passage(SourcePassage(passage_id, text))
        return counter.count(passage) <= config.evidence_tokens

    by_id = {segment.segment_id: segment for segment in segments}
    split_tokens = config.evidence_tokens // 2
    ids: list[str] = []
    texts: dict[str, str] = {}
    starts: dict[str, int] = {}
    passage_segments: list[SourceSegment] = []
    parents: dict[str, str] = {}
    next_order = max((segment.order for segment in segments), default=-1) + 1
    next_number = 1 + max(
        (
            int(segment.segment_id[1:])
            for segment in segments
            if segment.segment_id.startswith("S") and segment.segment_id[1:].isdigit()
        ),
        default=0,
    )
    for segment_id in dict.fromkeys(root.provenance):
        try:
            text = source_cores[segment_id]
        except KeyError as error:
            raise ValueError(f"source text is missing for segment {segment_id}") from error
        segment = by_id.get(segment_id)
        start = _passage_start(segment, text) if segment is not None else None
        if (
            segment is None
            or start is None
            or split_tokens < _MIN_PASSAGE_TOKENS
            or fits(segment_id, text)
        ):
            ids.append(segment_id)
            texts[segment_id] = text
            if start is not None:
                starts[segment_id] = start
            continue
        pieces = _fitting_ranges(
            text,
            source_id=segment.source_id,
            counter=counter,
            max_tokens=split_tokens,
            fits=partial(fits, f"S{next_number:06d}"),
        )
        for piece_start, piece_end, kind in pieces:
            passage_id = f"S{next_number:06d}"
            next_number += 1
            passage_text = text[piece_start:piece_end]
            token_count = counter.count(passage_text)
            passage_segments.append(
                SourceSegment(
                    segment_id=passage_id,
                    source_id=segment.source_id,
                    order=next_order,
                    text=passage_text,
                    core_start=start + piece_start,
                    core_end=start + piece_end,
                    context_start=start + piece_start,
                    context_end=start + piece_end,
                    core_token_count=token_count,
                    token_count=token_count,
                    leading_overlap_tokens=0,
                    trailing_overlap_tokens=0,
                    boundary_kind=kind,
                )
            )
            next_order += 1
            ids.append(passage_id)
            texts[passage_id] = passage_text
            starts[passage_id] = start + piece_start
            parents[passage_id] = segment_id
    return _EvidencePassages(
        ids=tuple(ids),
        texts=texts,
        starts=starts,
        segments=tuple(passage_segments),
        parents=parents,
    )


def _shorten(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[: limit - 1]}…"


def _sentence_failure(
    claims: Sequence[Claim], verdicts: Mapping[str, ClaimVerdict]
) -> tuple[str, str] | None:
    """Return the verdict and reason that remove a sentence, or None if it passes."""
    failing = [
        (claim, verdicts.get(claim.claim_id))
        for claim in claims
        if verdicts.get(claim.claim_id) is not ClaimVerdict.SUPPORTED
    ]
    if not claims or any(verdict is None for _, verdict in failing):
        return "unverified", _UNVERIFIED_REASON
    if not failing:
        return None
    worst = max(
        (verdict for _, verdict in failing if verdict is not None),
        key=_VERDICT_SEVERITY.__getitem__,
    )
    anchor = next(
        (
            claim.anchor
            for claim, verdict in failing
            if verdict is worst and not claim.is_fallback
        ),
        None,
    )
    whole, part = _REMOVAL_REASONS[worst]
    reason = whole if anchor is None else part.format(anchor=_shorten(anchor, 80))
    return worst.value, reason


class _SentenceLedger:
    """Every sentence verification saw, with the latest reason it failed."""

    def __init__(
        self,
        source_index: SourceLexicalIndex,
        *,
        strict_numbers: bool = False,
        strict_names: bool = False,
    ) -> None:
        self._source_index = source_index
        self._strict_numbers = strict_numbers
        self._strict_names = strict_names
        self._seen: dict[str, None] = {}
        self._failed: dict[str, tuple[str, str]] = {}

    def record(self, result: VerificationResult) -> None:
        for passed in result.pass_results:
            bundles_by_claim = {
                bundle.selection.claim_id: bundle for bundle in passed.bundles
            }
            claims_by_id = {claim.claim_id: claim for claim in passed.claims}
            verdicts = {
                assessment.claim_id: (
                    _effective_claim_verdict(
                        claims_by_id[assessment.claim_id],
                        assessment.verdict,
                        bundles_by_claim,
                        self._source_index,
                        strict_numbers=self._strict_numbers,
                        strict_names=self._strict_names,
                    )
                    if assessment.claim_id in claims_by_id
                    else assessment.verdict
                )
                for assessment in passed.assessments
            }
            claims_by_span: dict[str, list[Claim]] = {}
            for claim in passed.claims:
                claims_by_span.setdefault(claim.span_id, []).append(claim)
            for span in passed.spans:
                text = span.text.strip()
                if not text:
                    continue
                self._seen.setdefault(text, None)
                if passed.failed:
                    continue
                failure = _sentence_failure(claims_by_span.get(span.span_id, ()), verdicts)
                if failure is not None:
                    self._failed[text] = failure

    def seen(self) -> frozenset[str]:
        return frozenset(self._seen)

    def removed(
        self, published: Collection[str], *, abandoned: Collection[str] = ()
    ) -> tuple[AuditRemovedSentence, ...]:
        """Sentences left out of `published`, in the order they were first seen.

        A sentence counts once it failed a check, or when it belonged to an
        abandoned draft; the latest failing verdict wins.
        """
        removed: list[AuditRemovedSentence] = []
        for text in self._seen:
            if text in published:
                continue
            failure = self._failed.get(text)
            if failure is None:
                if text not in abandoned:
                    continue
                failure = ("unverified", _UNVERIFIED_REASON)
            verdict, reason = failure
            removed.append(AuditRemovedSentence(text=text, verdict=verdict, reason=reason))
        return tuple(removed)


def _published_sentences(
    text: str,
    verification: VerificationResult | None,
    *,
    passages: _EvidencePassages | None,
) -> tuple[AuditPublishedSentence, ...]:
    """Map every published sentence to its verdict and supporting quotations.

    With verification the sentences are the final pass's spans over exactly
    this text; a sentence with a claim that did not pass is refused, so no
    unsupported sentence can be recorded as published.
    """
    if verification is None:
        spans = split_draft_spans(text, pass_index=1)
        claims_by_span: dict[str, list[Claim]] = {}
        assessments: dict[str, ClaimAssessment] = {}
    else:
        final = verification.pass_results[-1] if verification.pass_results else None
        if final is None or final.failed or verification.text != text:
            raise FinalizationVerificationError(
                "the published text lacks its final verification pass"
            )
        spans = final.spans
        claims_by_span = {}
        for claim in final.claims:
            claims_by_span.setdefault(claim.span_id, []).append(claim)
        assessments = {assessment.claim_id: assessment for assessment in final.assessments}
    sentences: list[AuditPublishedSentence] = []
    for span in spans:
        stripped = span.text.strip()
        if not stripped:
            continue
        start = span.start + len(span.text) - len(span.text.lstrip())
        evidence: dict[tuple[str, str], AuditSentenceEvidence] = {}
        verdict = "unchecked"
        if verification is not None:
            claims = claims_by_span.get(span.span_id, [])
            if not claims or any(claim.claim_id not in assessments for claim in claims):
                raise FinalizationVerificationError("a published sentence lacks support")
            verdicts = {assessments[claim.claim_id].verdict for claim in claims}
            if not verdicts <= _PUBLISHABLE_VERDICTS:
                raise FinalizationVerificationError("a published sentence lacks support")
            verdict = (
                "supported"
                if verdicts == {ClaimVerdict.SUPPORTED}
                else "not_meaningfully_verifiable"
            )
            for claim in claims:
                assessment = assessments[claim.claim_id]
                if assessment.verdict is not ClaimVerdict.SUPPORTED:
                    continue
                for finding in assessment.findings:
                    if finding.verdict is not ClaimVerdict.SUPPORTED:
                        continue
                    for segment_id, quote in zip(
                        finding.evidence_ids, finding.exact_quotes, strict=True
                    ):
                        if (segment_id, quote) not in evidence:
                            evidence[(segment_id, quote)] = _sentence_evidence(
                                segment_id, quote, passages
                            )
        sentences.append(
            AuditPublishedSentence(
                index=len(sentences),
                paragraph=len(_PARAGRAPH_BREAK.findall(text, 0, start)),
                start=start,
                end=start + len(stripped),
                text=stripped,
                verdict=verdict,
                evidence=tuple(evidence.values()),
            )
        )
    return tuple(sentences)


def _sentence_evidence(
    segment_id: str, quote: str, passages: _EvidencePassages | None
) -> AuditSentenceEvidence:
    """Locate a verifier quotation in the canonical document when possible."""
    start = passages.starts.get(segment_id) if passages is not None else None
    offset = (
        passages.texts[segment_id].find(quote)
        if passages is not None and start is not None
        else -1
    )
    if start is None or offset < 0:
        return AuditSentenceEvidence(segment_id=segment_id, quote=quote)
    return AuditSentenceEvidence(
        segment_id=segment_id,
        quote=quote,
        start=start + offset,
        end=start + offset + len(quote),
    )


class PublicationError(RuntimeError):
    """A final output pair is absent, incomplete, or does not match its witness."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _path_key(summary_path: Path, audit_path: Path) -> tuple[str, str]:
    summary_key = str(summary_path.resolve(strict=False))
    audit_key = str(audit_path.resolve(strict=False))
    return (
        (summary_key, audit_key)
        if summary_key <= audit_key
        else (audit_key, summary_key)
    )


@contextmanager
def _publication_lock(summary_path: Path, audit_path: Path) -> Iterator[None]:
    key = _path_key(summary_path, audit_path)
    with _PUBLICATION_LOCKS_GUARD:
        lock = _PUBLICATION_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PUBLICATION_LOCKS[key] = lock
    with lock:
        yield


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _matches(path: Path, digest: str) -> bool:
    payload = _read_bytes(path)
    return payload is not None and _sha256(payload) == digest


def publish_final_output(
    result: FinalizationResult,
    *,
    summary_path: Path,
    audit_path: Path,
    session: CheckpointSession,
    atomic_replace: Callable[[Path, bytes], None] = _atomic_replace,
) -> None:
    """Publish a reliable audit before using the summary as its witness."""
    if summary_path.resolve(strict=False) == audit_path.resolve(strict=False):
        raise PublicationError("summary_path and audit_path must differ")
    if (
        result.audit is None
        or result.audit.schema_version not in {"audit/3", "audit/4"}
        or (
            result.audit.schema_version == "audit/4"
            and result.audit.reliability is None
        )
    ):
        raise PublicationError("reliable publication requires a validated reliable audit")
    audit_payload = serialize_audit(result.audit)
    summary_payload = result.text.encode("utf-8")
    audit_digest = _sha256(audit_payload)
    summary_digest = _sha256(summary_payload)
    with _publication_lock(summary_path, audit_path):
        manifest = session.manifest
        expected = (manifest.audit_sha256, manifest.summary_sha256)
        files_match = _matches(audit_path, audit_digest) and _matches(
            summary_path, summary_digest
        )
        if (
            expected == (audit_digest, summary_digest)
            and manifest.publication is PublicationState.COMPLETE
            and files_match
        ):
            return
        if (
            expected == (audit_digest, summary_digest)
            and manifest.publication is PublicationState.AUDIT_STAGED
            and files_match
        ):
            session.complete_publication()
            return

        atomic_replace(audit_path, audit_payload)
        session.stage_publication(
            audit_sha256=audit_digest, summary_sha256=summary_digest
        )
        atomic_replace(summary_path, summary_payload)
        session.complete_publication()


def read_published_summary(
    summary_path: Path, audit_path: Path, manifest: RunManifest
) -> str:
    """Read a final summary only when the manifest witnesses both exact files."""
    with _publication_lock(summary_path, audit_path):
        audit_payload = _read_bytes(audit_path)
        summary_payload = _read_bytes(summary_path)
        if (
            manifest.publication is not PublicationState.COMPLETE
            or manifest.audit_sha256 is None
            or manifest.summary_sha256 is None
            or audit_payload is None
            or summary_payload is None
            or _sha256(audit_payload) != manifest.audit_sha256
            or _sha256(summary_payload) != manifest.summary_sha256
        ):
            raise PublicationError("final output publication is incomplete")
        try:
            return summary_payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PublicationError("final summary is unreadable") from error


def _verification_runtime(
    *,
    provider: ModelProvider,
    counter: TokenCounter | None,
    model: str,
    timeout_seconds: float,
    context_window_tokens: int | None,
    injected: VerificationRuntime | None,
) -> VerificationRuntime:
    """Resolve the default dependencies or use an injected complete runtime."""
    if injected is None:
        if counter is None or context_window_tokens is None:
            raise ValueError(
                "enabled verification requires a counter and context window"
            )
        runtime = VerificationRuntime(
            provider=provider,
            counter=counter,
            model=model,
            timeout_seconds=timeout_seconds,
            context_window_tokens=context_window_tokens,
        )
    else:
        runtime = injected
    return runtime


def _build_audit(
    *,
    audit_path: Path | None,
    source_id: str,
    strategy: str,
    model: str,
    audit_configuration: Mapping[str, object] | None,
    segments: Sequence[SourceSegment],
    segment_parents: Mapping[str, str],
    nodes: Sequence[TreeNode],
    root_node_id: str,
    citations: Sequence[Citation],
    generations: Sequence[GenerationResult],
    warnings: Sequence[str],
    failures: Sequence[str],
    verification: VerificationResult | None,
    verification_enabled: bool,
    publication: AuditPublication | None,
    reliability_resume: Mapping[str, object] | None = None,
    reliability_tracker: ReliabilityTracker | None = None,
    materialize: bool,
) -> AuditArtifact | None:
    if audit_path is None:
        return None
    snapshot = reliability_tracker.snapshot() if reliability_tracker else None
    if snapshot is not None:
        reliability_resume = {
            "resumed": snapshot.resumed,
            "reused_count": snapshot.reused_count,
            "recomputed_count": snapshot.recomputed_count,
        }
    artifact = build_audit_artifact(
        source_id=source_id,
        strategy=strategy,
        model=model,
        configuration=audit_configuration or {},
        segments=segments,
        segment_parents=segment_parents,
        nodes=nodes,
        root_node_id=root_node_id,
        citations=citations,
        generations=generations,
        warnings=warnings,
        failures=failures,
        verification=verification,
        verification_enabled=verification_enabled,
        publication=publication,
        reliability_resume=reliability_resume,
        reliability_cache=snapshot.cache if snapshot is not None else None,
        reliability_attempts=snapshot.attempts if snapshot is not None else None,
    )
    if materialize:
        write_audit(audit_path, artifact)
    return artifact


@dataclass(frozen=True)
class _VerifiedPublication:
    kind: PublicationKind
    result: VerificationResult
    removed: tuple[AuditRemovedSentence, ...]
    generations: tuple[GenerationResult, ...]

    @property
    def detail(self) -> str:
        """The VERIFYING stage's closing description."""
        if self.kind != "verified_subset":
            return _COMPLETED_DETAIL[self.kind]
        count = len(self.removed)
        return f"Removed {count} unsupported sentence{'' if count == 1 else 's'}"


def _verify_publication(
    draft: str,
    root: SummaryNode,
    *,
    source_id: str,
    source_index: SourceLexicalIndex,
    runtime: VerificationRuntime,
    config: VerificationConfig,
    coordinator: CacheCoordinator | None,
    target_words: int,
    progress: VerificationProgress,
) -> _VerifiedPublication:
    """Verify the editorial draft once, else publish its passing sentences."""
    ledger = _SentenceLedger(
        source_index,
        strict_numbers=config.strict_numbers,
        strict_names=config.strict_names,
    )
    progress.phase("Checking the editorial draft")
    result = verify_and_repair(
        draft,
        source_id=source_id,
        source_index=source_index,
        runtime=runtime,
        config=config,
        coordinator=coordinator,
        progress=progress,
    )
    ledger.record(result)
    kind: PublicationKind = "editorial"
    generations: tuple[GenerationResult, ...] = ()
    if result.failed:
        progress.phase("Removing unsupported sentences")
        subset = _subset_from_first_pass(
            result,
            source_index=source_index,
            strict_numbers=config.strict_numbers,
            strict_names=config.strict_names,
        )
        if subset is not None:
            kind = "verified_subset"
            result = subset
            ledger.record(result)
    published = (
        frozenset(span.text.strip() for span in result.pass_results[-1].spans)
        if not result.failed and result.pass_results
        else frozenset()
    )
    return _VerifiedPublication(
        kind=kind,
        result=result,
        removed=ledger.removed(published),
        generations=generations,
    )


def _finalize_summary(
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
    max_output_tokens: int | None = None,
    include_citations: bool = False,
    audit_configuration: Mapping[str, object] | None = None,
    audit_path: Path | None = None,
    generations: Sequence[GenerationResult] = (),
    warnings: Sequence[str] = (),
    failures: Sequence[str] = (),
    counter: TokenCounter | None = None,
    source_cores: Mapping[str, str] | None = None,
    verification: VerificationConfig = _DEFAULT_VERIFICATION_CONFIG,
    verification_runtime: VerificationRuntime | None = None,
    verification_context_window_tokens: int | None = None,
    verification_coordinator: CacheCoordinator | None = None,
    reliability_resume: Mapping[str, object] | None = None,
    reliability_tracker: ReliabilityTracker | None = None,
    observer: RuntimeObserver | None = None,
    materialize_audit: bool,
) -> FinalizationResult:
    """Run the final editor, verify what is published, and build its audit.

    `observer` sees WRITING around the editorial call, then VERIFYING with a
    detail per phase and one event pair per assessed claim, and is polled for
    Stop before every model call. Citations come from the verifier's evidence
    or the root's own validated provenance, never from the writer, so output
    formatting cannot leave a citation dangling from the recorded sources.
    """
    runtime_observer = get_observer(observer)
    compression_generations: tuple[GenerationResult, ...] = ()
    if source_cores is not None:
        root, compression_generations = _prepare_root_for_editorial(
            root,
            provider,
            source_id=source_id,
            model=model,
            timeout_seconds=timeout_seconds,
            target_words=target_words,
            source_cores=source_cores,
            segments=segments,
            strict_numbers=verification.strict_numbers,
            strict_names=verification.strict_names,
        )
    runtime_observer.emit(StageEvent(StageName.WRITING, "active"))
    editorial = write_editorial(
        root,
        provider,
        source_id=source_id,
        model=model,
        timeout_seconds=timeout_seconds,
        target_words=target_words,
        max_output_tokens=max_output_tokens,
        observer=runtime_observer,
    )
    runtime_observer.emit(StageEvent(StageName.WRITING, "completed"))
    final_text = retain_sentences_with_missing_literals(
        root.summary,
        editorial.text,
        strict_numbers=verification.strict_numbers,
        strict_names=verification.strict_names,
    )
    audit_warnings = tuple(warnings)
    audit_segments = tuple(segments)
    passages: _EvidencePassages | None = None
    outcome: _VerifiedPublication | None = None
    if verification.enabled:
        if source_cores is None:
            raise ValueError(
                "enabled verification requires root-provenance source cores"
            )
        runtime = _verification_runtime(
            provider=provider,
            counter=counter,
            model=model,
            timeout_seconds=timeout_seconds,
            context_window_tokens=verification_context_window_tokens,
            injected=verification_runtime,
        )
        runtime_observer.raise_if_stopped("before verification")
        passages = _evidence_passages(
            root=root,
            segments=segments,
            source_cores=source_cores,
            counter=runtime.counter,
            config=verification,
        )
        audit_segments = (*segments, *passages.segments)
        outcome = _verify_publication(
            editorial.text,
            root,
            source_id=source_id,
            source_index=build_source_lexical_index(
                provenance_ids=passages.ids, source=passages.texts
            ),
            runtime=runtime,
            config=verification,
            coordinator=verification_coordinator,
            target_words=target_words,
            progress=VerificationProgress(runtime_observer),
        )
        if outcome.result.failed:
            _build_audit(
                audit_path=audit_path,
                source_id=source_id,
                strategy=strategy,
                model=model,
                audit_configuration=audit_configuration,
                segments=audit_segments,
                segment_parents=passages.parents,
                nodes=nodes,
                root_node_id=root_node_id,
                citations=(),
                generations=(
                    *generations,
                    *compression_generations,
                    editorial.generation,
                    *outcome.generations,
                ),
                warnings=audit_warnings,
                failures=failures,
                verification=outcome.result,
                verification_enabled=True,
                publication=None,
                reliability_resume=reliability_resume,
                reliability_tracker=reliability_tracker,
                materialize=True,
            )
            raise FinalizationVerificationError(
                "verification did not produce a safe final summary"
            )
        final_text = outcome.result.text
        if outcome.kind in PUBLICATION_WARNINGS:
            audit_warnings = (*audit_warnings, PUBLICATION_WARNINGS[outcome.kind])
        runtime_observer.emit(
            StageEvent(StageName.VERIFYING, "completed", detail=outcome.detail)
        )
    verification_result = outcome.result if outcome is not None else None

    citations = resolve_citations(
        citation_provenance_for_summary(
            root.provenance,
            verification_result,
            verification_enabled=verification.enabled,
        ),
        source_id=source_id,
        segments=audit_segments,
    )
    text = render_citations(final_text, citations) if include_citations else final_text

    artifact = _build_audit(
        audit_path=audit_path,
        source_id=source_id,
        strategy=strategy,
        model=model,
        audit_configuration=audit_configuration,
        segments=audit_segments,
        segment_parents=passages.parents if passages is not None else {},
        nodes=nodes,
        root_node_id=root_node_id,
        citations=citations,
        generations=(
            *generations,
            *compression_generations,
            editorial.generation,
            *(outcome.generations if outcome is not None else ()),
        ),
        warnings=audit_warnings,
        failures=failures,
        verification=verification_result,
        verification_enabled=verification.enabled,
        publication=(
            AuditPublication(
                kind=outcome.kind if outcome is not None else "editorial",
                sentences=_published_sentences(
                    final_text, verification_result, passages=passages
                ),
                removed_sentences=outcome.removed if outcome is not None else (),
            )
            if audit_path is not None
            else None
        ),
        reliability_resume=reliability_resume,
        reliability_tracker=reliability_tracker,
        materialize=materialize_audit,
    )
    return FinalizationResult(text=text, citations=citations, audit=artifact)


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
    max_output_tokens: int | None = None,
    include_citations: bool = False,
    audit_configuration: Mapping[str, object] | None = None,
    audit_path: Path | None = None,
    generations: Sequence[GenerationResult] = (),
    warnings: Sequence[str] = (),
    failures: Sequence[str] = (),
    counter: TokenCounter | None = None,
    source_cores: Mapping[str, str] | None = None,
    verification: VerificationConfig = _DEFAULT_VERIFICATION_CONFIG,
    verification_runtime: VerificationRuntime | None = None,
    verification_context_window_tokens: int | None = None,
    verification_coordinator: CacheCoordinator | None = None,
    reliability_resume: Mapping[str, object] | None = None,
    reliability_tracker: ReliabilityTracker | None = None,
    observer: RuntimeObserver | None = None,
) -> FinalizationResult:
    """Run finalization and materialize its requested audit output."""
    return _finalize_summary(
        root,
        provider,
        source_id=source_id,
        model=model,
        timeout_seconds=timeout_seconds,
        target_words=target_words,
        strategy=strategy,
        segments=segments,
        nodes=nodes,
        root_node_id=root_node_id,
        max_output_tokens=max_output_tokens,
        include_citations=include_citations,
        audit_configuration=audit_configuration,
        audit_path=audit_path,
        generations=generations,
        warnings=warnings,
        failures=failures,
        counter=counter,
        source_cores=source_cores,
        verification=verification,
        verification_runtime=verification_runtime,
        verification_context_window_tokens=verification_context_window_tokens,
        verification_coordinator=verification_coordinator,
        reliability_resume=reliability_resume,
        reliability_tracker=reliability_tracker,
        observer=observer,
        materialize_audit=True,
    )
