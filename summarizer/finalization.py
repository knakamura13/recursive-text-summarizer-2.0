"""Compose the final summary: editorial writing, verified publication, and audit.

With verification enabled, exactly one of two verified texts is published:

1. `editorial`: the editorial draft passes verification, repairs included.
2. `verified_subset`: otherwise the complete sentences the verifier supported
   in the last pass are published and the rest are dropped (warning
   `verified_sentence_subset`). With `strict_numbers` on, a rejected numbered
   sentence may first be swapped for its source sentence; the swapped draft is
   verified again, and only sentences that pass that check publish.

A sentence publishes only on the verifier's own supported verdict. Shared
names, numbers, or words never turn a rejected claim into a supported one.
When nothing passes, a failure audit is written and
`FinalizationVerificationError` is raised. The audit's `publication` records
the kind, each published sentence with its supporting quotations, every
removed sentence with its verdict, and every proposed source-sentence swap.
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
    AuditSection,
    AuditSectionCitation,
    AuditSectionPublication,
    AuditSubstitution,
    AuditSentenceEvidence,
    Citation,
    PublicationKind,
    SectionStatus,
    build_audit_artifact,
    citation_provenance_for_summary,
    render_citations,
    resolve_citations,
    serialize_audit,
    with_sections,
    write_audit,
)
from summarizer.checkpoint import (
    CheckpointSession,
    PublicationState,
    RunManifest,
)
from summarizer.budget import RequestLimits
from summarizer.compression import (
    compress_to_target,
    overlapping_numbered_source_sentences,
    retain_sentences_with_missing_literals,
    word_count,
    _above_ceiling,
    _in_band,
    _under_floor,
)
from summarizer.editorial import (
    EDITORIAL_WORK_ID,
    EditorialResult,
    SectionScope,
    section_work_id,
    write_editorial,
)
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
from summarizer.sections import PageExtent, PageLookup, SectionTree
from summarizer.segmentation import (
    BoundaryKind,
    CacheCoordinator,
    SegmentationConfig,
    SourceSegment,
    segment_document,
)
from summarizer.summaries import SummaryNode
from summarizer.text import (
    default_sentence_tokenizer,
    original_sentence_spans,
    split_unfinished_ending,
)
from summarizer.tokenization import TokenCounter
from summarizer.verification import (
    Claim,
    ClaimAssessment,
    DraftSpan,
    ClaimVerdict,
    SourceLexicalIndex,
    VerificationConfig,
    VerificationPassResult,
    VerificationProgress,
    VerificationResult,
    VerificationRuntime,
    REDACTED_QUOTE_PREFIX,
    build_source_lexical_index,
    split_draft_spans,
    verify_and_repair,
    verify_draft_once,
)

# Evidence passages are never split below this many tokens.
_MIN_PASSAGE_TOKENS = 32
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")
# Work the finalization stage plans for itself: compression passes, the
# editorial call and verification.
_FINALIZATION_WORK_ID = re.compile(r"C\d{2}K\d{6}|editorial-final|V\d+")
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
_UNFINISHED_REASON = "The draft ended before this sentence did."
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
    """Every span was decomposed and every claim was supported."""
    return (
        not result.failed
        and not getattr(result, "unresolved", ())
        and bool(result.assessments)
        and all(
            assessment.verdict is ClaimVerdict.SUPPORTED
            for assessment in result.assessments
        )
    )


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
    sentences = original_sentence_spans(unit_text)
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
    if (
        result.failed
        or getattr(result, "unresolved", ())
        or not result.assessments
        or not claims
        or not spans
    ):
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


def _span_supported(
    span_id: str,
    claims_by_span: Mapping[str, Sequence[str]],
    verdicts: Mapping[str, ClaimVerdict],
) -> bool:
    """True when the verifier rated every claim in the span supported."""
    claim_ids = claims_by_span.get(span_id, ())
    return bool(claim_ids) and all(
        verdicts.get(claim_id) is ClaimVerdict.SUPPORTED for claim_id in claim_ids
    )


def _pass_verdicts(
    passed: VerificationPassResult,
) -> tuple[dict[str, ClaimVerdict], dict[str, list[str]]]:
    verdicts = {
        assessment.claim_id: assessment.verdict for assessment in passed.assessments
    }
    claims_by_span: dict[str, list[str]] = {}
    for claim in passed.claims:
        claims_by_span.setdefault(claim.span_id, []).append(claim.claim_id)
    return verdicts, claims_by_span


def _sentence_groups(spans: Sequence[DraftSpan]) -> list[list[DraftSpan]]:
    """Draft spans grouped into sentences.

    The span splitter can break inside a sentence, for example after "U.S.";
    a span that starts in lowercase continues the sentence before it.
    """
    groups: list[list[DraftSpan]] = []
    for span in spans:
        text = span.text.strip()
        if not text:
            continue
        if groups and text[0].islower():
            groups[-1].append(span)
        else:
            groups.append([span])
    return groups


def _published_layout(
    passed: VerificationPassResult,
) -> tuple[str, list[tuple[DraftSpan, int]]] | None:
    """The text of the complete sentences the verifier supported, and where each span sits.

    A sentence publishes only when every claim in every one of its spans was
    supported. Kept sentences stay in order, and a paragraph break survives
    wherever the dropped text crossed one.
    """
    verdicts, claims_by_span = _pass_verdicts(passed)
    pieces: list[str] = []
    placed: list[tuple[DraftSpan, int]] = []
    cursor = 0
    paragraph_break = False
    for group in _sentence_groups(passed.spans):
        if not all(
            _span_supported(span.span_id, claims_by_span, verdicts) for span in group
        ):
            paragraph_break = paragraph_break or any(
                _PARAGRAPH_BREAK.search(span.text) for span in group
            )
            continue
        for position, span in enumerate(group):
            if pieces:
                separator = "\n\n" if paragraph_break and position == 0 else " "
                pieces.append(separator)
                cursor += len(separator)
            text = span.text.strip()
            placed.append((span, cursor))
            pieces.append(text)
            cursor += len(text)
        paragraph_break = bool(
            _PARAGRAPH_BREAK.search(group[-1].text[len(group[-1].text.rstrip()) :])
        )
    if not placed:
        return None
    return "".join(pieces), placed


def _passing_sentence_text(result: VerificationResult) -> str | None:
    """Keep the complete sentences the verifier supported, and drop the rest.

    A contract failure has no sentence verdicts to trust, so nothing is kept.
    """
    if not result.failed or not result.pass_results:
        return None
    passed = result.pass_results[-1]
    if passed.failed or not passed.spans or not passed.claims or not passed.assessments:
        return None
    layout = _published_layout(passed)
    return layout[0] if layout is not None else None


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


def _source_sentence_catalog(
    source_cores: Mapping[str, str] | None,
) -> list[tuple[str, str]]:
    if not source_cores:
        return []
    catalog: list[tuple[str, str]] = []
    for segment_id, core in source_cores.items():
        for sentence in default_sentence_tokenizer(core):
            text = sentence.strip()
            if text:
                catalog.append((text, segment_id))
    return catalog


@dataclass(frozen=True)
class _Substitution:
    """A rejected draft sentence and one source sentence proposed in its place."""

    original: str
    original_verdict: ClaimVerdict
    replacement: str


def _numbered_substitution_draft(
    result: VerificationResult,
    source_cores: Mapping[str, str],
) -> tuple[str, tuple[_Substitution, ...]] | None:
    """The draft with each rejected numbered sentence swapped for source sentences.

    A supported sentence stays as written and a contradicted one stays dropped.
    Another rejected sentence that shares a number and a content word with
    source sentences is replaced by them. This is only a proposal: the new
    draft must pass verification before any of it publishes.
    """
    catalog = _source_sentence_catalog(source_cores)
    if not catalog or not result.failed or not result.pass_results:
        return None
    passed = result.pass_results[-1]
    if passed.failed or not passed.spans or not passed.claims or not passed.assessments:
        return None
    verdicts, claims_by_span = _pass_verdicts(passed)
    source_sentences = [text for text, _segment_id in catalog]
    pieces: list[str] = []
    substitutions: list[_Substitution] = []
    used: set[str] = set()
    for group in _sentence_groups(passed.spans):
        text = " ".join(span.text.strip() for span in group)
        if all(_span_supported(span.span_id, claims_by_span, verdicts) for span in group):
            pieces.append(text)
            used.add(text)
            continue
        span_verdicts = [
            verdicts.get(claim_id)
            for span in group
            for claim_id in claims_by_span.get(span.span_id, ())
        ]
        if not span_verdicts or any(
            verdict is None or verdict is ClaimVerdict.CONTRADICTED
            for verdict in span_verdicts
        ):
            continue
        worst = max(
            (
                verdict
                for verdict in span_verdicts
                if verdict is not None and verdict is not ClaimVerdict.SUPPORTED
            ),
            key=_VERDICT_SEVERITY.__getitem__,
        )
        for sentence in overlapping_numbered_source_sentences(text, source_sentences):
            if sentence in used:
                continue
            used.add(sentence)
            pieces.append(sentence)
            substitutions.append(_Substitution(text, worst, sentence))
    if not substitutions:
        return None
    return " ".join(pieces), tuple(substitutions)


def _substitution_record(
    substitution: _Substitution,
    reassessed: VerificationPassResult | None,
) -> AuditSubstitution:
    """What the verifier decided about one proposed source sentence."""
    verdict = "unverified"
    if reassessed is not None:
        verdicts, claims_by_span = _pass_verdicts(reassessed)
        covering = [
            span
            for span in reassessed.spans
            if span.text.strip()
            and (
                span.text.strip() in substitution.replacement
                or substitution.replacement in span.text
            )
        ]
        if covering and all(
            _span_supported(span.span_id, claims_by_span, verdicts) for span in covering
        ):
            verdict = ClaimVerdict.SUPPORTED.value
        elif covering:
            failing = [
                verdicts.get(claim_id)
                for span in covering
                for claim_id in claims_by_span.get(span.span_id, ())
                if verdicts.get(claim_id) is not ClaimVerdict.SUPPORTED
            ]
            if failing and None not in failing:
                verdict = max(failing, key=_VERDICT_SEVERITY.__getitem__).value
    return AuditSubstitution(
        original_text=substitution.original,
        original_verdict=substitution.original_verdict.value,
        replacement_text=substitution.replacement,
        replacement_verdict=verdict,
        action="replaced" if verdict == ClaimVerdict.SUPPORTED.value else "removed",
    )


def _reassessed_substitutions(
    result: VerificationResult,
    *,
    source_cores: Mapping[str, str],
    source_id: str,
    source_index: SourceLexicalIndex,
    runtime: VerificationRuntime,
    config: VerificationConfig,
    progress: VerificationProgress,
) -> tuple[VerificationResult | None, tuple[AuditSubstitution, ...]]:
    """Verify the draft with source sentences swapped in as one more pass.

    A replacement publishes only when the verifier supports it in that pass,
    and each proposal is recorded with both verdicts and its outcome. With no
    proposal, or when the pass cannot run, nothing is returned to publish.
    """
    proposal = _numbered_substitution_draft(result, source_cores)
    if proposal is None:
        return None, ()
    draft, substitutions = proposal
    pass_index = len(result.pass_results) + 1
    reassessed: VerificationPassResult | None = None
    if pass_index <= 99:
        progress.phase("Checking source sentences")
        reassessed = verify_draft_once(
            draft,
            source_id=source_id,
            source_index=source_index,
            runtime=runtime,
            config=config,
            pass_index=pass_index,
            terminalize_errors=True,
            progress=progress,
        )
    if reassessed is None or reassessed.failed:
        return None, tuple(_substitution_record(item, None) for item in substitutions)
    records = tuple(_substitution_record(item, reassessed) for item in substitutions)
    candidate = VerificationResult(
        text=draft,
        passes=(*result.passes, reassessed.assessments),
        selections=(*result.selections, reassessed.selections),
        repairs=result.repairs,
        generations=(*result.generations, *reassessed.generations),
        diagnostic_codes=(*result.diagnostic_codes, *reassessed.diagnostic_codes),
        exhausted=result.exhausted,
        failed=True,
        pass_results=(*result.pass_results, reassessed),
        phase_generations=(*result.phase_generations, *reassessed.phase_generations),
        warning_codes=result.warning_codes,
        limitation_codes=result.limitation_codes,
        failure_codes=result.failure_codes,
    )
    return candidate, records


def _subset_from_first_pass(
    result: VerificationResult,
    *,
    changed: bool = False,
) -> VerificationResult | None:
    """Publish the sentences the verifier supported in the last pass.

    Earlier passes stay in the result so their verdicts reach the audit.
    `changed` marks a candidate whose text already differs from the editorial
    draft, so it is a subset even when every sentence passed.
    """
    if not result.failed or not result.pass_results:
        return None
    passed = result.pass_results[-1]
    if passed.failed or not passed.spans or not passed.claims or not passed.assessments:
        return None
    layout = _published_layout(passed)
    if layout is None:
        return None
    reduced, placed = layout
    # Each kept span keeps the text the verifier checked; its range moves to
    # where that text sits in the published summary.
    kept_spans = tuple(
        replace(span, start=start, end=start + len(span.text.strip()))
        if isinstance(span, DraftSpan)
        else span
        for span, start in placed
    )
    kept_span_ids = {span.span_id for span in kept_spans}
    kept_claims = tuple(
        claim for claim in passed.claims if claim.span_id in kept_span_ids
    )
    kept_claim_ids = {claim.claim_id for claim in kept_claims}
    kept_assessments = tuple(
        assessment
        for assessment in passed.assessments
        if assessment.claim_id in kept_claim_ids
    )
    kept_bundles = tuple(
        bundle for bundle in passed.bundles if bundle.selection.claim_id in kept_claim_ids
    )
    kept_spans, kept_claims = _renumber_pass_ordinals(kept_spans, kept_claims)
    dropped_a_sentence = len(kept_spans) < len(
        tuple(span for span in passed.spans if span.text.strip())
    )
    subset_codes = (
        ("verified_sentence_subset",) if dropped_a_sentence or changed else ()
    )
    filtered_pass = VerificationPassResult(
        spans=kept_spans,
        claims=kept_claims,
        assessments=kept_assessments,
        selections=tuple(bundle.selection for bundle in kept_bundles),
        bundles=kept_bundles,
        generations=getattr(passed, "generations", ()),
        phase_generations=getattr(passed, "phase_generations", ()),
        diagnostic_codes=getattr(passed, "diagnostic_codes", ()),
        # Unfinished spans and claims are never kept, so each entry names work
        # this publication withheld; the audit keeps its id and reason.
        unresolved=getattr(passed, "unresolved", ()),
    )
    prior = tuple(result.pass_results[:-1])
    return VerificationResult(
        text=reduced,
        passes=(*(item.assessments for item in prior), kept_assessments),
        selections=(*(item.selections for item in prior), filtered_pass.selections),
        repairs=getattr(result, "repairs", ()),
        generations=getattr(result, "generations", ()),
        diagnostic_codes=(
            *getattr(result, "diagnostic_codes", ()),
            *subset_codes,
        ),
        exhausted=getattr(result, "exhausted", False),
        failed=False,
        pass_results=(*prior, filtered_pass),
        phase_generations=getattr(result, "phase_generations", ()),
        warning_codes=(
            *getattr(result, "warning_codes", ()),
            *subset_codes,
        ),
    )


def _ordered_source_cores(
    source_cores: Mapping[str, str] | None,
    segments: Sequence[SourceSegment],
) -> dict[str, str]:
    """Document cores in segment order, used when a numbered paraphrase is rejected."""
    if not source_cores:
        return {}
    ordered: dict[str, str] = {}
    for segment in sorted(segments, key=lambda item: item.order):
        core = source_cores.get(segment.segment_id, "")
        if core.strip():
            ordered[segment.segment_id] = core
    return ordered


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
    limits: RequestLimits | None = None,
    reserve_work: Callable[[tuple[str, ...]], None] | None = None,
    work_prefix: str = "",
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
        limits=limits,
        reserve_work=reserve_work,
        work_prefix=work_prefix,
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

    def __init__(self) -> None:
        self._seen: dict[str, None] = {}
        self._failed: dict[str, tuple[str, str]] = {}

    def record(self, result: VerificationResult) -> None:
        for passed in result.pass_results:
            verdicts = {
                assessment.claim_id: assessment.verdict
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
                        if quote.startswith(REDACTED_QUOTE_PREFIX):
                            raise FinalizationVerificationError(
                                "a published sentence cites redacted evidence"
                            )
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
    substitutions: tuple[AuditSubstitution, ...] = ()

    @property
    def detail(self) -> str:
        """The VERIFYING stage's closing description."""
        if self.kind != "verified_subset":
            return _COMPLETED_DETAIL[self.kind]
        unfinished = sum(1 for item in self.removed if item.verdict == "unfinished")
        count = len(self.removed) - unfinished
        parts = [f"{count} unsupported sentence{'' if count == 1 else 's'}"] if count else []
        if unfinished:
            parts.append("1 unfinished sentence")
        return f"Removed {' and '.join(parts)}"


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
    source_cores: Mapping[str, str] | None = None,
    segments: Sequence[SourceSegment] = (),
    unfinished: str | None = None,
    work_id: str = "V01",
) -> _VerifiedPublication:
    """Verify the editorial draft once, else publish its passing sentences.

    With `strict_numbers` on, a rejected numbered sentence may be swapped for
    its source sentence, but only a verification pass over the new draft can
    publish the swap. `unfinished` is the editorial's cut-off last sentence,
    already removed from `draft`; it is recorded as removed.
    """
    ledger = _SentenceLedger()
    progress.phase("Checking the editorial draft")
    result = verify_and_repair(
        draft,
        source_id=source_id,
        source_index=source_index,
        runtime=runtime,
        config=config,
        coordinator=coordinator,
        progress=progress,
        work_id=work_id,
    )
    ledger.record(result)
    kind: PublicationKind = "editorial"
    generations: tuple[GenerationResult, ...] = ()
    substitutions: tuple[AuditSubstitution, ...] = ()
    if result.failed:
        progress.phase("Removing unsupported sentences")
        subset: VerificationResult | None = None
        swapped_draft_checked = False
        ordered_cores = _ordered_source_cores(source_cores, segments)
        if config.strict_numbers and ordered_cores:
            candidate, substitutions = _reassessed_substitutions(
                result,
                source_cores=ordered_cores,
                source_id=source_id,
                source_index=source_index,
                runtime=runtime,
                config=config,
                progress=progress,
            )
            if candidate is not None:
                # The swapped draft was checked, so only that newest check may
                # publish; earlier verdicts on its sentences no longer count.
                ledger.record(candidate)
                result = candidate
                subset = _subset_from_first_pass(candidate, changed=True)
                swapped_draft_checked = True
        if subset is None and not swapped_draft_checked:
            subset = _subset_from_first_pass(result)
        if subset is not None:
            result = subset
            if "verified_sentence_subset" in subset.warning_codes:
                kind = "verified_subset"
            ledger.record(result)
        else:
            substitutions = ()
    published = (
        frozenset(span.text.strip() for span in result.pass_results[-1].spans)
        if not result.failed and result.pass_results
        else frozenset()
    )
    removed = ledger.removed(published)
    if unfinished is not None and not result.failed:
        removed = (
            *removed,
            AuditRemovedSentence(
                text=unfinished, verdict="unfinished", reason=_UNFINISHED_REASON
            ),
        )
        if kind == "editorial":
            kind = "verified_subset"
            code = ("verified_sentence_subset",)
            result = replace(
                result,
                diagnostic_codes=(*result.diagnostic_codes, *code),
                warning_codes=(*result.warning_codes, *code),
            )
    return _VerifiedPublication(
        kind=kind,
        result=result,
        removed=removed,
        generations=generations,
        substitutions=substitutions,
    )


def _finalization_work_planner(
    coordinator: object,
) -> Callable[[tuple[str, ...]], None] | None:
    """Return a callback that appends finalization work to the run's plan.

    Compression, editorial and verification work is planned in the order it
    runs, each pass just before it starts, as merge levels are. The plan is
    append-only and completed work must follow it, so the number of
    compression passes need not be known in advance. On resume the recorded
    plan already holds these ids and every call is a no-op.
    """
    session = getattr(coordinator, "session", None)
    if session is None:
        return None
    plan = session.manifest.work_ids
    first = next(
        (index for index, work_id in enumerate(plan) if _FINALIZATION_WORK_ID.fullmatch(work_id)),
        len(plan),
    )
    prefix = plan[:first]
    planned: list[str] = []

    def reserve(work_ids: tuple[str, ...]) -> None:
        planned.extend(work_ids)
        session.ensure_work_prefix((*prefix, *planned))

    return reserve



@dataclass(frozen=True)
class _SectionStep:
    """What scopes one editorial-and-verification step to a section."""

    scope: SectionScope
    label: str

    @property
    def verification_work_id(self) -> str:
        return section_work_id(self.scope.section_id, "V01")

    @property
    def compression_prefix(self) -> str:
        return section_work_id(self.scope.section_id, "")


@dataclass(frozen=True)
class _Drafted:
    """A written editorial and, with verification on, its checked publication.

    `text` is None when the draft was one unfinished sentence, so nothing is
    left to verify. `root` is the record the editorial was written from,
    after any compression.
    """

    root: SummaryNode
    editorial: EditorialResult
    compression_generations: tuple[GenerationResult, ...]
    text: str | None
    passages: _EvidencePassages | None
    outcome: _VerifiedPublication | None


def _draft_and_verify(
    root: SummaryNode,
    provider: ModelProvider,
    *,
    source_id: str,
    model: str,
    timeout_seconds: float,
    target_words: int,
    segments: Sequence[SourceSegment],
    source_cores: Mapping[str, str] | None,
    request_limits: RequestLimits | None,
    counter: TokenCounter | None,
    verification: VerificationConfig,
    verification_runtime: VerificationRuntime | None,
    verification_context_window_tokens: int | None,
    verification_coordinator: CacheCoordinator | None,
    observer: RuntimeObserver,
    reserve_work: Callable[[tuple[str, ...]], None] | None,
    section: _SectionStep | None = None,
) -> _Drafted:
    """Prepare `root`, write its editorial, and verify the draft against its sources.

    The root's final summary and each section's prose take this one path. It
    plans the work, compresses toward the target, writes, takes the model's
    cut-off ending out and restores sentences that lost a literal. With
    verification on, it then builds the evidence from `root.provenance`
    inside `source_cores` only, so the caller bounds the evidence by what it
    passes, and verifies and repairs the draft. A `section` gives the work its
    own ids and its progress its label. The caller decides what a failed
    verification means.
    """
    scope = section.scope if section is not None else None
    detail = section.label if section is not None else None
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
            limits=request_limits,
            reserve_work=reserve_work,
            work_prefix=section.compression_prefix if section is not None else "",
        )
    if reserve_work is not None:
        reserve_work(
            (
                EDITORIAL_WORK_ID if section is None else section_work_id(scope.section_id, "editorial"),
                *(
                    (("V01",) if section is None else (section.verification_work_id,))
                    if verification.enabled
                    else ()
                ),
            )
        )
    observer.emit(StageEvent(StageName.WRITING, "active", detail=detail))
    editorial = write_editorial(
        root,
        provider,
        source_id=source_id,
        model=model,
        timeout_seconds=timeout_seconds,
        target_words=target_words,
        limits=request_limits,
        observer=observer,
        section=scope,
    )
    observer.emit(StageEvent(StageName.WRITING, "completed", detail=detail))
    # The cut-off ending is taken from the model's own draft, before literal
    # restoration appends source sentences that could hide where it stopped.
    editorial_text, unfinished = (
        split_unfinished_ending(editorial.text)
        if verification.enabled
        else (editorial.text, None)
    )
    if not editorial_text:
        return _Drafted(root, editorial, compression_generations, None, None, None)
    final_text = retain_sentences_with_missing_literals(
        root.summary,
        editorial_text,
        strict_numbers=verification.strict_numbers,
        strict_names=verification.strict_names,
    )
    if not verification.enabled:
        return _Drafted(root, editorial, compression_generations, final_text, None, None)
    if source_cores is None:
        raise ValueError("enabled verification requires root-provenance source cores")
    runtime = _verification_runtime(
        provider=provider,
        counter=counter,
        model=model,
        timeout_seconds=timeout_seconds,
        context_window_tokens=verification_context_window_tokens,
        injected=verification_runtime,
    )
    observer.raise_if_stopped("before verification")
    passages = _evidence_passages(
        root=root,
        segments=segments,
        source_cores=source_cores,
        counter=runtime.counter,
        config=verification,
    )
    outcome = _verify_publication(
        final_text,
        root,
        source_id=source_id,
        source_index=build_source_lexical_index(
            provenance_ids=passages.ids, source=passages.texts
        ),
        runtime=runtime,
        config=verification,
        coordinator=verification_coordinator,
        target_words=target_words,
        progress=VerificationProgress(observer, scope=detail),
        source_cores=source_cores,
        segments=segments,
        unfinished=unfinished,
        work_id="V01" if section is None else section.verification_work_id,
    )
    return _Drafted(root, editorial, compression_generations, final_text, passages, outcome)


@dataclass(frozen=True)
class SectionCitation:
    """A source segment a section's prose cites, with the pages it spans."""

    segment_id: str
    order: int
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class SectionPublication:
    """One section's reader-facing prose, checked against that section's source only.

    `status` is `verified` when verification kept the text, `unverified` when
    verification is off (the text is the written draft), and `empty` when
    nothing could be published; `reason` then says why. `segment_ids` are the
    section's own subtree segments, the only evidence its prose could draw on,
    and every citation is one of them. Pages are those of the whole subtree.
    """

    section_id: str
    node_id: str
    heading: str | None
    status: SectionStatus
    text: str
    sentences: tuple[AuditPublishedSentence, ...]
    removed_sentences: tuple[AuditRemovedSentence, ...]
    citations: tuple[SectionCitation, ...]
    segment_ids: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    target_words: int
    words: int
    kind: PublicationKind | None = None
    reason: str | None = None


def _append_work_planner(
    coordinator: object,
) -> Callable[[tuple[str, ...]], None] | None:
    """Return a callback that appends not yet planned work ids to the run's plan."""
    session = getattr(coordinator, "session", None)
    if session is None:
        return None

    def reserve(work_ids: tuple[str, ...]) -> None:
        current = session.manifest.work_ids
        fresh = tuple(item for item in dict.fromkeys(work_ids) if item not in current)
        if fresh:
            session.ensure_work_prefix((*current, *fresh))

    return reserve


def _section_label(index: int, total: int, heading: str | None) -> str:
    label = f"Section {index} of {total}"
    return label if heading is None else f"{label}, {_shorten(heading, 40)}"


def _section_citations(
    verification: VerificationResult | None,
    root: SummaryNode,
    *,
    source_id: str,
    segments: Sequence[SourceSegment],
    passages: _EvidencePassages | None,
    pages: PageLookup,
) -> tuple[SectionCitation, ...]:
    """Cite the section's own segments, a split passage by the segment it came from."""
    evidence_segments = (*segments, *(passages.segments if passages is not None else ()))
    cited = resolve_citations(
        citation_provenance_for_summary(
            root.provenance, verification, verification_enabled=verification is not None
        ),
        source_id=source_id,
        segments=evidence_segments,
    )
    parents = passages.parents if passages is not None else {}
    own = {segment.segment_id: segment for segment in segments}
    ids = dict.fromkeys(parents.get(item.segment_id, item.segment_id) for item in cited)
    return tuple(
        SectionCitation(
            segment_id=segment.segment_id,
            order=segment.order,
            page_start=pages.at(segment.core_start),
            page_end=pages.at(max(segment.core_end - 1, segment.core_start)),
        )
        for segment in sorted((own[item] for item in ids), key=lambda item: item.order)
    )


def _leaf_evidence(
    sentences: tuple[AuditPublishedSentence, ...], passages: _EvidencePassages | None
) -> tuple[AuditPublishedSentence, ...]:
    """Name each quotation's leaf segment, not the verification passage it was found in.

    A section's passages are numbered after its own segments only, so their ids
    can collide with other sections' segments; the quotation's offsets still
    locate it in the document.
    """
    if passages is None or not passages.parents:
        return sentences
    return tuple(
        sentence.model_copy(
            update={
                "evidence": tuple(
                    item.model_copy(
                        update={"segment_id": passages.parents.get(item.segment_id, item.segment_id)}
                    )
                    for item in sentence.evidence
                )
            }
        )
        for sentence in sentences
    )


def finalize_sections(
    tree: SectionTree,
    section_nodes: Mapping[str, str],
    nodes: Sequence[TreeNode],
    provider: ModelProvider,
    *,
    source_id: str,
    model: str,
    timeout_seconds: float,
    target_words: int,
    segments: Sequence[SourceSegment],
    source_cores: Mapping[str, str],
    pages: Sequence[PageExtent] | None = None,
    request_limits: RequestLimits | None = None,
    counter: TokenCounter | None = None,
    verification: VerificationConfig = _DEFAULT_VERIFICATION_CONFIG,
    verification_runtime: VerificationRuntime | None = None,
    verification_context_window_tokens: int | None = None,
    verification_coordinator: CacheCoordinator | None = None,
    observer: RuntimeObserver | None = None,
) -> dict[str, SectionPublication]:
    """Write and verify each section's prose against that section's own source.

    A section's prose is written from its node's summary, which covers the
    section's whole subtree (its own text and its subsections), and is verified
    against the source cores of the node's covered segments only: a claim that
    only another section's source supports is not supported here. Each
    section takes the editorial, verification and repair path of the final
    summary, under work ids of its own, with a target that is its share of
    `target_words`. Sentences that pass are published and the rest dropped. A
    section with no supported sentence publishes nothing and is reported with
    status `empty`, not raised. Without verification the written prose is
    published as is, marked `unverified`.

    `observer` sees WRITING and VERIFYING for every section, each detailed
    with the section's label, and is polled for Stop before every model call.
    """
    runtime_observer = get_observer(observer)
    reserve_work = _append_work_planner(getattr(provider, "cache_coordinator", None))
    by_node = {node.node_id: node for node in nodes}
    lookup = PageLookup(pages)
    ordered = [section for section in tree.nodes if section.id in section_nodes]
    publications: dict[str, SectionPublication] = {}
    for index, section in enumerate(ordered, start=1):
        runtime_observer.raise_if_stopped("before a section's prose")
        node = by_node[section_nodes[section.id]]
        covered = frozenset(node.covered_segments)
        own_segments = tuple(
            segment for segment in segments if segment.segment_id in covered
        )
        cores = {item: source_cores[item] for item in node.covered_segments}
        # The evidence is the node's provenance inside the section, so nothing
        # outside it can ever support a claim.
        provenance = tuple(
            item for item in node.summary.provenance if item in covered
        ) or node.covered_segments
        root = node.summary.model_copy(update={"provenance": provenance})
        target = tree.target_words(section.id, target_words)
        label = _section_label(index, len(ordered), section.heading)
        drafted = _draft_and_verify(
            root,
            provider,
            source_id=source_id,
            model=model,
            timeout_seconds=timeout_seconds,
            target_words=target,
            segments=own_segments,
            source_cores=cores,
            request_limits=request_limits,
            counter=counter,
            verification=verification,
            verification_runtime=verification_runtime,
            verification_context_window_tokens=verification_context_window_tokens,
            verification_coordinator=verification_coordinator,
            observer=runtime_observer,
            reserve_work=reserve_work,
            section=_SectionStep(SectionScope(section.id, section.heading), label),
        )
        subtree_pages = tree.subtree_pages(section.id)
        common = dict(
            section_id=section.id,
            node_id=node.node_id,
            heading=section.heading,
            segment_ids=tuple(segment.segment_id for segment in own_segments),
            page_start=subtree_pages[0] if subtree_pages else None,
            page_end=subtree_pages[1] if subtree_pages else None,
            target_words=target,
        )

        def empty(
            reason: str, removed: tuple[AuditRemovedSentence, ...] = ()
        ) -> SectionPublication:
            return SectionPublication(
                status="empty",
                text="",
                sentences=(),
                removed_sentences=removed,
                citations=(),
                words=0,
                reason=reason,
                **common,
            )

        if drafted.text is None:
            publications[section.id] = empty("the draft was one unfinished sentence")
            continue
        outcome = drafted.outcome
        if outcome is None:
            publications[section.id] = SectionPublication(
                status="unverified",
                text=drafted.text,
                sentences=_published_sentences(drafted.text, None, passages=None),
                removed_sentences=(),
                citations=_section_citations(
                    None,
                    drafted.root,
                    source_id=source_id,
                    segments=own_segments,
                    passages=None,
                    pages=lookup,
                ),
                words=word_count(drafted.text),
                **common,
            )
            continue
        if outcome.result.failed or not outcome.result.text.strip():
            runtime_observer.emit(
                StageEvent(StageName.VERIFYING, "completed", detail=f"{label}: nothing supported")
            )
            publications[section.id] = empty(
                "no sentence was supported by the section's source", outcome.removed
            )
            continue
        runtime_observer.emit(
            StageEvent(StageName.VERIFYING, "completed", detail=f"{label}: {outcome.detail}")
        )
        text = outcome.result.text
        publications[section.id] = SectionPublication(
            status="verified",
            text=text,
            sentences=_leaf_evidence(
                _published_sentences(text, outcome.result, passages=drafted.passages),
                drafted.passages,
            ),
            removed_sentences=outcome.removed,
            citations=_section_citations(
                outcome.result,
                drafted.root,
                source_id=source_id,
                segments=own_segments,
                passages=drafted.passages,
                pages=lookup,
            ),
            words=word_count(text),
            kind=outcome.kind,
            **common,
        )
    return publications


def _audit_section_publication(publication: SectionPublication) -> AuditSectionPublication:
    return AuditSectionPublication(
        status=publication.status,
        kind=publication.kind,
        reason=publication.reason,
        sentences=publication.sentences,
        removed_sentences=publication.removed_sentences,
        citations=tuple(
            AuditSectionCitation(
                segment_id=item.segment_id,
                order=item.order,
                page_start=item.page_start,
                page_end=item.page_end,
            )
            for item in publication.citations
        ),
        segment_ids=publication.segment_ids,
        page_start=publication.page_start,
        page_end=publication.page_end,
        target_words=publication.target_words,
        words=publication.words,
    )


def attach_section_records(
    result: FinalizationResult,
    tree: SectionTree,
    section_nodes: Mapping[str, str],
    publications: Mapping[str, SectionPublication],
    segments: Sequence[SourceSegment],
    *,
    rewrite_path: Path | None = None,
) -> FinalizationResult:
    """Add one audit record per section to the run's audit; `rewrite_path` rewrites a written file.

    Pass `rewrite_path` only when `_finalize_summary` already materialized the
    audit; a published (checkpointed) audit is written once, with the records.
    A run without an audit has nowhere to record sections and is returned as is.
    """
    if result.audit is None:
        return result
    records = tuple(
        AuditSection(
            section_id=section.id,
            heading=section.heading,
            level=section.level,
            page_start=section.page_start,
            page_end=section.page_end,
            parent_id=section.parent_id,
            child_ids=section.child_ids,
            folded_headings=section.folded_headings,
            segment_ids=tuple(
                segment.segment_id
                for segment in segments
                if section.start <= segment.core_start < section.end
            ),
            node_id=section_nodes.get(section.id),
            publication=(
                _audit_section_publication(publications[section.id])
                if section.id in publications
                else None
            ),
        )
        for section in tree.nodes
    )
    artifact = with_sections(result.audit, records)
    if rewrite_path is not None:
        write_audit(rewrite_path, artifact)
    return replace(result, audit=artifact)


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
    request_limits: RequestLimits | None = None,
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
    reserve_work = _finalization_work_planner(getattr(provider, "cache_coordinator", None))
    drafted = _draft_and_verify(
        root,
        provider,
        source_id=source_id,
        model=model,
        timeout_seconds=timeout_seconds,
        target_words=target_words,
        segments=segments,
        source_cores=source_cores,
        request_limits=request_limits,
        counter=counter,
        verification=verification,
        verification_runtime=verification_runtime,
        verification_context_window_tokens=verification_context_window_tokens,
        verification_coordinator=verification_coordinator,
        observer=runtime_observer,
        reserve_work=reserve_work,
    )
    root = drafted.root
    editorial = drafted.editorial
    compression_generations = drafted.compression_generations
    if drafted.text is None:
        _build_audit(
            audit_path=audit_path,
            source_id=source_id,
            strategy=strategy,
            model=model,
            audit_configuration=audit_configuration,
            segments=segments,
            segment_parents={},
            nodes=nodes,
            root_node_id=root_node_id,
            citations=(),
            generations=(*generations, *compression_generations, editorial.generation),
            warnings=tuple(warnings),
            failures=(*failures, "editorial_draft_unfinished"),
            verification=None,
            verification_enabled=True,
            publication=None,
            reliability_resume=reliability_resume,
            reliability_tracker=reliability_tracker,
            materialize=True,
        )
        raise FinalizationVerificationError(
            "the editorial draft is one unfinished sentence, so nothing is left to verify"
        )
    final_text = drafted.text
    audit_warnings = tuple(warnings)
    audit_segments = tuple(segments)
    passages = drafted.passages
    outcome = drafted.outcome
    if outcome is not None:
        assert passages is not None
        audit_segments = (*segments, *passages.segments)
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
                substitutions=outcome.substitutions if outcome is not None else (),
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
    request_limits: RequestLimits | None = None,
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
        request_limits=request_limits,
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
