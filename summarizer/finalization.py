"""Compose final writing, optional citations, and optional audit output."""

from __future__ import annotations

import hashlib
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from summarizer.audit import (
    _atomic_replace,
    AuditArtifact,
    Citation,
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
from summarizer.editorial import write_editorial
from summarizer.hierarchy import TreeNode
from summarizer.providers.base import GenerationResult, ModelProvider
from summarizer.reliability import ReliabilityTracker
from summarizer.segmentation import CacheCoordinator, SourceSegment
from summarizer.summaries import SummaryNode
from summarizer.text import default_sentence_tokenizer
from summarizer.tokenization import TokenCounter
from summarizer.verification import (
    ClaimVerdict,
    SourceLexicalIndex,
    VerificationConfig,
    VerificationPassResult,
    VerificationResult,
    VerificationRuntime,
    build_source_lexical_index,
    verify_and_repair,
    verify_draft_once,
)

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


def _passing_sentence_text(result: VerificationResult) -> str | None:
    """Keep sentences whose every claim passed, and drop the rest.

    A contract failure has no sentence verdicts to trust, so nothing is kept.
    The same text is not returned, which stops a failed draft from being checked again.
    """
    if not result.failed or not result.pass_results:
        return None
    passed = result.pass_results[-1]
    if passed.failed or not passed.spans or not passed.claims or not passed.assessments:
        return None
    verdicts = {
        assessment.claim_id: assessment.verdict for assessment in passed.assessments
    }
    claims_by_span: dict[str, list[str]] = {}
    for claim in passed.claims:
        claims_by_span.setdefault(claim.span_id, []).append(claim.claim_id)
    kept: list[str] = []
    for span in passed.spans:
        claim_ids = claims_by_span.get(span.span_id, [])
        if claim_ids and all(
            verdicts.get(claim_id) is ClaimVerdict.SUPPORTED for claim_id in claim_ids
        ):
            text = span.text.strip()
            if text:
                kept.append(text)
    if not kept:
        return None
    reduced = " ".join(kept)
    if reduced == result.text.strip():
        return None
    return reduced


def _recheck_fallback_draft(
    text: str,
    *,
    verify: Callable[[str], VerificationResult],
) -> VerificationResult:
    """Drop failing sentences and publish only a remainder that passes."""
    result = verify(text)
    for _ in range(7):
        if not result.failed:
            return result
        reduced = _passing_sentence_text(result)
        if reduced is None:
            return result
        result = verify(reduced)
    return result


def _verified_content_unit_draft(
    root: SummaryNode,
    *,
    source_id: str,
    source_index: SourceLexicalIndex,
    runtime: VerificationRuntime,
    config: VerificationConfig,
    target_words: int,
) -> tuple[str, tuple[GenerationResult, ...]]:
    """Select a bounded set of independently supported root assertions."""
    selected: list[str] = []
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
        )
        generations.extend(result.generations)
        return result

    for unit in root.content_units[:8]:
        if unit.uncertain:
            continue
        result = _verify(unit.text)
        if result.failed:
            continue
        candidate = _supported_fragment_text(result, unit.text)
        if candidate is None:
            continue
        if candidate != unit.text.strip() and not _all_claims_supported(_verify(candidate)):
            continue
        if selected and not _assembled_addition_supported(
            _verify(" ".join((*selected, candidate))),
            selected,
        ):
            continue
        selected.append(candidate)
        if sum(len(text.split()) for text in selected) >= target_words:
            break
    return " ".join(selected), tuple(generations)


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
    nodes: Sequence[TreeNode],
    root_node_id: str,
    citations: Sequence[Citation],
    generations: Sequence[GenerationResult],
    warnings: Sequence[str],
    failures: Sequence[str],
    verification: VerificationResult | None,
    verification_enabled: bool,
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
        nodes=nodes,
        root_node_id=root_node_id,
        citations=citations,
        generations=generations,
        warnings=warnings,
        failures=failures,
        verification=verification,
        verification_enabled=verification_enabled,
        reliability_resume=reliability_resume,
        reliability_cache=snapshot.cache if snapshot is not None else None,
        reliability_attempts=snapshot.attempts if snapshot is not None else None,
    )
    if materialize:
        write_audit(audit_path, artifact)
    return artifact


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
    verification_segments: Sequence[SourceSegment] = (),
    verification: VerificationConfig = _DEFAULT_VERIFICATION_CONFIG,
    verification_runtime: VerificationRuntime | None = None,
    verification_context_window_tokens: int | None = None,
    verification_coordinator: CacheCoordinator | None = None,
    reliability_resume: Mapping[str, object] | None = None,
    reliability_tracker: ReliabilityTracker | None = None,
    materialize_audit: bool,
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
        max_output_tokens=max_output_tokens,
    )
    verification_result: VerificationResult | None = None
    final_text = editorial.text
    fallback_generations: tuple[GenerationResult, ...] = ()
    audit_warnings = tuple(warnings)
    audit_segments = (*segments, *verification_segments)
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
        source_index = build_source_lexical_index(
            provenance_ids=(
                tuple(segment.segment_id for segment in verification_segments)
                if verification_segments else root.provenance
            ),
            source={
                **source_cores,
                **{segment.segment_id: segment.text for segment in verification_segments},
            },
        )
        verification_result = verify_and_repair(
            editorial.text,
            source_id=source_id,
            source_index=source_index,
            runtime=runtime,
            config=verification,
            coordinator=verification_coordinator,
        )
        if not verification_result.failed:
            final_text = verification_result.text
        if verification_result.failed:
            fallback_text, fallback_generations = _verified_content_unit_draft(
                root,
                source_id=source_id,
                source_index=source_index,
                runtime=runtime,
                config=verification,
                target_words=target_words,
            )
            if fallback_text:
                fallback_result = _recheck_fallback_draft(
                    fallback_text,
                    verify=lambda candidate: verify_and_repair(
                        candidate,
                        source_id=source_id,
                        source_index=source_index,
                        runtime=runtime,
                        config=verification,
                        coordinator=verification_coordinator,
                    ),
                )
                verification_result = fallback_result
                if not fallback_result.failed:
                    final_text = fallback_result.text
                    audit_warnings = (*audit_warnings, "verified_content_unit_fallback")
            if verification_result.failed:
                audit_warnings = (*audit_warnings, "verified_content_unit_fallback_failed")
        if verification_result.failed:
            _build_audit(
                audit_path=audit_path,
                source_id=source_id,
                strategy=strategy,
                model=model,
                audit_configuration=audit_configuration,
                segments=audit_segments,
                nodes=nodes,
                root_node_id=root_node_id,
                citations=(),
                generations=(*generations, editorial.generation, *fallback_generations),
                warnings=audit_warnings,
                failures=failures,
                verification=verification_result,
                verification_enabled=True,
                reliability_resume=reliability_resume,
                reliability_tracker=reliability_tracker,
                materialize=True,
            )
            raise FinalizationVerificationError(
                "verification did not produce a safe final summary"
            )

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
        nodes=nodes,
        root_node_id=root_node_id,
        citations=citations,
        generations=(*generations, editorial.generation, *fallback_generations),
        warnings=audit_warnings,
        failures=failures,
        verification=verification_result,
        verification_enabled=verification.enabled,
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
        materialize_audit=True,
    )
