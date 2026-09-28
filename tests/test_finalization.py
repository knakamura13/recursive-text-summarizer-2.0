import hashlib
import json
from types import SimpleNamespace

import pytest

from summarizer.direct import whole_document_segment
from summarizer.finalization import (
    FinalizationVerificationError,
    _subset_from_first_pass,
    _verified_content_unit_draft,
    finalize_summary,
)
from summarizer.hierarchy import TreeNode
from summarizer.ingestion import ingest_text
from summarizer.providers.base import GenerationResult
from summarizer.summaries import SummaryNode
from summarizer.tokenization import ConservativeUtf8TokenCounter
from summarizer.verification import (
    BatchFinding,
    Claim,
    ClaimAssessment,
    ClaimVerdict,
    DraftSpan,
    VerificationConfig,
    VerificationPassResult,
    VerificationResult,
    build_source_lexical_index,
    select_claim_evidence,
)


class ScriptedProvider:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if request.operation_id == "editorial-final":
            payload = {"text": "The source fact is correct."}
        elif request.operation_id == "verification-decompose:V01":
            payload = {"spans": [{"span_id": "V01S000001", "anchors": []}]}
        else:
            payload = {
                "findings": [
                    {
                        "claim_id": "V01C000001",
                        "verdict": "supported",
                        "evidence": [
                            {"segment_id": "D000001", "exact_quote": "source fact"}
                        ],
                    }
                ]
            }
        return GenerationResult(json.dumps(payload), "fake", request.model)


def test_finalize_summary_resolves_verification_runtime_without_injection() -> None:
    counter = ConservativeUtf8TokenCounter()
    document = ingest_text("The source fact is correct.")
    segment = whole_document_segment(document, counter)
    summary = SummaryNode.model_validate(
        {
            "summary": "The source fact is correct.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [segment.segment_id],
            "level": 0,
        }
    )
    root = TreeNode("L0N0001", 0, 0, summary, (), (segment.segment_id,))
    provider = ScriptedProvider()

    result = finalize_summary(
        summary,
        provider,
        source_id=document.source_id,
        model="test-model",
        timeout_seconds=30,
        target_words=20,
        strategy="direct",
        segments=(segment,),
        nodes=(root,),
        root_node_id=root.node_id,
        counter=counter,
        source_cores={segment.segment_id: segment.text},
        verification=VerificationConfig(enabled=True),
        verification_context_window_tokens=10_000,
    )

    assert result.text == "The source fact is correct."
    assert [request.operation_id for request in provider.requests] == [
        "editorial-final",
        "verification-decompose:V01",
        "verification-classify:V01",
    ]


def test_finalize_keeps_an_editorial_paraphrase_when_verification_is_off() -> None:
    counter = ConservativeUtf8TokenCounter()
    document = ingest_text("The crew counted 16 divers on the reef.")
    segment = whole_document_segment(document, counter)
    summary = SummaryNode.model_validate(
        {
            "summary": document.text,
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [segment.segment_id],
            "level": 0,
        }
    )
    root = TreeNode("L0N0001", 0, 0, summary, (), (segment.segment_id,))

    class Provider:
        def generate(self, request):
            return GenerationResult(
                json.dumps({"text": "The event impacted sixteen divers on the reef."}),
                "fake",
                request.model,
            )

    result = finalize_summary(
        summary,
        Provider(),
        source_id=document.source_id,
        model="test-model",
        timeout_seconds=30,
        target_words=8,
        strategy="direct",
        segments=(segment,),
        nodes=(root,),
        root_node_id=root.node_id,
        counter=counter,
        source_cores={segment.segment_id: document.text},
    )

    assert result.text == "The event impacted sixteen divers on the reef."


def test_verification_sees_the_editorial_paraphrase(monkeypatch) -> None:
    counter = ConservativeUtf8TokenCounter()
    document = ingest_text("The crew counted 16 divers on the reef.")
    segment = whole_document_segment(document, counter)
    summary = SummaryNode.model_validate(
        {
            "summary": document.text,
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [segment.segment_id],
            "level": 0,
        }
    )
    root = TreeNode("L0N0001", 0, 0, summary, (), (segment.segment_id,))
    checked: list[str] = []

    def verify(text, **kwargs):
        checked.append(text)
        return VerificationResult(
            text=text,
            passes=(),
            selections=(),
            repairs=(),
            generations=(),
            diagnostic_codes=(),
            exhausted=False,
            failed=False,
        )

    monkeypatch.setattr("summarizer.finalization.verify_and_repair", verify)

    class Provider:
        def generate(self, request):
            return GenerationResult(
                json.dumps({"text": "The event impacted sixteen divers on the reef."}),
                "fake",
                request.model,
            )

    result = finalize_summary(
        summary,
        Provider(),
        source_id=document.source_id,
        model="test-model",
        timeout_seconds=30,
        target_words=8,
        strategy="direct",
        segments=(segment,),
        nodes=(root,),
        root_node_id=root.node_id,
        counter=counter,
        source_cores={segment.segment_id: document.text},
        verification=VerificationConfig(enabled=True),
        verification_context_window_tokens=10_000,
    )

    assert checked == ["The event impacted sixteen divers on the reef."]
    assert result.text == "The event impacted sixteen divers on the reef."


def test_without_strict_numbers_a_rejected_numbered_paraphrase_is_not_swapped_for_the_source(
    monkeypatch,
) -> None:
    counter = ConservativeUtf8TokenCounter()
    document = ingest_text("The crew counted 16 divers on the reef.")
    segment = whole_document_segment(document, counter)
    summary = SummaryNode.model_validate(
        {
            "summary": document.text,
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [segment.segment_id],
            "level": 0,
        }
    )
    root = TreeNode("L0N0001", 0, 0, summary, (), (segment.segment_id,))
    paraphrase = "The event impacted sixteen divers on the reef."
    checked: list[str] = []

    def verify(text, **kwargs):
        checked.append(text)
        span = _draft_span("V01S000001", 1, text, 0)
        claim = Claim("V01C000001", "V01S000001", 1, text.strip(), False)
        index = build_source_lexical_index(
            provenance_ids=(segment.segment_id,),
            source={segment.segment_id: document.text},
        )
        bundle = select_claim_evidence(
            claim,
            source_index=index,
            counter=counter,
            max_tokens=1000,
        )
        assessment = ClaimAssessment(
            claim_id=claim.claim_id,
            verdict=ClaimVerdict.INSUFFICIENTLY_SUPPORTED,
            findings=(
                BatchFinding(
                    claim_id=claim.claim_id,
                    verdict=ClaimVerdict.INSUFFICIENTLY_SUPPORTED,
                    evidence_ids=(),
                    exact_quotes=(),
                ),
            ),
            pass_index=1,
            verifier_provider="test",
            verifier_model="test",
            prompt_version="verification-classification/1",
        )
        passed = VerificationPassResult(
            spans=(span,),
            claims=(claim,),
            assessments=(assessment,),
            selections=(bundle.selection,),
            bundles=(bundle,),
            generations=(),
            phase_generations=(),
            diagnostic_codes=(),
            failed=False,
        )
        return VerificationResult(
            text=text,
            passes=((assessment,),),
            selections=((bundle.selection,),),
            repairs=(),
            generations=(),
            diagnostic_codes=("insufficient_support",),
            exhausted=False,
            failed=True,
            pass_results=(passed,),
        )

    monkeypatch.setattr("summarizer.finalization.verify_and_repair", verify)

    class Provider:
        def generate(self, request):
            return GenerationResult(
                json.dumps({"text": paraphrase}),
                "fake",
                request.model,
            )

    with pytest.raises(FinalizationVerificationError):
        finalize_summary(
            summary,
            Provider(),
            source_id=document.source_id,
            model="test-model",
            timeout_seconds=30,
            target_words=8,
            strategy="direct",
            segments=(segment,),
            nodes=(root,),
            root_node_id=root.node_id,
            counter=counter,
            source_cores={segment.segment_id: document.text},
            verification=VerificationConfig(enabled=True),
            verification_context_window_tokens=10_000,
        )

    assert checked == [paraphrase]


def test_failed_editorial_without_passing_sentences_stays_unpublished(tmp_path) -> None:
    class FallbackProvider:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            if request.operation_id == "editorial-final":
                payload = {"text": "An unsupported outcome occurred."}
            elif request.operation_id == "verification-decompose:V01":
                payload = {"spans": [{"span_id": "V01S000001", "anchors": []}]}
            else:
                claim_text = json.loads(request.input_text.splitlines()[1])["claims"][0]["span_text"]
                supported = claim_text == "The source fact is correct."
                payload = {
                    "findings": [{
                        "claim_id": "V01C000001",
                        "verdict": "supported" if supported else "insufficiently_supported",
                        "evidence": (
                            [{"segment_id": "D000001", "exact_quote": "source fact"}]
                            if supported else []
                        ),
                    }]
                }
            return GenerationResult(json.dumps(payload), "fake", request.model)

    counter = ConservativeUtf8TokenCounter()
    document = ingest_text("The source fact is correct.")
    segment = whole_document_segment(document, counter)
    summary = SummaryNode.model_validate({
        "summary": "The source fact is correct.",
        "content_units": [{
            "text": "The source fact is correct.",
            "kind": "fact",
            "evidence": [{"segment_id": segment.segment_id, "quote": "source fact"}],
            "qualification": None,
            "uncertain": False,
        }],
        "entities": [], "qualifications": [], "contradictions": [],
        "quotations": [], "provenance": [segment.segment_id], "level": 0,
    })
    root = TreeNode("L0N0001", 0, 0, summary, (), (segment.segment_id,))
    provider = FallbackProvider()

    with pytest.raises(FinalizationVerificationError):
        finalize_summary(
            summary, provider,
            source_id=document.source_id,
            model="test-model",
            timeout_seconds=30,
            target_words=5,
            strategy="direct",
            segments=(segment,),
            nodes=(root,),
            root_node_id=root.node_id,
            audit_path=tmp_path / "audit.json",
            counter=counter,
            source_cores={segment.segment_id: segment.text},
            verification=VerificationConfig(enabled=True),
            verification_context_window_tokens=10_000,
        )


def test_failed_editorial_without_supported_units_stays_unpublished(tmp_path) -> None:
    class RejectingProvider(ScriptedProvider):
        def generate(self, request):
            self.requests.append(request)
            if request.operation_id == "editorial-final":
                payload = {"text": "Unsupported. Also unsupported."}
            elif request.operation_id.startswith("verification-decompose:"):
                spans = json.loads(request.input_text.splitlines()[1])
                payload = {"spans": [{"span_id": span["span_id"], "anchors": []} for span in spans]}
            else:
                claims = json.loads(request.input_text.splitlines()[1])["claims"]
                payload = {"findings": [
                    {"claim_id": claim["claim_id"], "verdict": "insufficiently_supported", "evidence": []}
                    for claim in claims
                ]}
            return GenerationResult(json.dumps(payload), "fake", request.model)

    counter = ConservativeUtf8TokenCounter()
    document = ingest_text("The source fact is correct.")
    segment = whole_document_segment(document, counter)
    summary = SummaryNode.model_validate({
        "summary": "Unsupported.",
        "content_units": [{"text": "Unsupported.", "kind": "fact", "evidence": [], "qualification": None, "uncertain": False}],
        "entities": [], "qualifications": [], "contradictions": [],
        "quotations": [], "provenance": [segment.segment_id], "level": 0,
    })
    root = TreeNode("L0N0001", 0, 0, summary, (), (segment.segment_id,))
    audit_path = tmp_path / "audit.json"
    with pytest.raises(FinalizationVerificationError):
        finalize_summary(
            summary, RejectingProvider(),
            source_id=document.source_id,
            model="test-model",
            timeout_seconds=30,
            target_words=1,
            strategy="direct",
            segments=(segment,),
            nodes=(root,),
            root_node_id=root.node_id,
            audit_path=audit_path,
            counter=counter,
            source_cores={segment.segment_id: segment.text},
            verification=VerificationConfig(enabled=True),
            verification_context_window_tokens=10_000,
        )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert "publication" not in audit
    assert audit["verification"]["failed"] is True
    assert audit["warnings"] == []
    assert audit["citations"] == []


def test_fallback_skips_unit_that_breaks_combined_verification(monkeypatch) -> None:
    summary = SummaryNode.model_validate({
        "summary": "First fact. Second fact. Third fact.",
        "content_units": [
            {"text": text, "kind": "fact", "evidence": [], "qualification": None, "uncertain": False}
            for text in ("First fact.", "Second fact.", "Third fact.")
        ],
        "entities": [], "qualifications": [], "contradictions": [],
        "quotations": [], "provenance": ["D000001"], "level": 0,
    })
    checked = []

    def verify(text, **kwargs):
        checked.append(text)
        verdict = (
            ClaimVerdict.INSUFFICIENTLY_SUPPORTED
            if text == "First fact. Second fact."
            else ClaimVerdict.SUPPORTED
        )
        return SimpleNamespace(
            failed=False,
            assessments=(SimpleNamespace(verdict=verdict),),
            generations=(),
        )

    monkeypatch.setattr("summarizer.finalization.verify_draft_once", verify)
    draft, _ = _verified_content_unit_draft(
        summary,
        source_id="a" * 64,
        source_index=None,
        runtime=None,
        config=VerificationConfig(enabled=True),
        target_words=100,
    )

    assert draft == "First fact. Third fact."
    assert "First fact. Second fact." in checked
    assert "First fact. Third fact." in checked


def _verdict(claim_id: str, verdict: ClaimVerdict, *, fallback: bool, anchor: str):
    return (
        SimpleNamespace(claim_id=claim_id, is_fallback=fallback, anchor=anchor),
        SimpleNamespace(claim_id=claim_id, verdict=verdict),
    )


def test_fallback_drops_a_mixed_sentence_instead_of_joining_fragments(monkeypatch) -> None:
    unit = (
        "Elyse Saugstad, a professional skier, deployed her airbag "
        "but remained conscious."
    )
    summary = SummaryNode.model_validate({
        "summary": unit,
        "content_units": [
            {"text": unit, "kind": "fact", "evidence": [], "qualification": None, "uncertain": False}
        ],
        "entities": [], "qualifications": [], "contradictions": [],
        "quotations": [], "provenance": ["D000001"], "level": 0,
    })
    checked = []

    def verify(text, **kwargs):
        checked.append(text)
        claims = []
        assessments = []
        for claim_id, verdict, fallback, anchor in (
            ("V01C000001", ClaimVerdict.SUPPORTED, False, "Elyse Saugstad"),
            ("V01C000002", ClaimVerdict.INSUFFICIENTLY_SUPPORTED, False, "professional skier"),
            ("V01C000003", ClaimVerdict.SUPPORTED, False, "deployed her airbag"),
            ("V01C000004", ClaimVerdict.INSUFFICIENTLY_SUPPORTED, False, "conscious"),
            ("V01C000005", ClaimVerdict.SUPPORTED, True, unit),
        ):
            claim, assessment = _verdict(claim_id, verdict, fallback=fallback, anchor=anchor)
            claims.append(claim)
            assessments.append(assessment)
        return SimpleNamespace(
            failed=False, assessments=tuple(assessments), claims=tuple(claims), generations=()
        )

    monkeypatch.setattr("summarizer.finalization.verify_draft_once", verify)
    draft, _ = _verified_content_unit_draft(
        summary,
        source_id="a" * 64,
        source_index=None,
        runtime=None,
        config=VerificationConfig(enabled=True),
        target_words=100,
    )

    assert draft == ""
    assert checked == [unit]


def test_fallback_keeps_a_supported_sentence_beside_a_mixed_sentence(monkeypatch) -> None:
    kept = "The avalanche slab was approximately 200 feet wide."
    mixed = "Elyse Saugstad, a professional skier, deployed her airbag but remained conscious."
    unit = f"{kept} {mixed}"
    summary = SummaryNode.model_validate({
        "summary": unit,
        "content_units": [
            {"text": unit, "kind": "fact", "evidence": [], "qualification": None, "uncertain": False}
        ],
        "entities": [], "qualifications": [], "contradictions": [],
        "quotations": [], "provenance": ["D000001"], "level": 0,
    })
    checked = []

    def verify(text, **kwargs):
        checked.append(text)
        if text == unit:
            claims = []
            assessments = []
            for claim_id, verdict, fallback, anchor in (
                ("V01C000001", ClaimVerdict.SUPPORTED, False, "200 feet wide"),
                ("V01C000002", ClaimVerdict.SUPPORTED, False, "Elyse Saugstad"),
                ("V01C000003", ClaimVerdict.INSUFFICIENTLY_SUPPORTED, False, "professional skier"),
                ("V01C000004", ClaimVerdict.SUPPORTED, False, "deployed her airbag"),
                ("V01C000005", ClaimVerdict.INSUFFICIENTLY_SUPPORTED, False, "conscious"),
                ("V01C000006", ClaimVerdict.SUPPORTED, True, unit),
            ):
                claim, assessment = _verdict(claim_id, verdict, fallback=fallback, anchor=anchor)
                claims.append(claim)
                assessments.append(assessment)
            return SimpleNamespace(
                failed=False, assessments=tuple(assessments), claims=tuple(claims), generations=()
            )
        return SimpleNamespace(
            failed=False,
            assessments=(SimpleNamespace(verdict=ClaimVerdict.SUPPORTED),),
            claims=(),
            generations=(),
        )

    monkeypatch.setattr("summarizer.finalization.verify_draft_once", verify)
    draft, _ = _verified_content_unit_draft(
        summary,
        source_id="a" * 64,
        source_index=None,
        runtime=None,
        config=VerificationConfig(enabled=True),
        target_words=100,
    )

    assert draft == kept
    assert checked == [unit, kept]


def test_fallback_drops_a_fragment_that_fails_recheck(monkeypatch) -> None:
    unit = "Supported clause but conscious."
    summary = SummaryNode.model_validate({
        "summary": unit,
        "content_units": [
            {"text": unit, "kind": "fact", "evidence": [], "qualification": None, "uncertain": False}
        ],
        "entities": [], "qualifications": [], "contradictions": [],
        "quotations": [], "provenance": ["D000001"], "level": 0,
    })

    def verify(text, **kwargs):
        if text == unit:
            claims = []
            assessments = []
            for claim_id, verdict, fallback, anchor in (
                ("V01C000001", ClaimVerdict.SUPPORTED, False, "Supported clause"),
                ("V01C000002", ClaimVerdict.INSUFFICIENTLY_SUPPORTED, False, "conscious"),
                ("V01C000003", ClaimVerdict.SUPPORTED, True, unit),
            ):
                claim, assessment = _verdict(claim_id, verdict, fallback=fallback, anchor=anchor)
                claims.append(claim)
                assessments.append(assessment)
            return SimpleNamespace(
                failed=False, assessments=tuple(assessments), claims=tuple(claims), generations=()
            )
        return SimpleNamespace(
            failed=False,
            assessments=(SimpleNamespace(verdict=ClaimVerdict.INSUFFICIENTLY_SUPPORTED),),
            claims=(),
            generations=(),
        )

    monkeypatch.setattr("summarizer.finalization.verify_draft_once", verify)
    draft, _ = _verified_content_unit_draft(
        summary,
        source_id="a" * 64,
        source_index=None,
        runtime=None,
        config=VerificationConfig(enabled=True),
        target_words=100,
    )

    assert draft == ""


def test_fallback_continues_after_a_failed_unit(monkeypatch) -> None:
    summary = SummaryNode.model_validate({
        "summary": "Broken fact. Later fact.",
        "content_units": [
            {"text": text, "kind": "fact", "evidence": [], "qualification": None, "uncertain": False}
            for text in ("Broken fact.", "Later fact.")
        ],
        "entities": [], "qualifications": [], "contradictions": [],
        "quotations": [], "provenance": ["D000001"], "level": 0,
    })

    def verify(text, **kwargs):
        failed = text == "Broken fact."
        return SimpleNamespace(
            failed=failed,
            assessments=() if failed else (SimpleNamespace(verdict=ClaimVerdict.SUPPORTED),),
            claims=(),
            generations=(),
        )

    monkeypatch.setattr("summarizer.finalization.verify_draft_once", verify)
    draft, _ = _verified_content_unit_draft(
        summary,
        source_id="a" * 64,
        source_index=None,
        runtime=None,
        config=VerificationConfig(enabled=True),
        target_words=100,
    )

    assert draft == "Later fact."


def test_fallback_drops_a_unit_with_a_contradicted_fragment(monkeypatch) -> None:
    unit = "True clause and a contradicted death."
    summary = SummaryNode.model_validate({
        "summary": unit,
        "content_units": [
            {"text": unit, "kind": "fact", "evidence": [], "qualification": None, "uncertain": False}
        ],
        "entities": [], "qualifications": [], "contradictions": [],
        "quotations": [], "provenance": ["D000001"], "level": 0,
    })
    checked = []

    def verify(text, **kwargs):
        checked.append(text)
        claims = []
        assessments = []
        for claim_id, verdict, fallback, anchor in (
            ("V01C000001", ClaimVerdict.SUPPORTED, False, "True clause"),
            ("V01C000002", ClaimVerdict.CONTRADICTED, False, "contradicted death"),
            ("V01C000003", ClaimVerdict.SUPPORTED, True, unit),
        ):
            claim, assessment = _verdict(claim_id, verdict, fallback=fallback, anchor=anchor)
            claims.append(claim)
            assessments.append(assessment)
        return SimpleNamespace(
            failed=False, assessments=tuple(assessments), claims=tuple(claims), generations=()
        )

    monkeypatch.setattr("summarizer.finalization.verify_draft_once", verify)
    draft, _ = _verified_content_unit_draft(
        summary,
        source_id="a" * 64,
        source_index=None,
        runtime=None,
        config=VerificationConfig(enabled=True),
        target_words=100,
    )

    assert draft == ""
    assert checked == [unit]


def _units(*texts: str) -> SummaryNode:
    return SummaryNode.model_validate({
        "summary": " ".join(texts),
        "content_units": [
            {"text": text, "kind": "fact", "evidence": [], "qualification": None, "uncertain": False}
            for text in texts
        ],
        "entities": [], "qualifications": [], "contradictions": [],
        "quotations": [], "provenance": ["D000001"], "level": 0,
    })


def test_fallback_uses_later_units_without_exceeding_target_words(monkeypatch) -> None:
    def verify(text, **kwargs):
        return SimpleNamespace(
            failed=False,
            assessments=(SimpleNamespace(verdict=ClaimVerdict.SUPPORTED),),
            claims=(),
            spans=(),
            generations=(),
        )

    monkeypatch.setattr("summarizer.finalization.verify_draft_once", verify)
    draft, _ = _verified_content_unit_draft(
        _units(
            "One two three four.",
            "Five six.",
            "Seven eight.",
            "Nine.",
        ),
        source_id="a" * 64,
        source_index=None,
        runtime=None,
        config=VerificationConfig(enabled=True),
        target_words=3,
    )

    assert draft == "Five six. Nine."
    assert len(draft.split()) == 3


def test_fallback_considers_root_units_after_the_first_eight(monkeypatch) -> None:
    last = "Ninth fact."

    def verify(text, **kwargs):
        verdict = (
            ClaimVerdict.SUPPORTED if text == last else ClaimVerdict.INSUFFICIENTLY_SUPPORTED
        )
        return SimpleNamespace(
            failed=text != last,
            assessments=(SimpleNamespace(verdict=verdict),),
            claims=(),
            spans=(),
            generations=(),
        )

    monkeypatch.setattr("summarizer.finalization.verify_draft_once", verify)
    draft, _ = _verified_content_unit_draft(
        _units(*(f"{ordinal}th fact." for ordinal in range(1, 9)), last),
        source_id="a" * 64,
        source_index=None,
        runtime=None,
        config=VerificationConfig(enabled=True),
        target_words=100,
    )

    assert draft == last


def _pass_result(rows: tuple[tuple[str, ClaimVerdict, bool, str, str], ...]):
    claims = []
    assessments = []
    spans = []
    span_ids: dict[str, str] = {}
    for claim_id, verdict, fallback, anchor, span_text in rows:
        span_id = span_ids.setdefault(span_text, f"V01S{len(span_ids) + 1:06d}")
        claims.append(SimpleNamespace(
            claim_id=claim_id, is_fallback=fallback, anchor=anchor, span_id=span_id,
        ))
        assessments.append(SimpleNamespace(claim_id=claim_id, verdict=verdict))
    for span_text, span_id in span_ids.items():
        spans.append(SimpleNamespace(span_id=span_id, text=span_text))
    return SimpleNamespace(
        failed=False,
        assessments=tuple(assessments),
        claims=tuple(claims),
        spans=tuple(spans),
        generations=(),
    )


def test_fallback_adds_a_supported_sentence_when_an_old_fragment_flips(monkeypatch) -> None:
    first = "A massive avalanche occurred in Tunnel Creek on February 19."
    second = "The slab was 200 feet wide."
    checked = []

    def verify(text, **kwargs):
        checked.append(text)
        if text == first or text == second:
            return SimpleNamespace(
                failed=False,
                assessments=(SimpleNamespace(verdict=ClaimVerdict.SUPPORTED),),
                claims=(),
                spans=(),
                generations=(),
            )
        return _pass_result((
            ("V01C000001", ClaimVerdict.INSUFFICIENTLY_SUPPORTED, False, "February 19", first),
            ("V01C000002", ClaimVerdict.SUPPORTED, True, first, first),
            ("V01C000003", ClaimVerdict.SUPPORTED, True, second, second),
        ))

    monkeypatch.setattr("summarizer.finalization.verify_draft_once", verify)
    draft, _ = _verified_content_unit_draft(
        _units(first, second),
        source_id="a" * 64,
        source_index=None,
        runtime=None,
        config=VerificationConfig(enabled=True),
        target_words=100,
    )

    assert draft == f"{first} {second}"
    assert checked[-1] == f"{first} {second}"


def test_fallback_rejects_a_new_insufficient_fragment(monkeypatch) -> None:
    first = "The slab was 200 feet wide."
    second = "The group had socialized the previous night."

    def verify(text, **kwargs):
        if text == first or text == second:
            return SimpleNamespace(
                failed=False,
                assessments=(SimpleNamespace(verdict=ClaimVerdict.SUPPORTED),),
                claims=(),
                spans=(),
                generations=(),
            )
        return _pass_result((
            ("V01C000001", ClaimVerdict.SUPPORTED, True, first, first),
            ("V01C000002", ClaimVerdict.INSUFFICIENTLY_SUPPORTED, False, "socialized the previous night", second),
            ("V01C000003", ClaimVerdict.SUPPORTED, True, second, second),
        ))

    monkeypatch.setattr("summarizer.finalization.verify_draft_once", verify)
    draft, _ = _verified_content_unit_draft(
        _units(first, second),
        source_id="a" * 64,
        source_index=None,
        runtime=None,
        config=VerificationConfig(enabled=True),
        target_words=100,
    )

    assert draft == first


def _sentence_pass(span_id: str, text: str, verdicts: tuple[ClaimVerdict, ...]):
    claims = tuple(
        SimpleNamespace(
            claim_id=f"{span_id}C{index}",
            span_id=span_id,
            is_fallback=index == len(verdicts),
        )
        for index, _ in enumerate(verdicts, start=1)
    )
    return (
        SimpleNamespace(span_id=span_id, text=text),
        claims,
        tuple(
            SimpleNamespace(claim_id=claim.claim_id, verdict=verdict)
            for claim, verdict in zip(claims, verdicts, strict=True)
        ),
    )


def _failed_draft(text: str, sentences: tuple[tuple[str, tuple[ClaimVerdict, ...]], ...]):
    spans = []
    claims = []
    assessments = []
    for index, (sentence, verdicts) in enumerate(sentences, start=1):
        span, span_claims, span_assessments = _sentence_pass(f"V01S{index:06d}", sentence, verdicts)
        spans.append(span)
        claims.extend(span_claims)
        assessments.extend(span_assessments)
    return SimpleNamespace(
        failed=True,
        text=text,
        pass_results=(
            SimpleNamespace(
                failed=False,
                spans=tuple(spans),
                claims=tuple(claims),
                assessments=tuple(assessments),
                bundles=(),
            ),
        ),
    )


def test_subset_publishes_passing_sentences_without_second_verification() -> None:
    draft = "Keep this sentence. Drop this sentence."
    result = _subset_from_first_pass(
        _failed_draft(
            draft,
            (
                ("Keep this sentence.", (ClaimVerdict.SUPPORTED, ClaimVerdict.SUPPORTED)),
                ("Drop this sentence.", (ClaimVerdict.INSUFFICIENTLY_SUPPORTED,)),
            ),
        ),
    )

    assert result is not None
    assert result.failed is False
    assert result.text == "Keep this sentence."


def test_subset_returns_none_when_every_sentence_fails() -> None:
    draft = "First failure. Second failure."
    result = _subset_from_first_pass(
        _failed_draft(
            draft,
            (
                ("First failure.", (ClaimVerdict.SUPPORTED, ClaimVerdict.INSUFFICIENTLY_SUPPORTED)),
                ("Second failure.", (ClaimVerdict.CONTRADICTED,)),
            ),
        ),
    )

    assert result is None


def test_subset_does_not_salvage_a_contract_failure() -> None:
    result = _subset_from_first_pass(
        SimpleNamespace(
            failed=True,
            text="Keep this sentence.",
            pass_results=(
                SimpleNamespace(failed=True, spans=(), claims=(), assessments=(), bundles=()),
            ),
        ),
    )

    assert result is None


def _draft_span(span_id: str, ordinal: int, text: str, start: int) -> DraftSpan:
    return DraftSpan(
        span_id=span_id,
        ordinal=ordinal,
        start=start,
        end=start + len(text),
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


class _ScriptedVerifier:
    """An editor that returns `draft`, and a verifier that asks `decide` per claim.

    `decide(claim, request)` returns a verdict. A supported or contradicted
    claim quotes its own sentence when the evidence contains it verbatim.
    """

    def __init__(self, draft: str, decide) -> None:
        self.draft = draft
        self.decide = decide
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if request.operation_id == "editorial-final":
            payload = {"text": self.draft}
        elif request.operation_id.startswith("verification-decompose:"):
            spans = json.loads(request.input_text.splitlines()[1])
            payload = {"spans": [{"span_id": span["span_id"], "anchors": []} for span in spans]}
        else:
            claims = json.loads(request.input_text.splitlines()[1])["claims"]
            findings = []
            for claim in claims:
                verdict = self.decide(claim, request)
                evidence = []
                if verdict in {"supported", "contradicted"}:
                    item = claim["evidence"][0]
                    quote = claim["span_text"].strip()
                    evidence = [{
                        "segment_id": item["segment_id"],
                        "exact_quote": quote if quote in item["text"] else item["text"],
                    }]
                findings.append(
                    {"claim_id": claim["claim_id"], "verdict": verdict, "evidence": evidence}
                )
            payload = {"findings": findings}
        return GenerationResult(json.dumps(payload), "fake", request.model)


def _finalize_draft(tmp_path, *, source: str, draft: str, decide, **switches):
    counter = ConservativeUtf8TokenCounter()
    document = ingest_text(source)
    segment = whole_document_segment(document, counter)
    summary = SummaryNode.model_validate({
        "summary": draft,
        "content_units": [],
        "entities": [], "qualifications": [], "contradictions": [],
        "quotations": [], "provenance": [segment.segment_id], "level": 0,
    })
    root = TreeNode("L0N0001", 0, 0, summary, (), (segment.segment_id,))
    provider = _ScriptedVerifier(draft, decide)
    audit_path = tmp_path / "audit.json"

    def run():
        return finalize_summary(
            summary,
            provider,
            source_id=document.source_id,
            model="test-model",
            timeout_seconds=30,
            target_words=len(draft.split()),
            strategy="direct",
            segments=(segment,),
            nodes=(root,),
            root_node_id=root.node_id,
            audit_path=audit_path,
            counter=counter,
            source_cores={segment.segment_id: segment.text},
            verification=VerificationConfig(enabled=True, **switches),
            verification_context_window_tokens=10_000,
        )

    return run, provider, audit_path


def _reassessment_requests(provider) -> list:
    return [
        request
        for request in provider.requests
        if "allowed_difference" in request.input_text
    ]


def test_a_reversed_payer_is_not_published_through_matching_words_and_number(
    tmp_path,
) -> None:
    def decide(claim, request):
        if "allowed_difference" in claim:
            return "contradicted"
        return "insufficiently_supported"

    run, provider, audit_path = _finalize_draft(
        tmp_path,
        source="Alpha paid Beta 7 dollars.",
        draft="Beta paid Alpha 7 dollars.",
        decide=decide,
    )

    with pytest.raises(FinalizationVerificationError):
        run()

    assert len(_reassessment_requests(provider)) == 1
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert "publication" not in audit
    assessment = audit["verification"]["passes"][-1]["assessments"][0]
    assert assessment["verdict"] == "contradicted"
    assert assessment["prompt_version"] == "verification-reassessment/1"
    assert assessment["reassessment"]["original_verdict"] == "insufficiently_supported"
    assert assessment["reassessment"]["tolerance"] == "rounded_number"


@pytest.mark.parametrize("strict_numbers", [False, True])
def test_a_rounded_number_publishes_only_on_the_verifiers_second_look(
    tmp_path, strict_numbers
) -> None:
    source = "They built a stunning 963.6 foot skyscraper beside the river."

    def decide(claim, request):
        return "supported" if "allowed_difference" in claim else "insufficiently_supported"

    run, provider, audit_path = _finalize_draft(
        tmp_path,
        source=source,
        draft="They built a stunning nearly 1000 foot skyscraper beside the river.",
        decide=decide,
        strict_numbers=strict_numbers,
    )

    if strict_numbers:
        with pytest.raises(FinalizationVerificationError):
            run()
        assert _reassessment_requests(provider) == []
        return

    result = run()
    assert "nearly 1000 foot" in result.text
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["publication"]["kind"] == "editorial"
    sentence = audit["publication"]["sentences"][0]
    assert sentence["verdict"] == "supported"
    assert sentence["evidence"][0]["quote"] == source
    assessment = audit["verification"]["passes"][-1]["assessments"][0]
    assert assessment["reassessment"]["original_verdict"] == "insufficiently_supported"


_HARBOUR = "The harbour opened in May."
_PARAPHRASE = "The event impacted sixteen divers on the reef."
_SOURCE_SENTENCE = "The crew counted 16 divers on the reef."


@pytest.mark.parametrize("replacement_verdict", ["supported", "insufficiently_supported"])
def test_strict_numbers_publishes_a_source_sentence_only_after_it_is_verified(
    tmp_path, replacement_verdict
) -> None:
    def decide(claim, request):
        text = claim["span_text"].strip()
        if text == _HARBOUR:
            return "supported"
        if text == _SOURCE_SENTENCE:
            return replacement_verdict
        return "insufficiently_supported"

    run, provider, audit_path = _finalize_draft(
        tmp_path,
        source=f"{_HARBOUR} {_SOURCE_SENTENCE}",
        draft=f"{_HARBOUR} {_PARAPHRASE}",
        decide=decide,
        strict_numbers=True,
    )

    result = run()

    assert "verification-decompose:V02" in [request.operation_id for request in provider.requests]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    publication = audit["publication"]
    assert publication["kind"] == "verified_subset"
    assert [item["pass_index"] for item in audit["verification"]["passes"]] == [1, 2]
    replaced = replacement_verdict == "supported"
    assert publication["substitutions"] == [{
        "original_text": _PARAPHRASE,
        "original_verdict": "insufficiently_supported",
        "replacement_text": _SOURCE_SENTENCE,
        "replacement_verdict": replacement_verdict,
        "action": "replaced" if replaced else "removed",
    }]
    assert _PARAPHRASE in [item["text"] for item in publication["removed_sentences"]]
    if replaced:
        assert result.text == f"{_HARBOUR} {_SOURCE_SENTENCE}"
        assert publication["sentences"][1]["evidence"][0]["quote"] == _SOURCE_SENTENCE
    else:
        assert result.text == _HARBOUR


def test_without_strict_numbers_a_rejected_numbered_sentence_is_only_dropped(
    tmp_path,
) -> None:
    def decide(claim, request):
        return "supported" if claim["span_text"].strip() == _HARBOUR else "insufficiently_supported"

    run, provider, audit_path = _finalize_draft(
        tmp_path,
        source=f"{_HARBOUR} {_SOURCE_SENTENCE}",
        draft=f"{_HARBOUR} {_PARAPHRASE}",
        decide=decide,
    )

    result = run()

    assert result.text == _HARBOUR
    assert "verification-decompose:V02" not in [
        request.operation_id for request in provider.requests
    ]
    assert "substitutions" not in json.loads(audit_path.read_text(encoding="utf-8"))["publication"]


def test_a_contradiction_elsewhere_keeps_the_supported_sentence_and_its_quote(
    tmp_path,
) -> None:
    def decide(claim, request):
        return "supported" if claim["span_text"].strip() == _HARBOUR else "contradicted"

    run, _provider, audit_path = _finalize_draft(
        tmp_path,
        source=f"{_HARBOUR} The mayor resigned.",
        draft=f"{_HARBOUR} The mayor stayed.",
        decide=decide,
    )

    result = run()

    assert result.text == _HARBOUR
    publication = json.loads(audit_path.read_text(encoding="utf-8"))["publication"]
    assert publication["sentences"][0]["evidence"][0]["quote"] == _HARBOUR
    assert publication["removed_sentences"][0]["verdict"] == "contradicted"
