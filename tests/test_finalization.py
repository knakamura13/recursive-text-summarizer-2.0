import hashlib
import json
from types import SimpleNamespace

import pytest

from summarizer.direct import whole_document_segment
from summarizer.finalization import (
    FinalizationVerificationError,
    _effective_claim_verdict,
    _published_sentences,
    _subset_from_first_pass,
    _verify_publication,
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
    index = build_source_lexical_index(
        provenance_ids=("D000001",), source={"D000001": "source"}
    )
    result = _subset_from_first_pass(
        _failed_draft(
            draft,
            (
                ("Keep this sentence.", (ClaimVerdict.SUPPORTED, ClaimVerdict.SUPPORTED)),
                ("Drop this sentence.", (ClaimVerdict.INSUFFICIENTLY_SUPPORTED,)),
            ),
        ),
        source_index=index,
    )

    assert result is not None
    assert result.failed is False
    assert result.text == "Keep this sentence."


def test_subset_returns_none_when_every_sentence_fails() -> None:
    draft = "First failure. Second failure."
    index = build_source_lexical_index(
        provenance_ids=("D000001",), source={"D000001": "source"}
    )
    result = _subset_from_first_pass(
        _failed_draft(
            draft,
            (
                ("First failure.", (ClaimVerdict.SUPPORTED, ClaimVerdict.INSUFFICIENTLY_SUPPORTED)),
                ("Second failure.", (ClaimVerdict.CONTRADICTED,)),
            ),
        ),
        source_index=index,
    )

    assert result is None


def test_subset_does_not_salvage_a_contract_failure() -> None:
    index = build_source_lexical_index(
        provenance_ids=("D000001",), source={"D000001": "source"}
    )
    result = _subset_from_first_pass(
        SimpleNamespace(
            failed=True,
            text="Keep this sentence.",
            pass_results=(
                SimpleNamespace(failed=True, spans=(), claims=(), assessments=(), bundles=()),
            ),
        ),
        source_index=index,
    )

    assert result is None


def _lenient_bundle(anchor: str, evidence: str):
    index = build_source_lexical_index(
        provenance_ids=("S000001",), source={"S000001": evidence}
    )
    item = Claim(
        claim_id="V01C000001",
        span_id="V01S000001",
        ordinal=1,
        anchor=anchor,
        is_fallback=True,
    )
    bundle = select_claim_evidence(
        item,
        source_index=index,
        counter=ConservativeUtf8TokenCounter(),
        max_tokens=1000,
    )
    return item, index, {item.claim_id: bundle}


def test_lenient_numbers_keep_a_rounded_sentence() -> None:
    item, index, bundles = _lenient_bundle(
        "They built a stunning nearly 1000 foot skyscraper.",
        "They built a stunning 963.6 foot skyscraper.",
    )
    assert _effective_claim_verdict(
        item,
        ClaimVerdict.CONTRADICTED,
        bundles,
        index,
        strict_numbers=False,
    ) is ClaimVerdict.SUPPORTED
    assert _effective_claim_verdict(
        item,
        ClaimVerdict.CONTRADICTED,
        bundles,
        index,
        strict_numbers=True,
    ) is ClaimVerdict.CONTRADICTED


def test_lenient_numbers_still_drop_a_different_fact() -> None:
    item, index, bundles = _lenient_bundle(
        "Nearly 1000 people died.",
        "963 people were rescued.",
    )
    assert _effective_claim_verdict(
        item,
        ClaimVerdict.CONTRADICTED,
        bundles,
        index,
        strict_numbers=False,
    ) is ClaimVerdict.CONTRADICTED


def test_lenient_names_keep_a_shortened_name() -> None:
    item, index, bundles = _lenient_bundle(
        "Professional skier Saugstad wore a backpack.",
        "Professional skier Elyse Saugstad wore a backpack.",
    )
    assert _effective_claim_verdict(
        item,
        ClaimVerdict.INSUFFICIENTLY_SUPPORTED,
        bundles,
        index,
        strict_names=False,
    ) is ClaimVerdict.SUPPORTED
    assert _effective_claim_verdict(
        item,
        ClaimVerdict.INSUFFICIENTLY_SUPPORTED,
        bundles,
        index,
        strict_names=True,
    ) is ClaimVerdict.INSUFFICIENTLY_SUPPORTED


def _draft_span(span_id: str, ordinal: int, text: str, start: int) -> DraftSpan:
    return DraftSpan(
        span_id=span_id,
        ordinal=ordinal,
        start=start,
        end=start + len(text),
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def test_lenient_subset_publishes_with_supported_evidence() -> None:
    kept = "They built a stunning nearly 1000 foot skyscraper."
    dropped = "Nearly 1000 people died."
    draft = f"{kept} {dropped}"
    kept_source = "They built a stunning 963.6 foot skyscraper."
    dropped_source = "963 people were rescued."
    index = build_source_lexical_index(
        provenance_ids=("S000001", "S000002"),
        source={"S000001": kept_source, "S000002": dropped_source},
    )
    spans = (
        _draft_span("V01S000001", 1, f"{kept} ", 0),
        _draft_span("V01S000002", 2, dropped, len(kept) + 1),
    )
    claims = (
        Claim("V01C000001", "V01S000001", 1, kept, False),
        Claim("V01C000002", "V01S000002", 1, dropped, False),
    )
    bundles = tuple(
        select_claim_evidence(
            claim,
            source_index=index,
            counter=ConservativeUtf8TokenCounter(),
            max_tokens=1000,
        )
        for claim in claims
    )
    assessments = tuple(
        ClaimAssessment(
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
        for claim in claims
    )
    passed = VerificationPassResult(
        spans=spans,
        claims=claims,
        assessments=assessments,
        selections=tuple(bundle.selection for bundle in bundles),
        bundles=bundles,
        generations=(),
        phase_generations=(),
        diagnostic_codes=(),
        failed=False,
    )
    result = _subset_from_first_pass(
        VerificationResult(
            text=draft,
            passes=(assessments,),
            selections=(passed.selections,),
            repairs=(),
            generations=(),
            diagnostic_codes=(),
            exhausted=False,
            failed=True,
            pass_results=(passed,),
        ),
        source_index=index,
    )

    assert result is not None
    assert result.text == kept
    published = _published_sentences(result.text, result, passages=None)
    assert published[0].text == kept
    assert published[0].verdict == "supported"
    assert published[0].evidence[0].quote == kept_source
