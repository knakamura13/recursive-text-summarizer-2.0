import json
import re

import pytest

from summarizer.providers.base import GenerationRequest, GenerationResult, ProviderResponseError
from summarizer.runtime.observers import RuntimeObserver
from summarizer.tokenization import ConservativeUtf8TokenCounter
from summarizer.verification import (
    BatchFinding,
    ClaimVerdict,
    GenerationPhase,
    UnresolvedWork,
    VerificationConfig,
    VerificationProgress,
    VerificationRuntime,
    _redact_finding,
    build_decomposition_request,
    build_source_lexical_index,
    reduce_batch_findings,
    split_draft_spans,
    verify_draft_once,
)


class Provider:
    def __init__(self, responses: list[str], providers: list[str] | None = None, models: list[str] | None = None) -> None:
        self.responses = iter(responses)
        self.providers = iter(providers or ["scripted"] * len(responses))
        self.models = iter(models or ["model"] * len(responses))
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(
            text=next(self.responses), provider=next(self.providers), model=next(self.models)
        )


def runtime(provider: Provider) -> VerificationRuntime:
    return VerificationRuntime(
        provider=provider,
        counter=ConservativeUtf8TokenCounter(),
        model="model",
        timeout_seconds=30,
        context_window_tokens=10_000,
    )


def index():
    return build_source_lexical_index(
        provenance_ids=("S000001",),
        source={"S000001": "The measured value is 42."},
    )


def test_terminal_redaction_keeps_distinct_quotes_from_one_segment_valid() -> None:
    finding = BatchFinding(
        claim_id="V01C000001",
        verdict=ClaimVerdict.SUPPORTED,
        evidence_ids=("S000001", "S000001"),
        exact_quotes=("first source quote", "second source quote"),
    )

    redacted = _redact_finding(finding)

    assert redacted.evidence_ids == finding.evidence_ids
    assert redacted.exact_quotes == ("[redacted:1]", "[redacted:2]")
    assert "source quote" not in repr(redacted)


def test_verify_once_decomposes_selects_and_classifies_every_claim() -> None:
    provider = Provider(
        [
            '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
            json.dumps(
                {
                    "findings": [
                        {
                            "claim_id": "V01C000001",
                            "verdict": "supported",
                            "evidence": [
                                {
                                    "segment_id": "S000001",
                                    "exact_quote": "measured value is 42",
                                }
                            ],
                        }
                    ]
                }
            ),
        ]
    )

    result = verify_draft_once(
        "The measured value is 42.",
        source_id="a" * 64,
        source_index=index(),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True),
        pass_index=1,
    )

    assert [assessment.verdict for assessment in result.assessments] == [
        ClaimVerdict.SUPPORTED
    ]
    assert result.assessments[0].findings[0].evidence_ids == ("S000001",)
    assert [request.operation_id for request in provider.requests] == [
        "verification-decompose:V01",
        "verification-classify:V01",
    ]
    assert len(result.generations) == 2


def test_verify_once_is_disabled_without_provider_calls() -> None:
    provider = Provider([])

    result = verify_draft_once(
        "The measured value is 42.",
        source_id="a" * 64,
        source_index=index(),
        runtime=runtime(provider),
        config=VerificationConfig(),
        pass_index=1,
    )

    assert result.assessments == ()
    assert result.generations == ()
    assert provider.requests == []


def test_verify_once_leaves_a_span_unresolved_after_malformed_answers() -> None:
    provider = Provider(["not json", "not json"])

    result = verify_draft_once(
        "The measured value is 42.",
        source_id="a" * 64,
        source_index=index(),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True),
        pass_index=1,
    )

    assert not result.failed
    assert result.claims == ()
    assert result.unresolved == (
        UnresolvedWork("V01S000001", GenerationPhase.DECOMPOSITION, "invalid_response"),
    )
    assert result.diagnostic_codes == ("decomposition_incomplete",)
    assert len(provider.requests) == 2


def test_a_supported_sentence_whose_quotes_share_too_few_of_its_words_is_not_supported() -> None:
    quote = '{"segment_id":"S000001","exact_quote":"measured value is 42"}'
    provider = Provider(
        [
            '{"spans":[{"span_id":"V01S000001","anchors":[]},{"span_id":"V01S000002","anchors":[]}]}',
            '{"findings":['
            f'{{"claim_id":"V01C000001","verdict":"supported","evidence":[{quote}]}},'
            f'{{"claim_id":"V01C000002","verdict":"supported","evidence":[{quote}]}}'
            "]}",
        ]
    )

    result = verify_draft_once(
        "The measured value is 42. The harbour in 北京 closed for repairs after the storm.",
        source_id="a" * 64,
        source_index=index(),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True),
        pass_index=1,
    )

    assert [assessment.verdict for assessment in result.assessments] == [
        ClaimVerdict.SUPPORTED,
        ClaimVerdict.INSUFFICIENTLY_SUPPORTED,
    ]
    assert "evidence_overlap_below_floor" in result.diagnostic_codes


def test_the_floor_keeps_a_supported_sentence_in_a_script_without_word_spaces() -> None:
    source = "港は五月に開港した。"
    provider = Provider(
        [
            '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
            '{"findings":[{"claim_id":"V01C000001","verdict":"supported","evidence":'
            '[{"segment_id":"S000001","exact_quote":"五月に開港"}]}]}',
        ]
    )

    result = verify_draft_once(
        "港は五月に開港した。",
        source_id="a" * 64,
        source_index=build_source_lexical_index(provenance_ids=("S000001",), source={"S000001": source}),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True),
        pass_index=1,
    )

    assert [assessment.verdict for assessment in result.assessments] == [ClaimVerdict.SUPPORTED]


def test_the_floor_also_rejects_a_poorly_quoted_claim_beside_an_uncheckable_one() -> None:
    provider = Provider(
        [
            '{"spans":[{"span_id":"V01S000001","anchors":["The harbour closed"]}]}',
            '{"findings":['
            '{"claim_id":"V01C000001","verdict":"supported","evidence":'
            '[{"segment_id":"S000001","exact_quote":"measured value is 42"}]},'
            '{"claim_id":"V01C000002","verdict":"not_meaningfully_verifiable","evidence":[]}'
            "]}",
        ]
    )
    events = []

    result = verify_draft_once(
        "The harbour closed for repairs after the storm.",
        source_id="a" * 64,
        source_index=index(),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True),
        pass_index=1,
        progress=VerificationProgress(RuntimeObserver(on_item=events.append)),
    )

    assert [assessment.verdict for assessment in result.assessments] == [
        ClaimVerdict.INSUFFICIENTLY_SUPPORTED,
        ClaimVerdict.NOT_MEANINGFULLY_VERIFIABLE,
    ]
    last_reported = {
        event.work_id: event.message for event in events if event.state == "completed"
    }
    assert last_reported == {
        "V01C000001": "insufficiently_supported",
        "V01C000002": "not_meaningfully_verifiable",
    }


def test_verify_once_retries_invalid_anchor_with_specific_feedback() -> None:
    provider = Provider(
        [
            '{"spans":[{"span_id":"V01S000001","anchors":["wrong value"]}]}',
            '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
            '{"findings":[{"claim_id":"V01C000001","verdict":"supported","evidence":[{"segment_id":"S000001","exact_quote":"measured value is 42"}]}]}',
        ]
    )

    result = verify_draft_once(
        "The measured value is 42.",
        source_id="a" * 64,
        source_index=index(),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True),
        pass_index=1,
    )

    assert not result.failed
    assert len(result.generations) == 3
    assert [request.operation_id for request in provider.requests] == [
        "verification-decompose:V01",
        "verification-decompose:V01",
        "verification-classify:V01",
    ]
    assert "exact substring" in provider.requests[1].instructions
    assert "wrong value" not in provider.requests[1].instructions


def test_verify_once_retries_unselected_evidence_with_specific_feedback() -> None:
    provider = Provider(
        [
            '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
            '{"findings":[{"claim_id":"V01C000001","verdict":"supported","evidence":[{"segment_id":"V01S000001","exact_quote":"measured value is 42"}]}]}',
            '{"findings":[{"claim_id":"V01C000001","verdict":"supported","evidence":[{"segment_id":"S000001","exact_quote":"measured value is 42"}]}]}',
        ]
    )

    result = verify_draft_once(
        "The measured value is 42.",
        source_id="a" * 64,
        source_index=index(),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True),
        pass_index=1,
    )

    assert not result.failed
    assert len(result.generations) == 3
    assert result.assessments[0].verdict is ClaimVerdict.SUPPORTED
    assert "segment_id" in provider.requests[2].instructions
    assert "selected evidence" in provider.requests[2].instructions
    assert "V01S000001" not in provider.requests[2].instructions


def test_verify_once_leaves_a_claim_unresolved_after_one_invalid_classification_retry() -> None:
    provider = Provider(
        [
            '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
            '{"findings":[{"claim_id":"V01C000001","verdict":"supported","evidence":[{"segment_id":"V01S000001","exact_quote":"measured value is 42"}]}]}',
            '{"findings":[{"claim_id":"V01C000001","verdict":"supported","evidence":[{"segment_id":"V01S000001","exact_quote":"measured value is 42"}]}]}',
        ]
    )

    result = verify_draft_once(
        "The measured value is 42.",
        source_id="a" * 64,
        source_index=index(),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True),
        pass_index=1,
        terminalize_errors=True,
    )

    assert not result.failed
    assert result.unresolved == (
        UnresolvedWork("V01C000001", GenerationPhase.CLASSIFICATION, "invalid_response"),
    )
    assert result.diagnostic_codes == ("classification_incomplete",)
    assert len(result.generations) == 3
    assert len(provider.requests) == 3
    assert result.assessments == ()


def test_verify_once_downgrades_nonexact_quotes_after_one_retry() -> None:
    invalid = ('{"findings":[{"claim_id":"V01C000001","verdict":"supported",'
               '"evidence":[{"segment_id":"S000001","exact_quote":"the value was 42"}]}]}')
    provider = Provider([
        '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
        invalid,
        invalid,
    ])

    result = verify_draft_once(
        "The measured value is 42.",
        source_id="a" * 64,
        source_index=index(),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True, strict_numbers=True, strict_names=True),
        pass_index=1,
        terminalize_errors=True,
    )

    assert not result.failed
    assert result.assessments[0].verdict is ClaimVerdict.INSUFFICIENTLY_SUPPORTED
    assert result.assessments[0].findings[0].evidence_ids == ()
    assert result.diagnostic_codes == ("invalid_evidence_quotes_downgraded",)
    assert len(provider.requests) == 3


def test_verify_once_omits_invalid_anchors_after_retry_but_checks_entire_span() -> None:
    provider = Provider(
        [
            '{"spans":[{"span_id":"V01S000001","anchors":["a different value"]}]}',
            '{"spans":[{"span_id":"V01S000001","anchors":["another different value"]}]}',
            '{"findings":[{"claim_id":"V01C000001","verdict":"supported","evidence":[{"segment_id":"S000001","exact_quote":"measured value is 42"}]}]}',
        ]
    )

    result = verify_draft_once(
        "The measured value is 42.",
        source_id="a" * 64,
        source_index=index(),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True),
        pass_index=1,
    )

    assert not result.failed
    assert len(result.claims) == 1
    assert result.claims[0].is_fallback
    assert result.claims[0].anchor == "The measured value is 42."
    assert result.assessments[0].verdict is ClaimVerdict.SUPPORTED
    assert result.diagnostic_codes == ("invalid_anchors_omitted",)


def test_verify_once_rejects_an_oversized_decomposition_span_before_calling() -> None:
    provider = Provider([])

    with pytest.raises(ValueError, match="single work item"):
        verify_draft_once(
            "x" * 200,
            source_id="a" * 64,
            source_index=index(),
            runtime=runtime(provider),
            config=VerificationConfig(enabled=True, request_tokens=20, output_reserve_tokens=1),
            pass_index=1,
        )

    assert provider.requests == []


def test_verify_once_counts_response_schema_before_decomposition_call() -> None:
    provider = Provider([])
    draft = "Small draft."
    request = build_decomposition_request(
        split_draft_spans(draft, pass_index=1),
        source_id="a" * 64,
        runtime=runtime(provider),
    )
    payload_cost = runtime(provider).counter.count(request.instructions) + runtime(provider).counter.count(request.input_text)

    with pytest.raises(ValueError, match="single work item"):
        verify_draft_once(draft, source_id="a" * 64, source_index=index(), runtime=runtime(provider), config=VerificationConfig(enabled=True, request_tokens=payload_cost + 1, output_reserve_tokens=1, safety_margin_tokens=0), pass_index=1)

    assert provider.requests == []


def test_verify_once_keeps_provider_provenance_per_classification_batch() -> None:
    provider = Provider(
        [
            '{"spans":[{"span_id":"V01S000001","anchors":[]},{"span_id":"V01S000002","anchors":[]}]}',
            '{"findings":[{"claim_id":"V01C000001","verdict":"supported","evidence":[{"segment_id":"S000001","exact_quote":"measured value is 42"}]}]}',
            '{"findings":[{"claim_id":"V01C000002","verdict":"supported","evidence":[{"segment_id":"S000001","exact_quote":"measured value is 42"}]}]}',
        ],
        providers=["decomposer", "verifier-a", "verifier-b"],
        models=["base", "resolved-a", "resolved-b"],
    )

    result = verify_draft_once(
        "The measured value is 42. The measured value is 42.",
        source_id="a" * 64,
        source_index=index(),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True, request_tokens=2400, output_reserve_tokens=500, safety_margin_tokens=0),
        pass_index=1,
    )

    assert [item.verifier_provider for item in result.assessments] == [
        "verifier-a",
        "verifier-b",
    ]
    assert [item.verifier_model for item in result.assessments] == ["resolved-a", "resolved-b"]


def test_verify_once_merges_multiple_decomposition_batches() -> None:
    sentence = "evidence " + "x" * 1_291 + "."
    provider = Provider(
        [
            '{"spans":[{"span_id":"V01S000001","anchors":[]}]}',
            '{"spans":[{"span_id":"V01S000002","anchors":[]}]}',
            '{"findings":[{"claim_id":"V01C000001","verdict":"supported","evidence":[{"segment_id":"S000001","exact_quote":"evidence"}]}]}',
            '{"findings":[{"claim_id":"V01C000002","verdict":"supported","evidence":[{"segment_id":"S000001","exact_quote":"evidence"}]}]}',
        ]
    )

    result = verify_draft_once(
        f"{sentence} {sentence}",
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "evidence"}
        ),
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True, request_tokens=3_660, output_reserve_tokens=1, safety_margin_tokens=0),
        pass_index=1,
    )

    assert [claim.claim_id for claim in result.claims] == ["V01C000001", "V01C000002"]
    assert [assessment.verdict for assessment in result.assessments] == [
        ClaimVerdict.SUPPORTED,
        ClaimVerdict.SUPPORTED,
    ]
    assert [request.operation_id for request in provider.requests] == [
        "verification-decompose:V01",
        "verification-decompose:V01",
        "verification-classify:V01",
        "verification-classify:V01",
    ]


def test_verify_once_rejects_oversized_classification_before_its_provider_call() -> None:
    provider = Provider(['{"spans":[{"span_id":"V01S000001","anchors":[]}]}'])
    oversized_index = build_source_lexical_index(
        provenance_ids=("S000001",), source={"S000001": "evidence " * 1_000}
    )

    with pytest.raises(ValueError, match="single work item"):
        verify_draft_once(
            "The measured value is 42.",
            source_id="a" * 64,
            source_index=oversized_index,
            runtime=runtime(provider),
            config=VerificationConfig(enabled=True, evidence_tokens=20_000, request_tokens=2_000, output_reserve_tokens=1, safety_margin_tokens=0),
            pass_index=1,
        )

    assert [request.operation_id for request in provider.requests] == ["verification-decompose:V01"]


@pytest.mark.parametrize(
    ("findings", "retrieval_complete", "expected", "codes"),
    (
        (
            (ClaimVerdict.SUPPORTED, ClaimVerdict.CONTRADICTED),
            True,
            ClaimVerdict.INSUFFICIENTLY_SUPPORTED,
            ("conflicting_evidence",),
        ),
        (
            (ClaimVerdict.CONTRADICTED,),
            False,
            ClaimVerdict.INSUFFICIENTLY_SUPPORTED,
            (),
        ),
        (
            (ClaimVerdict.NOT_MEANINGFULLY_VERIFIABLE,),
            True,
            ClaimVerdict.NOT_MEANINGFULLY_VERIFIABLE,
            (),
        ),
        (
            (ClaimVerdict.NOT_MEANINGFULLY_VERIFIABLE, ClaimVerdict.SUPPORTED),
            True,
            ClaimVerdict.INSUFFICIENTLY_SUPPORTED,
            ("inconsistent_meaningfulness",),
        ),
    ),
)
def test_batch_finding_reducer_is_conservative(
    findings, retrieval_complete: bool, expected: ClaimVerdict, codes: tuple[str, ...]
) -> None:
    batch_findings = tuple(
        BatchFinding(
            claim_id="V01C000001",
            verdict=verdict,
            evidence_ids=("S000001",)
            if verdict in {ClaimVerdict.SUPPORTED, ClaimVerdict.CONTRADICTED}
            else (),
            exact_quotes=("evidence",)
            if verdict in {ClaimVerdict.SUPPORTED, ClaimVerdict.CONTRADICTED}
            else (),
        )
        for verdict in findings
    )

    result, result_codes = reduce_batch_findings(
        "V01C000001", batch_findings, retrieval_complete=retrieval_complete
    )

    assert result is expected
    assert result_codes == codes


class _ByOperation:
    """Answer every span with its whole text and every claim as unsupported.

    `cut_off` decomposition calls raise as a provider does for an answer that
    stopped at its output reserve.
    """

    def __init__(self, cut_off: int = 0) -> None:
        self.cut_off = cut_off
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if request.operation_id.startswith("verification-decompose"):
            if self.cut_off:
                self.cut_off -= 1
                raise ProviderResponseError("Ollama stopped at the configured output token limit")
            spans = [
                {"span_id": span_id, "anchors": []}
                for span_id in dict.fromkeys(re.findall(r"V01S\d{6}", request.input_text))
            ]
            text = json.dumps({"spans": spans})
        else:
            findings = [
                {"claim_id": claim_id, "verdict": "insufficiently_supported", "evidence": []}
                for claim_id in dict.fromkeys(re.findall(r"V01C\d{6}", request.input_text))
            ]
            text = json.dumps({"findings": findings})
        return GenerationResult(text=text, provider="fake", model="model")


def _decompositions(provider: _ByOperation) -> list[list[str]]:
    return [
        list(dict.fromkeys(re.findall(r"V01S\d{6}", request.input_text)))
        for request in provider.requests
        if request.operation_id.startswith("verification-decompose")
    ]


_LONG_DRAFT = " ".join(f"Measurement {index} recorded a value near forty-two units." for index in range(3))


def test_decomposition_batches_are_sized_by_their_expected_answer() -> None:
    spans = split_draft_spans(_LONG_DRAFT, pass_index=1)
    one_answer = ConservativeUtf8TokenCounter().count(json.dumps(
        {"spans": [{"span_id": spans[0].span_id, "anchors": [spans[0].text]}]}, ensure_ascii=False
    ))
    provider = _ByOperation()

    result = verify_draft_once(
        _LONG_DRAFT, source_id="a" * 64, source_index=index(), runtime=runtime(provider),
        config=VerificationConfig(enabled=True, output_reserve_tokens=one_answer + 5, safety_margin_tokens=0),
        pass_index=1,
    )

    assert _decompositions(provider) == [[span.span_id] for span in spans]
    assert not result.failed


def test_a_cut_off_decomposition_answer_is_asked_again_in_smaller_groups() -> None:
    provider = _ByOperation(cut_off=1)

    result = verify_draft_once(
        _LONG_DRAFT, source_id="a" * 64, source_index=index(), runtime=runtime(provider),
        config=VerificationConfig(enabled=True, safety_margin_tokens=0),
        pass_index=1,
    )

    first, *retries = _decompositions(provider)
    assert len(first) == 3
    assert retries and all(len(group) < 3 for group in retries)
    assert not result.failed
    assert len(result.claims) == 3
