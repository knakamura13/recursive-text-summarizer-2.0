"""Bounded re-asks of only the verifier work a response left missing (#108).

The response shapes model the study's saved failures without their text: a
decomposer answering 14 of 24 spans, one bad anchor among 29 answered spans,
and a batch whose spans are never answered.
"""

import json
import re
from collections.abc import Callable

from summarizer.finalization import _passing_sentence_text
from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer.tokenization import ConservativeUtf8TokenCounter
from summarizer.verification import (
    ClaimVerdict,
    GenerationPhase,
    UnresolvedWork,
    VerificationConfig,
    VerificationRuntime,
    _measure_request_tokens,
    _resolve_work,
    _response_correction,
    build_source_lexical_index,
    verify_and_repair,
    verify_draft_once,
)

FACTS = tuple(f"Item {name} weighs {value} grams." for name, value in zip("ABCDEF", range(11, 17)))
DRAFT = " ".join(FACTS)
SOURCE = build_source_lexical_index(provenance_ids=("S000001",), source={"S000001": DRAFT})
_SPAN = re.compile(r"V\d{2}S\d{6}")
_CLAIM = re.compile(r"V\d{2}C\d{6}")


class Responder:
    """Answer each request from the ids it carries, recording what was asked."""

    def __init__(
        self,
        decompose: Callable[[int, list[str]], list[str]] = lambda _call, ids: ids,
        classify: Callable[[int, list[str]], list[str]] = lambda _call, ids: ids,
    ) -> None:
        self.decompose = decompose
        self.classify = classify
        self.asked: list[tuple[str, list[str], str]] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.operation_id.startswith("verification-decompose"):
            payload = request.input_text.splitlines()[1]
            ids = [span["span_id"] for span in json.loads(payload)]
            answered = self.decompose(self._calls("decompose"), ids)
            self.asked.append(("decompose", ids, request.instructions))
            text = json.dumps({"spans": [{"span_id": span_id, "anchors": []} for span_id in answered]})
        else:
            ids = list(dict.fromkeys(_CLAIM.findall(request.input_text)))
            answered = self.classify(self._calls("classify"), ids)
            self.asked.append(("classify", ids, request.instructions))
            text = json.dumps({
                "findings": [
                    {
                        "claim_id": claim_id,
                        "verdict": "supported",
                        "evidence": [{"segment_id": "S000001", "exact_quote": FACTS[0]}],
                    }
                    for claim_id in answered
                ]
            })
        return GenerationResult(text=text, provider="scripted", model="model")

    def _calls(self, kind: str) -> int:
        return sum(1 for asked_kind, _ids, _instructions in self.asked if asked_kind == kind)


def runtime(provider: Responder) -> VerificationRuntime:
    return VerificationRuntime(
        provider=provider,
        counter=ConservativeUtf8TokenCounter(),
        model="model",
        timeout_seconds=30,
        context_window_tokens=100_000,
    )


def verify(provider: Responder):
    return verify_draft_once(
        DRAFT,
        source_id="a" * 64,
        source_index=SOURCE,
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True),
        pass_index=1,
        terminalize_errors=True,
    )


def test_partial_decomposition_re_asks_only_the_omitted_spans() -> None:
    provider = Responder(
        decompose=lambda call, ids: [ids[0], ids[1], ids[4], ids[5]] if call == 0 else ids
    )

    result = verify(provider)

    decompositions = [ids for kind, ids, _ in provider.asked if kind == "decompose"]
    assert decompositions == [
        ["V01S000001", "V01S000002", "V01S000003", "V01S000004", "V01S000005", "V01S000006"],
        ["V01S000003", "V01S000004"],
    ]
    assert "Return exactly one result for every supplied span_id" in provider.asked[1][2]
    assert result.unresolved == ()
    assert [claim.span_id for claim in result.claims] == [
        f"V01S00000{index}" for index in range(1, 7)
    ]
    assert [assessment.verdict for assessment in result.assessments] == [ClaimVerdict.SUPPORTED] * 6


def test_one_invalid_anchor_re_asks_only_its_span() -> None:
    class BadAnchorOnce(Responder):
        def generate(self, request: GenerationRequest) -> GenerationResult:
            generation = super().generate(request)
            kind, ids, _ = self.asked[-1]
            if kind != "decompose" or self._calls("decompose") != 1:
                return generation
            body = json.loads(generation.text)
            body["spans"][2]["anchors"] = ["not in the span"]
            return GenerationResult(json.dumps(body), generation.provider, generation.model)

    provider = BadAnchorOnce()

    result = verify(provider)

    decompositions = [(ids, instructions) for kind, ids, instructions in provider.asked if kind == "decompose"]
    assert [ids for ids, _ in decompositions] == [
        [f"V01S00000{index}" for index in range(1, 7)],
        ["V01S000003"],
    ]
    assert "exact substring of its own span text" in decompositions[1][1]
    assert result.unresolved == ()
    assert "invalid_anchors_omitted" not in result.diagnostic_codes


def test_never_answered_spans_are_unresolved_within_three_asks_each() -> None:
    silent = {"V01S000002", "V01S000005"}
    provider = Responder(decompose=lambda _call, ids: [span_id for span_id in ids if span_id not in silent])

    result = verify(provider)

    decompositions = [ids for kind, ids, _ in provider.asked if kind == "decompose"]
    # One first batch, one re-ask at half its size, then one ask per span.
    assert decompositions == [
        [f"V01S00000{index}" for index in range(1, 7)],
        ["V01S000002", "V01S000005"],
        ["V01S000002"],
        ["V01S000005"],
    ]
    assert result.unresolved == (
        UnresolvedWork("V01S000002", GenerationPhase.DECOMPOSITION, "omitted"),
        UnresolvedWork("V01S000005", GenerationPhase.DECOMPOSITION, "omitted"),
    )
    assert {claim.span_id for claim in result.claims}.isdisjoint(silent)
    assert len(result.assessments) == 4
    assert result.diagnostic_codes == ("decomposition_incomplete",)


def test_partial_classification_keeps_answered_verdicts_and_re_asks_the_rest() -> None:
    provider = Responder(classify=lambda call, ids: ids[:3] if call == 0 else ids)

    result = verify(provider)

    classifications = [ids for kind, ids, _ in provider.asked if kind == "classify"]
    assert classifications[0] == [f"V01C00000{index}" for index in range(1, 7)]
    assert classifications[1:] == [["V01C000004", "V01C000005", "V01C000006"]]
    assert result.unresolved == ()
    assert [assessment.claim_id for assessment in result.assessments] == [
        f"V01C00000{index}" for index in range(1, 7)
    ]


def test_an_unfinished_sentence_is_never_published_but_checked_ones_are() -> None:
    provider = Responder(decompose=lambda _call, ids: [span_id for span_id in ids if span_id != "V01S000003"])

    result = verify_and_repair(
        DRAFT,
        source_id="a" * 64,
        source_index=SOURCE,
        runtime=runtime(provider),
        config=VerificationConfig(enabled=True),
    )

    assert result.failed
    assert result.failure_codes == ("verification_incomplete",)
    assert _passing_sentence_text(result) == " ".join(
        fact for index, fact in enumerate(FACTS) if index != 2
    )


def test_an_item_near_capacity_is_re_asked_alone_with_its_own_correction() -> None:
    """A group's longer correction note must not turn a fitting item into a capacity failure."""
    notes = {
        "b": "claim-decomposition: anchor not in span",
        "d": "claim-decomposition: missing span result",
    }
    asked: list[tuple[str, ...]] = []

    def build(items, correction):
        return GenerationRequest(
            model="model",
            instructions="x" * (2_000 if "b" in items else 10) + correction,
            input_text="|".join(items),
            timeout_seconds=30,
            operation_id="verification-decompose:V01",
        )

    def parse(_text, items):
        # "a" and "c" answer at once, "d" on its re-ask, "b" never.
        answered = {"a", "c"} | ({"d"} if len(asked) > 1 else set())
        return (
            {item: [] for item in items if item in answered},
            {},
            {item: notes[item] for item in items if item not in answered},
        )

    class Provider:
        def generate(self, request):
            asked.append(tuple(request.input_text.split("|")))
            return GenerationResult(text="{}", provider="scripted", model="model")

    counter = ConservativeUtf8TokenCounter()
    own = _measure_request_tokens(
        build(("b",), _response_correction([notes["b"]], phase=GenerationPhase.DECOMPOSITION)), counter
    )
    joined = _measure_request_tokens(
        build(("b",), _response_correction(list(notes.values()), phase=GenerationPhase.DECOMPOSITION)), counter
    )
    assert own < joined
    config = VerificationConfig(
        enabled=True, request_tokens=(own + joined) // 2, output_reserve_tokens=1, safety_margin_tokens=0
    )

    resolution = _resolve_work(
        (("a", "b", "c", "d"),),
        item_id=lambda item: item,
        build_request=build,
        parse=parse,
        phase=GenerationPhase.DECOMPOSITION,
        runtime=VerificationRuntime(
            provider=Provider(),
            counter=counter,
            model="model",
            timeout_seconds=30,
            context_window_tokens=100_000,
        ),
        config=config,
        before_call=lambda _items, _retry: None,
        generations=[],
    )

    assert ("b",) in asked
    assert set(resolution.values) == {"a", "c", "d"}
    assert resolution.unresolved == {"b": "invalid_response"}
