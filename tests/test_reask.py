import json

import pytest

from summarizer.leaf import LeafSummaryError
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderConnectionError,
    ProviderResponseError,
)
from summarizer.reask import generate_validated, rejection_reason
from summarizer.runtime.observers import PipelineStopped
from summarizer.summaries import SummaryNode

REQUEST = GenerationRequest(
    model="m",
    instructions="Return one JSON object.",
    input_text="-----BEGIN-----\nThe archive moved in March.\n-----END-----",
    timeout_seconds=30,
    operation_id="S000001",
    response_schema={"type": "object"},
    schema_name="leaf_summary",
    audit_work_id="S000001",
)


class ScriptedProvider:
    """Answer each call with the next scripted text or raise the scripted error."""

    def __init__(self, *answers: str | Exception) -> None:
        self.answers = list(answers)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return GenerationResult(text=answer, provider="fake", model=request.model)


def parse(result: GenerationResult) -> dict[str, object]:
    try:
        value = json.loads(result.text)
    except json.JSONDecodeError as error:
        raise LeafSummaryError(f"S000001: response was not JSON ({error.msg})") from error
    if "summary" not in value:
        raise LeafSummaryError("S000001: response failed validation (summary: missing)")
    return value


def test_an_invalid_answer_is_reasked_with_its_validation_error() -> None:
    provider = ScriptedProvider('{"text": "wrong shape"}', '{"summary": "fixed"}')
    retries: list[tuple[int, str]] = []

    value = generate_validated(
        provider, REQUEST, parse, on_retry=lambda attempt, reason: retries.append((attempt, reason))
    )

    assert value == {"summary": "fixed"}
    assert retries == [(1, "S000001: response failed validation (summary: missing)")]
    first, second = provider.requests
    assert first == REQUEST
    assert second.instructions.startswith(REQUEST.instructions)
    assert "rejected: S000001: response failed validation (summary: missing)." in (
        second.instructions
    )
    # Only the instructions change, so the item keeps its identity and cache key.
    assert second.input_text == REQUEST.input_text
    assert second.response_schema == REQUEST.response_schema
    assert (second.operation_id, second.audit_work_id) == ("S000001", "S000001")


def test_spent_reasks_reraise_the_last_validation_error() -> None:
    provider = ScriptedProvider("not json", "still not json", '{"text": "no summary"}')
    retries: list[int] = []

    with pytest.raises(LeafSummaryError, match="summary: missing"):
        generate_validated(
            provider, REQUEST, parse, on_retry=lambda attempt, reason: retries.append(attempt)
        )

    assert len(provider.requests) == 3
    assert retries == [1, 2]
    # Each re-ask states only the latest error; notes never pile up.
    assert provider.requests[2].instructions.count("was rejected") == 1


def test_an_incomplete_provider_response_is_reasked() -> None:
    provider = ScriptedProvider(
        ProviderResponseError("Ollama stopped at the model's output limit; the response is incomplete"),
        '{"summary": "shorter"}',
    )
    retries: list[str] = []

    value = generate_validated(
        provider, REQUEST, parse, on_retry=lambda attempt, reason: retries.append(reason)
    )

    assert value == {"summary": "shorter"}
    assert retries == [
        "Ollama stopped at the model's output limit; the response is incomplete"
    ]


def test_other_failures_propagate_without_a_reask() -> None:
    provider = ScriptedProvider(ProviderConnectionError("unreachable"))
    retries: list[int] = []

    with pytest.raises(ProviderConnectionError):
        generate_validated(
            provider, REQUEST, parse, on_retry=lambda attempt, reason: retries.append(attempt)
        )

    assert len(provider.requests) == 1
    assert retries == []


def test_a_stop_raised_before_a_reask_prevents_the_next_call() -> None:
    provider = ScriptedProvider("not json", '{"summary": "never requested"}')

    def stop(attempt: int, reason: str) -> None:
        raise PipelineStopped("stopped before re-asking S000001")

    with pytest.raises(PipelineStopped):
        generate_validated(provider, REQUEST, parse, on_retry=stop)

    assert len(provider.requests) == 1


def test_a_raw_validation_error_is_described_without_payload_values() -> None:
    with pytest.raises(ValueError) as raised:
        SummaryNode.model_validate(
            {"summary": 7, "level": "SECRET-PAYLOAD-VALUE", "unexpected": "SECRET-TOO"}
        )

    reason = rejection_reason(raised.value)

    assert reason.startswith("response failed validation (")
    assert "level" in reason
    assert "SECRET" not in reason
    assert "\n" not in reason
