import json

import pytest

from summarizer.editorial import EditorialError, build_editorial_request, write_editorial
from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer.runtime.observers import ItemFailedError, RuntimeObserver, StageName
from summarizer.summaries import SummaryNode


SOURCE_ID = "a" * 64


def root() -> SummaryNode:
    return SummaryNode.model_validate(
        {
            "summary": "The source describes a qualified change.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": ["S000001"],
            "level": 1,
        }
    )


class Provider:
    def __init__(self, response: str = '{"text":"A coherent final summary."}') -> None:
        self.requests: list[GenerationRequest] = []
        self.response = response

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(text=self.response, provider="fake", model=request.model)


def test_request_is_a_dedicated_genre_neutral_fenced_final_call() -> None:
    request = build_editorial_request(
        root(), source_id=SOURCE_ID, model="m", timeout_seconds=30, target_words=120
    )

    assert request.operation_id == "editorial-final"
    assert request.response_schema is not None
    assert "near the intended length of about" in request.instructions
    assert "120 words" in request.instructions
    assert "without shortening it materially" in request.instructions
    assert "unsupported" in request.instructions
    assert "qualifications" in request.instructions
    assert "article" not in request.instructions.lower()
    assert "report" not in request.instructions.lower()
    assert root().summary not in request.instructions
    assert root().summary in request.input_text
    assert "GROUNDED-ROOT" in request.input_text


def test_final_writer_returns_redacted_plain_text_and_is_deterministic() -> None:
    first = Provider('{"text":"Keep sk-12345678901234567890 out."}')
    second = Provider('{"text":"Keep sk-12345678901234567890 out."}')

    one = write_editorial(
        root(), first, source_id=SOURCE_ID, model="m", timeout_seconds=30, target_words=50
    )
    two = write_editorial(
        root(), second, source_id=SOURCE_ID, model="m", timeout_seconds=30, target_words=50
    )

    assert one.text == "Keep [REDACTED] out."
    assert one == two
    assert first.requests == second.requests


@pytest.mark.parametrize(
    "secret",
    (
        "ghp_123456789012345678901234567890123456",
        "xoxb-1234567890-abcdefghij",
        "Authorization: Basic dXNlcjpwYXNzd29yZA==",
        "Authorization: Basic YTpi",
    ),
)
def test_final_writer_redacts_common_provider_credentials(secret: str) -> None:
    result = write_editorial(
        root(),
        Provider(json.dumps({"text": f"Do not disclose {secret}."})),
        source_id=SOURCE_ID,
        model="m",
        timeout_seconds=30,
        target_words=50,
    )

    assert secret not in result.text
    assert "[REDACTED]" in result.text


class ScriptedProvider:
    def __init__(self, *responses: str) -> None:
        self.requests: list[GenerationRequest] = []
        self.responses = list(responses)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(
            text=self.responses.pop(0), provider="fake", model=request.model
        )


def _recording_observer() -> tuple[RuntimeObserver, list[tuple[str, int | None, str | None]]]:
    events: list[tuple[str, int | None, str | None]] = []
    observer = RuntimeObserver(
        on_item=lambda event: events.append((event.state, event.attempt, event.message))
        if event.kind == "editorial" and event.work_id == "editorial-final"
        else None
    )
    return observer, events


def test_invalid_draft_is_reasked_with_the_validation_error_then_accepted() -> None:
    provider = ScriptedProvider("not json", '{"text":"A coherent final summary."}')
    observer, events = _recording_observer()

    result = write_editorial(
        root(),
        provider,
        source_id=SOURCE_ID,
        model="m",
        timeout_seconds=30,
        target_words=50,
        observer=observer,
    )

    assert result.text == "A coherent final summary."
    first, second = provider.requests
    assert second.operation_id == first.operation_id == "editorial-final"
    assert second.input_text == first.input_text
    assert "rejected" in second.instructions
    assert "not json" not in second.instructions
    assert [state for state, _, _ in events] == ["active", "retrying", "completed"]
    assert events[1][1] == 1
    assert events[1][2]


@pytest.mark.parametrize("response", ["not json", json.dumps({"text": " "}), "{}"])
def test_draft_still_invalid_after_reasks_fails_the_editorial_item(response: str) -> None:
    provider = ScriptedProvider(response, response, response)
    observer, events = _recording_observer()

    with pytest.raises(ItemFailedError) as error:
        write_editorial(
            root(),
            provider,
            source_id=SOURCE_ID,
            model="m",
            timeout_seconds=30,
            target_words=50,
            observer=observer,
        )

    assert len(provider.requests) == 3
    assert error.value.stage is StageName.WRITING
    assert (error.value.kind, error.value.work_id) == ("editorial", "editorial-final")
    assert isinstance(error.value.__cause__, EditorialError)
    assert response not in str(error.value)
    assert [state for state, _, _ in events] == ["active", "retrying", "retrying", "failed"]
    assert events[-1][2] == str(error.value)
