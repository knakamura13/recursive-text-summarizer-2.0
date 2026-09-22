import json

import pytest

from summarizer.editorial import EditorialError, build_editorial_request, write_editorial
from summarizer.providers.base import GenerationRequest, GenerationResult
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
    assert "about 120 words" in request.instructions
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


@pytest.mark.parametrize("response", ["not json", json.dumps({"text": " "}), "{}"])
def test_invalid_final_responses_are_rejected_without_echoing_them(response: str) -> None:
    with pytest.raises(EditorialError) as error:
        write_editorial(
            root(), Provider(response), source_id=SOURCE_ID, model="m", timeout_seconds=30, target_words=50
        )

    assert response not in str(error.value)
