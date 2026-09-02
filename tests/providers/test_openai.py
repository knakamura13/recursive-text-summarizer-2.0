from types import SimpleNamespace
import importlib

import httpx2
import openai
import pytest

from summarizer.providers.base import (
    GenerationRequest,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from summarizer.providers.openai import OpenAIProvider


REQUEST = GenerationRequest(
    model="gpt-4o-mini",
    instructions="Summarize accurately.",
    input_text='\n"""\nSensitive source\n"""\n',
    timeout_seconds=30,
    operation_id="leaf-17",
)


class FakeResponses:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def test_adapts_request_response_and_constructs_client_lazily() -> None:
    response = SimpleNamespace(
        output_text="  concise\nsummary  ",
        model="gpt-4o-mini-2024-07-18",
        status="completed",
        usage=SimpleNamespace(input_tokens=100, output_tokens=12),
        _request_id="req_123",
    )
    responses = FakeResponses(response)
    factory_calls: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> object:
        factory_calls.append(kwargs)
        return SimpleNamespace(responses=responses)

    provider = OpenAIProvider(client_factory=client_factory)
    assert factory_calls == []

    first = provider.generate(REQUEST)
    second = provider.generate(REQUEST)

    assert factory_calls == [{"max_retries": 0}]
    assert responses.calls == [
        {
            "model": "gpt-4o-mini",
            "instructions": "Summarize accurately.",
            "input": '\n"""\nSensitive source\n"""\n',
            "timeout": 30,
        },
        {
            "model": "gpt-4o-mini",
            "instructions": "Summarize accurately.",
            "input": '\n"""\nSensitive source\n"""\n',
            "timeout": 30,
        },
    ]
    assert first.text == "concise summary"
    assert first.provider == "openai"
    assert first.model == "gpt-4o-mini-2024-07-18"
    assert first.input_tokens == 100
    assert first.output_tokens == 12
    assert first.finish_status == "completed"
    assert first.request_id == "req_123"
    assert second == first


@pytest.mark.parametrize("output_text", [None, "", "   ", 42])
def test_rejects_missing_or_invalid_output_text(output_text: object) -> None:
    provider = OpenAIProvider(
        client_factory=lambda **_kwargs: SimpleNamespace(
            responses=FakeResponses(SimpleNamespace(output_text=output_text))
        )
    )

    with pytest.raises(ProviderResponseError, match="text"):
        provider.generate(REQUEST)


def test_allows_absent_optional_response_metadata() -> None:
    provider = OpenAIProvider(
        client_factory=lambda **_kwargs: SimpleNamespace(
            responses=FakeResponses(SimpleNamespace(output_text="summary"))
        )
    )

    result = provider.generate(REQUEST)

    assert result.model == REQUEST.model
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.finish_status is None
    assert result.request_id is None


def test_import_does_not_construct_an_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fail_if_called(**kwargs: object) -> object:
        calls.append(kwargs)
        raise AssertionError("OpenAI client constructed during import")

    monkeypatch.setattr(openai, "OpenAI", fail_if_called)
    module = importlib.import_module("summarizer.providers.openai")

    importlib.reload(module)

    assert calls == []


def _sdk_error(name: str) -> BaseException:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    if name == "timeout":
        return openai.APITimeoutError(request)
    if name == "connection":
        return openai.APIConnectionError(request=request)
    status_by_name = {
        "rate_limit": 429,
        "authentication": 401,
        "request": 400,
        "server": 500,
    }
    response = httpx2.Response(status_by_name[name], request=request)
    class_by_name = {
        "rate_limit": openai.RateLimitError,
        "authentication": openai.AuthenticationError,
        "request": openai.BadRequestError,
        "server": openai.InternalServerError,
    }
    return class_by_name[name](
        "sentinel credential and Sensitive source",
        response=response,
        body=None,
    )


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("timeout", ProviderTimeoutError),
        ("rate_limit", ProviderRateLimitError),
        ("connection", ProviderConnectionError),
        ("authentication", ProviderAuthenticationError),
        ("request", ProviderRequestError),
        ("server", ProviderServerError),
    ],
)
def test_translates_sdk_errors_without_sensitive_content(
    name: str,
    expected_type: type[BaseException],
) -> None:
    sdk_error = _sdk_error(name)
    provider = OpenAIProvider(
        client_factory=lambda **_kwargs: SimpleNamespace(
            responses=FakeResponses(sdk_error)
        )
    )

    with pytest.raises(expected_type) as exc_info:
        provider.generate(REQUEST)

    message = str(exc_info.value)
    assert "Sensitive source" not in message
    assert "sentinel credential" not in message
    assert exc_info.value.__cause__ is sdk_error
