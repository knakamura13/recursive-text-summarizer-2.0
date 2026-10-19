"""OpenRouter adapter tests.

Requests go through a real `httpx.Client` with a mock transport, so the URL,
headers and JSON body asserted here are the bytes that would be sent. The
responses are built from OpenRouter's documented formats (see
`tests/fixtures/openrouter/README.md`); they were not recorded from live
traffic.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
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
from summarizer.providers.openrouter import (
    OPENROUTER_BASE_URL,
    OpenRouterProvider,
    openrouter_api_key_is_set,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "openrouter"
KEY = "sk-or-v1-testsentinel0123456789abcdef0123456789abcdef"
SOURCE_TEXT = "Distinctive source passage that must never appear in an error."
MODEL = "meta-llama/llama-3.3-70b-instruct"

REQUEST = GenerationRequest(
    model=MODEL,
    instructions="Summarize accurately.",
    input_text=SOURCE_TEXT,
    timeout_seconds=30,
    operation_id="leaf-17",
)
SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}}
SCHEMA_REQUEST = GenerationRequest(
    model=MODEL,
    instructions="Summarize accurately.",
    input_text=SOURCE_TEXT,
    timeout_seconds=30,
    response_schema=SCHEMA,
    schema_name="leaf_summary",
)


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)


def fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def provider_for(
    handler: object, *, upstream: str = ""
) -> tuple[OpenRouterProvider, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)  # type: ignore[operator]

    def client_factory(**kwargs: object) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(respond), **kwargs)

    return OpenRouterProvider(upstream=upstream, client_factory=client_factory), requests


def replying(status: int, name: str) -> object:
    return lambda request: httpx.Response(status, json=fixture(name))


def sent_body(request: httpx.Request) -> dict[str, object]:
    return json.loads(request.content)


# --- request shape -----------------------------------------------------------


def test_prose_request_is_a_chat_completion_with_system_and_user_messages() -> None:
    provider, requests = provider_for(replying(200, "chat_completion_ok.json"))

    provider.generate(REQUEST)

    (request,) = requests
    assert request.method == "POST"
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["content-type"] == "application/json"
    assert sent_body(request) == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Summarize accurately."},
            {"role": "user", "content": SOURCE_TEXT},
        ],
        "stream": False,
    }


def test_structured_request_sends_a_strict_json_schema_and_requires_parameters() -> None:
    provider, requests = provider_for(
        replying(200, "chat_completion_structured_ok.json")
    )

    provider.generate(SCHEMA_REQUEST)

    body = sent_body(requests[0])
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "leaf_summary", "strict": True, "schema": SCHEMA},
    }
    # Without this a request can route to an endpoint that ignores the schema.
    assert body["provider"] == {"require_parameters": True}


def test_prose_request_does_not_restrict_routing_by_parameters() -> None:
    provider, requests = provider_for(replying(200, "chat_completion_ok.json"))

    provider.generate(REQUEST)

    assert "provider" not in sent_body(requests[0])
    assert "response_format" not in sent_body(requests[0])


def test_forwards_the_output_token_limit_as_max_tokens() -> None:
    provider, requests = provider_for(replying(200, "chat_completion_ok.json"))
    request = GenerationRequest(**{**REQUEST.__dict__, "max_output_tokens": 512})

    provider.generate(request)

    assert sent_body(requests[0])["max_tokens"] == 512


@pytest.mark.parametrize(
    ("request_", "routing"),
    [
        (REQUEST, {"order": ["deepinfra/turbo"], "allow_fallbacks": False}),
        (
            SCHEMA_REQUEST,
            {
                "order": ["deepinfra/turbo"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        ),
    ],
    ids=["prose", "schema"],
)
def test_pinned_upstream_disables_fallbacks_for_every_request(
    request_: GenerationRequest, routing: dict[str, object]
) -> None:
    provider, requests = provider_for(
        replying(200, "chat_completion_structured_ok.json"), upstream="deepinfra/turbo"
    )

    provider.generate(request_)

    assert sent_body(requests[0])["provider"] == routing


def test_hosted_only_constraints_are_not_sent_to_the_model() -> None:
    provider, requests = provider_for(
        replying(200, "chat_completion_structured_ok.json")
    )
    request = GenerationRequest(
        **{
            **SCHEMA_REQUEST.__dict__,
            "quote_candidates_by_segment": {"S1": ("a quote",)},
            "allowed_summary_segment_ids": ("S1",),
            "expected_summary_level": 0,
            "audit_work_id": "internal-work-id",
            "operation_id": "leaf-17",
        }
    )

    provider.generate(request)

    wire = requests[0].content.decode("utf-8")
    for internal in ("a quote", "internal-work-id", "leaf-17", "allowed_summary"):
        assert internal not in wire


def test_the_openai_environment_cannot_redirect_or_decorate_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-be-sent-0123456789")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-must-not-be-sent")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-must-not-be-sent")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer openai-secret")
    provider, requests = provider_for(replying(200, "chat_completion_ok.json"))

    provider.generate(REQUEST)

    (request,) = requests
    assert request.url.host == "openrouter.ai"
    sent_headers = "\n".join(f"{k}: {v}" for k, v in request.headers.items())
    for foreign in ("must-not-be-sent", "openai-secret", "attacker"):
        assert foreign not in sent_headers
    assert request.headers["authorization"] == f"Bearer {KEY}"


def test_default_client_targets_the_documented_base_url() -> None:
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    created: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> httpx.Client:
        created.append(kwargs)
        return httpx.Client(
            transport=httpx.MockTransport(replying(200, "chat_completion_ok.json")),
            **kwargs,
        )

    OpenRouterProvider(client_factory=client_factory).generate(REQUEST)

    assert str(created[0]["base_url"]).rstrip("/") == OPENROUTER_BASE_URL
    assert created[0]["follow_redirects"] is False


# --- response mapping ----------------------------------------------------------


def test_prose_response_is_normalized_and_usage_is_recorded() -> None:
    provider, _ = provider_for(replying(200, "chat_completion_ok.json"))

    result = provider.generate(REQUEST)

    assert result.text == "A concise summary."
    assert result.provider == "openrouter"
    assert result.model == MODEL
    assert result.input_tokens == 412
    assert result.output_tokens == 87
    assert result.finish_status == "stop"
    assert result.request_id == "gen-1760000000-AbCdEfGhIjKlMnOpQrSt"


def test_structured_response_keeps_its_whitespace() -> None:
    provider, _ = provider_for(replying(200, "chat_completion_structured_ok.json"))

    result = provider.generate(SCHEMA_REQUEST)

    assert json.loads(result.text) == {"summary": "line one\n  line two", "level": 0}


def test_missing_or_invalid_usage_is_recorded_as_unknown() -> None:
    for usage in (None, {}, {"prompt_tokens": -1, "completion_tokens": True},
                  {"prompt_tokens": "12", "completion_tokens": 1.5}, "oops"):
        body = fixture("chat_completion_ok.json")
        body["usage"] = usage
        provider, _ = provider_for(lambda request: httpx.Response(200, json=body))

        result = provider.generate(REQUEST)

        assert (result.input_tokens, result.output_tokens) == (None, None), usage


def test_a_response_model_that_is_blank_falls_back_to_the_requested_model() -> None:
    body = fixture("chat_completion_ok.json")
    body["model"] = "  "
    provider, _ = provider_for(lambda request: httpx.Response(200, json=body))

    assert provider.generate(REQUEST).model == MODEL


def test_an_unsafe_generation_id_is_not_recorded() -> None:
    body = fixture("chat_completion_ok.json")
    body["id"] = "gen-1 with spaces and sk-or-v1-" + "a" * 40
    provider, _ = provider_for(lambda request: httpx.Response(200, json=body))

    assert provider.generate(REQUEST).request_id is None


@pytest.mark.parametrize(
    ("finish_reason", "expected", "match"),
    [
        ("length", ProviderResponseError, "output token limit"),
        ("content_filter", ProviderResponseError, "content_filter"),
        ("tool_calls", ProviderResponseError, "tool_calls"),
        ("error", ProviderServerError, "upstream"),
    ],
)
def test_terminal_states_other_than_stop_are_not_accepted(
    finish_reason: str, expected: type[BaseException], match: str
) -> None:
    body = fixture("chat_completion_ok.json")
    body["choices"][0]["finish_reason"] = finish_reason  # type: ignore[index]
    provider, _ = provider_for(lambda request: httpx.Response(200, json=body))

    with pytest.raises(expected, match=match):
        provider.generate(
            GenerationRequest(**{**REQUEST.__dict__, "max_output_tokens": 4096})
        )


def test_length_stop_names_the_configured_limit_rather_than_calling_it_malformed() -> None:
    provider, _ = provider_for(replying(200, "chat_completion_length.json"))
    request = GenerationRequest(**{**SCHEMA_REQUEST.__dict__, "max_output_tokens": 4096})

    with pytest.raises(ProviderResponseError, match="4096"):
        provider.generate(request)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body.update(choices=[]),
        lambda body: body.pop("choices"),
        lambda body: body.update(choices=["not a mapping"]),
        lambda body: body["choices"][0].pop("message"),
        lambda body: body["choices"][0]["message"].update(content=None),
        lambda body: body["choices"][0]["message"].update(content="   "),
        lambda body: body["choices"][0]["message"].update(content=["parts"]),
        lambda body: body["choices"][0]["message"].update(refusal="I cannot help."),
    ],
    ids=[
        "no-choices", "choices-missing", "choice-not-mapping", "message-missing",
        "content-null", "content-blank", "content-not-text", "refusal",
    ],
)
def test_responses_without_usable_text_are_rejected(mutate: object) -> None:
    body = fixture("chat_completion_ok.json")
    mutate(body)  # type: ignore[operator]
    provider, _ = provider_for(lambda request: httpx.Response(200, json=body))

    with pytest.raises(ProviderResponseError):
        provider.generate(REQUEST)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>gateway splash</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, content=b""),
        httpx.Response(307, headers={"location": "https://elsewhere.example/"}),
    ],
    ids=["html", "json-list", "empty", "redirect"],
)
def test_malformed_or_redirected_responses_are_response_errors(
    response: httpx.Response,
) -> None:
    provider, requests = provider_for(lambda request: response)

    with pytest.raises(ProviderResponseError):
        provider.generate(REQUEST)

    assert len(requests) == 1  # a redirect is never followed with the key


# --- error mapping -------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "name", "expected"),
    [
        (401, "error_401.json", ProviderAuthenticationError),
        (402, "error_402_credits.json", ProviderRequestError),
        (402, "error_402_key_limit.json", ProviderRequestError),
        (402, "error_402_in_flight.json", ProviderRateLimitError),
        (404, "error_404_no_endpoint.json", ProviderRequestError),
        (429, "error_429.json", ProviderRateLimitError),
        (502, "error_502.json", ProviderServerError),
        (503, "error_502.json", ProviderServerError),
        (408, "error_502.json", ProviderTimeoutError),
        (400, "error_404_no_endpoint.json", ProviderRequestError),
        (403, "error_401.json", ProviderRequestError),
    ],
)
def test_http_errors_map_onto_provider_error_classes(
    status: int, name: str, expected: type[BaseException]
) -> None:
    provider, _ = provider_for(replying(status, name))

    with pytest.raises(expected) as caught:
        provider.generate(REQUEST)

    assert type(caught.value) is expected


def test_the_two_non_retryable_402_causes_say_which_limit_was_hit() -> None:
    credits, _ = provider_for(replying(402, "error_402_credits.json"))
    key_limit, _ = provider_for(replying(402, "error_402_key_limit.json"))

    with pytest.raises(ProviderRequestError, match="credits"):
        credits.generate(REQUEST)
    with pytest.raises(ProviderRequestError, match="spend limit|credit limit"):
        key_limit.generate(REQUEST)


def test_an_error_inside_a_200_response_is_mapped_by_its_own_code() -> None:
    provider, _ = provider_for(replying(200, "error_in_200.json"))

    with pytest.raises(ProviderServerError):
        provider.generate(REQUEST)


def test_error_bodies_that_are_not_json_still_map_by_status() -> None:
    provider, _ = provider_for(
        lambda request: httpx.Response(502, text="<html>Bad gateway</html>")
    )

    with pytest.raises(ProviderServerError):
        provider.generate(REQUEST)


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"code": "not-a-number", "message": "x"}},
        {"error": "just a string"},
        {"error": None},
    ],
)
def test_a_200_with_an_unusable_error_object_is_a_response_error(body: object) -> None:
    provider, _ = provider_for(lambda request: httpx.Response(200, json=body))

    with pytest.raises(ProviderResponseError):
        provider.generate(REQUEST)


@pytest.mark.parametrize(
    "name", ["error_429.json", "error_402_credits.json", "error_502.json", "error_401.json"]
)
def test_error_messages_carry_no_provider_text_key_or_source(name: str) -> None:
    status = int(fixture(name)["error"]["code"])  # type: ignore[index]
    provider, _ = provider_for(replying(status, name))

    with pytest.raises(Exception) as caught:
        provider.generate(REQUEST)

    chain = []
    error: BaseException | None = caught.value
    while error is not None:
        chain.append(f"{error!s} {error!r}")
        error = error.__cause__
    text = "\n".join(chain)
    for forbidden in (KEY, SOURCE_TEXT, "echoed request text", "upstream detail", "Insufficient credits. Add"):
        assert forbidden not in text


def test_the_upstream_provider_name_is_reported_only_when_it_is_a_plain_name() -> None:
    plain, _ = provider_for(replying(429, "error_429.json"))
    with pytest.raises(ProviderRateLimitError, match="DeepInfra"):
        plain.generate(REQUEST)

    body = fixture("error_429.json")
    body["error"]["metadata"]["provider_name"] = "evil\nInjected: header\x1b[31m"  # type: ignore[index]
    hostile, _ = provider_for(lambda request: httpx.Response(429, json=body))
    with pytest.raises(ProviderRateLimitError) as caught:
        hostile.generate(REQUEST)
    assert "Injected" not in str(caught.value)
    assert "\x1b" not in str(caught.value)


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (httpx.ReadTimeout("slow"), ProviderTimeoutError),
        (httpx.ConnectTimeout("slow"), ProviderTimeoutError),
        (httpx.ConnectError("refused"), ProviderConnectionError),
        (httpx.RemoteProtocolError("reset"), ProviderConnectionError),
    ],
)
def test_transport_failures_map_onto_provider_error_classes(
    raised: Exception, expected: type[BaseException]
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise raised

    provider, _ = provider_for(fail)

    with pytest.raises(expected):
        provider.generate(REQUEST)


# --- key handling ----------------------------------------------------------------


def test_missing_key_is_an_actionable_error_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    provider, requests = provider_for(replying(200, "chat_completion_ok.json"))

    with pytest.raises(ProviderAuthenticationError, match="OPENROUTER_API_KEY"):
        provider.generate(REQUEST)

    assert requests == []


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_blank_key_is_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", value)
    provider, requests = provider_for(replying(200, "chat_completion_ok.json"))

    with pytest.raises(ProviderAuthenticationError, match="not set"):
        provider.generate(REQUEST)

    assert requests == []
    assert openrouter_api_key_is_set() is False


@pytest.mark.parametrize(
    "value",
    [
        "sk-or-v1-abc\ndef",
        "sk-or-v1-abc def",
        "sk-or-v1-abc\x07def",
        "sk-or-v1-ключ",
    ],
)
def test_a_malformed_key_is_rejected_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", value)
    provider, requests = provider_for(replying(200, "chat_completion_ok.json"))

    with pytest.raises(ProviderAuthenticationError) as caught:
        provider.generate(REQUEST)

    assert "abc" not in str(caught.value) and "ключ" not in str(caught.value)
    assert requests == []


def test_surrounding_whitespace_in_the_key_is_trimmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", f"  {KEY}\n")
    provider, requests = provider_for(replying(200, "chat_completion_ok.json"))

    provider.generate(REQUEST)

    assert requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_the_key_never_appears_in_the_provider_repr_or_attributes() -> None:
    provider, _ = provider_for(replying(200, "chat_completion_ok.json"))
    provider.generate(REQUEST)

    assert KEY not in repr(provider)
    assert KEY not in repr(vars(provider))


def test_the_key_is_read_at_first_use_not_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    provider, requests = provider_for(replying(200, "chat_completion_ok.json"))
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)

    provider.generate(REQUEST)

    assert requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_importing_the_module_does_not_read_the_key_or_build_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    import summarizer.providers.openrouter as module

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("client constructed during import")

    monkeypatch.setattr(httpx, "Client", fail)
    monkeypatch.delenv("OPENROUTER_API_KEY")

    importlib.reload(module)


# --- concurrency and accounting -------------------------------------------------


def test_concurrent_first_calls_construct_one_client() -> None:
    constructed: list[object] = []
    gate = threading.Barrier(4)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fixture("chat_completion_ok.json"))

    def client_factory(**kwargs: object) -> httpx.Client:
        constructed.append(kwargs)
        time.sleep(0.05)
        return httpx.Client(transport=httpx.MockTransport(respond), **kwargs)

    provider = OpenRouterProvider(client_factory=client_factory)

    def call(_: int) -> None:
        gate.wait(timeout=5)
        provider.generate(REQUEST)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(call, range(4)))

    assert len(constructed) == 1
    assert provider.request_count == 4


def test_request_count_includes_failed_attempts() -> None:
    outcomes = iter([429, 200])

    def respond(request: httpx.Request) -> httpx.Response:
        if next(outcomes) == 429:
            return httpx.Response(429, json=fixture("error_429.json"))
        return httpx.Response(200, json=fixture("chat_completion_ok.json"))

    provider, _ = provider_for(respond)

    with pytest.raises(ProviderRateLimitError):
        provider.generate(REQUEST)
    provider.generate(REQUEST)

    assert provider.request_count == 2
