import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import httpx
import ollama
import pytest
from pydantic import ValidationError

from summarizer.providers.base import (
    GenerationRequest,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from summarizer.providers.ollama import OllamaProvider
from summarizer.summaries import (
    MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES,
    leaf_summary_schema,
    summary_schema,
)

REQUEST = GenerationRequest(
    model="gemma3:4b",
    instructions="Summarize accurately.",
    input_text="Source material",
    timeout_seconds=42,
)


class FakeClient:
    def __init__(
        self,
        outcome: object,
        modelinfo: dict[str, object] | None = None,
        show_error: BaseException | None = None,
    ) -> None:
        self.outcome = outcome
        self.modelinfo = modelinfo or {
            "general.architecture": "gemma3",
            "gemma3.context_length": 131_072,
        }
        self.show_error = show_error
        self.calls: list[dict[str, object]] = []
        self.show_calls: list[str] = []

    def show(self, model: str) -> SimpleNamespace:
        self.show_calls.append(model)
        if self.show_error is not None:
            raise self.show_error
        return SimpleNamespace(modelinfo=self.modelinfo)

    def chat(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def response(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "message": SimpleNamespace(content=" local\n summary "),
        "model": "gemma3:4b-q4_K_M",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 42,
        "eval_count": 11,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_adapts_native_chat_request_and_response_lazily() -> None:
    client = FakeClient(response())
    constructions: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> FakeClient:
        constructions.append(kwargs)
        return client

    provider = OllamaProvider(
        host="http://ollama.internal:11434",
        client_factory=client_factory,
    )

    assert constructions == []

    result = provider.generate(REQUEST)

    assert constructions == [
        {"host": "http://ollama.internal:11434", "timeout": 42}
    ]
    assert client.calls == [
        {
            "model": "gemma3:4b",
            "messages": [
                {"role": "system", "content": "Summarize accurately."},
                {"role": "user", "content": "Source material"},
            ],
            "stream": False,
            "think": False,
        }
    ]
    assert result.text == "local summary"
    assert result.provider == "ollama"
    assert result.model == "gemma3:4b-q4_K_M"
    assert result.input_tokens == 42
    assert result.output_tokens == 11
    assert result.finish_status == "stop"
    assert result.request_id is None

    provider.generate(REQUEST)
    assert len(constructions) == 1



def test_configures_and_enforces_discovered_context_window() -> None:
    client = FakeClient(response())
    provider = OllamaProvider(client_factory=lambda **_kwargs: client)

    selected = provider.configure_context_window(
        REQUEST.model,
        None,
        timeout_seconds=REQUEST.timeout_seconds,
    )
    provider.generate(REQUEST)

    assert selected == 32_768
    assert client.show_calls == [REQUEST.model]
    assert client.calls[0]["options"] == {"num_ctx": 32_768}


def test_forwards_output_token_limit_with_context_window() -> None:
    client = FakeClient(response())
    provider = OllamaProvider(client_factory=lambda **_kwargs: client)
    provider.configure_context_window(REQUEST.model, 32_768, timeout_seconds=42)

    provider.generate(replace(REQUEST, max_output_tokens=321))

    assert client.calls[0]["options"] == {"num_ctx": 32_768, "num_predict": 321}


def test_reports_a_response_cut_off_at_the_output_limit() -> None:
    client = FakeClient(response(done_reason="length"))
    provider = OllamaProvider(client_factory=lambda **_kwargs: client)

    with pytest.raises(ProviderResponseError, match="output token limit of 321"):
        provider.generate(replace(REQUEST, max_output_tokens=321))


def test_rejects_context_larger_than_model_architecture() -> None:
    client = FakeClient(
        response(),
        modelinfo={
            "general.architecture": "tiny",
            "tiny.context_length": 8_192,
        },
    )
    provider = OllamaProvider(client_factory=lambda **_kwargs: client)

    with pytest.raises(ProviderRequestError, match="exceeds the model maximum"):
        provider.configure_context_window(
            REQUEST.model,
            16_384,
            timeout_seconds=REQUEST.timeout_seconds,
        )



def test_sanitizes_malformed_model_metadata() -> None:
    client = FakeClient(
        response(),
        show_error=json.JSONDecodeError("secret response body", "secret", 0),
    )
    provider = OllamaProvider(client_factory=lambda **_kwargs: client)

    with pytest.raises(ProviderResponseError) as error:
        provider.configure_context_window(
            REQUEST.model,
            None,
            timeout_seconds=REQUEST.timeout_seconds,
        )

    assert str(error.value) == "Ollama returned malformed model metadata"


def test_uses_a_client_with_each_distinct_request_timeout() -> None:
    constructions: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> FakeClient:
        constructions.append(kwargs)
        return FakeClient(response())

    provider = OllamaProvider(client_factory=client_factory)

    provider.generate(REQUEST)
    provider.generate(
        GenerationRequest(
            model=REQUEST.model,
            instructions=REQUEST.instructions,
            input_text=REQUEST.input_text,
            timeout_seconds=90,
        )
    )
    provider.generate(REQUEST)

    assert constructions == [
        {"host": "http://localhost:11434", "timeout": 42},
        {"host": "http://localhost:11434", "timeout": 90},
    ]


def test_concurrent_first_calls_construct_one_client_per_timeout_without_deadlock() -> None:
    constructions: list[dict[str, object]] = []
    factory_barrier = threading.Barrier(2)
    start = threading.Barrier(3)

    def client_factory(**kwargs: object) -> FakeClient:
        constructions.append(kwargs)
        try:
            factory_barrier.wait(timeout=0.1)
        except threading.BrokenBarrierError:
            pass
        return FakeClient(response())

    provider = OllamaProvider(client_factory=client_factory)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lambda: (start.wait(), provider.generate(REQUEST))[1]) for _ in range(2)]
        start.wait()
        results = [future.result(timeout=2) for future in futures]

    provider.generate(
        GenerationRequest(
            model=REQUEST.model,
            instructions=REQUEST.instructions,
            input_text=REQUEST.input_text,
            timeout_seconds=90,
        )
    )

    assert constructions == [
        {"host": "http://localhost:11434", "timeout": 42},
        {"host": "http://localhost:11434", "timeout": 90},
    ]
    assert [result.text for result in results] == ["local summary", "local summary"]


@pytest.mark.parametrize(
    "bad_response",
    [
        response(message=SimpleNamespace(content=" ")),
        response(message=None),
        response(done=False),
        response(done=None),
    ],
)
def test_rejects_nonterminal_or_missing_content(bad_response: object) -> None:
    provider = OllamaProvider(
        client_factory=lambda **_kwargs: FakeClient(bad_response)
    )

    with pytest.raises(ProviderResponseError):
        provider.generate(REQUEST)


@pytest.mark.parametrize(
    ("error", "expected_type", "message"),
    [
        (
            httpx.ReadTimeout("secret source"),
            ProviderTimeoutError,
            "timed out",
        ),
        (
            ConnectionError("secret source"),
            ProviderConnectionError,
            "connection failed",
        ),
        (
            ollama.ResponseError("model secret-model not found", 404),
            ProviderRequestError,
            "model was not found",
        ),
        (
            ollama.ResponseError("secret source", 429),
            ProviderRateLimitError,
            "rate limited",
        ),
        (
            ollama.ResponseError("secret source", 500),
            ProviderServerError,
            "server request failed",
        ),
        (
            ollama.ResponseError("secret source", 502),
            ProviderServerError,
            "server request failed",
        ),
        (
            ollama.ResponseError("secret source", 400),
            ProviderRequestError,
            "rejected",
        ),
    ],
)
def test_translates_native_errors_without_sensitive_content(
    error: Exception,
    expected_type: type[Exception],
    message: str,
) -> None:
    provider = OllamaProvider(
        client_factory=lambda **_kwargs: FakeClient(error)
    )

    with pytest.raises(expected_type, match=message) as exc_info:
        provider.generate(REQUEST)

    assert exc_info.value.__cause__ is error
    assert "secret" not in str(exc_info.value)


def test_rejects_invalid_metadata() -> None:
    provider = OllamaProvider(
        client_factory=lambda **_kwargs: FakeClient(
            response(prompt_eval_count=-1)
        )
    )

    with pytest.raises(ProviderResponseError, match="metadata"):
        provider.generate(REQUEST)


def test_rejects_non_string_model_metadata() -> None:
    provider = OllamaProvider(
        client_factory=lambda **_kwargs: FakeClient(response(model=123))
    )

    with pytest.raises(ProviderResponseError, match="metadata"):
        provider.generate(REQUEST)


@pytest.mark.parametrize(
    "error",
    [
        json.JSONDecodeError("secret response body", "secret response body", 0),
        ValidationError.from_exception_data("secret schema", []),
    ],
)
def test_translates_malformed_successful_response_without_leaking_body(
    error: Exception,
) -> None:
    provider = OllamaProvider(
        client_factory=lambda **_kwargs: FakeClient(error)
    )

    with pytest.raises(ProviderResponseError, match="malformed") as exc_info:
        provider.generate(REQUEST)

    assert exc_info.value.__cause__ is error
    assert "secret" not in str(exc_info.value)


SCHEMA_REQUEST = GenerationRequest(
    model="gemma3:4b",
    instructions="Summarize accurately.",
    input_text="Source material",
    timeout_seconds=42,
    response_schema={"type": "object", "properties": {}},
    schema_name="leaf_summary",
)


def _provider_for(client: FakeClient) -> OllamaProvider:
    return OllamaProvider(client_factory=lambda **kwargs: client)


def test_omits_format_when_no_schema_is_requested() -> None:
    client = FakeClient(response())

    _provider_for(client).generate(REQUEST)

    assert "format" not in client.calls[0]


def test_strengthens_summary_schema_for_local_constrained_decoding() -> None:
    schema = leaf_summary_schema()
    request = GenerationRequest(
        model="gemma3:4b",
        instructions="Summarize accurately.",
        input_text="Source material",
        timeout_seconds=42,
        operation_id="S000001",
        response_schema=schema,
        schema_name="leaf_summary",
        quote_candidates_by_segment={"S000001": ("quote from one",)},
        expected_summary_level=0,
        allowed_summary_segment_ids=("S000001",),
    )
    client = FakeClient(response(message=SimpleNamespace(content="{}")))

    _provider_for(client).generate(request)

    emitted = client.calls[0]["format"]
    assert isinstance(emitted, dict)
    definitions = emitted["$defs"]
    pair_variants = definitions["EvidenceItem"]["anyOf"]
    assert all(
        variant["required"] == ["segment_id", "quote"]
        and variant["additionalProperties"] is False
        for variant in pair_variants
    )
    assert pair_variants[0]["properties"] == {
        "segment_id": {
            "type": "string",
            "minLength": 1,
            "enum": ["S000001"],
        },
        "quote": {"type": "null"},
    }
    constrained_pairs = {
        variant["properties"]["quote"]["enum"][0]: variant["properties"][
            "segment_id"
        ]["enum"]
        for variant in pair_variants[1:]
    }
    assert constrained_pairs == {"quote from one": ["S000001"]}
    assert emitted["properties"]["level"]["enum"] == [0]
    assert emitted["properties"]["provenance"]["items"]["enum"] == ["S000001"]
    assert emitted["properties"]["quotations"]["maxItems"] == 5
    assert definitions["ContentUnit"]["properties"]["evidence"]["minItems"] == 1
    assert emitted["properties"]["summary"]["minLength"] == 1
    base_bytes = len(json.dumps(schema, separators=(",", ":"), sort_keys=True))
    emitted_bytes = len(json.dumps(emitted, separators=(",", ":"), sort_keys=True))
    assert emitted_bytes - base_bytes <= MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES
    assert schema == leaf_summary_schema()


def test_constrains_each_merge_quote_to_its_source_segment() -> None:
    quotes = ("quote from one", "quote from two")
    request = GenerationRequest(
        model="gemma3:4b",
        instructions="Summarize accurately.",
        input_text="Source material",
        timeout_seconds=42,
        response_schema=summary_schema(candidates=quotes),
        schema_name="merged_summary",
        quote_candidates_by_segment={
            "S000001": (quotes[0],),
            "S000002": (quotes[1],),
        },
        expected_summary_level=1,
        allowed_summary_segment_ids=("S000001", "S000002"),
    )
    client = FakeClient(response(message=SimpleNamespace(content="{}")))

    _provider_for(client).generate(request)

    emitted = client.calls[0]["format"]
    variants = emitted["$defs"]["EvidenceItem"]["anyOf"]
    constrained_pairs = {
        variant["properties"]["quote"]["enum"][0]: variant["properties"][
            "segment_id"
        ]["enum"]
        for variant in variants[1:]
    }
    assert constrained_pairs == {
        "quote from one": ["S000001"],
        "quote from two": ["S000002"],
    }
    assert variants[0]["properties"]["segment_id"]["enum"] == [
        "S000001",
        "S000002",
    ]
    assert emitted["properties"]["provenance"]["items"]["enum"] == [
        "S000001",
        "S000002",
    ]
    assert emitted["properties"]["level"]["enum"] == [1]


def test_bounds_pair_schema_when_a_quote_repeats_across_many_sources() -> None:
    quote = "A repeated exact quotation."
    schema = summary_schema(candidates=(quote,))
    source_count = 10_000
    request = GenerationRequest(
        model="gemma3:4b",
        instructions="Summarize accurately.",
        input_text="Source material",
        timeout_seconds=42,
        response_schema=schema,
        schema_name="merged_summary",
        quote_candidates_by_segment={
            f"S{index:06}": (quote,) for index in range(source_count)
        },
        allowed_summary_segment_ids=tuple(
            f"S{index:06}" for index in range(source_count)
        ),
    )
    client = FakeClient(response(message=SimpleNamespace(content="{}")))

    _provider_for(client).generate(request)

    emitted = client.calls[0]["format"]
    emitted_bytes = len(json.dumps(emitted, separators=(",", ":"), sort_keys=True))
    base_bytes = len(json.dumps(schema, separators=(",", ":"), sort_keys=True))
    variants = emitted["$defs"]["EvidenceItem"]["anyOf"]
    represented_sources = variants[0]["properties"]["segment_id"]["enum"]
    provenance_sources = emitted["properties"]["provenance"]["items"]["enum"]
    assert emitted_bytes - base_bytes <= MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES
    assert represented_sources == provenance_sources
    assert 0 < len(represented_sources) < source_count


def test_passes_a_requested_schema_as_the_native_format_argument() -> None:
    client = FakeClient(response(message=SimpleNamespace(content="{}")))

    _provider_for(client).generate(SCHEMA_REQUEST)

    assert client.calls[0]["format"] == {"type": "object", "properties": {}}


def test_preserves_response_whitespace_only_for_structured_requests() -> None:
    payload = '{\n  "summary": "a  b",\n  "quote": "line\\nbreak"\n}'
    client = FakeClient(response(message=SimpleNamespace(content=payload)))

    structured = _provider_for(client).generate(SCHEMA_REQUEST)

    assert structured.text == payload

    client = FakeClient(response())
    prose = _provider_for(client).generate(REQUEST)

    assert prose.text == "local summary"
