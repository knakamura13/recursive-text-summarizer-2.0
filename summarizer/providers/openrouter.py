"""OpenRouter adapter over its chat-completions endpoint.

This talks to OpenRouter with `httpx` directly rather than through the OpenAI
SDK, for three reasons that were checked against the installed SDK source:

- The SDK reads `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` and
  `OPENAI_CUSTOM_HEADERS` from the environment. Pointed at a third party it
  would forward whatever a user has set for OpenAI, including an
  `Authorization` header. A client built here sends exactly one credential,
  read from `OPENROUTER_API_KEY`, to exactly one host.
- OpenRouter's provider routing (`provider.require_parameters`, pinning) is not
  an SDK parameter and would travel as free-form extra body anyway.
- OpenRouter documents structured outputs for chat completions. Its Responses
  endpoint is documented as stateless, and the page read for this change does
  not document `text.format` schemas or its usage fields.

Documented behavior this module relies on, with the pages read on 2026-10-19:
`POST /api/v1/chat/completions` with a bearer key; `response_format` of type
`json_schema`; `provider.order`, `provider.allow_fallbacks` and
`provider.require_parameters`; `usage.prompt_tokens` and
`usage.completion_tokens`; the error shape `{"error": {"code", "message",
"metadata"}}`, which can also arrive inside a `200` response; and 402
responses whose `metadata.limit_source` distinguishes an exhausted key limit,
an exhausted balance and an in-flight budget. None of it was exercised live.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from threading import Lock
from typing import Any

import httpx

from summarizer.config import normalize_openrouter_upstream
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    normalize_output_text,
)
from summarizer.safety import redact_text

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

# Printable ASCII without whitespace. A value outside this would make the HTTP
# layer raise an error whose text can contain the header value, so it is
# refused before a client exists.
_KEY_CHARACTERS = re.compile(r"^[\x21-\x7e]+$")
_PLAIN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._/-]{0,63}$")
_GENERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9~][A-Za-z0-9._/:-]{0,255}$")
_FINISH_REASON = re.compile(r"^[a-z_]{1,32}$")
_LIMIT_SOURCES = frozenset(
    {"openrouter_in_flight_budget", "openrouter_key_limit", "openrouter_credits"}
)


def _create_client(**kwargs: object) -> httpx.Client:
    return httpx.Client(**kwargs)  # type: ignore[arg-type]


def openrouter_api_key_is_set() -> bool:
    """Report whether the key variable is non-blank, without exposing it."""
    return bool(os.environ.get(API_KEY_ENV, "").strip())


def _read_api_key() -> str:
    value = os.environ.get(API_KEY_ENV, "").strip()
    if not value:
        raise ProviderAuthenticationError(f"{API_KEY_ENV} is not set")
    if _KEY_CHARACTERS.fullmatch(value) is None:
        raise ProviderAuthenticationError(
            f"{API_KEY_ENV} contains characters that cannot be part of a credential"
        )
    return value


def _count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _json_or_none(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return None


def _status_code(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _translate_error(code: int, error: Mapping[str, Any] | None) -> ProviderError:
    """Map an OpenRouter status or in-body code onto a provider error.

    Only fixed text and two allow-listed fields from the error's metadata are
    used. `error.message` and `metadata.raw` are never copied: either can echo
    request content or an upstream's own text.
    """
    metadata = error.get("metadata") if error is not None else None
    metadata = metadata if isinstance(metadata, Mapping) else {}
    name = metadata.get("provider_name")
    upstream = (
        f" (upstream {name})"
        if isinstance(name, str) and _PLAIN_NAME.fullmatch(name)
        else ""
    )
    if code == 401:
        return ProviderAuthenticationError(
            f"OpenRouter authentication failed; check {API_KEY_ENV}"
        )
    if code == 402:
        source = metadata.get("limit_source")
        source = source if source in _LIMIT_SOURCES else None
        if source == "openrouter_in_flight_budget":
            return ProviderRateLimitError(
                "OpenRouter in-flight spend budget is exhausted; retry shortly"
            )
        if source == "openrouter_key_limit":
            return ProviderRequestError(
                "OpenRouter key credit limit is used up; raise the key's spend limit"
            )
        return ProviderRequestError("OpenRouter account has insufficient credits")
    if code == 408:
        return ProviderTimeoutError(f"OpenRouter request timed out{upstream}")
    if code == 429:
        return ProviderRateLimitError(f"OpenRouter request was rate limited{upstream}")
    if code >= 500:
        return ProviderServerError(
            f"OpenRouter server request failed (status {code}){upstream}"
        )
    if 400 <= code < 500:
        hints = {
            400: ": invalid or missing parameters",
            403: ": insufficient permissions, a guardrail or moderation",
        }
        return ProviderRequestError(
            f"OpenRouter rejected the request (status {code}){hints.get(code, '')}"
            f"{upstream}"
        )
    return ProviderResponseError(f"OpenRouter returned unexpected status {code}")


def _listed_model(
    data: list[object], model: str
) -> Mapping[str, Any] | None:
    """Find a model's listing entry, retrying without a trailing `:variant`.

    Routing shortcuts such as `:nitro` or `:online` are not listed models, but
    they route to a listed one. A listed variant such as `:free` is matched
    exactly first because its window can differ from the base model's.
    """
    candidates = [model]
    if ":" in model:
        candidates.append(model.rsplit(":", 1)[0])
    for candidate in candidates:
        for entry in data:
            if isinstance(entry, Mapping) and candidate in (
                entry.get("id"),
                entry.get("canonical_slug"),
            ):
                return entry
    return None


class OpenRouterProvider:
    """Generate through OpenRouter, optionally pinned to one upstream provider."""

    def __init__(
        self,
        upstream: str = "",
        client_factory: Callable[..., httpx.Client] = _create_client,
    ) -> None:
        self._upstream = normalize_openrouter_upstream(upstream)
        self._client_factory = client_factory
        self._client: httpx.Client | None = None
        self._client_lock = Lock()
        self._request_count = 0
        self._count_lock = Lock()
        self._window_source: str | None = None
        self._window_notice: str | None = None

    @property
    def upstream(self) -> str:
        return self._upstream

    @property
    def window_source(self) -> str | None:
        """Where the last configured window came from, or None before that.

        One of `explicit`, `model_listing` or `assumed`.
        """
        return self._window_source

    @property
    def window_notice(self) -> str | None:
        """A caveat about the last configured window, if it has one."""
        return self._window_notice

    def configure_context_window(
        self,
        model: str,
        requested: int | None,
        *,
        timeout_seconds: float,
    ) -> int | None:
        """Resolve the window from an explicit value, bounded by the listing.

        An explicit value is authoritative, as for every provider, but cannot
        exceed the length OpenRouter lists for the model. Without a listed
        length the explicit value stands, and without either the result is
        None so the caller's assumed window applies. The listing is read with
        the same key and can fail without stopping a run, except for a
        rejected key.
        """
        client = self._get_client()
        listed, problem = self._listed_context_length(client, model, timeout_seconds)
        if listed is None:
            self._window_source = "explicit" if requested is not None else "assumed"
            self._window_notice = None if requested is not None else problem
            return requested
        if requested is not None and requested > listed:
            raise ProviderRequestError(
                f"requested context window {requested} exceeds the model's "
                f"listed maximum {listed}"
            )
        if requested is not None:
            self._window_source, self._window_notice = "explicit", None
            return requested
        self._window_source = "model_listing"
        self._window_notice = (
            f"The context window is the model's listed maximum, not that of the "
            f"pinned upstream {self._upstream}, which may be smaller; pass "
            f"--context-window to set it."
            if self._upstream
            else None
        )
        return listed

    def _listed_context_length(
        self, client: httpx.Client, model: str, timeout_seconds: float
    ) -> tuple[int | None, str]:
        """Return the listed length, or None with a reason that is safe to show."""
        try:
            response = client.get("/models", timeout=timeout_seconds)
        except httpx.TimeoutException:
            return None, "OpenRouter's model listing timed out"
        except httpx.RequestError:
            return None, "OpenRouter's model listing could not be reached"
        if response.status_code == 401:
            raise _translate_error(401, None)
        if response.status_code != 200:
            return None, (
                "OpenRouter's model listing could not be read "
                f"(status {response.status_code})"
            )
        payload = _json_or_none(response)
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, list):
            return None, "OpenRouter's model listing was not in the expected form"
        entry = _listed_model(data, model)
        if entry is None:
            return None, "OpenRouter's model listing does not include this model"
        top = entry.get("top_provider")
        lengths = [
            value
            for value in (
                entry.get("context_length"),
                top.get("context_length") if isinstance(top, Mapping) else None,
            )
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        ]
        if not lengths:
            return None, "OpenRouter's model listing gives no context length for this model"
        # The model-wide and top-provider figures can differ; the smaller one
        # is the safer bound for a request that may land on either.
        return min(lengths), ""

    @property
    def request_count(self) -> int:
        """Chat requests sent so far, counting retries and failures."""
        with self._count_lock:
            return self._request_count

    def generate(self, request: GenerationRequest) -> GenerationResult:
        client = self._get_client()
        with self._count_lock:
            self._request_count += 1
        try:
            response = client.post(
                "/chat/completions",
                json=self._request_body(request),
                timeout=request.timeout_seconds,
            )
        except httpx.TimeoutException as error:
            raise ProviderTimeoutError("OpenRouter request timed out") from error
        except httpx.RequestError as error:
            raise ProviderConnectionError("OpenRouter connection failed") from error
        return self._result(response, request)

    def _request_body(self, request: GenerationRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.instructions},
                {"role": "user", "content": request.input_text},
            ],
            "stream": False,
        }
        if request.max_output_tokens is not None:
            # `max_tokens` rather than `max_completion_tokens`: both are
            # documented, and this is the name models list in their
            # `supported_parameters`, which `require_parameters` matches on.
            body["max_tokens"] = request.max_output_tokens
        routing: dict[str, Any] = {}
        if self._upstream:
            routing["order"] = [self._upstream]
            routing["allow_fallbacks"] = False
        if request.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
            # Without this a request can route to an endpoint that silently
            # ignores the schema, which would defeat the point of sending it.
            routing["require_parameters"] = True
        if routing:
            body["provider"] = routing
        return body

    def _result(
        self, response: httpx.Response, request: GenerationRequest
    ) -> GenerationResult:
        payload = _json_or_none(response)
        error = payload.get("error") if isinstance(payload, Mapping) else None
        error = error if isinstance(error, Mapping) else None
        status = response.status_code
        if status >= 400:
            raise _translate_error(status, error)
        if not 200 <= status < 300:
            raise ProviderResponseError(f"OpenRouter returned unexpected status {status}")
        if not isinstance(payload, Mapping):
            raise ProviderResponseError("OpenRouter returned a malformed response")
        if payload.get("error") is not None and not payload.get("choices"):
            # A failure after the request was accepted arrives as `200` with
            # only an `error` object, so the status alone cannot be trusted.
            code = _status_code(error.get("code")) if error is not None else None
            if code is None:
                raise ProviderResponseError("OpenRouter returned an error response")
            raise _translate_error(code, error)

        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not isinstance(choice, Mapping):
            raise ProviderResponseError("OpenRouter response did not contain a choice")
        self._check_finish(choice.get("finish_reason"), request)
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise ProviderResponseError("OpenRouter response did not contain a message")
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise ProviderResponseError("OpenRouter model refused the request")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError("OpenRouter response did not contain valid text")

        usage = payload.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        model = payload.get("model")
        generation_id = payload.get("id")
        finish_reason = choice.get("finish_reason")
        return GenerationResult(
            text=normalize_output_text(content, request),
            provider="openrouter",
            model=(
                model
                if isinstance(model, str) and _MODEL_NAME.fullmatch(model)
                else request.model
            ),
            input_tokens=_count(usage.get("prompt_tokens")),
            output_tokens=_count(usage.get("completion_tokens")),
            finish_status=finish_reason if isinstance(finish_reason, str) else None,
            request_id=(
                generation_id
                if isinstance(generation_id, str)
                and _GENERATION_ID.fullmatch(generation_id)
                and redact_text(generation_id) == generation_id
                else None
            ),
        )

    @staticmethod
    def _check_finish(finish_reason: object, request: GenerationRequest) -> None:
        if finish_reason in (None, "stop"):
            return
        if finish_reason == "error":
            raise ProviderServerError(
                "OpenRouter upstream provider failed while generating the response"
            )
        if finish_reason == "length":
            # A record cut off part-way is unparseable JSON, and calling it
            # malformed output hides the one fact that explains it.
            limit = (
                f"the configured output token limit of {request.max_output_tokens}"
                if request.max_output_tokens is not None
                else "the model's output limit"
            )
            raise ProviderResponseError(
                f"OpenRouter stopped at {limit}; the response is incomplete"
            )
        reason = (
            finish_reason
            if isinstance(finish_reason, str) and _FINISH_REASON.fullmatch(finish_reason)
            else "an unrecognized reason"
        )
        raise ProviderResponseError(f"OpenRouter response finished with {reason}")

    def _get_client(self) -> httpx.Client:
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                self._client = self._client_factory(
                    base_url=OPENROUTER_BASE_URL,
                    headers={
                        "Authorization": f"Bearer {_read_api_key()}",
                        "Accept": "application/json",
                    },
                    follow_redirects=False,
                )
            return self._client
