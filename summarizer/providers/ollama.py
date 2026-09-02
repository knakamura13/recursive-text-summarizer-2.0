from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import httpx
import ollama

from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)


def _create_client(**kwargs: object) -> object:
    return ollama.Client(**kwargs)


class OllamaProvider:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        client_factory: Callable[..., object] = _create_client,
    ) -> None:
        self._host = host
        self._client_factory = client_factory
        self._client: Any | None = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            response = self._get_client(request.timeout_seconds).chat(
                model=request.model,
                messages=[
                    {
                        "role": "system",
                        "content": request.instructions,
                    },
                    {"role": "user", "content": request.input_text},
                ],
                stream=False,
                think=False,
            )
        except httpx.TimeoutException as error:
            raise ProviderTimeoutError("Ollama request timed out") from error
        except (ConnectionError, httpx.TransportError) as error:
            raise ProviderConnectionError(
                "Ollama connection failed; confirm the service is running"
            ) from error
        except ollama.ResponseError as error:
            status_code = error.status_code
            if status_code == 404:
                raise ProviderRequestError(
                    "Ollama model was not found; pull it before retrying"
                ) from error
            if status_code == 429:
                raise ProviderRateLimitError(
                    "Ollama request was rate limited"
                ) from error
            if status_code >= 500:
                raise ProviderServerError(
                    "Ollama server request failed"
                ) from error
            raise ProviderRequestError(
                "Ollama rejected the request"
            ) from error
        except ollama.RequestError as error:
            raise ProviderRequestError(
                "Ollama request was invalid"
            ) from error

        if getattr(response, "done", None) is not True:
            raise ProviderResponseError(
                "Ollama response did not complete"
            )
        message = getattr(response, "message", None)
        output_text = getattr(message, "content", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ProviderResponseError(
                "Ollama response did not contain valid text"
            )

        try:
            return GenerationResult(
                text=re.sub(r"\s+", " ", output_text.strip()).strip(),
                provider="ollama",
                model=getattr(response, "model", None) or request.model,
                input_tokens=getattr(response, "prompt_eval_count", None),
                output_tokens=getattr(response, "eval_count", None),
                finish_status=getattr(response, "done_reason", None),
                request_id=None,
            )
        except (TypeError, ValueError) as error:
            raise ProviderResponseError(
                "Ollama response contained invalid metadata"
            ) from error

    def _get_client(self, timeout_seconds: float) -> Any:
        if self._client is None:
            self._client = self._client_factory(
                host=self._host,
                timeout=timeout_seconds,
            )
        return self._client
