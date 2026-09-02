from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import openai

from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)


def _create_client(**kwargs: object) -> object:
    return openai.OpenAI(**kwargs)


class OpenAIProvider:
    def __init__(
        self,
        client_factory: Callable[..., object] = _create_client,
    ) -> None:
        self._client_factory = client_factory
        self._client: Any | None = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            response = self._get_client().responses.create(
                model=request.model,
                instructions=request.instructions,
                input=request.input_text,
                timeout=request.timeout_seconds,
            )
        except openai.APITimeoutError as error:
            raise ProviderTimeoutError("OpenAI request timed out") from error
        except openai.RateLimitError as error:
            raise ProviderRateLimitError("OpenAI request was rate limited") from error
        except openai.APIConnectionError as error:
            raise ProviderConnectionError("OpenAI connection failed") from error
        except openai.AuthenticationError as error:
            raise ProviderAuthenticationError(
                "OpenAI authentication failed"
            ) from error
        except openai.APIStatusError as error:
            request_id = getattr(error, "request_id", None)
            detail = f" (request {request_id})" if request_id else ""
            if error.status_code >= 500:
                raise ProviderServerError(
                    f"OpenAI server request failed{detail}"
                ) from error
            raise ProviderRequestError(
                f"OpenAI rejected the request{detail}"
            ) from error

        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ProviderResponseError(
                "OpenAI response did not contain valid text"
            )

        usage = getattr(response, "usage", None)
        return GenerationResult(
            text=re.sub(r"\s+", " ", output_text.strip()).strip(),
            provider="openai",
            model=getattr(response, "model", None) or request.model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            finish_status=getattr(response, "status", None),
            request_id=getattr(response, "_request_id", None),
        )

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(max_retries=0)
        return self._client
