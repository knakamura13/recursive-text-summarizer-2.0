from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    instructions: str
    input_text: str
    timeout_seconds: float
    operation_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("model", "instructions", "input_text"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_status: str | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("text", "provider", "model"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must not be negative")


@runtime_checkable
class ModelProvider(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class ProviderError(RuntimeError):
    """Base class for provider-independent generation failures."""


class TransientProviderError(ProviderError):
    """A provider failure that may succeed when retried."""


class ProviderTimeoutError(TransientProviderError):
    pass


class ProviderRateLimitError(TransientProviderError):
    pass


class ProviderConnectionError(TransientProviderError):
    pass


class ProviderServerError(TransientProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderRequestError(ProviderError):
    pass


class ProviderResponseError(ProviderError):
    pass


class ProviderRetriesExhaustedError(ProviderError):
    def __init__(self, attempts: int) -> None:
        self.attempts = attempts
        super().__init__(f"Provider request failed after {attempts} attempts")
