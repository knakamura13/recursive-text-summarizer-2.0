from __future__ import annotations

from collections.abc import Callable
from time import sleep

from summarizer.config import RetryPolicy
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelProvider,
    ProviderRetriesExhaustedError,
    TransientProviderError,
)


class RetryingProvider:
    def __init__(
        self,
        provider: ModelProvider,
        policy: RetryPolicy,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._sleeper = sleeper

    def generate(self, request: GenerationRequest) -> GenerationResult:
        delay = self._policy.initial_delay_seconds
        for attempt in range(1, self._policy.max_attempts + 1):
            try:
                return self._provider.generate(request)
            except TransientProviderError as error:
                if attempt == self._policy.max_attempts:
                    raise ProviderRetriesExhaustedError(
                        attempt,
                        str(error),
                    ) from error
                self._sleeper(delay)
                delay *= self._policy.backoff_multiplier
        raise AssertionError("retry loop completed without a result")
