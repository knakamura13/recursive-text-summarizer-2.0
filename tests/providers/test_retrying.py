from collections import deque
from dataclasses import dataclass, field

from summarizer.config import RetryPolicy
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderRetriesExhaustedError,
)
from summarizer.providers.retrying import RetryingProvider


REQUEST = GenerationRequest("model", "instructions", "input", 30)
RESULT = GenerationResult("summary", "fake", "model")


@dataclass
class ScriptedProvider:
    outcomes: deque[object]
    calls: list[GenerationRequest] = field(default_factory=list)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, GenerationResult)
        return outcome


def test_transient_failures_retry_with_configured_backoff() -> None:
    provider = ScriptedProvider(
        deque(
            [
                ProviderConnectionError("first"),
                ProviderConnectionError("second"),
                RESULT,
            ]
        )
    )
    delays: list[float] = []
    retrying = RetryingProvider(
        provider,
        RetryPolicy(max_attempts=3),
        sleeper=delays.append,
    )

    result = retrying.generate(REQUEST)

    assert result is RESULT
    assert provider.calls == [REQUEST, REQUEST, REQUEST]
    assert delays == [1, 2]


def test_fatal_failure_is_not_retried() -> None:
    failure = ProviderRequestError("invalid request")
    provider = ScriptedProvider(deque([failure]))
    delays: list[float] = []
    retrying = RetryingProvider(provider, RetryPolicy(), sleeper=delays.append)

    try:
        retrying.generate(REQUEST)
    except ProviderRequestError as caught:
        assert caught is failure
    else:
        raise AssertionError("fatal provider failure was not raised")

    assert provider.calls == [REQUEST]
    assert delays == []


def test_exhaustion_preserves_last_failure_as_cause() -> None:
    first = ProviderConnectionError("first")
    last = ProviderConnectionError("last")
    provider = ScriptedProvider(deque([first, last]))
    delays: list[float] = []
    retrying = RetryingProvider(
        provider,
        RetryPolicy(max_attempts=2),
        sleeper=delays.append,
    )

    try:
        retrying.generate(REQUEST)
    except ProviderRetriesExhaustedError as caught:
        assert caught.attempts == 2
        assert caught.__cause__ is last
    else:
        raise AssertionError("retry exhaustion was not raised")

    assert delays == [1]


def test_single_attempt_never_sleeps() -> None:
    failure = ProviderConnectionError("offline")
    provider = ScriptedProvider(deque([failure]))
    delays: list[float] = []
    retrying = RetryingProvider(
        provider,
        RetryPolicy(max_attempts=1),
        sleeper=delays.append,
    )

    try:
        retrying.generate(REQUEST)
    except ProviderRetriesExhaustedError:
        pass
    else:
        raise AssertionError("retry exhaustion was not raised")

    assert provider.calls == [REQUEST]
    assert delays == []
