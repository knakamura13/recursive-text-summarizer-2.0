"""Bounded re-asks for structured model output that fails validation.

A local model sometimes answers a structured request with a record that does
not validate: a missing field, a citation it was not given, a quotation that is
not in the source, or an object cut off at the output limit. Sending the
identical request again tends to reproduce the same mistake, so a re-ask adds
the validator's one-line explanation to the instructions and asks again.

The follow-up is derived from the original request every time, so correction
notes never accumulate, and callers key their caches on the original request:
a result obtained on a re-ask is stored under the item's original descriptor,
exactly as a first-try result would be.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

from pydantic import ValidationError

from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelProvider,
    ProviderResponseError,
)

DEFAULT_REASK_ATTEMPTS = 2

# Every parser in this package raises a `ValueError` subclass for a response it
# rejects (`LeafSummaryError`, `EditorialError`, pydantic's `ValidationError`,
# `json.JSONDecodeError`), and adapters raise `ProviderResponseError` for an
# incomplete or malformed completion. Nothing else is an invalid-output signal.
INVALID_OUTPUT_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    ProviderResponseError,
)

_MAX_REASON_CHARS = 400
_MAX_LOCATION_PART_CHARS = 40

_Parsed = TypeVar("_Parsed")

_STRUCTURED_REASK = (
    "The previous response to this request was rejected: {reason}. Return one "
    "corrected JSON object that conforms to the supplied schema and follows "
    "every rule above, and nothing else."
)
_PROSE_REASK = (
    "The previous response to this request was rejected: {reason}. Return a "
    "corrected response that follows every rule above."
)


def _collapse(value: object, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def rejection_reason(error: BaseException) -> str:
    """Describe a rejected response in one bounded line.

    The text is sent back to the model and shown to the operator, so it never
    carries payload values: a raw pydantic error is reduced to locations and
    error types, and every message is collapsed onto one line and capped.
    """
    if isinstance(error, ValidationError):
        details = "; ".join(
            "{}: {}".format(
                ".".join(
                    _collapse(part, _MAX_LOCATION_PART_CHARS)
                    for part in failure["loc"]
                )
                or "(root)",
                failure["type"],
            )
            for failure in error.errors(include_url=False)
        )
        text = f"response failed validation ({details})"
    else:
        text = str(error)
    return _collapse(text, _MAX_REASON_CHARS) or type(error).__name__


def reask_request(request: GenerationRequest, reason: str) -> GenerationRequest:
    """Return the original request with a note stating why its answer failed."""
    template = (
        _STRUCTURED_REASK if request.response_schema is not None else _PROSE_REASK
    )
    note = template.format(reason=reason.rstrip(". "))
    return replace(request, instructions=f"{request.instructions}\n\n{note}")


def generate_validated(
    provider: ModelProvider,
    request: GenerationRequest,
    parse: Callable[[GenerationResult], _Parsed],
    *,
    on_retry: Callable[[int, str], None] | None = None,
    reask_attempts: int = DEFAULT_REASK_ATTEMPTS,
) -> _Parsed:
    """Generate and parse one structured result, re-asking while it is invalid.

    `parse` receives the whole `GenerationResult`, so a caller that records
    generation metadata can return it alongside the parsed value. A
    `ProviderResponseError` from the provider, or a `ValueError` subclass or
    `ProviderResponseError` from `parse`, counts as invalid output: the request
    is re-asked up to `reask_attempts` times, with `on_retry(attempt, reason)`
    called before each re-ask (`attempt` counts re-asks from 1). The last
    error is re-raised unchanged once the re-asks are spent. Any other
    exception, including one raised by `on_retry`, propagates immediately.
    """
    if (
        isinstance(reask_attempts, bool)
        or not isinstance(reask_attempts, int)
        or reask_attempts < 0
    ):
        raise ValueError("reask_attempts must be a non-negative integer")
    current = request
    reasks = 0
    while True:
        try:
            result = provider.generate(current)
        except ProviderResponseError as error:
            if reasks >= reask_attempts:
                raise
            reason = rejection_reason(error)
        else:
            try:
                return parse(result)
            except INVALID_OUTPUT_ERRORS as error:
                if reasks >= reask_attempts:
                    raise
                reason = rejection_reason(error)
        reasks += 1
        if on_retry is not None:
            on_retry(reasks, reason)
        current = reask_request(request, reason)
