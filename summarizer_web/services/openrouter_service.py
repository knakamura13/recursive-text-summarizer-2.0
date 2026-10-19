"""OpenRouter for the web app: whether the server has a key, and a provider factory.

The key is read from the server's environment and nowhere else. It is not a
setting, not part of a Run configuration and never in a response; only whether
it is set is reported. `create_provider` is the one place the web app builds an
OpenRouter provider for preflight, so tests replace it rather than the network.
"""

from __future__ import annotations

from summarizer.providers.openrouter import (
    API_KEY_ENV,
    OpenRouterProvider,
    openrouter_api_key_is_set,
    openrouter_key_state,
)
from summarizer_web.models.api import OpenRouterStatusResponse

KEY_MISSING_MESSAGE = (
    f"OpenRouter needs an API key. Set {API_KEY_ENV} in the environment the server "
    "runs in, then restart the server."
)


def key_is_set() -> bool:
    return openrouter_api_key_is_set()


KEY_INVALID_MESSAGE = (
    f"{API_KEY_ENV} is set but cannot be used: it contains whitespace or characters that "
    "cannot be part of a key. Fix it, then restart the server."
)


def key_problem_message() -> str:
    """What to tell the user when the key cannot be used: unset, or set but malformed."""
    return KEY_INVALID_MESSAGE if openrouter_key_state() == "invalid" else KEY_MISSING_MESSAGE


def key_status() -> OpenRouterStatusResponse:
    state = openrouter_key_state()
    if state == "ok":
        return OpenRouterStatusResponse(key_set=True, message="An OpenRouter API key is set.")
    message = KEY_INVALID_MESSAGE if state == "invalid" else KEY_MISSING_MESSAGE
    return OpenRouterStatusResponse(key_set=False, message=message)


def create_provider(upstream: str) -> OpenRouterProvider:
    return OpenRouterProvider(upstream=upstream)
