from __future__ import annotations

from dataclasses import dataclass

# Context windows have no offline source of truth, and for OpenAI no online one
# either: the installed openai client exposes only {id, created, object,
# owned_by, shutdown_date} per model, and tiktoken maps names to encodings
# rather than to sizes. Ollama does report an architectural length through
# show(), but the key is architecture-prefixed rather than tag-named (the tag
# `qwen3.8` reports `qwen35.context_length`) and reaching it is a network call
# that is unavailable before a pull. So this table is maintained by hand, and
# anything missing from it is reported as assumed rather than as knowledge.
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
}

# Matched by longest prefix, mirroring the two-tier lookup tiktoken already
# uses and which is what makes a dated or suffixed model name resolve at all.
_MODEL_PREFIX_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-5": 400_000,
    "o1": 200_000,
    "o3": 200_000,
    "o4": 200_000,
}

# Deliberately small: an assumed window should not let an unknown model gamble
# a large request. Selection routes an assumed window to hierarchical rather
# than trusting it.
ASSUMED_CONTEXT_WINDOW = 8_192


@dataclass(frozen=True)
class ContextWindow:
    """A model's total context size, and whether that size is actually known."""

    tokens: int
    assumed: bool

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError("context window tokens must be positive")


def resolve_context_window(
    *,
    provider: str,
    model: str,
    explicit: int | None = None,
) -> ContextWindow:
    """Resolve a model's context window without contacting a provider.

    An explicit value is authoritative. Otherwise the table is consulted by
    exact name and then by longest matching prefix. An unresolved model yields
    an assumed window flagged as such, so a caller can decline to rely on it.
    """
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("context window must be positive")
        return ContextWindow(tokens=explicit, assumed=False)

    if provider.strip().lower() == "openai":
        name = model.strip()
        exact = _MODEL_CONTEXT_WINDOWS.get(name)
        if exact is not None:
            return ContextWindow(tokens=exact, assumed=False)

        matches = [
            prefix
            for prefix in _MODEL_PREFIX_CONTEXT_WINDOWS
            if name.startswith(prefix)
        ]
        if matches:
            longest = max(matches, key=len)
            return ContextWindow(
                tokens=_MODEL_PREFIX_CONTEXT_WINDOWS[longest], assumed=False
            )

    return ContextWindow(tokens=ASSUMED_CONTEXT_WINDOW, assumed=True)
