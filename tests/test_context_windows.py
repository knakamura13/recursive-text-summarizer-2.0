import pytest

from summarizer.budget import (
    ASSUMED_CONTEXT_WINDOW,
    ContextWindow,
    resolve_context_window,
)


def test_resolves_a_known_model_exactly() -> None:
    window = resolve_context_window(provider="openai", model="gpt-4o-mini")

    assert window.tokens > 0
    assert window.assumed is False


def test_resolves_a_known_model_family_by_prefix() -> None:
    """`gpt-4o-mini` has no exact entry in tiktoken either; prefixes carry it."""
    exact = resolve_context_window(provider="openai", model="gpt-4o")
    variant = resolve_context_window(provider="openai", model="gpt-4o-2024-11-20")

    assert variant.tokens == exact.tokens
    assert variant.assumed is False


def test_prefers_the_longest_matching_prefix() -> None:
    """A more specific family must win over a shorter one that also matches."""
    windows = {
        resolve_context_window(provider="openai", model=name).tokens
        for name in ("gpt-4-turbo", "gpt-4")
    }

    assert len(windows) == 2


def test_an_explicit_window_overrides_the_table() -> None:
    window = resolve_context_window(
        provider="openai", model="gpt-4o-mini", explicit=4_096
    )

    assert window == ContextWindow(tokens=4_096, assumed=False)


def test_an_unknown_openai_model_is_assumed() -> None:
    window = resolve_context_window(provider="openai", model="gpt-nonexistent-9")

    assert window.tokens == ASSUMED_CONTEXT_WINDOW
    assert window.assumed is True


def test_an_arbitrary_ollama_tag_is_assumed() -> None:
    """Neither the installed clients nor tiktoken expose a window offline.

    An arbitrary local tag therefore cannot be known, and the epic requires
    such tags keep working, so it resolves to an assumed window rather than
    failing.
    """
    window = resolve_context_window(provider="ollama", model="qwen3.8")

    assert window.tokens == ASSUMED_CONTEXT_WINDOW
    assert window.assumed is True


def test_an_explicit_window_rescues_an_unknown_model() -> None:
    window = resolve_context_window(
        provider="ollama", model="qwen3.8", explicit=262_144
    )

    assert window == ContextWindow(tokens=262_144, assumed=False)


def test_rejects_a_non_positive_explicit_window() -> None:
    for value in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            resolve_context_window(
                provider="openai", model="gpt-4o-mini", explicit=value
            )


def test_context_window_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        ContextWindow(tokens=0, assumed=True)
