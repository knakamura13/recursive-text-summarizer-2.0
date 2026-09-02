from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from summarizer.tokenization import (
    ConservativeUtf8TokenCounter,
    TiktokenCounter,
    TokenAccountingError,
    TokenCounter,
    resolve_token_counter,
)


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True

    def count(self, text: str) -> int:
        return len(text)


def test_protocol_accepts_an_injected_deterministic_counter() -> None:
    counter: TokenCounter = CharacterCounter()

    assert counter.count("café") == 4
    assert counter.identity == "test:characters"
    assert counter.exact is True


def test_tiktoken_counter_uses_known_model_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiktoken

    encoding = SimpleNamespace(name="o200k_base", encode=lambda text: list(text))

    def encoding_for_model(model: str) -> SimpleNamespace:
        assert model == "gpt-4o-mini"
        return encoding

    monkeypatch.setattr(tiktoken, "encoding_for_model", encoding_for_model)

    counter = TiktokenCounter.for_model("gpt-4o-mini")

    assert counter.count("summary") == 7
    assert counter.identity == "tiktoken:o200k_base"
    assert counter.exact is True


def test_openai_unknown_model_requires_explicit_encoding_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiktoken

    def reject_unknown_model(model: str) -> None:
        assert model == "future-model"
        raise KeyError(model)

    monkeypatch.setattr(tiktoken, "encoding_for_model", reject_unknown_model)

    with pytest.raises(TokenAccountingError, match="encoding"):
        resolve_token_counter(provider="openai", model="future-model")


def test_openai_explicit_encoding_fallback_is_exact_for_that_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiktoken

    encoding = SimpleNamespace(name="cl100k_base", encode=lambda text: [text])

    def get_encoding(name: str) -> SimpleNamespace:
        assert name == "cl100k_base"
        return encoding

    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)

    counter = resolve_token_counter(
        provider="openai",
        model="future-model",
        encoding_name="cl100k_base",
    )

    assert isinstance(counter, TiktokenCounter)
    assert counter.identity == "tiktoken:cl100k_base"
    assert counter.exact is True


def test_conservative_counter_uses_utf8_byte_length() -> None:
    counter = ConservativeUtf8TokenCounter()

    assert counter.count("abc") == 3
    assert counter.count("é") == 2
    assert counter.count("🙂") == 4
    assert counter.identity == "estimate:utf8-bytes"
    assert counter.exact is False


@pytest.mark.parametrize("provider", ["ollama", "custom"])
def test_non_openai_provider_uses_offline_conservative_counter(
    provider: str,
) -> None:
    counter = resolve_token_counter(provider=provider, model="org/model:tag")

    assert isinstance(counter, ConservativeUtf8TokenCounter)
    assert counter.identity == "estimate:utf8-bytes"
    assert counter.count("é") == 2
