"""Provider-independent token accounting for segmentation budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import tiktoken
from tiktoken.core import Encoding


class TokenAccountingError(ValueError):
    """Raised when a token counter cannot provide a valid budget value."""


class TokenCounter(Protocol):
    """Minimal injectable token-counting boundary."""

    @property
    def identity(self) -> str: ...

    @property
    def exact(self) -> bool: ...

    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class TiktokenCounter:
    """Count exactly for a selected tiktoken encoding."""

    encoding: Encoding

    @classmethod
    def for_model(cls, model: str) -> TiktokenCounter:
        try:
            return cls(tiktoken.encoding_for_model(model))
        except KeyError as error:
            raise TokenAccountingError(
                f"no tiktoken encoding is registered for model {model!r}; "
                "provide encoding_name explicitly"
            ) from error

    @classmethod
    def for_encoding(cls, encoding_name: str) -> TiktokenCounter:
        try:
            return cls(tiktoken.get_encoding(encoding_name))
        except ValueError as error:
            raise TokenAccountingError(
                f"unknown tiktoken encoding {encoding_name!r}"
            ) from error

    @property
    def identity(self) -> str:
        return f"tiktoken:{self.encoding.name}"

    @property
    def exact(self) -> bool:
        return True

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))


@dataclass(frozen=True)
class ConservativeUtf8TokenCounter:
    """Use UTF-8 bytes as a conservative offline token estimate."""

    identity: str = "estimate:utf8-bytes"
    exact: bool = False

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))


def resolve_token_counter(
    *,
    provider: str,
    model: str,
    encoding_name: str | None = None,
) -> TokenCounter:
    """Resolve a counter without constructing or calling a model provider.

    Constructing an OpenAI counter may download an uncached tiktoken vocabulary.
    Once constructed, counter calls perform local encoding only.
    """
    if provider.strip().lower() != "openai":
        return ConservativeUtf8TokenCounter()
    if encoding_name is not None:
        return TiktokenCounter.for_encoding(encoding_name)
    return TiktokenCounter.for_model(model)
