"""Provider-independent token accounting for segmentation budgets."""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

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

    @property
    def monotonic(self) -> bool: ...

    def count(self, text: str) -> int: ...


@runtime_checkable
class PrefixTokenCounter(Protocol):
    def fitting_prefix(
        self,
        text: str,
        start: int,
        end: int,
        max_tokens: int,
    ) -> int: ...


@runtime_checkable
class SuffixTokenCounter(Protocol):
    def fitting_suffix(
        self,
        text: str,
        floor: int,
        end: int,
        max_tokens: int,
    ) -> int: ...


# Width of the window searched exactly at the end of a boundary search. BPE
# token counts are not monotonic in text length, and the instability is *not*
# bounded by any single token's byte length: with cl100k_base, "a" * 3000 at a
# 12-token budget fits through 96 characters even though 93 characters already
# exceeds it. No window makes a coarse search exact, so this is a packing
# density knob only. Every returned boundary is verified by an exact recount,
# and callers must not assume a returned boundary is the largest one possible.
_UNSTABLE_TOKEN_WINDOW_CHARS = 16


@dataclass(frozen=True)
class TiktokenCounter:
    """Count exactly for a selected tiktoken encoding."""

    encoding: Encoding

    @classmethod
    def for_model(cls, model: str) -> TiktokenCounter:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError as error:
            raise TokenAccountingError(
                f"no tiktoken encoding is registered for model {model!r}; "
                "provide encoding_name explicitly"
            ) from error
        return cls(encoding)

    @classmethod
    def for_encoding(cls, encoding_name: str) -> TiktokenCounter:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except ValueError as error:
            raise TokenAccountingError(
                f"unknown tiktoken encoding {encoding_name!r}"
            ) from error
        return cls(encoding)

    @property
    def identity(self) -> str:
        return f"tiktoken:{self.encoding.name}"

    @property
    def exact(self) -> bool:
        return True

    @property
    def monotonic(self) -> bool:
        return False

    def count(self, text: str) -> int:
        return len(self.encoding.encode_ordinary(text))

    def fitting_prefix(
        self,
        text: str,
        start: int,
        end: int,
        max_tokens: int,
    ) -> int:
        """Return an end offset whose prefix is verified to fit `max_tokens`.

        The result is exact for its own slice but is not guaranteed to be the
        largest fitting offset: a coarse search cannot be exact against a
        non-monotonic encoding. Under-packing is safe; callers must never treat
        the result as maximal, nor assume a smaller offset also fits.
        """
        return _search_prefix(self.count, text, start, end, max_tokens)

    def fitting_suffix(
        self,
        text: str,
        floor: int,
        end: int,
        max_tokens: int,
    ) -> int:
        """Return a start offset whose suffix is verified to fit `max_tokens`.

        The backward mirror of `fitting_prefix`, carrying the same contract: the
        returned slice is verified, but is not guaranteed to be the longest
        fitting suffix.
        """
        return _search_suffix(self.count, text, floor, end, max_tokens)


def _search_prefix(
    count: Callable[[str], int], text: str, start: int, end: int, max_tokens: int
) -> int:
    """Binary-search a verified fitting prefix end for a non-monotonic counter."""
    if count(text[start:end]) <= max_tokens:
        return end

    low, high = start, end
    while high - low > _UNSTABLE_TOKEN_WINDOW_CHARS:
        mid = (low + high) // 2
        if count(text[start:mid]) <= max_tokens:
            low = mid
        else:
            high = mid

    best = low
    for candidate in range(low + 1, high + 1):
        if count(text[start:candidate]) <= max_tokens:
            best = candidate
    return best


def _search_suffix(
    count: Callable[[str], int], text: str, floor: int, end: int, max_tokens: int
) -> int:
    """The backward mirror of `_search_prefix`."""
    if count(text[floor:end]) <= max_tokens:
        return floor

    low, high = floor, end
    while high - low > _UNSTABLE_TOKEN_WINDOW_CHARS:
        mid = (low + high) // 2
        if count(text[mid:end]) <= max_tokens:
            high = mid
        else:
            low = mid

    best = high
    for candidate in range(high - 1, low - 1, -1):
        if count(text[candidate:end]) <= max_tokens:
            best = candidate
    return best


# GGUF tokenizers whose counts were checked against Ollama's prompt_eval_count:
# (tokenizer.ggml.model, tokenizer.ggml.pre). gemma4 matched every one of 1,742
# recorded requests at system + user tokens + a fixed chat template.
_VERIFIED_GGUF_TOKENIZERS = frozenset({("llama", "gemma4")})
_GGUF_MAGIC = b"GGUF"
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class GgufTokenCounter:
    """Count exactly with a model's own tokenizer, read from its local GGUF file.

    Special tokens are not recognised in counted text, so a marker such as
    ``<|turn>`` inside a document counts as its pieces. That can only count
    more than the model does, never fewer.

    Ollama wraps the system and user messages in its chat template, 14 tokens
    for gemma4, which this counter does not see. Request measurement also
    counts the JSON schema, which Ollama sends as a grammar rather than
    prompt tokens. On all 1,742 recorded requests the schema count exceeded
    the template by at least 19 tokens, before any safety margin.
    """

    tokenizer: Any
    model_digest: str

    @classmethod
    def for_ollama_model(cls, model: str, *, models_dir: Path | None = None) -> GgufTokenCounter:
        """Load the tokenizer of a locally installed Ollama model."""
        root = models_dir or Path(
            os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models"
        )
        manifest = root / "manifests" / _ollama_manifest_path(model)
        try:
            layers = json.loads(manifest.read_text(encoding="utf-8"))["layers"]
            digest = next(
                layer["digest"]
                for layer in layers
                if layer["mediaType"] == "application/vnd.ollama.image.model"
            )
        except (OSError, ValueError, KeyError, StopIteration) as error:
            raise TokenAccountingError(f"no local Ollama model file for {model!r}") from error
        return cls(_gguf_tokenizer(root / "blobs" / digest.replace(":", "-")), digest)

    @property
    def identity(self) -> str:
        return "gguf:" + self.model_digest.removeprefix("sha256:")[:16]

    @property
    def exact(self) -> bool:
        return True

    @property
    def monotonic(self) -> bool:
        return False

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def fitting_prefix(self, text: str, start: int, end: int, max_tokens: int) -> int:
        """Same contract as `TiktokenCounter.fitting_prefix`."""
        return _search_prefix(self.count, text, start, end, max_tokens)

    def fitting_suffix(self, text: str, floor: int, end: int, max_tokens: int) -> int:
        """Same contract as `TiktokenCounter.fitting_suffix`."""
        return _search_suffix(self.count, text, floor, end, max_tokens)


def _ollama_manifest_path(model: str) -> Path:
    name, _, tag = model.strip().partition(":")
    parts = name.split("/")
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library", *parts]
    elif len(parts) == 2:
        parts = ["registry.ollama.ai", *parts]
    if len(parts) != 3 or not all(parts) or any(part in (".", "..") for part in [*parts, tag]):
        raise TokenAccountingError(f"unrecognised Ollama model name {model!r}")
    return Path(*parts, tag or "latest")


@lru_cache(maxsize=4)
def _gguf_tokenizer(path: Path) -> Any:
    """Build a tokenizer from a GGUF file's metadata; tensors are never read."""
    metadata = _gguf_tokenizer_metadata(path)
    kind = (metadata.get("tokenizer.ggml.model"), metadata.get("tokenizer.ggml.pre"))
    if kind not in _VERIFIED_GGUF_TOKENIZERS:
        raise TokenAccountingError(f"GGUF tokenizer {kind} has not been checked against Ollama")
    try:
        from tokenizers import Tokenizer, models, normalizers
    except ImportError as error:
        raise TokenAccountingError("the tokenizers package is not installed") from error

    tokens = metadata["tokenizer.ggml.tokens"]
    merges = [tuple(merge.split(" ", 1)) for merge in metadata["tokenizer.ggml.merges"]]
    tokenizer = Tokenizer(
        models.BPE(
            vocab={token: index for index, token in enumerate(tokens)},
            merges=merges,
            byte_fallback=True,
            unk_token="<unk>",
            fuse_unk=True,
        )
    )
    tokenizer.normalizer = normalizers.Replace(" ", "\u2581")
    return tokenizer


def _gguf_tokenizer_metadata(path: Path) -> dict[str, Any]:
    """Read the ``tokenizer.*`` metadata of a GGUF v2/v3 file."""
    try:
        with path.open("rb") as file:
            if file.read(4) != _GGUF_MAGIC:
                raise TokenAccountingError(f"{path.name} is not a GGUF file")
            version = struct.unpack("<I", file.read(4))[0]
            if version not in (2, 3):
                raise TokenAccountingError(f"unsupported GGUF version {version}")
            _, entries = struct.unpack("<QQ", file.read(16))

            def string() -> str:
                return file.read(struct.unpack("<Q", file.read(8))[0]).decode("utf-8")

            def value(kind: int) -> Any:
                if kind == _GGUF_STRING:
                    return string()
                if kind == _GGUF_ARRAY:
                    item_kind, length = struct.unpack("<IQ", file.read(12))
                    if item_kind == _GGUF_STRING:
                        return [string() for _ in range(length)]
                    code = _GGUF_SCALARS[item_kind]
                    return struct.unpack(
                        f"<{length}{code}", file.read(struct.calcsize(code) * length)
                    )
                code = _GGUF_SCALARS[kind]
                return struct.unpack(f"<{code}", file.read(struct.calcsize(code)))[0]

            metadata: dict[str, Any] = {}
            for _ in range(entries):
                key = string()
                item = value(struct.unpack("<I", file.read(4))[0])
                if key.startswith("tokenizer."):
                    metadata[key] = item
            return metadata
    except (OSError, KeyError, struct.error, UnicodeDecodeError) as error:
        raise TokenAccountingError(f"cannot read GGUF metadata from {path.name}") from error


def _is_loopback(host: str) -> bool:
    hostname = urlsplit(host if "://" in host else f"http://{host}").hostname
    return hostname in _LOOPBACK_HOSTS


@dataclass(frozen=True)
class ConservativeUtf8TokenCounter:
    """Use UTF-8 bytes as a conservative offline token estimate."""

    identity: str = "estimate:utf8-bytes"
    exact: bool = False
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def fitting_prefix(
        self,
        text: str,
        start: int,
        end: int,
        max_tokens: int,
    ) -> int:
        byte_count = 0
        for index in range(start, end):
            byte_count += len(text[index].encode("utf-8"))
            if byte_count > max_tokens:
                return index
        return end

    def fitting_suffix(
        self,
        text: str,
        floor: int,
        end: int,
        max_tokens: int,
    ) -> int:
        byte_count = 0
        for index in range(end - 1, floor - 1, -1):
            byte_count += len(text[index].encode("utf-8"))
            if byte_count > max_tokens:
                return index + 1
        return floor


def resolve_token_counter(
    *,
    provider: str,
    model: str,
    encoding_name: str | None = None,
    ollama_host: str | None = None,
) -> TokenCounter:
    """Resolve a counter without constructing or calling a model provider.

    Constructing a tiktoken counter may download an uncached vocabulary.
    For an Ollama host on this machine, the model's own tokenizer is read from
    its local model file when that tokenizer has been checked against Ollama.
    Any other Ollama model or host falls back to the conservative byte count.
    Once constructed, counter calls perform local encoding only.
    """
    if encoding_name is not None:
        return TiktokenCounter.for_encoding(encoding_name)
    if provider.strip().lower() == "openai":
        return TiktokenCounter.for_model(model)
    if provider.strip().lower() == "ollama" and ollama_host and _is_loopback(ollama_host):
        try:
            return GgufTokenCounter.for_ollama_model(model)
        except TokenAccountingError:
            pass
    return ConservativeUtf8TokenCounter()
