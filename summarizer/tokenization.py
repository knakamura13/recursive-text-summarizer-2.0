"""Provider-independent token accounting for segmentation budgets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
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


def estimate_notice(counter: TokenCounter) -> str | None:
    """Return a user-facing notice when a counter's numbers are estimates.

    Budgets, strategy choice and segment sizes all follow these counts, so a
    run that depends on an estimate should say so wherever it reports them.
    """
    if counter.exact:
        return None
    notice = (
        f"Token counts are estimates ({counter.identity}), not the model's own "
        "tokenizer. Budgets are conservative, so segments may be smaller than "
        "the context window requires."
    )
    reason = getattr(counter, "reason", "")
    return f"{notice} No exact counter was used: {reason}." if reason else notice


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

    Ollama wraps the system and user messages in its chat template, which this
    counter does not see. #120 recorded 14 tokens for gemma4 (Ollama 0.34.4,
    llama runner). The MLX runner on Ollama 0.40.6 reported 18 with think false
    and 16 with it unset (#190), so the figure belongs to a runner and a
    version. No budget term charges the template by name. Request measurement
    counts the JSON schema as prompt tokens, which Ollama sends as a grammar
    rather than prompt tokens, so every request is charged its schema term, and
    the safety margin, beyond what the template costs. The smallest schema term,
    33 tokens for the editorial and compression requests, exceeded a 14-token
    template by 19 on #120's 1,742 requests and exceeds an 18-token one by 15,
    before any safety margin.
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


# Ollama's tensor layout has no GGUF layer. The model's Hugging Face
# tokenizer.json is one of several JSON layers beside its configuration, so it
# is told apart by content. These are the sections `Tokenizer.to_str()` always
# writes; the library has the last word when the layer is loaded. A manifest
# can hold more than one tokenizer.json (gemma4:31b lists two, which differ by
# one special token), so layers are compared by what they count, not by file.
_OLLAMA_JSON_LAYER = "application/vnd.ollama.image.json"
_OLLAMA_LAYER_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_TOKENIZER_JSON_SECTIONS = frozenset(
    {"added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder", "model"}
)

# Counting fingerprints of tokenizer.json layers whose counts were checked
# against Ollama's prompt_eval_count (see `_counting_fingerprint`). A tokenizer
# that is not listed is never used silently: the byte counter is used and the
# notice names its fingerprint. Add a fingerprint only with the comparison made
# by experiments/token_counter_verification, since the counter must never count
# fewer tokens than Ollama processes (#120). A tokenizer that counts differently
# has another fingerprint and stays refused.
_VERIFIED_TOKENIZER_JSON_FINGERPRINTS: frozenset[str] = frozenset(
    {
        # gemma4:31b, checked on 2026-10-19 against Ollama 0.40.6 on the MLX
        # runner with think false (#190). Short set: 19 probes, 11 clean ones
        # exact at a template overhead of 18, 7 edge-whitespace probes exactly
        # 1 lower (Ollama trims message edges), markers 7 lower. Long set: 9
        # requests, 5 clean ones from 2,454 to 6,445 tokens exact at 18 (three
        # are the app's leaf requests with the Ollama wire schema), blank-line
        # edges 1 lower, markers 10 lower. No request counted higher by Ollama
        # than counter plus overhead. Not compared: other Ollama versions or
        # runners, requests above about 6,500 tokens, thinking enabled, and
        # gemma4:31b-mlx by name (same fingerprint, not run).
        "63e8558fe07c03e6c38b58646e1db85e6d5f72bcd551f0a1f1f28f7114ab6c2a",
    }
)


class _NoTokenizerLayerError(TokenAccountingError):
    """The model has no tokenizer.json layer, so an earlier refusal is the better reason."""


@dataclass(frozen=True)
class TokenizerJsonCounter:
    """Count exactly with a model's own tokenizer, read from its tokenizer.json layer.

    For Ollama models in the tensor layout, which carry no GGUF file. Like
    `GgufTokenCounter`, it does not recognise special tokens in counted text,
    so a marker such as ``<|turn>`` inside a document counts as its pieces,
    which can only count more than the model does. It never adds the start
    token or the chat template, which Ollama wraps around the messages.

    It is not trusted until its layer is listed in
    `_VERIFIED_TOKENIZER_JSON_FINGERPRINTS`; `allow_unverified` exists only for the
    comparison against Ollama that earns that listing.
    """

    tokenizer: Any
    fingerprint: str

    @classmethod
    def for_ollama_model(
        cls,
        model: str,
        *,
        models_dir: Path | None = None,
        allow_unverified: bool = False,
    ) -> TokenizerJsonCounter:
        """Load the tokenizer of a locally installed Ollama model."""
        root = models_dir or Path(
            os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models"
        )
        manifest = root / "manifests" / _ollama_manifest_path(model)
        try:
            layers = json.loads(manifest.read_text(encoding="utf-8"))["layers"]
            digests = [
                layer["digest"] for layer in layers if layer["mediaType"] == _OLLAMA_JSON_LAYER
            ]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise _NoTokenizerLayerError(f"no local Ollama model file for {model!r}") from error

        found: dict[str, Path] = {}
        for digest in dict.fromkeys(digests):
            if not isinstance(digest, str) or _OLLAMA_LAYER_DIGEST.fullmatch(digest) is None:
                raise TokenAccountingError(f"malformed layer digest in the manifest of {model!r}")
            path = root / "blobs" / digest.replace(":", "-")
            try:
                fingerprint = _tokenizer_json_fingerprint(path)
            except OSError as error:
                raise TokenAccountingError(f"cannot read layer {digest[:19]} of {model!r}") from error
            if fingerprint is not None:
                found.setdefault(fingerprint, path)
        if not found:
            raise _NoTokenizerLayerError(f"no tokenizer.json layer in the model {model!r}")
        if len(found) > 1:
            raise TokenAccountingError(
                f"more than one tokenizer.json layer in the model {model!r}, and they count differently"
            )
        [(fingerprint, path)] = found.items()

        if not allow_unverified and fingerprint not in _VERIFIED_TOKENIZER_JSON_FINGERPRINTS:
            raise TokenAccountingError(
                f"the tokenizer.json of {model!r} (counting fingerprint {fingerprint[:12]}) "
                "has not been checked against Ollama"
            )
        return cls(_tokenizer_from_json(path), fingerprint)

    @property
    def identity(self) -> str:
        return "tokenizer-json:" + self.fingerprint[:16]

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


@lru_cache(maxsize=64)
def _tokenizer_json_fingerprint(path: Path) -> str | None:
    """Return the counting fingerprint of a JSON layer if it has a tokenizer's shape, else None.

    Layers are content-addressed, so what a path holds does not change and the
    answer is kept. A layer that cannot be read raises and is asked again.
    """
    try:
        document = json.loads(path.read_bytes())
    except ValueError:
        return None
    if not _has_tokenizer_shape(document):
        return None
    return _counting_fingerprint(document)


def _counting_fingerprint(document: dict[str, Any]) -> str:
    """SHA-256 of the parts of a tokenizer definition that decide how many tokens a text costs.

    That is the model, the normalizer, the pre-tokenizer and the added tokens
    that are matched in ordinary text. Special tokens are left out because the
    counter splits them into pieces, and so are the post-processor and decoder,
    which never change how many tokens a counted text takes. Truncation and
    padding are switched off when the tokenizer is loaded, so they are left out
    too. Two layers with one fingerprint therefore count every text alike.
    """
    counting = {
        "model": document["model"],
        "normalizer": document["normalizer"],
        "pre_tokenizer": document["pre_tokenizer"],
        "added_tokens": [
            token
            for token in document["added_tokens"]
            if not (isinstance(token, dict) and token.get("special"))
        ],
    }
    canonical = json.dumps(counting, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _has_tokenizer_shape(document: object) -> bool:
    if not isinstance(document, dict) or not _TOKENIZER_JSON_SECTIONS <= document.keys():
        return False
    model = document["model"]
    return (
        isinstance(model, dict)
        and isinstance(model.get("type"), str)
        and isinstance(document["added_tokens"], list)
    )


@lru_cache(maxsize=2)
def _tokenizer_from_json(path: Path) -> Any:
    """Build a tokenizer from a tokenizer.json layer."""
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise TokenAccountingError("the tokenizers package is not installed") from error
    try:
        tokenizer = Tokenizer.from_str(path.read_text(encoding="utf-8"))
        # Match special-token markers in text as their pieces, as the GGUF
        # counter does, so a document cannot count fewer than the model does.
        tokenizer.encode_special_tokens = True
        # A truncation or padding rule in the file would clip or pad the ids.
        tokenizer.no_truncation()
        tokenizer.no_padding()
    except Exception as error:  # the library raises a bare Exception for a bad definition
        raise TokenAccountingError(f"cannot load the tokenizer in {path.name}") from error
    return tokenizer


def _is_loopback(host: str) -> bool:
    hostname = urlsplit(host if "://" in host else f"http://{host}").hostname
    return hostname in _LOOPBACK_HOSTS


@dataclass(frozen=True)
class ConservativeUtf8TokenCounter:
    """Use UTF-8 bytes as a conservative offline token estimate.

    `reason` says why no exact counter was used. It feeds `estimate_notice`
    and is never part of the counter's identity or equality.
    """

    identity: str = "estimate:utf8-bytes"
    exact: bool = False
    monotonic: bool = True
    reason: str = field(default="", compare=False)

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

    A model in the tensor layout has no model file, so its tokenizer.json layer
    is used instead, and only once its tokenizer is in
    `_VERIFIED_TOKENIZER_JSON_FINGERPRINTS`. Every fallback to the byte count says
    why in `reason`, which `estimate_notice` reports.
    """
    if encoding_name is not None:
        return TiktokenCounter.for_encoding(encoding_name)
    if provider.strip().lower() == "openai":
        return TiktokenCounter.for_model(model)
    if provider.strip().lower() == "ollama":
        return _local_ollama_counter(model, ollama_host)
    return ConservativeUtf8TokenCounter()


def _local_ollama_counter(model: str, host: str | None) -> TokenCounter:
    """The model's own tokenizer when it can be read and trusted, else bytes and why not."""
    if not host:
        return ConservativeUtf8TokenCounter(reason="no Ollama host was given")
    if not _is_loopback(host):
        return ConservativeUtf8TokenCounter(
            reason="the Ollama host is not on this machine, so its model files cannot be read"
        )
    try:
        return GgufTokenCounter.for_ollama_model(model)
    except TokenAccountingError as gguf_error:
        reason = str(gguf_error)
    try:
        return TokenizerJsonCounter.for_ollama_model(model)
    except _NoTokenizerLayerError:
        pass  # Not a tensor-layout model, so the GGUF refusal is the reason.
    except TokenAccountingError as error:
        reason = str(error)
    return ConservativeUtf8TokenCounter(reason=reason)
