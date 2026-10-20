"""Exact counting for Ollama models stored in the tensor layout (#190).

Models pulled in the newer layout have no GGUF layer. Their tokenizer is one of
several JSON layers, so it is found by its content and counted through the
`tokenizers` library. It stays behind a verified set until its counts have been
checked against Ollama's `prompt_eval_count`, so by default it is refused and
the byte counter is used with a notice that says why.

Every tokenizer here is a few tokens built in the test, never a real vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, processors

from summarizer import tokenization
from summarizer.cache import CacheDescriptor
from summarizer.tokenization import (
    ConservativeUtf8TokenCounter,
    GgufTokenCounter,
    TokenAccountingError,
    TokenizerJsonCounter,
    estimate_notice,
    resolve_token_counter,
)
from tests.test_cache import _descriptor
from tests.test_tokenization import _install_ollama_model

TENSOR = "application/vnd.ollama.image.tensor"
JSON_LAYER = "application/vnd.ollama.image.json"
LICENSE = "application/vnd.ollama.image.license"
LOCAL = "http://127.0.0.1:11434"


def _tokenizer_json(
    *,
    extra_piece: str | None = None,
    video_special: bool = False,
    td_added: bool = False,
    lowercase: bool = False,
    digits: bool = False,
    clip: bool = False,
) -> bytes:
    """A tiny SentencePiece-style BPE tokenizer in Hugging Face's JSON shape.

    `extra_piece` changes the vocabulary. `video_special` and `td_added`
    declare a piece that every vocabulary holds as a special or an ordinary
    added token (only one of gemma4:31b's two layers lists `<|video|>`), so
    each changes nothing but the added tokens. `lowercase` and `digits` change
    only the normalizer and the pre-tokenizer. `clip` writes a truncation and a
    padding rule into the file.
    """
    pieces = ["<unk>", "<bos>", "<sp>", "<", ">", "s", "p", "a", "b", "ab", "▁", "▁ab", "<|video|>", "<td>"]
    if extra_piece is not None:
        pieces.append(extra_piece)
    tokenizer = Tokenizer(
        models.BPE(
            vocab={piece: index for index, piece in enumerate(pieces)},
            merges=[("a", "b"), ("▁", "ab")],
            unk_token="<unk>",
        )
    )
    replace = normalizers.Replace(" ", "▁")
    tokenizer.normalizer = (
        normalizers.Sequence([normalizers.Lowercase(), replace]) if lowercase else replace
    )
    if digits:
        tokenizer.pre_tokenizer = pre_tokenizers.Digits(individual_digits=True)
    specials = [AddedToken("<bos>", special=True), AddedToken("<sp>", special=True)]
    if video_special:
        specials.append(AddedToken("<|video|>", special=True))
    tokenizer.add_special_tokens(specials)
    if td_added:
        tokenizer.add_tokens([AddedToken("<td>", special=False)])
    if clip:
        tokenizer.enable_truncation(max_length=3)
        tokenizer.enable_padding(length=50)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    return tokenizer.to_str().encode("utf-8")


def _config_json(pad: int = 0) -> bytes:
    return json.dumps(
        {"model_type": "test", "architectures": ["Test"], "vocab_size": 12, "pad": "x" * pad}
    ).encode("utf-8")


def _install_layers(root: Path, name: str, tag: str, layers: list[tuple[str, bytes]]) -> list[str]:
    """Write a tensor-layout model: each layer's blob is named by its real SHA-256."""
    digests = []
    for media_type, content in layers:
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        blob = root / "blobs" / digest.replace(":", "-")
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        digests.append(digest)
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "layers": [
                    {"mediaType": media_type, "digest": digest, "size": len(content)}
                    for (media_type, content), digest in zip(layers, digests, strict=True)
                ]
            }
        )
    )
    return digests


def _tensor_model(root: Path, *, tokenizer: bytes | None = None) -> str:
    """Install `big:31b` with the tokenizer among its other layers; return the tokenizer's digest.

    The first JSON layer is the largest and the tokenizer is neither first nor
    last, so only its content identifies it.
    """
    tokenizer = tokenizer if tokenizer is not None else _tokenizer_json()
    layers = [
        (TENSOR, b"weights-1"),
        (JSON_LAYER, _config_json(pad=200_000)),
        (TENSOR, b"weights-2"),
        (JSON_LAYER, json.dumps({"temperature": 1.0, "top_k": 64}).encode()),
        (JSON_LAYER, tokenizer),
        (JSON_LAYER, json.dumps({"quantization": "q4"}).encode()),
        (LICENSE, b"license text"),
    ]
    return _install_layers(root, "big", "31b", layers)[4]


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    return tmp_path


def _fingerprint(model: str = "big:31b") -> str:
    return TokenizerJsonCounter.for_ollama_model(model, allow_unverified=True).fingerprint


def _verify(monkeypatch: pytest.MonkeyPatch, fingerprint: str) -> None:
    monkeypatch.setattr(
        tokenization, "_VERIFIED_TOKENIZER_JSON_FINGERPRINTS", frozenset({fingerprint})
    )


# The counting fingerprint of gemma4:31b's tokenizer.json, as the verification
# script printed it on 2026-10-19.
GEMMA4_31B_FINGERPRINT = "63e8558fe07c03e6c38b58646e1db85e6d5f72bcd551f0a1f1f28f7114ab6c2a"


def test_only_the_checked_gemma4_31b_tokenizer_is_verified() -> None:
    # Adding a fingerprint is the act of verification and needs the recorded
    # comparison described in the experiments README, so this pins the whole set.
    assert tokenization._VERIFIED_TOKENIZER_JSON_FINGERPRINTS == frozenset({GEMMA4_31B_FINGERPRINT})


def test_a_layer_with_the_verified_fingerprint_is_accepted_without_asking(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A synthetic layer cannot hash to the real fingerprint, so its fingerprint is
    # forced; the gate itself and the constant are the real ones.
    _tensor_model(models_dir)
    monkeypatch.setattr(tokenization, "_counting_fingerprint", lambda document: GEMMA4_31B_FINGERPRINT)

    counter = TokenizerJsonCounter.for_ollama_model("big:31b")
    resolved = resolve_token_counter(provider="ollama", model="big:31b", ollama_host=LOCAL)

    assert counter.fingerprint == GEMMA4_31B_FINGERPRINT and counter.exact is True
    assert isinstance(resolved, TokenizerJsonCounter) and estimate_notice(resolved) is None


@pytest.mark.parametrize(
    "other",
    [
        GEMMA4_31B_FINGERPRINT[:-1] + "b",
        "0" + GEMMA4_31B_FINGERPRINT[1:],
        GEMMA4_31B_FINGERPRINT[:-1],
        GEMMA4_31B_FINGERPRINT.upper(),
        "",
    ],
)
def test_a_fingerprint_that_differs_in_any_way_is_still_refused(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch, other: str
) -> None:
    _tensor_model(models_dir)
    monkeypatch.setattr(tokenization, "_counting_fingerprint", lambda document: other)

    with pytest.raises(TokenAccountingError, match="has not been checked against Ollama"):
        TokenizerJsonCounter.for_ollama_model("big:31b")
    resolved = resolve_token_counter(provider="ollama", model="big:31b", ollama_host=LOCAL)

    assert isinstance(resolved, ConservativeUtf8TokenCounter) and "has not been checked" in resolved.reason


def test_a_synthetic_tokenizer_is_not_verified_by_the_real_entry(models_dir: Path) -> None:
    _tensor_model(models_dir)

    with pytest.raises(TokenAccountingError, match="has not been checked against Ollama"):
        TokenizerJsonCounter.for_ollama_model("big:31b")


def test_the_tokenizer_layer_is_found_by_its_content_not_its_position_or_size(
    models_dir: Path,
) -> None:
    _tensor_model(models_dir)

    counter = TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)

    assert counter.exact is True
    assert counter.monotonic is False
    assert len(counter.fingerprint) == 64
    assert counter.identity == "tokenizer-json:" + counter.fingerprint[:16]
    # Merges apply: "ab ab" is "ab" + "▁ab".
    assert counter.count("ab ab") == 2
    # Characters outside the vocabulary count as the unknown token, never zero.
    assert counter.count("\U0001f642") == 1


def test_a_counted_text_never_receives_the_template_start_token(models_dir: Path) -> None:
    _tensor_model(models_dir)
    counter = TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)

    # The post-processor would add `<bos>`; the chat template is not counted here.
    assert counter.count("ab") == 1
    assert counter.count("") == 0


def test_a_special_token_marker_inside_text_counts_as_its_pieces(models_dir: Path) -> None:
    _tensor_model(models_dir)
    counter = TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)

    # Matching it as one token would count fewer than Ollama may; pieces count more.
    assert counter.count("<sp>") == 4
    assert counter.count("a <sp> b") > counter.count("a  b")


def test_json_layers_that_only_resemble_a_tokenizer_are_ignored(models_dir: Path) -> None:
    lookalikes = [
        json.dumps({"model": {"type": "BPE"}, "added_tokens": []}).encode(),  # missing sections
        json.dumps({"model": "gemma", "added_tokens": [], "normalizer": None, "pre_tokenizer": None,
                    "post_processor": None, "decoder": None}).encode(),  # model is not an object
        json.dumps([1, 2, 3]).encode(),
        b"\xff\xfe not json at all",
        b"",
    ]
    layers = [(JSON_LAYER, content) for content in lookalikes]
    layers.insert(2, (JSON_LAYER, _tokenizer_json()))
    _install_layers(models_dir, "big", "31b", layers)

    counter = TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)

    assert counter.count("ab ab") == 2


def test_two_tokenizer_layers_that_count_differently_are_refused_as_ambiguous(
    models_dir: Path,
) -> None:
    _install_layers(
        models_dir,
        "big",
        "31b",
        [(JSON_LAYER, _tokenizer_json()), (JSON_LAYER, _tokenizer_json(extra_piece="zz"))],
    )

    with pytest.raises(TokenAccountingError, match="more than one.*count differently"):
        TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)


def test_two_layers_that_differ_only_in_a_special_token_are_one_tokenizer(
    models_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # gemma4:31b lists two tokenizer.json layers that differ by one special
    # token, which the counter never matches.
    _install_layers(
        models_dir,
        "big",
        "31b",
        [
            (JSON_LAYER, _tokenizer_json()),
            (JSON_LAYER, _tokenizer_json(video_special=True)),
        ],
    )
    alone = tmp_path_factory.mktemp("alone")
    _install_layers(alone, "big", "31b", [(JSON_LAYER, _tokenizer_json())])

    counter = TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)
    reference = TokenizerJsonCounter.for_ollama_model(
        "big:31b", models_dir=alone, allow_unverified=True
    )

    assert counter.fingerprint == reference.fingerprint
    assert counter.identity == reference.identity
    for text in ("ab ab", "<|video|>", "a <sp> b", "\U0001f642"):
        assert counter.count(text) == reference.count(text)


@pytest.mark.parametrize(
    "variant", [{"td_added": True}, {"lowercase": True}, {"digits": True}, {"extra_piece": "zz"}]
)
def test_two_layers_that_differ_in_anything_a_count_depends_on_are_ambiguous(
    models_dir: Path, variant: dict[str, object]
) -> None:
    # An ordinary added token is matched in text, and the normalizer and
    # pre-tokenizer shape every piece, so each alone changes what a layer counts.
    _install_layers(
        models_dir,
        "big",
        "31b",
        [(JSON_LAYER, _tokenizer_json()), (JSON_LAYER, _tokenizer_json(**variant))],
    )

    with pytest.raises(TokenAccountingError, match="count differently"):
        TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)


def test_an_ordinary_added_token_counts_as_one_token(models_dir: Path) -> None:
    _install_layers(models_dir, "big", "31b", [(JSON_LAYER, _tokenizer_json(td_added=True))])
    counter = TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)

    assert counter.count("<td>") == 1


def test_a_truncation_or_padding_rule_in_the_file_never_changes_a_count(models_dir: Path) -> None:
    # Truncating would undercount; padding would overcount. Neither belongs in a budget.
    _install_layers(models_dir, "big", "31b", [(JSON_LAYER, _tokenizer_json(clip=True))])
    counter = TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)

    # 20 words and the trailing space: truncation would give 3, padding 50.
    assert counter.count("ab " * 20) == 21
    assert counter.count("ab") == 1


def test_the_same_layer_listed_twice_is_one_layer(models_dir: Path) -> None:
    tokenizer = _tokenizer_json()
    _install_layers(models_dir, "big", "31b", [(JSON_LAYER, tokenizer), (JSON_LAYER, tokenizer)])

    assert TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True).count("ab") == 1


def test_a_model_without_a_tokenizer_layer_is_refused(models_dir: Path) -> None:
    _install_layers(models_dir, "big", "31b", [(TENSOR, b"w"), (JSON_LAYER, _config_json())])

    with pytest.raises(TokenAccountingError, match="no tokenizer"):
        TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)


@pytest.mark.parametrize("model", ["absent:31b", "../escape:1", "a/b/c/d:1"])
def test_a_model_that_is_not_installed_or_not_named_validly_is_refused(
    models_dir: Path, model: str
) -> None:
    with pytest.raises(TokenAccountingError):
        TokenizerJsonCounter.for_ollama_model(model, allow_unverified=True)


UNHASHABLE_DIGESTS = [["sha256:" + "ab" * 32], {"sha256": "ab" * 32}, [], {}]


@pytest.mark.parametrize(
    "digest",
    ["sha256:../../../etc/passwd", "sha256:ab", "md5:" + "ab" * 16, 7, None, True, *UNHASHABLE_DIGESTS],
)
def test_a_malformed_layer_digest_is_refused_before_any_path_is_built(
    models_dir: Path, digest: object
) -> None:
    manifest = models_dir / "manifests" / "registry.ollama.ai" / "library" / "big" / "31b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [{"mediaType": JSON_LAYER, "digest": digest}]}))

    with pytest.raises(TokenAccountingError, match="malformed layer digest"):
        TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)


@pytest.mark.parametrize("digest", UNHASHABLE_DIGESTS)
def test_a_digest_that_cannot_be_hashed_falls_back_to_bytes_with_the_reason_not_a_crash(
    models_dir: Path, digest: object
) -> None:
    # A list or object digest used to reach dict.fromkeys and raise TypeError,
    # which the resolver does not catch, so preflight and the CLI crashed.
    good = hashlib.sha256(b"x").hexdigest()
    manifest = models_dir / "manifests" / "registry.ollama.ai" / "library" / "big" / "31b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "layers": [
                    {"mediaType": JSON_LAYER, "digest": "sha256:" + good},
                    {"mediaType": JSON_LAYER, "digest": digest},
                ]
            }
        )
    )

    counter = resolve_token_counter(provider="ollama", model="big:31b", ollama_host=LOCAL)

    assert isinstance(counter, ConservativeUtf8TokenCounter)
    assert "malformed layer digest" in counter.reason
    notice = estimate_notice(counter)
    assert notice is not None and "malformed layer digest" in notice


def test_a_valid_digest_listed_twice_is_still_one_layer_after_the_digests_are_checked(
    models_dir: Path,
) -> None:
    tokenizer = _tokenizer_json()
    _install_layers(models_dir, "big", "31b", [(JSON_LAYER, tokenizer), (JSON_LAYER, tokenizer)])

    assert TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True).count("ab") == 1


def test_an_unverified_tokenizer_is_refused_by_default(models_dir: Path) -> None:
    _tensor_model(models_dir)
    fingerprint = _fingerprint()

    with pytest.raises(TokenAccountingError, match="has not been checked against Ollama") as raised:
        TokenizerJsonCounter.for_ollama_model("big:31b")

    # The message carries the fingerprint the operator adds once it has been checked.
    assert fingerprint[:12] in str(raised.value)


def test_a_verified_tokenizer_loads_without_asking(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tensor_model(models_dir)
    _verify(monkeypatch, _fingerprint())

    counter = TokenizerJsonCounter.for_ollama_model("big:31b")

    assert counter.count("ab ab") == 2


def test_verification_is_per_tokenizer_so_a_different_one_stays_refused(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tensor_model(models_dir)
    _verify(monkeypatch, hashlib.sha256(b"some other tokenizer").hexdigest())

    with pytest.raises(TokenAccountingError, match="has not been checked"):
        TokenizerJsonCounter.for_ollama_model("big:31b")


def test_a_missing_layer_blob_is_a_counting_error_not_an_os_error(models_dir: Path) -> None:
    digest = _tensor_model(models_dir)
    (models_dir / "blobs" / digest.replace(":", "-")).unlink()

    with pytest.raises(TokenAccountingError, match="cannot read"):
        TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)


def test_a_missing_tokenizers_package_is_a_counting_error_not_a_crash(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tensor_model(models_dir)
    monkeypatch.setitem(sys.modules, "tokenizers", None)

    with pytest.raises(TokenAccountingError, match="tokenizers package"):
        TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)


def test_a_tokenizer_the_library_cannot_load_is_a_counting_error(models_dir: Path) -> None:
    # Right shape, but a model the library rejects.
    broken = json.loads(_tokenizer_json())
    broken["model"] = {"type": "NotAModel", "vocab": {}}
    _tensor_model(models_dir, tokenizer=json.dumps(broken).encode())

    with pytest.raises(TokenAccountingError, match="cannot load"):
        TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)


def test_prefix_and_suffix_search_return_verified_fitting_slices(models_dir: Path) -> None:
    _tensor_model(models_dir)
    counter = TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)
    text = "ab " * 40

    end = counter.fitting_prefix(text, 0, len(text), 10)
    start = counter.fitting_suffix(text, 0, len(text), 10)

    assert 0 < end < len(text) and counter.count(text[:end]) <= 10
    assert 0 < start < len(text) and counter.count(text[start:]) <= 10


def test_its_identity_is_accepted_in_a_cache_descriptor(models_dir: Path) -> None:
    _tensor_model(models_dir)
    counter = TokenizerJsonCounter.for_ollama_model("big:31b", allow_unverified=True)

    descriptor = _descriptor(
        provider="ollama",
        model="big:31b",
        ollama_host="http://localhost:11434",
        counter_identity=counter.identity,
        counter_exact=True,
    )

    assert isinstance(descriptor, CacheDescriptor)
    assert descriptor.counter_identity == counter.identity


# --- resolve_token_counter -------------------------------------------------


def test_resolution_refuses_an_unverified_tokenizer_and_says_why(models_dir: Path) -> None:
    _tensor_model(models_dir)
    fingerprint = _fingerprint()

    counter = resolve_token_counter(provider="ollama", model="big:31b", ollama_host=LOCAL)

    assert isinstance(counter, ConservativeUtf8TokenCounter)
    assert counter.identity == "estimate:utf8-bytes" and counter.exact is False
    assert "has not been checked against Ollama" in counter.reason
    assert fingerprint[:12] in counter.reason
    notice = estimate_notice(counter)
    assert notice is not None and counter.reason in notice


def test_resolution_uses_a_verified_tokenizer_json_counter(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tensor_model(models_dir)
    _verify(monkeypatch, _fingerprint())

    counter = resolve_token_counter(provider="ollama", model="big:31b", ollama_host=LOCAL)

    assert isinstance(counter, TokenizerJsonCounter)
    assert counter.exact is True
    assert estimate_notice(counter) is None


def test_the_gguf_layer_still_wins_when_a_model_has_one(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gguf_digest = _install_ollama_model(models_dir, "both", "latest")
    manifest = models_dir / "manifests" / "registry.ollama.ai" / "library" / "both" / "latest"
    tokenizer = _tokenizer_json()
    tokenizer_digest = "sha256:" + hashlib.sha256(tokenizer).hexdigest()
    (models_dir / "blobs" / tokenizer_digest.replace(":", "-")).write_bytes(tokenizer)
    layers = json.loads(manifest.read_text())["layers"]
    layers.append({"mediaType": JSON_LAYER, "digest": tokenizer_digest})
    manifest.write_text(json.dumps({"layers": layers}))
    _verify(monkeypatch, _fingerprint("both:latest"))

    counter = resolve_token_counter(provider="ollama", model="both", ollama_host=LOCAL)

    assert isinstance(counter, GgufTokenCounter)
    assert counter.identity == "gguf:" + gguf_digest.removeprefix("sha256:")[:16]


def test_an_unverified_gguf_tokenizer_reports_its_own_reason_not_a_missing_layer(
    models_dir: Path,
) -> None:
    _install_ollama_model(models_dir, "tiny", "latest", pre="llama-bpe")

    counter = resolve_token_counter(provider="ollama", model="tiny", ollama_host=LOCAL)

    assert isinstance(counter, ConservativeUtf8TokenCounter)
    assert "GGUF tokenizer" in counter.reason and "has not been checked" in counter.reason
    assert "tokenizer.json" not in counter.reason


def test_a_model_that_is_not_installed_reports_the_missing_model_file(models_dir: Path) -> None:
    counter = resolve_token_counter(provider="ollama", model="absent:1b", ollama_host=LOCAL)

    assert isinstance(counter, ConservativeUtf8TokenCounter)
    assert "no local Ollama model file" in counter.reason


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("http://10.0.0.5:11434", "not on this machine"),
        (None, "no Ollama host"),
        ("", "no Ollama host"),
    ],
)
def test_a_host_whose_files_cannot_be_read_reports_that(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch, host: str | None, expected: str
) -> None:
    _tensor_model(models_dir)
    _verify(monkeypatch, _fingerprint())

    counter = resolve_token_counter(provider="ollama", model="big:31b", ollama_host=host)

    assert isinstance(counter, ConservativeUtf8TokenCounter)
    assert expected in counter.reason


def test_a_missing_tokenizers_package_falls_back_with_that_reason(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tensor_model(models_dir)
    _verify(monkeypatch, _fingerprint())
    # Loading it above cached the tokenizer; start as a process that never had one.
    tokenization._tokenizer_from_json.cache_clear()
    monkeypatch.setitem(sys.modules, "tokenizers", None)

    counter = resolve_token_counter(provider="ollama", model="big:31b", ollama_host=LOCAL)

    assert isinstance(counter, ConservativeUtf8TokenCounter)
    assert "tokenizers package" in counter.reason


def test_the_reason_does_not_change_what_the_byte_counter_counts_or_is_called() -> None:
    plain = ConservativeUtf8TokenCounter()
    explained = ConservativeUtf8TokenCounter(reason="because")

    assert plain == explained
    assert (explained.identity, explained.exact, explained.monotonic) == (
        "estimate:utf8-bytes",
        False,
        True,
    )
    assert explained.count("é") == 2


# --- estimate_notice -------------------------------------------------------


def test_an_exact_counter_has_no_notice() -> None:
    class Exact:
        identity = "test:exact"
        exact = True
        monotonic = True

        def count(self, text: str) -> int:
            return len(text)

    assert estimate_notice(Exact()) is None


def test_the_notice_names_the_counter_and_what_the_estimate_costs() -> None:
    notice = estimate_notice(ConservativeUtf8TokenCounter())

    assert notice is not None
    assert "estimate:utf8-bytes" in notice
    assert "not the model's own tokenizer" in notice
    assert "segments may be smaller than the context window requires" in notice


def test_a_notice_with_a_reason_ends_with_it() -> None:
    notice = estimate_notice(ConservativeUtf8TokenCounter(reason="the host is elsewhere"))

    assert notice is not None and notice.endswith("the host is elsewhere.")
