from summarizer.compression import (
    BAND_TOLERANCE,
    RETENTION_RATIO,
    build_compression_request,
    compress_to_target,
    retain_sentences_with_missing_literals,
    word_count,
    _above_ceiling,
    _in_band,
    _under_floor,
)
from summarizer.providers.base import GenerationResult


class Provider:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        text = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        return GenerationResult(text=text, provider="fake", model=request.model)


def test_compression_prompt_avoids_legacy_fragments() -> None:
    request = build_compression_request(
        "Sample chunk.",
        source_id="a" * 64,
        model="m",
        timeout_seconds=30,
        target_word_count=7,
        operation_id="compression:C01K000001",
    )
    assert "Ignore complete sentences" not in request.instructions
    assert "Abbreviate" not in request.instructions
    assert "complete sentences" in request.instructions.lower()


def test_stop_rules_without_model_calls() -> None:
    provider = Provider([])
    target = 100
    low = int(target * (1 - BAND_TOLERANCE))
    in_band = compress_to_target(
        " ".join(["word"] * low),
        provider,
        source_id="a" * 64,
        model="m",
        timeout_seconds=30,
        target_words=target,
    )
    assert in_band.passes == 0
    assert provider.calls == 0
    assert _in_band(word_count(in_band.text), target)

    under = compress_to_target(
        " ".join(["word"] * (low - 5)),
        provider,
        source_id="a" * 64,
        model="m",
        timeout_seconds=30,
        target_words=target,
    )
    assert under.passes == 0
    assert _under_floor(word_count(under.text), target)


def test_compress_pass_invokes_provider() -> None:
    provider = Provider(['{"text":"Shorter summary text."}'])
    result = compress_to_target(
        " ".join(["word"] * 500),
        provider,
        source_id="a" * 64,
        model="m",
        timeout_seconds=30,
        target_words=100,
    )
    assert provider.calls >= 1
    assert result.text


def test_retention_ratio_constant() -> None:
    assert RETENTION_RATIO == 0.70


def test_above_ceiling_helper() -> None:
    assert _above_ceiling(120, 100)
    assert not _above_ceiling(105, 100)


def test_compression_keeps_a_dropped_number_from_any_sentence() -> None:
    source = (
        "Alpha reviewed the harbour plan without figures. "
        "The council approved 42 units in March. "
        "Beta closed the meeting."
    )
    result = compress_to_target(
        source,
        Provider(['{"text":"Alpha reviewed the harbour plan. Beta closed the meeting."}']),
        source_id="a" * 64,
        model="m",
        timeout_seconds=30,
        target_words=4,
    )
    assert "approved 42 units in March" in result.text
    assert result.text.index("Beta closed") < result.text.index("approved 42")


def test_compression_may_drop_sentences_without_names_or_numbers() -> None:
    source = "Alpha reviewed the harbour plan. Beta closed the meeting after a long debate."
    shortened = "Alpha reviewed the harbour plan."
    assert retain_sentences_with_missing_literals(source, shortened) == shortened


def test_compression_does_not_duplicate_a_kept_literal() -> None:
    source = "The council approved 42 units in March. Beta closed the meeting."
    shortened = "The council approved 42 units in March."
    assert retain_sentences_with_missing_literals(source, shortened) == shortened
