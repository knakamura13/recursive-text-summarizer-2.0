from summarizer.compression import (
    BAND_TOLERANCE,
    RETENTION_RATIO,
    build_compression_request,
    compress_to_target,
    restore_paraphrased_number_sentences,
    retain_sentences_with_missing_literals,
    retain_sentences_with_omitted_numbers,
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
        strict_numbers=True,
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


def test_literal_retention_is_off_unless_a_switch_is_on() -> None:
    source = "They built a stunning 963.6 foot skyscraper beside the river."
    shortened = "They built a nearly 1000 foot skyscraper beside the river."
    assert retain_sentences_with_missing_literals(source, shortened) == shortened

    named = "Professional skier Elyse Saugstad wore a backpack."
    shortened_name = "Professional skier Saugstad wore a backpack."
    assert retain_sentences_with_missing_literals(named, shortened_name) == shortened_name


def test_strict_numbers_restores_a_rounded_sentence() -> None:
    source = "They built a stunning 963.6 foot skyscraper beside the river."
    shortened = "They built a nearly 1000 foot skyscraper beside the river."
    restored = retain_sentences_with_missing_literals(
        source, shortened, strict_numbers=True
    )
    assert "963.6" in restored


def test_strict_names_restores_a_shortened_name() -> None:
    source = "Professional skier Elyse Saugstad wore a backpack."
    shortened = "Professional skier Saugstad wore a backpack."
    restored = retain_sentences_with_missing_literals(
        source, shortened, strict_names=True
    )
    assert "Elyse Saugstad" in restored


def test_a_paraphrase_that_keeps_numbers_is_replaced_by_the_source_sentences() -> None:
    source = (
        "The crew counted 16 divers on the reef. "
        "A slab nearly 200 feet across and 3 feet deep broke loose."
    )
    rewritten = (
        "The event impacted sixteen divers on the reef, originating from a slab "
        "that cracked a meadow, measuring nearly 200 feet across and 3 feet deep."
    )

    restored = restore_paraphrased_number_sentences(source, rewritten)

    assert "The crew counted 16 divers on the reef." in restored
    assert "A slab nearly 200 feet across and 3 feet deep broke loose." in restored
    assert "The event impacted" not in restored


def test_a_rounded_sentence_with_the_same_words_stays() -> None:
    source = "They built a stunning 963.6 foot skyscraper beside the river."
    rewritten = "They built a stunning nearly 1000 foot skyscraper beside the river."

    assert restore_paraphrased_number_sentences(source, rewritten) == rewritten


def test_omitting_a_numbered_sentence_does_not_restore_it() -> None:
    source = "They built a stunning 963.6 foot skyscraper beside the river. The river flooded."
    rewritten = "The river flooded."

    assert restore_paraphrased_number_sentences(source, rewritten) == rewritten


def test_a_different_year_is_not_treated_as_a_rounded_source_sentence() -> None:
    source = "The lake froze in 1910."
    rewritten = "The lake froze in 1911."

    restored = restore_paraphrased_number_sentences(source, rewritten)

    assert restored == rewritten


def test_a_shortening_that_keeps_the_words_stays() -> None:
    source = "The council approved 42 units in March after a long debate."
    rewritten = "The council approved 42 units in March."

    assert restore_paraphrased_number_sentences(source, rewritten) == rewritten


def test_compression_stops_when_shortening_would_drop_a_number() -> None:
    source = (
        "Alpha reviewed the harbour plan in detail before lunch. "
        "The council approved 42 units in March after a long debate about funding."
    )
    provider = Provider(['{"text":"Alpha reviewed the harbour plan."}'])
    target = 4

    result = compress_to_target(
        source,
        provider,
        source_id="a" * 64,
        model="m",
        timeout_seconds=30,
        target_words=target,
    )

    assert result.text == source
    assert word_count(result.text) > target * (1 + BAND_TOLERANCE)
    assert provider.calls == 1


def test_an_omitted_number_is_restored_without_the_strict_switch() -> None:
    source = "The council approved 42 units in March. Beta closed the meeting."
    shortened = "Beta closed the meeting."

    restored = retain_sentences_with_omitted_numbers(source, shortened)

    assert "The council approved 42 units in March." in restored
    assert "Beta closed the meeting." in restored


def test_a_rounded_number_is_not_treated_as_omitted() -> None:
    source = "They built a stunning 963.6 foot skyscraper beside the river."
    shortened = "They built a stunning nearly 1000 foot skyscraper beside the river."

    assert retain_sentences_with_omitted_numbers(source, shortened) == shortened


def test_a_changed_fact_around_a_rounded_number_restores_the_source() -> None:
    source = "963 people were rescued."
    rewritten = "Nearly 1000 people died."

    restored = restore_paraphrased_number_sentences(source, rewritten)

    assert "963 people were rescued." in restored
    assert "died" not in restored
