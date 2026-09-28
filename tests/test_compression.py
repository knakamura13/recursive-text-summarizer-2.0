from summarizer.compression import (
    BAND_TOLERANCE,
    RETENTION_RATIO,
    build_compression_request,
    compress_to_target,
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


def _drop_the_numbered_sentence(*, strict_numbers: bool):
    source = (
        "Alpha reviewed the harbour plan in detail before lunch. "
        "The council approved 42 units in March after a long debate about funding."
    )
    provider = Provider(['{"text":"Alpha reviewed the harbour plan."}'])
    result = compress_to_target(
        source,
        provider,
        source_id="a" * 64,
        model="m",
        timeout_seconds=30,
        target_words=4,
        strict_numbers=strict_numbers,
    )
    return result, provider


def test_strict_numbers_keeps_the_numbered_sentence_when_a_pass_drops_it() -> None:
    result, _provider = _drop_the_numbered_sentence(strict_numbers=True)

    assert "The council approved 42 units in March" in result.text
    assert word_count(result.text) > 4 * (1 + BAND_TOLERANCE)
    assert "Alpha reviewed the harbour plan" in result.text


def test_without_strict_numbers_a_pass_may_drop_a_numbered_sentence() -> None:
    result, _provider = _drop_the_numbered_sentence(strict_numbers=False)

    assert result.text == "Alpha reviewed the harbour plan."
    assert result.passes == 1


def test_an_omitted_number_puts_its_source_sentence_back() -> None:
    source = "The council approved 42 units in March. Beta closed the meeting."
    shortened = "Beta closed the meeting."

    restored = retain_sentences_with_omitted_numbers(source, shortened)

    assert "The council approved 42 units in March." in restored
    assert "Beta closed the meeting." in restored


def test_a_rounded_number_is_not_treated_as_omitted() -> None:
    source = "They built a stunning 963.6 foot skyscraper beside the river."
    shortened = "They built a stunning nearly 1000 foot skyscraper beside the river."

    assert retain_sentences_with_omitted_numbers(source, shortened) == shortened


def test_an_unpunctuated_run_is_sent_in_bounded_chunks_with_their_allowances() -> None:
    """One long "sentence" or unspaced run would otherwise be one request of any size."""
    from summarizer.budget import ContextWindow, RequestLimits
    from summarizer.compression import CHUNK_CHAR_LIMIT, compression_work_ids_for_text
    from summarizer.config import StrategyConfig

    class CharacterCounter:
        identity = "test:characters"
        exact = True
        monotonic = True

        def count(self, text: str) -> int:
            return len(text)

    class Recording:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            chunk = request.input_text.split("\n", 1)[1].rsplit("\n", 1)[0]
            words = chunk.split()
            text = " ".join(words[: max(1, int(len(words) * RETENTION_RATIO))])
            return GenerationResult(
                text=f'{{"text": "{text}"}}', provider="fake", model=request.model
            )

    source = "alpha " * 1_000 + "\u5b57" * 2_500
    provider = Recording()
    result = compress_to_target(
        source,
        provider,
        source_id="a" * 64,
        model="m",
        timeout_seconds=30,
        target_words=600,
        limits=RequestLimits(
            window=ContextWindow(tokens=8_192, assumed=False),
            config=StrategyConfig(),
            counter=CharacterCounter(),
            correction_headroom=0,
        ),
    )

    chunks = [
        request.input_text.split("\n", 1)[1].rsplit("\n", 1)[0]
        for request in provider.requests
    ]
    first_pass = [
        chunk
        for request, chunk in zip(provider.requests, chunks)
        if request.operation_id.startswith("compression:C01")
    ]
    assert len(first_pass) == len(compression_work_ids_for_text(source, max_passes=1))
    assert "".join("".join(first_pass).split()) == "".join(source.split())
    assert all(len(chunk) <= CHUNK_CHAR_LIMIT for chunk in chunks)
    assert [request.max_output_tokens for request in provider.requests] == [
        len(chunk) + 64 for chunk in chunks
    ]
    # Slices of the unspaced run rejoin without a separator, and pieces of the
    # one long sentence rejoin with a space rather than a paragraph break.
    assert "\u5b57" * 2_500 in result.text
    assert "\n\n" not in result.text
