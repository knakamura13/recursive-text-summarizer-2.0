import json

import pytest

from summarizer.cache import CacheStore
from summarizer.editorial import EditorialError, build_editorial_request, write_editorial
from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer.runtime.observers import ItemFailedError, RuntimeObserver, StageName
from summarizer.segmentation import CacheCoordinator
from summarizer.summaries import SummaryNode


SOURCE_ID = "a" * 64


def root() -> SummaryNode:
    return SummaryNode.model_validate(
        {
            "summary": "The source describes a qualified change.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": ["S000001"],
            "level": 1,
        }
    )


class Provider:
    def __init__(self, response: str = '{"text":"A coherent final summary."}') -> None:
        self.requests: list[GenerationRequest] = []
        self.response = response

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(text=self.response, provider="fake", model=request.model)


def test_request_is_a_dedicated_genre_neutral_fenced_final_call() -> None:
    request = build_editorial_request(
        root(), source_id=SOURCE_ID, model="m", timeout_seconds=30, target_words=120
    )

    assert request.operation_id == "editorial-final"
    assert request.response_schema is not None
    assert "120 words" in request.instructions
    assert "Do not drop a number" in request.instructions
    assert "unsupported" in request.instructions
    assert "qualifications" in request.instructions
    assert "article" not in request.instructions.lower()
    assert "report" not in request.instructions.lower()
    assert root().summary not in request.instructions
    assert root().summary in request.input_text
    assert "GROUNDED-ROOT" in request.input_text


def test_final_writer_returns_redacted_plain_text_and_is_deterministic() -> None:
    first = Provider('{"text":"Keep sk-12345678901234567890 out."}')
    second = Provider('{"text":"Keep sk-12345678901234567890 out."}')

    one = write_editorial(
        root(), first, source_id=SOURCE_ID, model="m", timeout_seconds=30, target_words=50
    )
    two = write_editorial(
        root(), second, source_id=SOURCE_ID, model="m", timeout_seconds=30, target_words=50
    )

    assert one.text == "Keep [REDACTED] out."
    assert one == two
    assert first.requests == second.requests


@pytest.mark.parametrize(
    "secret",
    (
        "ghp_123456789012345678901234567890123456",
        "xoxb-1234567890-abcdefghij",
        "Authorization: Basic dXNlcjpwYXNzd29yZA==",
        "Authorization: Basic YTpi",
    ),
)
def test_final_writer_redacts_common_provider_credentials(secret: str) -> None:
    result = write_editorial(
        root(),
        Provider(json.dumps({"text": f"Do not disclose {secret}."})),
        source_id=SOURCE_ID,
        model="m",
        timeout_seconds=30,
        target_words=50,
    )

    assert secret not in result.text
    assert "[REDACTED]" in result.text


class ScriptedProvider:
    def __init__(self, *responses: str) -> None:
        self.requests: list[GenerationRequest] = []
        self.responses = list(responses)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(
            text=self.responses.pop(0), provider="fake", model=request.model
        )


def _recording_observer() -> tuple[RuntimeObserver, list[tuple[str, int | None, str | None]]]:
    events: list[tuple[str, int | None, str | None]] = []
    observer = RuntimeObserver(
        on_item=lambda event: events.append((event.state, event.attempt, event.message))
        if event.kind == "editorial" and event.work_id == "editorial-final"
        else None
    )
    return observer, events


def test_invalid_draft_is_reasked_with_the_validation_error_then_accepted() -> None:
    provider = ScriptedProvider("not json", '{"text":"A coherent final summary."}')
    observer, events = _recording_observer()

    result = write_editorial(
        root(),
        provider,
        source_id=SOURCE_ID,
        model="m",
        timeout_seconds=30,
        target_words=50,
        observer=observer,
    )

    assert result.text == "A coherent final summary."
    first, second = provider.requests
    assert second.operation_id == first.operation_id == "editorial-final"
    assert second.input_text == first.input_text
    assert "rejected" in second.instructions
    assert "not json" not in second.instructions
    assert [state for state, _, _ in events] == ["active", "retrying", "completed"]
    assert events[1][1] == 1
    assert events[1][2]


@pytest.mark.parametrize("response", ["not json", json.dumps({"text": " "}), "{}"])
def test_draft_still_invalid_after_reasks_fails_the_editorial_item(response: str) -> None:
    provider = ScriptedProvider(response, response, response)
    observer, events = _recording_observer()

    with pytest.raises(ItemFailedError) as error:
        write_editorial(
            root(),
            provider,
            source_id=SOURCE_ID,
            model="m",
            timeout_seconds=30,
            target_words=50,
            observer=observer,
        )

    assert len(provider.requests) == 3
    assert error.value.stage is StageName.WRITING
    assert (error.value.kind, error.value.work_id) == ("editorial", "editorial-final")
    assert isinstance(error.value.__cause__, EditorialError)
    assert response not in str(error.value)
    assert [state for state, _, _ in events] == ["active", "retrying", "retrying", "failed"]
    assert events[-1][2] == str(error.value)


class _Characters:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


def _limits():
    from summarizer.budget import ContextWindow, RequestLimits
    from summarizer.config import StrategyConfig

    return RequestLimits(
        window=ContextWindow(tokens=32_768, assumed=False),
        config=StrategyConfig(),
        counter=_Characters(),
        correction_headroom=0,
    )


def _root_with_summary(summary: str) -> SummaryNode:
    return root().model_copy(update={"summary": summary})


def test_a_draft_longer_than_the_output_allowance_is_refused_before_any_call() -> None:
    from summarizer.budget import BudgetFailure, RequestBudgetError

    provider = Provider()
    # 4,100 counted tokens plus the answer object exceed the 4,096 allowance.
    with pytest.raises(RequestBudgetError) as caught:
        write_editorial(
            _root_with_summary("x" * 4_100),
            provider,
            source_id=SOURCE_ID,
            model="m",
            timeout_seconds=30,
            target_words=100,
            limits=_limits(),
        )

    assert caught.value.failure is BudgetFailure.OUTPUT_CANNOT_HOLD_DRAFT
    assert "4164 output tokens" in str(caught.value)
    assert provider.requests == []


def test_a_draft_that_fits_the_output_allowance_is_sent() -> None:
    provider = Provider()

    write_editorial(
        _root_with_summary("x" * 4_000),
        provider,
        source_id=SOURCE_ID,
        model="m",
        timeout_seconds=30,
        target_words=100,
        limits=_limits(),
    )

    assert [request.max_output_tokens for request in provider.requests] == [4_096]


def test_an_estimated_count_does_not_refuse_a_draft_as_too_long_to_rewrite() -> None:
    from summarizer.budget import ContextWindow, RequestLimits
    from summarizer.config import StrategyConfig
    from summarizer.tokenization import ConservativeUtf8TokenCounter

    provider = Provider()
    # A target-length draft is far more bytes than the allowance's tokens.
    write_editorial(
        _root_with_summary("word " * 1_000),
        provider,
        source_id=SOURCE_ID,
        model="m",
        timeout_seconds=30,
        target_words=1_000,
        limits=RequestLimits(
            window=ContextWindow(tokens=65_536, assumed=False),
            config=StrategyConfig(),
            counter=ConservativeUtf8TokenCounter(),
            correction_headroom=0,
        ),
    )

    assert len(provider.requests) == 1


def test_an_estimated_count_still_refuses_a_draft_with_more_words_than_output_tokens() -> None:
    from summarizer.budget import BudgetFailure, ContextWindow, RequestBudgetError, RequestLimits
    from summarizer.config import StrategyConfig
    from summarizer.tokenization import ConservativeUtf8TokenCounter

    provider = Provider()
    # The #121 case: 4,641 words to rewrite within a 4,096-token allowance.
    with pytest.raises(RequestBudgetError) as caught:
        write_editorial(
            _root_with_summary("word " * 4_641),
            provider,
            source_id=SOURCE_ID,
            model="m",
            timeout_seconds=30,
            target_words=980,
            limits=RequestLimits(
                window=ContextWindow(tokens=65_536, assumed=False),
                config=StrategyConfig(),
                counter=ConservativeUtf8TokenCounter(),
                correction_headroom=0,
            ),
        )

    assert caught.value.failure is BudgetFailure.OUTPUT_CANNOT_HOLD_DRAFT
    assert provider.requests == []


# A local model writing JSON under a grammar ends the text field at an
# unescaped double quotation mark, so its draft stops where a quotation began.
_CUT = json.dumps({"text": "The team met weekly. The lead said the goal was to "})
_COMPLETE = json.dumps({"text": "The team met weekly. The lead named one goal."})


def _write(provider):
    observer, events = _recording_observer()
    result = write_editorial(
        root(),
        provider,
        source_id=SOURCE_ID,
        model="m",
        timeout_seconds=30,
        target_words=50,
        observer=observer,
    )
    return result, events


def test_a_draft_that_stops_mid_sentence_is_asked_for_again_with_the_reason() -> None:
    provider = ScriptedProvider(_CUT, _COMPLETE)

    result, events = _write(provider)

    assert result.text == "The team met weekly. The lead named one goal."
    first, second = provider.requests
    assert second.input_text == first.input_text
    assert "stops mid-sentence" in second.instructions
    assert [state for state, _, _ in events] == ["active", "retrying", "completed"]
    assert "stops mid-sentence" in events[1][2]


def test_the_last_unfinished_draft_is_kept_when_no_reask_finishes() -> None:
    provider = ScriptedProvider(_CUT, _CUT, _CUT)

    result, events = _write(provider)

    assert len(provider.requests) == 3
    assert result.text == "The team met weekly. The lead said the goal was to"
    assert [state for state, _, _ in events] == ["active", "retrying", "retrying", "completed"]
    assert "stops mid-sentence" in events[-1][2]


def test_an_invalid_reask_answer_does_not_cost_the_unfinished_draft() -> None:
    provider = ScriptedProvider(_CUT, "not json", "not json")

    result, events = _write(provider)

    assert result.text == "The team met weekly. The lead said the goal was to"
    assert events[-1][0] == "completed"


@pytest.mark.parametrize(
    "text",
    ["The lead named one goal.", "The lead said 'one goal.'", "**The lead named one goal.**"],
)
def test_a_complete_draft_is_not_asked_for_again(text: str) -> None:
    provider = ScriptedProvider(json.dumps({"text": text}))

    result, _ = _write(provider)

    assert result.text == text
    assert len(provider.requests) == 1


def _cached_write(tmp_path, *responses: str) -> ScriptedProvider:
    provider = ScriptedProvider(*responses)
    provider.cache_coordinator = CacheCoordinator(
        store=CacheStore(tmp_path / "cache"), source_id=SOURCE_ID,
        provider="openai", model="m", ollama_host="",
        counter_identity="test:characters",
        counter_exact=True, context_window_tokens=100_000,
        behavior={},
    )
    _write(provider)
    return provider


def test_a_kept_unfinished_draft_is_not_cached(tmp_path) -> None:
    _cached_write(tmp_path, _CUT, _CUT, _CUT)

    later = _cached_write(tmp_path, _COMPLETE)

    assert len(later.requests) == 1


def test_a_complete_draft_is_cached(tmp_path) -> None:
    _cached_write(tmp_path, _CUT, _COMPLETE)

    later = _cached_write(tmp_path)

    assert later.requests == []
