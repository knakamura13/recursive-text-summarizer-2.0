from dataclasses import dataclass

import pytest

from summarizer.budget import (
    BudgetFailure,
    ContextWindow,
    OverheadMeasurement,
    RequestBudgetError,
    correction_headroom,
    measure_overhead,
    plan_request,
    safety_margin,
)
from summarizer.config import StrategyConfig


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


def test_overhead_counts_instructions_schema_and_fencing() -> None:
    overhead = measure_overhead(CharacterCounter(), with_overlap=False)

    assert overhead.instructions > 0
    assert overhead.schema > 0
    assert overhead.fencing > 0
    assert overhead.total == (
        overhead.instructions + overhead.schema + overhead.fencing
    )


def test_overlap_increases_overhead() -> None:
    """Overhead is not one constant: overlap adds a second instruction block."""
    plain = measure_overhead(CharacterCounter(), with_overlap=False)
    overlapping = measure_overhead(CharacterCounter(), with_overlap=True)

    assert overlapping.total > plain.total
    assert overlapping.instructions > plain.instructions


def test_overhead_is_measured_not_assumed() -> None:
    """A hard-coded constant would silently rot when the prompt changes.

    Counting with a character counter makes the measurement equal the actual
    character length of what is sent, so this fails if the schema or the
    instruction template stops being measured at all.
    """
    from summarizer.summaries import (
        MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES,
        MAX_QUOTE_CANDIDATE_JSON_BYTES,
        leaf_summary_schema,
    )
    import json

    overhead = measure_overhead(
        CharacterCounter(),
        with_overlap=False,
        provider_schema_reserve=MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES,
    )
    schema_chars = len(json.dumps(leaf_summary_schema(), separators=(",", ":")))

    assert overhead.schema == (
        schema_chars
        + MAX_QUOTE_CANDIDATE_JSON_BYTES
        + MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES
    )


def test_overhead_with_a_real_encoding_matches_the_measured_scale() -> None:
    """One case against a real tokenizer, since every other test uses a fake.

    The skip covers vocabulary availability alone, so a broken counter still
    fails loudly rather than skipping.
    """
    import tiktoken

    from summarizer.summaries import MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES
    from summarizer.tokenization import TiktokenCounter

    try:
        tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception as error:  # pragma: no cover - depends on the local cache
        pytest.skip(f"tiktoken vocabulary is unavailable offline: {error}")

    overhead = measure_overhead(
        TiktokenCounter.for_model("gpt-4o-mini"),
        with_overlap=False,
        provider_schema_reserve=MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES,
    )

    # Pinned exactly rather than banded: a band this wide would not notice the
    # prompt or the record changing, which is the thing worth noticing.
    assert (overhead.instructions, overhead.schema, overhead.fencing) == (
        345,
        1_890,
        27,
    )
    assert overhead.total == 2_262

def test_safety_margin_takes_the_larger_term() -> None:
    config = StrategyConfig(safety_margin_tokens=256, safety_margin_fraction=0.02)

    # Fixed term dominates a small window.
    assert safety_margin(4_096, config) == 256
    # Fractional term dominates a large one.
    assert safety_margin(128_000, config) == 2_560


# A fixed margin keeps the arithmetic in these tests visible term by term.
NO_MARGIN = StrategyConfig(safety_margin_tokens=0, safety_margin_fraction=0)
NO_OVERHEAD = OverheadMeasurement(instructions=0, schema=0, fencing=0)


def window(tokens: int) -> ContextWindow:
    return ContextWindow(tokens=tokens, assumed=False)


@pytest.mark.parametrize(
    ("stage", "overhead", "evidence", "headroom", "capacity"),
    [
        # 10_000 - margin 100 - output 1_000 = 8_900 before stage reserves.
        ("direct", OverheadMeasurement(300, 1_900, 30), 0, 0, 6_670),
        ("leaf", OverheadMeasurement(420, 1_900, 30), 0, 170, 6_380),
        ("merge", OverheadMeasurement(500, 2_400, 60), 1_500, 170, 4_270),
        ("editorial", OverheadMeasurement(600, 300, 20), 0, 170, 7_810),
    ],
)
def test_each_stage_is_charged_its_own_reserves(
    stage, overhead, evidence, headroom, capacity
) -> None:
    config = StrategyConfig(safety_margin_tokens=100, safety_margin_fraction=0)

    budget = plan_request(
        stage,
        window=window(10_000),
        overhead=overhead,
        output_allowance=1_000,
        correction_headroom=headroom,
        config=config,
        evidence=evidence,
    )

    assert budget.input_capacity == capacity
    assert budget.arithmetic().endswith(f"= {capacity} input tokens")


def test_input_capacity_of_one_is_feasible_and_zero_is_refused() -> None:
    overhead = OverheadMeasurement(instructions=10, schema=20, fencing=3)
    fixed = overhead.total + 5 + 7 + 100  # evidence, headroom, output

    def plan(tokens: int):
        return plan_request(
            "leaf",
            window=window(tokens),
            overhead=overhead,
            output_allowance=100,
            correction_headroom=7,
            config=NO_MARGIN,
            evidence=5,
        )

    assert plan(fixed + 1).input_capacity == 1
    with pytest.raises(RequestBudgetError) as error:
        plan(fixed)

    assert error.value.failure is BudgetFailure.NO_INPUT_CAPACITY
    assert "= 0 input tokens" in str(error.value)


def test_output_cannot_take_the_whole_usable_context() -> None:
    """An allowance equal to the usable context leaves no room for a prompt."""
    config = StrategyConfig(safety_margin_tokens=100, safety_margin_fraction=0)

    def plan(output: int):
        return plan_request(
            "editorial",
            window=window(1_000),
            overhead=NO_OVERHEAD,
            output_allowance=output,
            correction_headroom=0,
            config=config,
        )

    assert plan(899).input_capacity == 1
    with pytest.raises(RequestBudgetError) as error:
        plan(900)

    assert error.value.failure is BudgetFailure.OUTPUT_EXCEEDS_CONTEXT


def test_output_larger_than_context_is_a_configuration_failure() -> None:
    """Recorded transcript-50% shape: the allowance alone exceeds the window.

    The allowance is refused, not lowered to fit, and the error says the
    limit is the configuration rather than the requested target.
    """
    config = StrategyConfig(safety_margin_tokens=256, safety_margin_fraction=0.02)

    with pytest.raises(RequestBudgetError) as error:
        plan_request(
            "editorial",
            window=window(32_768),
            overhead=OverheadMeasurement(instructions=1_046, schema=4_400, fencing=100),
            output_allowance=40_882,
            correction_headroom=0,
            config=config,
        )

    assert error.value.failure is BudgetFailure.OUTPUT_EXCEEDS_CONTEXT
    assert error.value.budget.output_allowance_tokens == 40_882
    assert error.value.budget.input_capacity == -14_315
    message = str(error.value)
    for term in (
        "context window 32768",
        "safety margin 655",
        "output allowance 40882",
        "= -14315 input tokens",
        "usable context of 32113",
        "not of the requested output target",
    ):
        assert term in message


def test_a_margin_that_consumes_the_window_is_not_blamed_on_output() -> None:
    config = StrategyConfig(safety_margin_tokens=256, safety_margin_fraction=0)

    with pytest.raises(RequestBudgetError) as error:
        plan_request(
            "direct",
            window=window(200),
            overhead=NO_OVERHEAD,
            output_allowance=1,
            correction_headroom=0,
            config=config,
        )

    assert error.value.failure is BudgetFailure.NO_INPUT_CAPACITY
    assert "safety margin of 256 tokens leaves no usable context" in str(error.value)



def test_non_positive_capacity_reports_every_term() -> None:
    """Reachable on default local configuration, so it must be a named error.

    The byte estimator charges over four times a real tokenizer on the
    project's own ASCII, which drives a small window negative.
    """
    config = StrategyConfig(max_output_tokens=1_024, safety_margin_fraction=0.05)
    overhead = measure_overhead(CharacterCounter(), with_overlap=False)

    with pytest.raises(RequestBudgetError) as error:
        plan_request(
            "leaf",
            window=ContextWindow(tokens=4_096, assumed=True),
            overhead=overhead,
            output_allowance=1_024,
            correction_headroom=0,
            config=config,
        )

    assert error.value.failure is BudgetFailure.NO_INPUT_CAPACITY
    message = str(error.value)
    # Each term's own value, not just its label: "1024" alone would match the
    # window as well as the reserve.
    assert f"instructions {overhead.instructions}" in message
    assert f"schema {overhead.schema}" in message
    assert f"fencing {overhead.fencing}" in message
    assert "output allowance 1024" in message
    # The fixed floor dominates at this window: max(256, 0.05 * 4096) == 256.
    assert "safety margin 256" in message


def test_measured_input_is_refused_one_token_over_capacity() -> None:
    budget = plan_request(
        "direct",
        window=window(1_000),
        overhead=NO_OVERHEAD,
        output_allowance=100,
        correction_headroom=0,
        config=NO_MARGIN,
    )

    budget.require_input(900)
    with pytest.raises(RequestBudgetError) as error:
        budget.require_input(901)

    assert error.value.failure is BudgetFailure.INPUT_EXCEEDS_CAPACITY
    assert "input of 901 tokens exceeds the input capacity of 900" in str(error.value)


def merge_budget(capacity: int):
    """A 32,768-token merge budget whose input capacity is `capacity`."""
    config = StrategyConfig(safety_margin_tokens=256, safety_margin_fraction=0.02)
    overhead = OverheadMeasurement(instructions=800, schema=2_500, fencing=44)
    return plan_request(
        "merge",
        window=window(32_768),
        overhead=overhead,
        output_allowance=32_768 - 655 - overhead.total - capacity,
        correction_headroom=0,
        config=config,
    )


@pytest.mark.parametrize(
    ("capacity", "largest_child"),
    [
        # Recorded statistical-learning merge shapes, as token counts only.
        (12_269, 6_997),
        (578, 4_746),
    ],
)
def test_a_merge_that_cannot_hold_two_children_is_refused(
    capacity: int, largest_child: int
) -> None:
    budget = merge_budget(capacity)

    with pytest.raises(RequestBudgetError) as error:
        budget.require_merge_pair(largest_child)

    assert error.value.failure is BudgetFailure.MERGE_PAIR_EXCEEDS_CAPACITY
    assert (
        f"pair of {2 * largest_child} against an input capacity of {capacity}"
        in str(error.value)
    )


def test_merge_pair_boundary_is_exact() -> None:
    budget = merge_budget(12_268)

    budget.require_merge_pair(6_134)
    with pytest.raises(RequestBudgetError):
        budget.require_merge_pair(6_135)


@pytest.mark.parametrize("structured", [True, False])
def test_correction_headroom_covers_the_longest_reask(structured: bool) -> None:
    """A re-ask with the longest, widest-encoded reason still fits the reserve."""
    from summarizer.providers.base import GenerationRequest
    from summarizer.reask import reask_request, rejection_reason
    from summarizer.tokenization import ConservativeUtf8TokenCounter

    counter = ConservativeUtf8TokenCounter()
    request = GenerationRequest(
        model="m",
        instructions="Summarize the passage.",
        input_text="passage",
        timeout_seconds=1,
        response_schema={"type": "object"} if structured else None,
        schema_name="summary" if structured else None,
    )
    reason = rejection_reason(ValueError("\U0010ffff" * 1_000))

    grown = counter.count(reask_request(request, reason).instructions) - counter.count(
        request.instructions
    )

    assert 0 < grown <= correction_headroom(counter)


def test_plan_rejects_a_non_positive_output_allowance() -> None:
    with pytest.raises(ValueError, match="output allowance"):
        plan_request(
            "leaf",
            window=window(1_000),
            overhead=NO_OVERHEAD,
            output_allowance=0,
            correction_headroom=0,
            config=NO_MARGIN,
        )


def test_fencing_excludes_the_probe_text_it_measured_with() -> None:
    """The fencing term is defined by that subtraction, so pin it.

    Without this, dropping the subtraction leaves every assertion satisfied
    while the fencing figure silently absorbs the probe's own text.
    """
    counter = CharacterCounter()
    overhead = measure_overhead(counter, with_overlap=False)

    from summarizer.budget import _overhead_probe_segment
    from summarizer.leaf import build_leaf_request

    probe = _overhead_probe_segment(with_overlap=False)
    request = build_leaf_request(probe, model="probe", timeout_seconds=1)

    assert overhead.fencing == len(request.input_text) - len(probe.text)
    assert "probe" not in str(overhead.fencing)


def test_fencing_is_clamped_to_zero_when_negative() -> None:
    """If counter returns fewer tokens for input_text than probe.text, fencing clamps to 0."""

    class MockNegativeFencingCounter:
        identity: str = "test:negative_fencing"
        exact: bool = True
        monotonic: bool = True

        def count(self, text: str) -> int:
            if "probe" in text and len(text) <= 10:
                return 100
            return 10

    overhead = measure_overhead(MockNegativeFencingCounter(), with_overlap=False)  # type: ignore[arg-type]

    assert overhead.fencing == 0
