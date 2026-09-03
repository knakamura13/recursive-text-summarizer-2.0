from dataclasses import dataclass

import pytest

from summarizer.budget import BudgetError, measure_overhead, select_strategy
from summarizer.config import StrategyConfig
from summarizer.ingestion import ingest_text


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


def config_for(document_tokens: int, *, slack: int, **overrides: object) -> StrategyConfig:
    """Build a configuration whose capacity is `document_tokens + slack`."""
    counter = CharacterCounter()
    overhead = measure_overhead(counter, with_overlap=False)
    base: dict[str, object] = {
        "max_output_tokens": 1,
        "safety_margin_tokens": 0,
        "safety_margin_fraction": 0,
    }
    base.update(overrides)
    window = overhead.total + 1 + document_tokens + slack
    base["context_window"] = window
    return StrategyConfig(**base)  # type: ignore[arg-type]


def select(text: str, config: StrategyConfig):
    return select_strategy(
        ingest_text(text),
        CharacterCounter(),
        provider="openai",
        model="gpt-4o-mini",
        config=config,
    )


def test_selects_direct_when_the_document_fits() -> None:
    text = "a" * 100
    report = select(text, config_for(100, slack=1))

    assert report.strategy == "direct"
    assert report.document_tokens == 100
    assert report.fits is True


def test_selects_hierarchical_one_token_over_capacity() -> None:
    """The boundary is exact, so both sides of it are pinned."""
    text = "a" * 100
    report = select(text, config_for(100, slack=-1))

    assert report.strategy == "hierarchical"
    assert report.fits is False


def test_selects_direct_exactly_at_capacity() -> None:
    text = "a" * 100
    report = select(text, config_for(100, slack=0))

    assert report.strategy == "direct"
    assert report.document_tokens == report.usable_input_capacity


def test_an_assumed_window_routes_auto_to_hierarchical() -> None:
    """A guessed window must not be gambled on a direct request."""
    report = select_strategy(
        ingest_text("short"),
        CharacterCounter(),
        provider="ollama",
        model="qwen3.8",
        config=StrategyConfig(
            max_output_tokens=1, safety_margin_tokens=0, safety_margin_fraction=0
        ),
    )

    assert report.context_window_assumed is True
    assert report.strategy == "hierarchical"
    assert "assumed" in report.reason


def test_a_direct_cap_forces_hierarchical_even_when_it_fits() -> None:
    text = "a" * 100
    report = select(text, config_for(100, slack=10_000, max_direct_tokens=50))

    assert report.fits is True
    assert report.strategy == "hierarchical"
    assert "cap" in report.reason


def test_explicit_hierarchical_is_honoured_even_when_the_document_fits() -> None:
    report = select(
        "a" * 100, config_for(100, slack=10_000, strategy="hierarchical")
    )

    assert report.fits is True
    assert report.strategy == "hierarchical"
    assert "explicitly" in report.reason


@pytest.mark.parametrize("strategy", ["auto", "direct", "hierarchical"])
def test_a_window_with_no_usable_capacity_fails_for_every_strategy(
    strategy: str,
) -> None:
    """No strategy can proceed without positive capacity.

    Hierarchical execution sizes each of its own requests against the same
    capacity, so a window this small is a dead configuration rather than a
    reason to prefer one path over another.
    """
    with pytest.raises(BudgetError, match="no usable input capacity"):
        select(
            "a" * 100,
            StrategyConfig(
                strategy=strategy,  # type: ignore[arg-type]
                context_window=10,
                max_output_tokens=1,
                safety_margin_tokens=0,
                safety_margin_fraction=0,
            ),
        )


def test_explicit_direct_over_capacity_fails_with_its_arithmetic() -> None:
    """Rejection happens before any provider call and shows the numbers used."""
    with pytest.raises(BudgetError) as error:
        select(
            "a" * 100,
            config_for(100, slack=-1, strategy="direct"),
        )

    message = str(error.value)
    assert "100" in message
    for term in ("capacity", "direct"):
        assert term in message


def test_report_carries_the_counter_and_overhead_it_used() -> None:
    report = select("a" * 10, config_for(10, slack=100))

    assert report.counter_identity == "test:characters"
    assert report.counter_exact is True
    assert report.overhead.total > 0
    assert report.reserved_output_tokens == 1
    assert report.reason


def test_reports_are_deterministic() -> None:
    text = "a" * 100

    assert select(text, config_for(100, slack=5)) == select(
        text, config_for(100, slack=5)
    )
