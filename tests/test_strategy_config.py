import pytest

from summarizer.config import AppConfig, StrategyConfig


def test_defaults_to_auto() -> None:
    assert StrategyConfig().strategy == "auto"


def test_rejects_an_unknown_strategy_at_the_record() -> None:
    """Argparse `choices` catches the CLI path, so the record needs its own test.

    A library caller passing a mis-cased or invented name would otherwise fall
    through to the auto path silently.
    """
    for name in ("sideways", "Direct", "DIRECT", ""):
        with pytest.raises(ValueError, match="strategy must be one of"):
            StrategyConfig(strategy=name)  # type: ignore[arg-type]


def test_app_config_still_accepts_positional_construction() -> None:
    """A commit exists in this history solely to repair this after a field moved."""
    from pathlib import Path

    config = AppConfig(Path("in.txt"), Path("out.txt"), "model", 45)

    assert config.model == "model"
    assert config.timeout_seconds == 45
