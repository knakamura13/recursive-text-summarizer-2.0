from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from summarizer.config import AppConfig, LegacyWorkflowConfig, RetryPolicy


def test_configuration_defaults_are_legacy_compatible() -> None:
    app = AppConfig()
    retry = RetryPolicy()
    workflow = LegacyWorkflowConfig()

    assert app.input_path == Path("input.txt")
    assert app.output_path == Path("output.txt")
    assert app.model == "gpt-4o-mini"
    assert app.timeout_seconds == 180
    assert retry.max_attempts == 5
    assert retry.initial_delay_seconds == 1
    assert retry.backoff_multiplier == 2
    assert workflow.chunk_size == 1000
    assert workflow.max_chunks == -1
    assert workflow.dry_run is False


@pytest.mark.parametrize(
    ("configuration", "field_name"),
    [
        (AppConfig(), "model"),
        (RetryPolicy(), "max_attempts"),
        (LegacyWorkflowConfig(), "chunk_size"),
    ],
)
def test_configuration_is_immutable(
    configuration: object,
    field_name: str,
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(configuration, field_name, "changed")


@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (lambda: AppConfig(model=" "), "model"),
        (lambda: AppConfig(input_path=Path("")), "input_path"),
        (lambda: AppConfig(output_path=Path("")), "output_path"),
        (
            lambda: AppConfig(
                input_path=Path("same.txt"),
                output_path=Path("./same.txt"),
            ),
            "output_path",
        ),
        (lambda: AppConfig(timeout_seconds=0), "timeout_seconds"),
        (lambda: RetryPolicy(max_attempts=0), "max_attempts"),
        (
            lambda: RetryPolicy(initial_delay_seconds=0),
            "initial_delay_seconds",
        ),
        (lambda: RetryPolicy(backoff_multiplier=0), "backoff_multiplier"),
        (lambda: LegacyWorkflowConfig(chunk_size=0), "chunk_size"),
        (lambda: LegacyWorkflowConfig(max_chunks=0), "max_chunks"),
        (lambda: LegacyWorkflowConfig(max_chunks=-2), "max_chunks"),
    ],
)
def test_configuration_rejects_invalid_values(
    factory: object,
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        factory()
