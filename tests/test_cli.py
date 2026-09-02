from collections.abc import Callable
from pathlib import Path

import pytest

from summarizer.cli import main, parse_args
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelProvider,
    ProviderRequestError,
)


class RecordingProvider:
    def __init__(self, outcome: object = "summary") -> None:
        self.outcome = outcome
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        assert isinstance(self.outcome, str)
        return GenerationResult(self.outcome, "fake", request.model)


def test_parse_args_returns_exact_defaults() -> None:
    parsed = parse_args([])

    assert parsed.app.input_path == Path("input.txt")
    assert parsed.app.output_path == Path("output.txt")
    assert parsed.app.model == "gpt-4o-mini"
    assert parsed.app.timeout_seconds == 180
    assert parsed.retry.max_attempts == 5
    assert parsed.workflow.chunk_size == 1000
    assert parsed.workflow.max_chunks == -1
    assert parsed.workflow.dry_run is False


def test_parse_args_supports_all_overrides() -> None:
    parsed = parse_args(
        [
            "--input",
            "source.txt",
            "--output",
            "summary.txt",
            "--model",
            "custom-model",
            "--timeout",
            "42.5",
            "--max-retries",
            "3",
            "--chunk-size",
            "2048",
            "--max-chunks",
            "7",
            "--dry-run",
        ]
    )

    assert parsed.app.input_path == Path("source.txt")
    assert parsed.app.output_path == Path("summary.txt")
    assert parsed.app.model == "custom-model"
    assert parsed.app.timeout_seconds == 42.5
    assert parsed.retry.max_attempts == 3
    assert parsed.workflow.chunk_size == 2048
    assert parsed.workflow.max_chunks == 7
    assert parsed.workflow.dry_run is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["--timeout", "0"],
        ["--max-retries", "0"],
        ["--chunk-size", "0"],
        ["--max-chunks", "0"],
        ["--input", "same.txt", "--output", "same.txt"],
    ],
)
def test_parse_args_reports_invalid_configuration(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args(arguments)

    assert exc_info.value.code == 2


def test_main_runs_default_file_workflow_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("Source.", encoding="utf-8")
    provider = RecordingProvider(" concise\nsummary ")

    exit_code = main(
        [],
        provider_factory=lambda: provider,
        sentence_tokenizer=lambda text: [text],
    )

    assert exit_code == 0
    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == (
        "concise summary"
    )
    assert len(provider.calls) == 1
    assert provider.calls[0].model == "gpt-4o-mini"
    assert provider.calls[0].timeout_seconds == 180


def test_main_reports_provider_failure_and_preserves_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("Source.", encoding="utf-8")
    (tmp_path / "output.txt").write_text("previous", encoding="utf-8")
    provider = RecordingProvider(ProviderRequestError("invalid request"))

    exit_code = main(
        [],
        provider_factory=lambda: provider,
        sentence_tokenizer=lambda text: [text],
    )

    assert exit_code == 1
    assert "invalid request" in capsys.readouterr().err
    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "previous"
    assert len(provider.calls) == 1


def test_main_reports_missing_input_without_creating_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main([], provider_factory=RecordingProvider)

    assert exit_code == 1
    assert "input.txt" in capsys.readouterr().err
    assert not (tmp_path / "output.txt").exists()


def test_dry_run_does_not_construct_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("Source.", encoding="utf-8")
    constructions: list[object] = []

    def provider_factory() -> ModelProvider:
        constructions.append(object())
        raise AssertionError("provider constructed during dry run")

    exit_code = main(
        ["--dry-run"],
        provider_factory=provider_factory,
        sentence_tokenizer=lambda text: [text],
    )

    assert exit_code == 0
    assert constructions == []
    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "Source."
