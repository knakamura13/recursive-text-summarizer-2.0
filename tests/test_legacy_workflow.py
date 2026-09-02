from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from summarizer.config import AppConfig, LegacyWorkflowConfig
from summarizer.legacy_workflow import LegacyWorkflow, run_file_workflow
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderRequestError,
)


@dataclass
class ScriptedProvider:
    outcomes: deque[object]
    calls: list[GenerationRequest] = field(default_factory=list)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, GenerationResult)
        return outcome


def result(text: str) -> GenerationResult:
    return GenerationResult(text, "fake", "resolved-model")


def make_workflow(
    provider: ScriptedProvider,
    *,
    max_chunks: int = -1,
    dry_run: bool = False,
) -> LegacyWorkflow:
    return LegacyWorkflow(
        provider=provider,
        app_config=AppConfig(
            input_path=Path("source.txt"),
            output_path=Path("summary.txt"),
            model="model",
            timeout_seconds=45,
        ),
        workflow_config=LegacyWorkflowConfig(
            chunk_size=12,
            max_chunks=max_chunks,
            dry_run=dry_run,
        ),
        sentence_tokenizer=lambda _text: ["Alpha.", "Beta.", "Gamma."],
    )


def test_summarizes_chunks_in_source_order_and_retains_metadata() -> None:
    provider = ScriptedProvider(deque([result(" first\nsummary "), result("second")]))
    workflow = make_workflow(provider)

    workflow_result = workflow.summarize("ignored")

    assert workflow_result.text == "first summary\nsecond"
    assert workflow_result.generations == (
        result("first summary"),
        result("second"),
    )
    assert [call.operation_id for call in provider.calls] == ["chunk-1", "chunk-2"]
    assert [call.model for call in provider.calls] == ["model", "model"]
    assert [call.timeout_seconds for call in provider.calls] == [45, 45]
    assert provider.calls[0].input_text.endswith('\n"""\n Alpha. Beta.\n"""\n')
    assert provider.calls[1].input_text.endswith('\n"""\nGamma.\n"""\n')


def test_positive_max_chunks_keeps_only_source_prefix() -> None:
    provider = ScriptedProvider(deque([result("first"), result("unused")]))
    workflow = make_workflow(provider, max_chunks=1)

    workflow_result = workflow.summarize("ignored")

    assert workflow_result.text == "first"
    assert len(provider.calls) == 1
    assert list(provider.outcomes) == [result("unused")]


def test_dry_run_normalizes_chunks_without_provider_calls() -> None:
    provider = ScriptedProvider(deque())
    workflow = make_workflow(provider, dry_run=True)

    workflow_result = workflow.summarize("ignored")

    assert workflow_result.text == "Alpha. Beta.\nGamma."
    assert provider.calls == []
    assert [item.provider for item in workflow_result.generations] == [
        "dry-run",
        "dry-run",
    ]


@pytest.mark.parametrize("preexisting", [None, "previous summary"])
def test_provider_failure_does_not_create_or_replace_output(
    tmp_path: Path,
    preexisting: str | None,
) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "summary.txt"
    source.write_text("Alpha.", encoding="utf-8")
    if preexisting is not None:
        output.write_text(preexisting, encoding="utf-8")
    provider = ScriptedProvider(deque([ProviderRequestError("failed")]))
    app_config = AppConfig(source, output, "model", 45)
    workflow = LegacyWorkflow(
        provider,
        app_config,
        LegacyWorkflowConfig(),
        sentence_tokenizer=lambda _text: ["Alpha."],
    )

    with pytest.raises(ProviderRequestError, match="failed"):
        run_file_workflow(app_config, workflow)

    if preexisting is None:
        assert not output.exists()
    else:
        assert output.read_text(encoding="utf-8") == preexisting


def test_file_workflow_round_trips_unicode(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "nested" / "summary.txt"
    source.write_text("Résumé, 東京.", encoding="utf-8")
    provider = ScriptedProvider(deque([result("Résumé, 東京")]))
    app_config = AppConfig(source, output, "model", 45)
    workflow = LegacyWorkflow(
        provider,
        app_config,
        LegacyWorkflowConfig(),
        sentence_tokenizer=lambda text: [text],
    )

    workflow_result = run_file_workflow(app_config, workflow)

    assert workflow_result.text == "Résumé, 東京"
    assert output.read_text(encoding="utf-8") == "Résumé, 東京"


def test_large_input_reaches_tokenizer_without_unconfigured_truncation() -> None:
    source = "Signal. " * 100_000
    observed: list[str] = []
    provider = ScriptedProvider(deque([result("summary")]))

    def tokenizer(text: str) -> list[str]:
        observed.append(text)
        return [text]

    workflow = LegacyWorkflow(
        provider,
        AppConfig(Path("source"), Path("output"), "model", 45),
        LegacyWorkflowConfig(chunk_size=len(source) + 1),
        sentence_tokenizer=tokenizer,
    )

    workflow.summarize(source)

    assert observed == [source]
    assert source in provider.calls[0].input_text
