from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from nltk.tokenize import sent_tokenize

from summarizer.config import AppConfig, LegacyWorkflowConfig
from summarizer.providers.base import GenerationResult, ModelProvider
from summarizer.text import (
    build_generation_request,
    chunk_text_by_sentences,
    normalize_whitespace,
)


@dataclass(frozen=True)
class WorkflowResult:
    text: str
    generations: tuple[GenerationResult, ...]


class LegacyWorkflow:
    def __init__(
        self,
        provider: ModelProvider,
        app_config: AppConfig,
        workflow_config: LegacyWorkflowConfig,
        sentence_tokenizer: Callable[[str], list[str]] = sent_tokenize,
    ) -> None:
        self._provider = provider
        self._app_config = app_config
        self._workflow_config = workflow_config
        self._sentence_tokenizer = sentence_tokenizer

    def summarize(self, text: str) -> WorkflowResult:
        chunks = chunk_text_by_sentences(
            text,
            self._workflow_config.chunk_size,
            self._sentence_tokenizer,
        )
        if self._workflow_config.max_chunks > 0:
            chunks = chunks[: self._workflow_config.max_chunks]

        generations: list[GenerationResult] = []
        for index, chunk in enumerate(chunks, start=1):
            if self._workflow_config.dry_run:
                generation = GenerationResult(
                    text=normalize_whitespace(chunk),
                    provider="dry-run",
                    model=self._app_config.model,
                )
            else:
                request = build_generation_request(
                    chunk,
                    model=self._app_config.model,
                    timeout_seconds=self._app_config.timeout_seconds,
                    operation_id=f"chunk-{index}",
                )
                generated = self._provider.generate(request)
                generation = replace(
                    generated,
                    text=normalize_whitespace(generated.text),
                )
            generations.append(generation)

        return WorkflowResult(
            text="\n".join(item.text for item in generations),
            generations=tuple(generations),
        )


def run_file_workflow(
    app_config: AppConfig,
    workflow: LegacyWorkflow,
) -> WorkflowResult:
    source = app_config.input_path.read_text(encoding="utf-8")
    result = workflow.summarize(source)
    app_config.output_path.parent.mkdir(parents=True, exist_ok=True)
    app_config.output_path.write_text(result.text, encoding="utf-8")
    return result
