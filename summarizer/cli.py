from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from summarizer.config import AppConfig, LegacyWorkflowConfig, RetryPolicy
from summarizer.legacy_workflow import LegacyWorkflow, run_file_workflow
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelProvider,
    ProviderError,
)
from summarizer.providers.openai import OpenAIProvider
from summarizer.providers.retrying import RetryingProvider


@dataclass(frozen=True)
class ParsedConfig:
    app: AppConfig
    retry: RetryPolicy
    workflow: LegacyWorkflowConfig


class _UnavailableProvider:
    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise AssertionError("provider called during dry run")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a UTF-8 text file")
    parser.add_argument("--input", type=Path, default=Path("input.txt"))
    parser.add_argument("--output", type=Path, default=Path("output.txt"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--max-chunks", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> ParsedConfig:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return ParsedConfig(
            app=AppConfig(
                input_path=args.input,
                output_path=args.output,
                model=args.model,
                timeout_seconds=args.timeout,
            ),
            retry=RetryPolicy(max_attempts=args.max_retries),
            workflow=LegacyWorkflowConfig(
                chunk_size=args.chunk_size,
                max_chunks=args.max_chunks,
                dry_run=args.dry_run,
            ),
        )
    except ValueError as error:
        parser.error(str(error))


def main(
    argv: list[str] | None = None,
    *,
    provider_factory: Callable[[], ModelProvider] = OpenAIProvider,
    sentence_tokenizer: Callable[[str], list[str]] | None = None,
) -> int:
    config = parse_args(argv)
    logging.basicConfig(
        filename="summarizer.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
    )
    provider: ModelProvider
    if config.workflow.dry_run:
        provider = _UnavailableProvider()
    else:
        provider = RetryingProvider(provider_factory(), config.retry)

    workflow_kwargs: dict[str, object] = {}
    if sentence_tokenizer is not None:
        workflow_kwargs["sentence_tokenizer"] = sentence_tokenizer
    workflow = LegacyWorkflow(
        provider,
        config.app,
        config.workflow,
        **workflow_kwargs,
    )
    try:
        run_file_workflow(config.app, workflow)
    except (OSError, ProviderError) as error:
        logging.getLogger(__name__).error("Summarization failed: %s", error)
        print(f"Summarization failed: {error}", file=sys.stderr)
        return 1
    return 0
