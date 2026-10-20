"""An OpenRouter run through the real pipeline, cache and audit."""

import json
from dataclasses import replace
from pathlib import Path

from summarizer.config import AppConfig, CacheConfig, ReliabilityConfig, StrategyConfig
from summarizer.ingestion import ingest_text
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import GenerationRequest, GenerationResult
from tests.test_pipeline import CharacterCounter, PipelineProvider

MODEL = "meta-llama/llama-3.3-70b-instruct:free"


class OpenRouterShapedProvider(PipelineProvider):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        return replace(super().generate(request), provider="openrouter")


def _strategy() -> StrategyConfig:
    return StrategyConfig(
        context_window=100_000,
        max_output_tokens=1,
        safety_margin_tokens=0,
        safety_margin_fraction=0,
    )


def _run(tmp_path: Path, *, upstream: str, run_id: str, provider: PipelineProvider):
    audit_path = tmp_path / f"{run_id}.json"
    run_pipeline(
        ingest_text("A short source for the OpenRouter pipeline."),
        provider,
        CharacterCounter(),
        app=AppConfig(
            output_path=tmp_path / f"{run_id}.txt",
            provider="openrouter",
            model=MODEL,
            timeout_seconds=30,
            openrouter_upstream=upstream,
        ),
        strategy=_strategy(),
        config=PipelineConfig(
            target_words=40,
            audit_path=audit_path,
            cache=CacheConfig(enabled=True, root=tmp_path / "cache"),
            reliability=ReliabilityConfig(run_id=run_id),
        ),
    )
    return json.loads(audit_path.read_text())


def test_audit_records_provider_model_pinned_upstream_and_estimate_status(
    tmp_path: Path,
) -> None:
    audit = _run(
        tmp_path, upstream="deepinfra/turbo", run_id="run-one", provider=OpenRouterShapedProvider()
    )

    assert audit["configuration"]["app"] == {
        "model": MODEL,
        "openrouter_upstream": "deepinfra/turbo",
        "provider": "openrouter",
        "timeout_seconds": 30,
    }
    assert {usage["provider"] for usage in audit["usage"]} == {"openrouter"}
    assert {usage["model"] for usage in audit["usage"]} == {MODEL}
    assert audit["configuration"]["budget"]["counter_exact"] is True


def test_an_unpinned_run_leaves_the_upstream_out_of_the_audit(tmp_path: Path) -> None:
    audit = _run(tmp_path, upstream="", run_id="run-one", provider=OpenRouterShapedProvider())

    assert "openrouter_upstream" not in audit["configuration"]["app"]


def test_cached_results_are_not_reused_across_a_different_pinned_upstream(
    tmp_path: Path,
) -> None:
    first = OpenRouterShapedProvider()
    _run(tmp_path, upstream="deepinfra", run_id="run-one", provider=first)
    same = OpenRouterShapedProvider()
    _run(tmp_path, upstream="deepinfra", run_id="run-two", provider=same)
    other = OpenRouterShapedProvider()
    _run(tmp_path, upstream="together", run_id="run-three", provider=other)

    assert len(first.requests) == 2
    assert same.requests == []  # same identity: every stage came from the cache
    assert len(other.requests) == 2  # a different upstream answers differently


# --- review: the pin reaches every identity the verifier and the checkpoint use ---------------


def test_the_verifier_cache_and_the_run_descriptor_follow_the_pinned_upstream(
    tmp_path: Path,
) -> None:
    from dataclasses import replace as replace_

    from summarizer.verification import VerificationConfig
    from tests.test_verification_integration import VerificationPipelineProvider

    class Shaped(VerificationPipelineProvider):
        def generate(self, request):
            return replace_(super().generate(request), provider="openrouter")

    def run(upstream: str, run_id: str) -> Shaped:
        provider = Shaped(verification="supported")
        run_pipeline(
            ingest_text("The source confirms the value is 41."),
            provider,
            CharacterCounter(),
            app=AppConfig(
                output_path=tmp_path / f"{run_id}.txt",
                provider="openrouter",
                model=MODEL,
                timeout_seconds=30,
                openrouter_upstream=upstream,
            ),
            strategy=_strategy(),
            config=PipelineConfig(
                target_words=40,
                audit_path=tmp_path / f"{run_id}.json",
                verification=VerificationConfig(enabled=True),
                cache=CacheConfig(enabled=True, root=tmp_path / "cache"),
                reliability=ReliabilityConfig(run_id=run_id),
            ),
        )
        return provider

    def verification_ops(provider: Shaped) -> list[str]:
        return [
            request.operation_id
            for request in provider.requests
            if (request.operation_id or "").startswith("verification-")
        ]

    first = run("deepinfra", "run-one")
    same = run("deepinfra", "run-two")
    other = run("together", "run-three")

    assert verification_ops(first)
    assert same.requests == []
    # A different pinned upstream answers differently, so no verification
    # result may be reused from the other one.
    assert verification_ops(other) == verification_ops(first)
    assert len(other.requests) == len(first.requests)


def test_resuming_a_run_with_a_different_pinned_upstream_is_refused(tmp_path: Path) -> None:
    import pytest

    from summarizer.checkpoint import CheckpointError

    _run(tmp_path, upstream="deepinfra", run_id="resumable", provider=OpenRouterShapedProvider())

    with pytest.raises(CheckpointError):
        run_pipeline(
            ingest_text("A short source for the OpenRouter pipeline."),
            OpenRouterShapedProvider(),
            CharacterCounter(),
            app=AppConfig(
                output_path=tmp_path / "resumed.txt",
                provider="openrouter",
                model=MODEL,
                timeout_seconds=30,
                openrouter_upstream="together",
            ),
            strategy=_strategy(),
            config=PipelineConfig(
                target_words=40,
                audit_path=tmp_path / "resumed.json",
                cache=CacheConfig(enabled=True, root=tmp_path / "cache"),
                reliability=ReliabilityConfig(run_id="resumable", run_mode="resume"),
            ),
        )
