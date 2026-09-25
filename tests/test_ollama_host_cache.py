"""Regression test for issue #64: Cache key omits ollama_host, so a run can silently reuse another host's output."""

import json
from pathlib import Path

from summarizer.config import AppConfig, CacheConfig, ReliabilityConfig, StrategyConfig
from summarizer.direct import DOCUMENT_SEGMENT_ID
from summarizer.ingestion import ingest_text
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import ModelProvider, GenerationRequest, GenerationResult
from summarizer.tokenization import resolve_token_counter
from tests.support.compression_provider import compression_generation_payload


def payload(**overrides: object) -> str:
    body: dict[str, object] = {
        "summary": "The source text to summarize.",
        "content_units": [],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": [DOCUMENT_SEGMENT_ID],
        "level": 0,
    }
    body.update(overrides)
    return json.dumps(body)


class HostSensitiveProvider(ModelProvider):
    def __init__(self, host: str) -> None:
        self._host = host
        self.requests = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if request.operation_id == "editorial-final":
            # Return proper FinalDraft format
            return GenerationResult(
                text=json.dumps({"text": f"ANSWER-FROM-{self._host}"}),
                provider="fake",
                model=request.model,
                input_tokens=10,
                output_tokens=10,
                finish_status="stop",
            )
        if (request.operation_id or "").startswith("compression:"):
            return GenerationResult(
                text=json.dumps(compression_generation_payload(request)),
                provider="fake",
                model=request.model,
                input_tokens=10,
                output_tokens=10,
                finish_status="stop",
            )
        # Return leaf summary format
        host_text = payload(summary=f"ANSWER-FROM-{self._host}")
        return GenerationResult(
            text=host_text,
            provider="fake",
            model=request.model,
            input_tokens=10,
            output_tokens=10,
            finish_status="stop",
        )


def test_cross_host_cache_collision_is_prevented(tmp_path: Path) -> None:
    """Two runs sharing a cache dir with different --ollama-host must not share cache."""
    cache_dir = tmp_path / ".summarizer-cache"
    source = ingest_text("The source text to summarize.")
    counter = resolve_token_counter(provider="ollama", model="llama3.2:3b")
    strategy = StrategyConfig(
        strategy="direct", context_window=100_000, max_output_tokens=1,
        safety_margin_tokens=0, safety_margin_fraction=0
    )

    # First run: host A
    host_a = "http://gpu-box-a:11434"
    provider_a = HostSensitiveProvider(host_a)
    app_a = AppConfig(
        provider="ollama",
        model="llama3.2:3b",
        ollama_host=host_a,
        timeout_seconds=30,
    )
    config_a = PipelineConfig(
        cache=CacheConfig(enabled=True, root=cache_dir),
        reliability=ReliabilityConfig(run_mode="new", run_id="host-a-test"),
        target_words=40,
    )

    result_a = run_pipeline(source, provider_a, counter, app=app_a, strategy=strategy, config=config_a)

    assert len(provider_a.requests) == 2, "First run should call provider twice (leaf + editorial)"
    assert "ANSWER-FROM-" + host_a in result_a.final.text

    # Second run: host B (different host, same cache dir, same model)
    host_b = "http://gpu-box-b:11434"
    provider_b = HostSensitiveProvider(host_b)
    app_b = AppConfig(
        provider="ollama",
        model="llama3.2:3b",
        ollama_host=host_b,
        timeout_seconds=30,
    )
    config_b = PipelineConfig(
        cache=CacheConfig(enabled=True, root=cache_dir),
        reliability=ReliabilityConfig(run_mode="new", run_id="host-b-test"),
        target_words=40,
    )

    result_b = run_pipeline(source, provider_b, counter, app=app_b, strategy=strategy, config=config_b)

    # Second run's provider MUST be called (cache miss due to host difference)
    assert len(provider_b.requests) == 2, "Second run should call provider (cache miss due to different host)"
    assert "ANSWER-FROM-" + host_b in result_b.final.text

    # First run's provider should still have only 2 calls
    assert len(provider_a.requests) == 2, "First run's provider should not be called again"