"""Child process entry point for summarization runs."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from summarizer.cli import build_counter, build_provider
from summarizer.config import AppConfig, CacheConfig, ReliabilityConfig, StrategyConfig
from summarizer.ingestion import SourceDocument
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.retrying import RetryingProvider
from summarizer.runtime.observers import RuntimeObserver, StageEvent, StageName
from summarizer.segmentation import SegmentationConfig
from summarizer.verification import VerificationConfig
from summarizer_web.config import load_paths
from summarizer_web.db.connection import init_database
from summarizer_web.services.events_service import record_event


def _emit(run_id: str, event: StageEvent) -> None:
    record_event(
        run_id,
        "stage",
        {
            "stage": event.stage.value,
            "state": event.state,
            "completed": event.completed,
            "total": event.total,
            "detail": event.detail,
        },
    )


def _cancel_flag_path(run_id: str) -> Path:
    return load_paths().runs / run_id / "cancel.flag"


def run_job(run_id: str, *, resume: bool = False) -> None:
    init_database()
    from summarizer_web.db.connection import get_database

    db = get_database()
    row = db.fetchone("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if row is None:
        raise RuntimeError(f"Unknown run {run_id}")
    revision = db.fetchone("SELECT * FROM source_revisions WHERE revision_id = ?", (row["revision_id"],))
    if revision is None:
        raise RuntimeError("Missing revision")
    config = json.loads(row["config_json"])
    text = Path(revision["canonical_path"]).read_text(encoding="utf-8")
    document = SourceDocument(text=text, source_id=revision["source_sha256"])

    run_dir = load_paths().runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root = load_paths().cache
    summary_path = run_dir / "summary.txt"
    audit_path = run_dir / "audit.json"

    from summarizer_web.services.settings_service import get_settings

    settings = get_settings()
    app = AppConfig(
        input_path=run_dir / "source.txt",
        output_path=summary_path,
        model=config["model"],
        provider="ollama",
        ollama_host=settings.ollama_host,
        timeout_seconds=config.get("timeout_seconds", 180.0),
    )
    strategy = StrategyConfig(
        strategy=config.get("strategy", "auto"),
        context_window=config.get("context_window"),
        max_output_tokens=config.get("max_output_tokens", 1024),
        safety_margin_tokens=config.get("safety_margin_tokens", 256),
        safety_margin_fraction=config.get("safety_margin_fraction", 0.02),
    )
    segmentation = None
    if config.get("chunk_tokens") is not None:
        segmentation = SegmentationConfig(
            max_tokens=config["chunk_tokens"],
            overlap_tokens=config.get("overlap_tokens", 0),
        )
    pipeline_config = PipelineConfig(
        target_words=config.get("target_words", 300),
        segmentation=segmentation,
        max_merge_children=config.get("max_merge_children"),
        include_citations=config.get("citations", True),
        audit_path=audit_path,
        verification=VerificationConfig(
            enabled=config.get("verify", True),
            max_repair_passes=config.get("max_repair_passes", 1),
        ),
        cache=CacheConfig(enabled=True, root=cache_root),
        reliability=ReliabilityConfig(
            max_in_flight=config.get("max_concurrency", 1),
            run_mode="resume" if resume else "new",
            run_id=run_id,
        ),
    )
    observer = RuntimeObserver(
        on_stage=lambda event: _emit(run_id, event),
        should_cancel=lambda: _cancel_flag_path(run_id).exists(),
    )
    provider = RetryingProvider(build_provider(app))
    counter = build_counter(app)
    result = run_pipeline(
        document,
        provider,
        counter,
        app=app,
        strategy=strategy,
        config=pipeline_config,
        observer=observer,
    )
    summary_path.write_text(result.final.text, encoding="utf-8")
    from summarizer_web.worker.projections import persist_projections

    persist_projections(run_id, result.root, result.nodes, document.text)
    verification_payload = {"state": "completed" if config.get("verify", True) else "not_run"}
    (run_dir / "verification.json").write_text(json.dumps(verification_payload), encoding="utf-8")


def main() -> None:
    run_id = sys.argv[1]
    resume = "--resume" in sys.argv
    run_job(run_id, resume=resume)


if __name__ == "__main__":
    main()
