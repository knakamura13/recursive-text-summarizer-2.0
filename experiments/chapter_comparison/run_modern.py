"""Run one pinned modern summarization case for chapter-comparison studies.

The parent process only orchestrates local files. Model traffic goes exclusively
through the caller-supplied capture proxy.

``--version worktree`` runs this repository's checkout instead of a pinned
one, with the pipeline's own default request allowances. ``--replay-from``
points at a saved trial directory: its cached segmentation and leaf summaries
are fed to the merge stage unchanged, so no leaf is regenerated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKOUTS = {
    "pre_experiments": (Path("/private/tmp/rts-study-20260926/checkouts/pre_experiments"), "c41febc"),
    "current": (Path("/private/tmp/rts-study-20260926/checkouts/current"), "9d31e4e"),
    # Revision is read at run time; the working tree may be ahead of any pin.
    "worktree": (_REPO_ROOT, None),
}
_MODEL = "gemma4:latest"
_MODEL_DIGEST = "c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb"
_CONTEXT_WINDOW = 32_768


def _output_allowance(target_words: int) -> int:
    return max(2_048, math.ceil(3 * target_words) + 1_024)


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=checkout, text=True).strip()


def run_case(
    version: str,
    source: str | Path,
    target: int,
    seed: int,
    output_dir: str | Path,
    proxy_url: str,
    replay_from: str | Path | None = None,
    strategy: str = "auto",
    context_window: int = _CONTEXT_WINDOW,
) -> dict[str, Any]:
    """Execute one case; all artifacts are written beneath ``output_dir``.

    ``version`` selects the pinned checkout (``pre_experiments`` or ``current``).
    A unique trial directory is created for each call, so its cache is always
    independent, even if the caller reuses an output root.
    """
    if version not in _CHECKOUTS:
        raise ValueError("version must be one of: " + ", ".join(_CHECKOUTS))
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        raise ValueError("target must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not proxy_url.strip():
        raise ValueError("proxy_url must not be empty")
    source_path = Path(source).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"source is not a file: {source_path}")
    checkout, revision = _CHECKOUTS[version]
    if not checkout.is_dir():
        raise FileNotFoundError(f"pinned checkout does not exist: {checkout}")
    worktree_dirty = None
    if revision is None:
        revision = _git(checkout, "rev-parse", "HEAD")
        # Only the summarizer package decides behavior; experiment edits do not.
        worktree_dirty = bool(_git(checkout, "status", "--porcelain=v1", "--", "summarizer"))
    replay_path = None
    if replay_from is not None:
        replay_path = Path(replay_from).expanduser().resolve(strict=True)
        if not any(replay_path.glob("cache/objects/*/*.json")):
            raise ValueError(f"replay directory has no saved cache objects: {replay_path}")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    trial = root / f"{version}-target{target}-seed{seed}-{uuid.uuid4().hex[:12]}"
    trial.mkdir()
    request = {
        "version": version,
        "revision": revision,
        "worktree_dirty": worktree_dirty,
        "checkout": str(checkout),
        "source": str(source_path),
        "target_words": target,
        "seed": seed,
        "proxy_url": proxy_url,
        "model": _MODEL,
        "model_digest": _MODEL_DIGEST,
        "context_window": context_window,
        # Pinned revisions reserve the protocol allowance on every request. The
        # working tree budgets each stage itself, so it keeps its defaults.
        "max_output_tokens": None if version == "worktree" else _output_allowance(target),
        "replay_from": None if replay_path is None else str(replay_path),
        "strategy": strategy,
        "trial_dir": str(trial),
    }
    config_path = trial / "worker_config.json"
    config_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--_worker", str(config_path)],
        cwd=trial,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(checkout)},
    )
    elapsed = time.monotonic() - started
    (trial / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (trial / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    metadata_path = trial / "run.json"
    if metadata_path.exists():
        result = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        result = {**request, "status": "failed", "failure": "worker exited without run metadata"}
    result["elapsed_seconds"] = elapsed
    result["exit_code"] = completed.returncode
    result["stdout_log"] = str(trial / "stdout.log")
    result["stderr_log"] = str(trial / "stderr.log")
    metadata_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _jsonable(asdict(value))
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _saved_merge_inputs(replay_dir: Path, document: Any) -> tuple[list[Any], tuple[Any, ...]]:
    """Load a saved trial's segments and leaves, revalidated against ``document``.

    Segments must be the document's own text at their recorded offsets, and
    every leaf must pass the current leaf provenance check for its segment.
    """
    from summarizer.leaf import core_text, validate_provenance
    from summarizer.segmentation import BoundaryKind, SourceSegment
    from summarizer.summaries import SummaryNode

    envelopes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(replay_dir.glob("cache/objects/*/*.json"))
    ]
    segmentations = [item["payload"] for item in envelopes if item["descriptor"]["stage"] == "segmentation"]
    if len(segmentations) != 1:
        raise ValueError(f"expected one saved segmentation, found {len(segmentations)}")
    segments = [
        SourceSegment(**{**item, "boundary_kind": BoundaryKind(item["boundary_kind"])})
        for item in segmentations[0]
    ]
    for segment in segments:
        if segment.source_id != document.source_id:
            raise ValueError(f"{segment.segment_id} belongs to another source")
        if document.text[segment.context_start : segment.context_end] != segment.text:
            raise ValueError(f"{segment.segment_id} no longer matches the source text")
    by_id = {segment.segment_id: segment for segment in segments}
    saved = {
        item["descriptor"]["work_id"]: item["payload"]
        for item in envelopes
        if item["descriptor"]["stage"] == "leaf"
    }
    if set(saved) != set(by_id):
        raise ValueError(
            f"saved leaves cover {len(saved)} of {len(by_id)} segments; replay needs all of them"
        )
    leaves = []
    for segment in sorted(segments, key=lambda item: item.order):
        node = SummaryNode.model_validate(saved[segment.segment_id])
        validate_provenance(
            node, legal={segment.segment_id: core_text(segment)}, subject=segment.segment_id
        )
        leaves.append(node)
    return segments, tuple(leaves)


def _worker(config_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trial = Path(config["trial_dir"])
    checkout = Path(config["checkout"])
    # This worker is a new interpreter per trial; the pinned checkout therefore
    # owns every imported summarizer module, rather than this working tree.
    sys.path.insert(0, str(checkout))
    event_path = trial / "stages.jsonl"
    event_stream = event_path.open("w", encoding="utf-8", buffering=1)
    metadata: dict[str, Any] = {**config, "status": "running", "stages_log": str(event_path)}

    def record(kind: str, event: Any) -> None:
        value = _jsonable(event)
        if kind == "stage":
            metadata["terminal_stage"] = {"stage": value.get("stage"), "state": value.get("state")}
        event_stream.write(json.dumps({"kind": kind, "event": value}, default=str) + "\n")

    (trial / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        from summarizer.budget import select_strategy
        from summarizer.config import AppConfig, CacheConfig, ReliabilityConfig, RetryPolicy, StrategyConfig
        from summarizer.ingestion import read_source
        from summarizer.pipeline import PipelineConfig, run_pipeline
        from summarizer.providers.ollama import OllamaProvider
        from summarizer.providers.retrying import RetryingProvider
        from summarizer.runtime.observers import RuntimeObserver
        from summarizer.tokenization import resolve_token_counter
        from summarizer.verification import VerificationConfig

        source = Path(config["source"])
        output_path = trial / "summary.txt"
        audit_path = trial / "audit.json"
        cache_path = trial / "cache"
        app = AppConfig(
            input_path=source,
            output_path=output_path,
            model=config["model"],
            provider="ollama",
            ollama_host=config["proxy_url"],
            timeout_seconds=600,
        )
        strategy = StrategyConfig(
            strategy=config.get("strategy", "auto"),
            context_window=config["context_window"],
            safety_margin_tokens=256,
            safety_margin_fraction=0.02,
            **(
                {}
                if config["max_output_tokens"] is None
                else {"max_output_tokens": config["max_output_tokens"]}
            ),
        )
        metadata["max_output_tokens"] = strategy.max_output_tokens
        pipeline = PipelineConfig(
            target_words=config["target_words"],
            include_citations=True,
            audit_path=audit_path,
            verification=VerificationConfig(enabled=True, max_repair_passes=1),
            cache=CacheConfig(enabled=True, root=cache_path),
            reliability=ReliabilityConfig(
                max_in_flight=1,
                run_mode="new",
                run_id=f"{config['version'].replace('_', '-')}-{config['seed']}-{uuid.uuid4().hex}",
            ),
        )
        source_document = read_source(source)
        if config.get("replay_from"):
            import summarizer.pipeline as pipeline_module

            saved_segments, saved_leaves = _saved_merge_inputs(
                Path(config["replay_from"]), source_document
            )
            # Only the leaf inputs are substituted; merge onwards is unchanged.
            pipeline_module.cached_segment_document = lambda *_args, **_kwargs: saved_segments
            pipeline_module.summarize_segments = lambda *_args, **_kwargs: saved_leaves
            metadata["replayed_leaves"] = len(saved_leaves)
        counter = resolve_token_counter(provider="ollama", model=config["model"])
        provider = OllamaProvider(host=config["proxy_url"])
        selected_context = provider.configure_context_window(
            config["model"], strategy.context_window, timeout_seconds=app.timeout_seconds
        )
        from dataclasses import replace

        strategy = replace(strategy, context_window=selected_context)
        selected = select_strategy(
            source_document, counter, provider=app.provider, model=app.model, config=strategy
        )
        metadata.update({
            "selected_strategy": selected.strategy,
            "strategy_report": _jsonable(selected),
            "counter_identity": counter.identity,
            "counter_exact": counter.exact,
            "effective_context_window": selected_context,
            "verification_enabled": pipeline.verification.enabled,
            "verification_max_repair_passes": pipeline.verification.max_repair_passes,
            "audit_path": str(audit_path),
            "cache_path": str(cache_path),
            "summary_path": str(output_path),
        })
        (trial / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        observer = RuntimeObserver(
            on_stage=lambda event: record("stage", event),
            on_segments=lambda segments: record("segments", segments),
            on_item=lambda event: record("item", event),
        )
        result = run_pipeline(
            source_document,
            RetryingProvider(provider, RetryPolicy(max_attempts=5)),
            counter,
            app=app,
            strategy=strategy,
            config=pipeline,
            observer=observer,
        )
        metadata.update({
            "status": "succeeded",
            "selected_strategy": result.strategy.strategy,
            "strategy_report": _jsonable(result.strategy),
            "word_count": len(result.final.text.split()),
            "verification_enabled": pipeline.verification.enabled,
            "audit_path": str(audit_path),
            "cache_path": str(cache_path),
            "summary_path": str(output_path),
        })
        return_code = 0
    except Exception as error:
        failure = getattr(error, "failure", None)
        budget = getattr(error, "budget", None)
        metadata.update({
            "status": "failed",
            "failure": f"{type(error).__name__}: {error}",
            "failure_class": None if failure is None else failure.value,
            "failure_stage": None if budget is None else getattr(budget.stage, "value", str(budget.stage)),
            "traceback": traceback.format_exc(),
            "audit_path": str(trial / "audit.json"),
            "cache_path": str(trial / "cache"),
        })
        print(f"Modern experiment failed: {type(error).__name__}: {error}", file=sys.stderr)
        return_code = 1
    finally:
        event_stream.close()
    (trial / "run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return return_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one pinned modern chapter-comparison case")
    parser.add_argument("--version", choices=tuple(_CHECKOUTS))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--proxy-url")
    parser.add_argument("--replay-from", type=Path, help="saved trial directory whose segments and leaves feed the merge stage")
    parser.add_argument("--strategy", choices=("auto", "direct", "hierarchical"), default="auto")
    parser.add_argument("--context-window", type=int, default=_CONTEXT_WINDOW, help="must match the proxy's --num-ctx")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._worker is not None:
        return _worker(args._worker)
    missing = [name for name in ("version", "source", "target", "seed", "output_dir", "proxy_url") if getattr(args, name) is None]
    if missing:
        _parser().error("required arguments missing: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    try:
        result = run_case(
            args.version, args.source, args.target, args.seed, args.output_dir, args.proxy_url,
            replay_from=args.replay_from, strategy=args.strategy, context_window=args.context_window,
        )
    except Exception as error:
        print(f"Modern experiment failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
