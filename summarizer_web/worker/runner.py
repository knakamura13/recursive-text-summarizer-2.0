"""Worker process for one Attempt of a Run.

The supervisor moves an Attempt to running, then spawns `worker_main(run_id,
parent_pid)` in a fresh interpreter. The worker records everything the
pipeline reports (run_events, node projections, run_segments, the selected
strategy, and the progress snapshot) and then how the Attempt ended:
completed, stopped (`stop.flag`), or failed with a classified RunFailure,
which it also writes to `failure.json`.

While alive it holds an exclusive lock on the Run's `worker.lock` (which
names its PID), so startup reconciliation can tell a live worker of this Run
from an unrelated process that reused the PID. It leaves with `os._exit` so
in-flight model calls abandoned by a Stop cannot keep it alive, and it exits
on its own when its parent process dies.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

from summarizer.budget import BudgetError
from summarizer.cli import build_counter, build_provider
from summarizer.config import AppConfig, CacheConfig, ReliabilityConfig, RetryPolicy, StrategyConfig
from summarizer.finalization import FinalizationVerificationError
from summarizer.ingestion import SourceDocument
from summarizer.pipeline import PipelineConfig, PipelineResult, run_pipeline
from summarizer.providers.base import (
    ContextWindowProvider,
    ProviderConnectionError,
    ProviderError,
    ProviderRetriesExhaustedError,
    ProviderTimeoutError,
    RetryErrorCategory,
)
from summarizer.providers.retrying import RetryingProvider
from summarizer.runtime.observers import (
    ItemEvent,
    ItemFailedError,
    PipelineStopped,
    RuntimeObserver,
    SegmentInfo,
    StageEvent,
    StageName,
)
from summarizer.segmentation import SegmentationConfig, SegmentationError
from summarizer.verification import VerificationCapacityError, VerificationConfig
from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database, init_database
from summarizer_web.models.api import RunConfig, RunFailure
from summarizer_web.services.events_service import (
    ACTIVE_ATTEMPT_STATES,
    checkpoint_manifest_path,
    checkpoint_run_id,
    end_attempt,
    failure_path,
    record_event,
    stop_flag_path,
    worker_lock_path,
)
from summarizer_web.services.progress_service import ProgressTracker, now_iso
from summarizer_web.services.settings_service import get_settings
from summarizer_web.worker.projections import (
    TREE_KINDS,
    NodeLabeler,
    PageMap,
    apply_item_event,
    label_root,
    parse_node_id,
    reconcile_final_tree,
    replace_segments,
)

EXIT_ORPHANED = 3
PROGRESS_INTERVAL_SECONDS = 0.5
MAX_FAILURE_DETAIL_BYTES = 4096
# One first call plus the two re-asks of D12.
ITEM_TRIES = 3


# --- Configuration mapping (shared with preflight) -------------------------------------


def strategy_config(config: RunConfig) -> StrategyConfig:
    """Strategy inputs of a Run before its context window is resolved with Ollama."""
    return StrategyConfig(
        strategy=config.strategy,
        context_window=config.context_window,
        max_output_tokens=config.max_output_tokens,
        safety_margin_tokens=config.safety_margin_tokens,
        safety_margin_fraction=config.safety_margin_fraction,
    )


def segmentation_config(config: RunConfig) -> SegmentationConfig | None:
    """Explicit segment sizing, or None to let the pipeline size segments from capacity."""
    if config.chunk_tokens is None:
        return None
    return SegmentationConfig(max_tokens=config.chunk_tokens, overlap_tokens=config.overlap_tokens)


# --- Failure classification ------------------------------------------------------------


def _chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return chain


def _truncate_utf8(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[: limit - 3].decode("utf-8", errors="ignore") + "..."


def error_detail(error: BaseException) -> str:
    """The error and its causes, one per line, bounded to 4 KB."""
    lines = [
        "".join(traceback.format_exception_only(type(item), item)).strip() for item in _chain(error)
    ]
    return _truncate_utf8("\nCaused by: ".join(lines), MAX_FAILURE_DETAIL_BYTES)


def _last_retry_category(error: BaseException) -> RetryErrorCategory | None:
    if isinstance(error, ProviderRetriesExhaustedError) and error.retry_attempts:
        return error.retry_attempts[-1].error_category
    return None


def _is_unreachable(error: BaseException) -> bool:
    if isinstance(error, (ProviderConnectionError, ConnectionRefusedError)):
        return True
    return _last_retry_category(error) is RetryErrorCategory.CONNECTION or type(error).__name__ in {
        "ConnectError",
        "ConnectTimeout",
    }


def _is_model_missing(error: BaseException) -> bool:
    if getattr(error, "status_code", None) == 404:
        return True
    return isinstance(error, ProviderError) and "not found" in str(error).lower()


def _is_timeout(error: BaseException) -> bool:
    return isinstance(error, ProviderTimeoutError) or (
        _last_retry_category(error) is RetryErrorCategory.TIMEOUT
    )


def classify_failure(
    error: BaseException,
    *,
    host: str,
    model: str,
    timeout_seconds: float,
    stage: str | None,
    item: str | None,
) -> RunFailure:
    """Map an Attempt's exception to a RunFailure the UI can explain (contract 6b)."""
    detail = error_detail(error)
    if isinstance(error, ItemFailedError):
        label = item or error.work_id
        return RunFailure(
            code="item_invalid_output",
            message=f"{label} produced invalid output after {ITEM_TRIES} tries: {error}",
            stage=error.stage.value,
            item=label,
            detail=detail,
            hint="Resume to retry this item, or try another model.",
        )
    chain = _chain(error)

    def failure(code: str, message: str, hint: str) -> RunFailure:
        return RunFailure(
            code=code, message=message, stage=stage, item=item, detail=detail, hint=hint
        )

    if any(_is_unreachable(cause) for cause in chain):
        return failure(
            "ollama_unreachable",
            f"Ollama is not reachable at {host}.",
            "Start Ollama with `ollama serve` or fix the host in Settings, then Resume.",
        )
    if any(_is_model_missing(cause) for cause in chain):
        return failure(
            "model_not_found",
            f"Model {model} is not installed.",
            f"Run `ollama pull {model}` or choose another model.",
        )
    if any(_is_timeout(cause) for cause in chain):
        return failure(
            "provider_timeout",
            f"The model did not answer within {timeout_seconds:g} s.",
            "Resume, or raise the timeout for a new Run.",
        )
    if isinstance(error, ProviderError):
        return failure("provider_error", str(error), "Resume to retry.")
    if isinstance(error, (BudgetError, SegmentationError, VerificationCapacityError)):
        return failure(
            "budget_error",
            str(error),
            "Lower max output tokens or set a larger context window.",
        )
    if isinstance(error, FinalizationVerificationError):
        return failure(
            "verification_failed",
            "Verification could not confirm any sentence of the summary.",
            "Start a new Run with another model, or turn verification off.",
        )
    return failure("internal_error", f"{type(error).__name__}: {error}", "Check the server log.")


# --- Recording observer callbacks ------------------------------------------------------


def _item_payload(event: ItemEvent, label: str) -> dict[str, object]:
    return {
        "kind": event.kind,
        "work_id": event.work_id,
        "state": event.state,
        "stage": event.stage.value,
        "label": label,
        "level": event.level,
        "order": event.order,
        "total": event.total,
        "child_ids": list(event.child_ids),
        "covered_segment_ids": list(event.covered_segment_ids),
        "attempt": event.attempt,
        "message": event.message,
    }


class RunRecorder:
    """Persists one Attempt's observer callbacks; safe to call from pipeline threads.

    Each callback is one transaction: the run_events row, then the rows it
    changes (node projections carry the event id). The progress snapshot is
    written with it on stage changes and at most every 0.5 s otherwise; a
    background flush writes the last throttled change.
    """

    def __init__(
        self,
        *,
        run_id: str,
        attempt_id: str,
        tracker: ProgressTracker,
        labeler: NodeLabeler,
    ) -> None:
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.tracker = tracker
        self.labeler = labeler
        self._lock = threading.Lock()
        self._labels: dict[str, str] = {}
        self._cursor = 0
        self._dirty = False
        self._last_progress_write = float("-inf")
        self._flusher_stop = threading.Event()

    # Observer callbacks: they must never raise into the pipeline.

    def on_stage(self, event: StageEvent) -> None:
        self._guarded(self._record_stage, event)

    def on_segments(self, segments: tuple[SegmentInfo, ...]) -> None:
        self._guarded(self._record_segments, segments)

    def on_item(self, event: ItemEvent) -> None:
        self._guarded(self._record_item, event)

    def _guarded(self, record, argument) -> None:
        try:
            with self._lock:
                record(argument)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    def _record_stage(self, event: StageEvent) -> None:
        changed = self.tracker.on_stage(event)
        with get_database().transaction() as connection:
            event_id = record_event(
                connection,
                self.run_id,
                "stage",
                {
                    "stage": event.stage.value,
                    "state": event.state,
                    "completed": event.completed,
                    "total": event.total,
                    "detail": event.detail,
                },
            )
            if event.stage is StageName.PREPARING and event.state == "completed":
                if event.detail in ("direct", "hierarchical"):
                    connection.execute(
                        "UPDATE runs SET selected_strategy = ?, updated_at = ? WHERE run_id = ?",
                        (event.detail, now_iso(), self.run_id),
                    )
            if event.stage is StageName.MERGING and event.state == "completed":
                label_root(connection, self.run_id, record_event)
            self._progress(connection, event_id, force=changed)

    def _record_segments(self, segments: tuple[SegmentInfo, ...]) -> None:
        self.tracker.on_segments(segments)
        pages = self.labeler.set_segments(segments)
        with get_database().transaction() as connection:
            event_id = record_event(
                connection, self.run_id, "segments", {"count": len(segments)}
            )
            replace_segments(connection, self.run_id, segments, pages)
            self._progress(connection, event_id)

    def _record_item(self, event: ItemEvent) -> None:
        label = self.labeler.label_event(event)
        self._labels[event.work_id] = label
        self.tracker.on_item(event, label)
        with get_database().transaction() as connection:
            event_id = record_event(connection, self.run_id, "item", _item_payload(event, label))
            apply_item_event(connection, self.run_id, event_id, event, label, now_iso())
            self._progress(connection, event_id)

    def _progress(self, connection, event_id: int, *, force: bool = False) -> None:
        self._cursor = event_id
        now = time.monotonic()
        if not force and now - self._last_progress_write < PROGRESS_INTERVAL_SECONDS:
            self._dirty = True
            return
        connection.execute(
            "UPDATE runs SET progress_json = ? WHERE run_id = ?",
            (self.tracker.snapshot(event_id).model_dump_json(), self.run_id),
        )
        self._last_progress_write = now
        self._dirty = False

    def flush(self) -> None:
        """Write a throttled progress change that no later callback carried."""
        try:
            with self._lock:
                if not self._dirty:
                    return
                with get_database().transaction() as connection:
                    self._progress(connection, self._cursor, force=True)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    def start_flusher(self) -> threading.Thread:
        def flush_until_stopped() -> None:
            while not self._flusher_stop.wait(PROGRESS_INTERVAL_SECONDS):
                self.flush()

        thread = threading.Thread(target=flush_until_stopped, name="progress-flush", daemon=True)
        thread.start()
        return thread

    def stop_flusher(self) -> None:
        self._flusher_stop.set()

    # Failure context and the end of the Attempt.

    def item_label(self, error: ItemFailedError) -> str:
        known = self._labels.get(error.work_id)
        if known is not None:
            return known
        if error.kind in TREE_KINDS:
            level, order = parse_node_id(error.work_id) or (0, 0)
            return self.labeler.label(error.kind, level, order, error.covered_segment_ids)
        return "Final summary draft" if error.kind == "editorial" else f"Claim {error.work_id}"

    def finish(
        self,
        state: str,
        failure: RunFailure | None = None,
        result: PipelineResult | None = None,
    ) -> bool:
        """Record how the Attempt ended, with the final projections and progress."""
        with self._lock:
            self.tracker.clear_current()
            with get_database().transaction() as connection:
                if result is not None:
                    attempt = connection.execute(
                        "SELECT state FROM run_attempts WHERE attempt_id = ? AND run_id = ?",
                        (self.attempt_id, self.run_id),
                    ).fetchone()
                    if attempt is None or attempt["state"] not in ACTIVE_ATTEMPT_STATES:
                        return False
                    reconcile_final_tree(
                        connection,
                        self.run_id,
                        record_event,
                        result.root,
                        result.nodes,
                        self.labeler,
                        now_iso(),
                    )
                event_id = end_attempt(
                    connection,
                    run_id=self.run_id,
                    attempt_id=self.attempt_id,
                    state=state,
                    failure=failure,
                )
                if event_id is None:
                    return False
                connection.execute(
                    "UPDATE runs SET progress_json = ? WHERE run_id = ?",
                    (self.tracker.snapshot(event_id).model_dump_json(), self.run_id),
                )
            self._dirty = False
            return True


# --- The Attempt ------------------------------------------------------------------------


def _write_failure_file(run_id: str, failure: RunFailure) -> None:
    path = failure_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(failure.model_dump_json(), encoding="utf-8")
    os.replace(temporary, path)


def _execute(
    run_id: str,
    run_dir: Path,
    config: RunConfig,
    revision,
    host: str,
    observer: RuntimeObserver,
) -> PipelineResult:
    text = Path(revision["canonical_path"]).read_text(encoding="utf-8")
    document = SourceDocument(text=text, source_id=revision["source_sha256"])
    app = AppConfig(
        input_path=run_dir / "source.txt",
        output_path=run_dir / "summary.txt",
        model=config.model,
        provider="ollama",
        ollama_host=host,
        timeout_seconds=config.timeout_seconds,
    )
    strategy = strategy_config(config)
    raw_provider = build_provider(app)
    if isinstance(raw_provider, ContextWindowProvider):
        window = raw_provider.configure_context_window(
            app.model, strategy.context_window, timeout_seconds=app.timeout_seconds
        )
        if window is not None:
            strategy = replace(strategy, context_window=window)
    provider = RetryingProvider(raw_provider, RetryPolicy(max_attempts=config.max_retries))
    # Resume reuses the checkpoint of earlier Attempts; an Attempt that failed
    # before the pipeline opened its checkpoint leaves none to resume.
    resume = checkpoint_manifest_path(run_id).exists()
    pipeline_config = PipelineConfig(
        target_words=config.target_words,
        segmentation=segmentation_config(config),
        max_merge_children=config.max_merge_children,
        include_citations=config.citations,
        audit_path=run_dir / "audit.json",
        verification=VerificationConfig(
            enabled=config.verify,
            max_repair_passes=config.max_repair_passes,
            strict_numbers=config.strict_numbers,
            strict_names=config.strict_names,
        ),
        cache=CacheConfig(enabled=True, root=load_paths().cache),
        reliability=ReliabilityConfig(
            max_in_flight=config.max_concurrency,
            run_mode="resume" if resume else "new",
            run_id=checkpoint_run_id(run_id),
        ),
    )
    # With the cache enabled the pipeline publishes summary.txt and audit.json
    # together, witnessed by the checkpoint manifest.
    return run_pipeline(
        document,
        provider,
        build_counter(app),
        app=app,
        strategy=strategy,
        config=pipeline_config,
        observer=observer,
    )


def run_job(run_id: str) -> str | None:
    """Execute the running Attempt of a Run; return the state it ended in.

    Returns None when the Run has no running Attempt (it was stopped or
    deleted before the worker started).
    """
    db = init_database(load_paths())
    run = db.fetchone("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    attempt = db.fetchone(
        "SELECT * FROM run_attempts WHERE run_id = ? ORDER BY attempt_number DESC LIMIT 1",
        (run_id,),
    )
    if run is None or attempt is None or attempt["state"] not in ("running", "stopping"):
        return None
    config = RunConfig.model_construct(**json.loads(run["config_json"]))
    revision = db.fetchone(
        "SELECT * FROM source_revisions WHERE revision_id = ?", (run["revision_id"],)
    )
    run_dir = load_paths().runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = ProgressTracker(
        attempt_number=attempt["attempt_number"],
        started_at=attempt["started_at"] or now_iso(),
        verify=bool(config.verify),
        target_words=config.target_words,
        max_merge_children=config.max_merge_children,
    )
    recorder = RunRecorder(
        run_id=run_id,
        attempt_id=attempt["attempt_id"],
        tracker=tracker,
        labeler=NodeLabeler(PageMap.from_json(revision["page_map_json"] if revision else None)),
    )
    stop_flag = stop_flag_path(run_id)
    observer = RuntimeObserver(
        on_stage=recorder.on_stage,
        on_segments=recorder.on_segments,
        on_item=recorder.on_item,
        should_stop=stop_flag.exists,
    )
    host = ""
    recorder.start_flusher()
    try:
        # Provider setup belongs to preparing, so failures there name it.
        recorder.on_stage(StageEvent(StageName.PREPARING, "active"))
        host = get_settings().ollama_host
        observer.raise_if_stopped("before preparing")
        if revision is None:
            raise RuntimeError("the Document's source revision is missing")
        result = _execute(run_id, run_dir, config, revision, host, observer)
    except PipelineStopped:
        recorder.finish("stopped")
        return "stopped"
    except Exception as error:
        if stop_flag.exists():
            # A Stop can surface as an error from an abandoned call.
            recorder.finish("stopped")
            return "stopped"
        traceback.print_exc(file=sys.stderr)
        item = recorder.item_label(error) if isinstance(error, ItemFailedError) else tracker.current_label()
        failure = classify_failure(
            error,
            host=host,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            stage=tracker.stage,
            item=item,
        )
        _write_failure_file(run_id, failure)
        recorder.finish("failed", failure)
        return "failed"
    finally:
        recorder.stop_flusher()
    recorder.finish("completed", result=result)
    return "completed"


def watch_parent(parent_pid: int, *, interval_seconds: float = 1.0) -> threading.Thread:
    """Exit this process once its parent is gone (the worker gets re-parented)."""

    def watch() -> None:
        while True:
            if os.getppid() != parent_pid:
                os._exit(EXIT_ORPHANED)
            time.sleep(interval_seconds)

    thread = threading.Thread(target=watch, name="parent-watch", daemon=True)
    thread.start()
    return thread


def hold_worker_lock(run_id: str) -> int:
    """Lock the Run's worker.lock for this process's lifetime and record the PID in it.

    The kernel releases the lock when the process dies, however it dies.
    """
    path = worker_lock_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError(f"another worker is already running Run {run_id}") from None
    os.ftruncate(descriptor, 0)
    os.write(descriptor, str(os.getpid()).encode())
    return descriptor


def worker_main(run_id: str, parent_pid: int) -> None:
    """Entry point of the spawned worker: run the Attempt, then exit the process."""
    # A new session keeps terminal signals (Ctrl-C) away from the worker; the
    # application's shutdown settles the Run instead.
    os.setsid()
    watch_parent(parent_pid)
    code = 1
    try:
        hold_worker_lock(run_id)
        code = 1 if run_job(run_id) == "failed" else 0
    except BaseException:
        traceback.print_exc(file=sys.stderr)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # Abandoned in-flight model calls run on non-daemon threads; exiting
        # here aborts them instead of waiting for their timeouts.
        os._exit(code)
