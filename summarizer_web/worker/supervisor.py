"""Single-Run worker supervisor.

One Run executes at a time, each Attempt in its own worker process, created
with the multiprocessing "spawn" context (see `summarizer_web.worker.runner`).
The supervisor thread starts queued Attempts, watches the worker, and settles
an Attempt the worker could not finish itself. Stop never waits for the worker:
`stop_grace_seconds` after the stop flag the worker gets SIGTERM, then SIGKILL
after `kill_grace_seconds`; per-item checkpoints make a kill at any point safe.
A worker that died without recording its end fails with its `failure.json` or
`worker_crashed`. Shutdown terminates the worker and marks the Run interrupted,
so it can be resumed later.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from time import monotonic

from pydantic import ValidationError

from summarizer_web.db.connection import get_database
from summarizer_web.models.api import RunFailure
from summarizer_web.services.events_service import (
    application_stopped_failure,
    end_attempt,
    failure_path,
    latest_attempt,
    request_stop_flag,
    set_worker_pid,
    start_attempt,
    stop_flag_path,
    stored_stage,
    worker_crashed_failure,
)
from summarizer_web.worker.reconcile import reconcile_on_startup

logger = logging.getLogger(__name__)

_SPAWN = multiprocessing.get_context("spawn")
_supervisor: RunSupervisor | None = None


def run_worker(run_id: str, parent_pid: int) -> None:
    """Target of the spawned worker; the pipeline is imported only in the child."""
    from summarizer_web.worker.runner import worker_main

    worker_main(run_id, parent_pid)


@dataclass
class _Worker:
    run_id: str
    attempt_id: str
    process: BaseProcess
    stop_requested_at: float | None = None
    terminated_at: float | None = None
    killed: bool = False


class RunSupervisor:
    def __init__(
        self,
        *,
        stop_grace_seconds: float = 2.0,
        kill_grace_seconds: float = 2.0,
        poll_seconds: float = 0.2,
        worker_target: Callable[[str, int], None] = run_worker,
    ) -> None:
        self.stop_grace_seconds = stop_grace_seconds
        self.kill_grace_seconds = kill_grace_seconds
        self.poll_seconds = poll_seconds
        # A module-level function: the spawn context pickles it by name.
        self.worker_target = worker_target
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._wake = threading.Event()
        self._queue: deque[str] = deque()
        self._worker: _Worker | None = None
        self._thread: threading.Thread | None = None
        self._closing = False

    def start(self) -> None:
        """Start the supervisor thread once, settling Runs a previous process left active."""
        with self._start_lock:
            with self._lock:
                if self._thread is not None and self._thread.is_alive():
                    return
                self._closing = False
                pending = set(self._queue)
            reconcile_on_startup(exclude_run_ids=pending)
            thread = threading.Thread(target=self._loop, name="run-supervisor", daemon=True)
            with self._lock:
                self._thread = thread
            thread.start()

    def enqueue(self, run_id: str) -> None:
        with self._lock:
            if run_id not in self._queue:
                self._queue.append(run_id)
        self.start()
        self._wake.set()

    def discard(self, run_id: str) -> None:
        """Forget a queued Run that was stopped before its worker started."""
        with self._lock:
            if run_id in self._queue:
                self._queue.remove(run_id)

    def request_stop(self, run_id: str) -> None:
        """Signal the worker; never waits. The supervisor thread escalates if it must."""
        request_stop_flag(run_id)
        self._wake.set()

    def shutdown(self) -> None:
        """Terminate the worker and mark every active Run interrupted (a Stop in flight: stopped)."""
        with self._lock:
            self._closing = True
            thread = self._thread
        self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)
        with self._lock:
            worker, self._worker = self._worker, None
            queued = list(self._queue)
            self._queue.clear()
            self._thread = None
        if worker is not None:
            self._terminate(worker.process)
            if worker.stop_requested_at is not None or stop_flag_path(worker.run_id).exists():
                self._settle(worker.run_id, worker.attempt_id, "stopped", None)
            else:
                self._settle(
                    worker.run_id,
                    worker.attempt_id,
                    "interrupted",
                    application_stopped_failure(stored_stage(worker.run_id)),
                )
        for run_id in queued:
            with get_database().transaction() as connection:
                attempt = latest_attempt(connection, run_id)
                if attempt is not None:
                    end_attempt(
                        connection,
                        run_id=run_id,
                        attempt_id=attempt["attempt_id"],
                        state="interrupted",
                        failure=application_stopped_failure(),
                    )

    # --- supervisor thread -------------------------------------------------------

    def _loop(self) -> None:
        while True:
            with self._lock:
                if self._closing:
                    return
                worker = self._worker
                run_id = self._queue.popleft() if worker is None and self._queue else None
            try:
                if run_id is not None:
                    self._launch(run_id)
                elif worker is not None:
                    self._supervise(worker)
            except Exception:
                logger.exception("Run supervisor step failed")
            if run_id is None:
                self._wake.wait(self.poll_seconds)
                self._wake.clear()

    def _launch(self, run_id: str) -> None:
        attempt = start_attempt(run_id)
        if attempt is None:
            return
        process = _SPAWN.Process(
            target=self.worker_target,
            args=(run_id, os.getpid()),
            name=f"run-worker-{run_id}",
            daemon=True,
        )
        try:
            process.start()
        except Exception as error:
            logger.exception("Could not start the worker of Run %s", run_id)
            self._settle(
                run_id,
                attempt["attempt_id"],
                "failed",
                RunFailure(
                    code="internal_error",
                    message=f"{type(error).__name__}: {error}",
                    detail=str(error),
                    hint="Check the server log.",
                ),
            )
            return
        assert process.pid is not None
        set_worker_pid(attempt["attempt_id"], process.pid)
        with self._lock:
            self._worker = _Worker(run_id=run_id, attempt_id=attempt["attempt_id"], process=process)

    def _supervise(self, worker: _Worker) -> None:
        if not worker.process.is_alive():
            self._finish(worker, worker.process.exitcode)
            with self._lock:
                if self._worker is worker:
                    self._worker = None
            return
        now = monotonic()
        if worker.stop_requested_at is None and stop_flag_path(worker.run_id).exists():
            worker.stop_requested_at = now
        if worker.stop_requested_at is None:
            return
        if worker.terminated_at is None:
            if now - worker.stop_requested_at >= self.stop_grace_seconds:
                worker.process.terminate()
                worker.terminated_at = now
        elif not worker.killed and now - worker.terminated_at >= self.kill_grace_seconds:
            worker.process.kill()
            worker.killed = True

    def _finish(self, worker: _Worker, exit_code: int | None) -> None:
        """Settle an Attempt whose worker exited without recording its end."""
        if worker.stop_requested_at is not None or stop_flag_path(worker.run_id).exists():
            self._settle(worker.run_id, worker.attempt_id, "stopped", None)
            return
        failure: RunFailure | None = None
        path = failure_path(worker.run_id)
        if path.exists():
            try:
                failure = RunFailure.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValidationError):
                failure = None
        if failure is None:
            failure = worker_crashed_failure(
                exit_code if exit_code is not None else -1, stored_stage(worker.run_id)
            )
        self._settle(worker.run_id, worker.attempt_id, "failed", failure)

    def _terminate(self, process: BaseProcess) -> None:
        if not process.is_alive():
            return
        process.terminate()
        process.join(self.kill_grace_seconds)
        if process.is_alive():
            process.kill()
            process.join()

    @staticmethod
    def _settle(run_id: str, attempt_id: str, state: str, failure: RunFailure | None) -> None:
        with get_database().transaction() as connection:
            end_attempt(
                connection, run_id=run_id, attempt_id=attempt_id, state=state, failure=failure
            )


def get_supervisor() -> RunSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = RunSupervisor()
    return _supervisor
