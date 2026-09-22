"""Single-run worker supervisor."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database, init_database
from summarizer_web.worker.reconcile import reconcile_on_startup
from summarizer_web.worker.runner import run_job

_supervisor: RunSupervisor | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunSupervisor:
    queue: list[tuple[str, bool]] = field(default_factory=list)
    active_run_id: str | None = None
    process: mp.Process | None = None
    cancel_requested: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: threading.Thread | None = None

    def start(self) -> None:
        init_database()
        reconcile_on_startup()
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def enqueue(self, run_id: str, *, resume: bool = False) -> None:
        with self.lock:
            self.queue.append((run_id, resume))
        self.start()

    def request_cancel(self, run_id: str) -> None:
        with self.lock:
            self.cancel_requested.add(run_id)
        cancel_flag = load_paths().runs / run_id / "cancel.flag"
        cancel_flag.parent.mkdir(parents=True, exist_ok=True)
        cancel_flag.write_text("1", encoding="utf-8")
        if self.process is not None and self.process.is_alive():
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()

    def _loop(self) -> None:
        while True:
            with self.lock:
                if self.active_run_id is None and self.queue:
                    run_id, resume = self.queue.pop(0)
                    self.active_run_id = run_id
                    self._start_process(run_id, resume=resume)
            if self.process is not None and not self.process.is_alive():
                self._finish_active()
            time.sleep(0.5)

    def _start_process(self, run_id: str, *, resume: bool) -> None:
        db = get_database()
        db.execute(
            "UPDATE runs SET state = 'running', updated_at = ? WHERE run_id = ?",
            (_now(), run_id),
        )
        db.execute(
            """
            UPDATE run_attempts SET state = 'running'
            WHERE run_id = ? AND attempt_number = (
                SELECT MAX(attempt_number) FROM run_attempts WHERE run_id = ?
            )
            """,
            (run_id, run_id),
        )
        self.process = mp.Process(target=run_job, args=(run_id,), kwargs={"resume": resume})
        self.process.start()

    def _finish_active(self) -> None:
        run_id = self.active_run_id
        if run_id is None:
            return
        db = get_database()
        exit_code = self.process.exitcode if self.process is not None else 1
        cancel_flag = load_paths().runs / run_id / "cancel.flag"
        if run_id in self.cancel_requested or cancel_flag.exists():
            state = "cancelled"
            failure = "Cancelled by user"
        elif exit_code == 0:
            state = "completed"
            failure = None
        else:
            state = "failed"
            failure = f"Worker exited with code {exit_code}"
        db.execute(
            "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?",
            (state, _now(), run_id),
        )
        db.execute(
            """
            UPDATE run_attempts
            SET state = ?, ended_at = ?, failure_reason = ?
            WHERE run_id = ? AND attempt_number = (
                SELECT MAX(attempt_number) FROM run_attempts WHERE run_id = ?
            )
            """,
            (state, _now(), failure, run_id, run_id),
        )
        self.active_run_id = None
        self.process = None
        if cancel_flag.exists():
            cancel_flag.unlink(missing_ok=True)


def get_supervisor() -> RunSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = RunSupervisor()
        _supervisor.start()
    return _supervisor
