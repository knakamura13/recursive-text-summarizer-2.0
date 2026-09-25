"""Run event log and Attempt lifecycle.

`run_events` rows are the cursor space of the activity stream: every item,
stage, Attempt, and Run-state transition is recorded there, and projection rows
changed by an event carry its id. State transitions run in one transaction with
their event, so a reader that sees a new Run state also sees every row it
implies.

Each Run directory holds the files that pass signals between processes:
`stop.flag` (Stop requested; the worker polls it between items),
`failure.json` (the worker's RunFailure, read if it could not record it), and
`worker.lock` (held by the live worker, naming its PID).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database
from summarizer_web.models.api import RunFailure
from summarizer_web.services.progress_service import now_iso
from summarizer_web.worker.projections import reset_active_nodes

ACTIVE_ATTEMPT_STATES = frozenset({"queued", "running", "stopping"})
STOPPED_REASON = "Stopped by user"


def run_directory(run_id: str) -> Path:
    return load_paths().runs / run_id


def stop_flag_path(run_id: str) -> Path:
    return run_directory(run_id) / "stop.flag"


def failure_path(run_id: str) -> Path:
    return run_directory(run_id) / "failure.json"


def worker_lock_path(run_id: str) -> Path:
    """Locked by the Run's live worker for its lifetime; holds the worker's PID."""
    return run_directory(run_id) / "worker.lock"


def request_stop_flag(run_id: str) -> None:
    path = stop_flag_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1", encoding="utf-8")


def checkpoint_run_id(run_id: str) -> str:
    """The pipeline checkpoint id of a Run; its manifest witnesses finished work."""
    return f"run-{run_id}"


def checkpoint_manifest_path(run_id: str) -> Path:
    return load_paths().cache / "runs" / f"{checkpoint_run_id(run_id)}.json"


def clear_attempt_flags(run_id: str) -> None:
    stop_flag_path(run_id).unlink(missing_ok=True)
    failure_path(run_id).unlink(missing_ok=True)


def record_event(
    connection: sqlite3.Connection,
    run_id: str,
    event_type: str,
    payload: Mapping[str, object],
) -> int:
    """Append an event inside the caller's transaction and return its id."""
    cursor = connection.execute(
        "INSERT INTO run_events (run_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
        (run_id, event_type, json.dumps(payload), now_iso()),
    )
    return int(cursor.lastrowid)


def record_run_state(connection: sqlite3.Connection, run_id: str, state: str) -> int:
    """Append the cursor-bearing event for a Run state transition."""
    return record_event(connection, run_id, "run_state", {"state": state})


def latest_attempt(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM run_attempts WHERE run_id = ? ORDER BY attempt_number DESC LIMIT 1",
        (run_id,),
    ).fetchone()


def start_attempt(run_id: str) -> sqlite3.Row | None:
    """Move a queued Run and its latest Attempt to running; None if it is no longer queued.

    Flags of an earlier Attempt are cleared first, while the Run is still
    queued: a Stop arriving meanwhile ends the queued Run directly instead of
    writing a flag that this clear could erase.
    """
    clear_attempt_flags(run_id)
    with get_database().transaction() as connection:
        run = connection.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        attempt = latest_attempt(connection, run_id)
        if run is None or attempt is None or run["state"] != "queued" or attempt["state"] != "queued":
            return None
        now = now_iso()
        record_event(
            connection,
            run_id,
            "attempt",
            {"attempt_number": attempt["attempt_number"], "state": "running"},
        )
        connection.execute(
            "UPDATE run_attempts SET state = 'running', started_at = ?, ended_at = NULL WHERE attempt_id = ?",
            (now, attempt["attempt_id"]),
        )
        connection.execute(
            "UPDATE runs SET state = 'running', progress_json = NULL, updated_at = ? WHERE run_id = ?",
            (now, run_id),
        )
        record_run_state(connection, run_id, "running")
        return connection.execute(
            "SELECT * FROM run_attempts WHERE attempt_id = ?", (attempt["attempt_id"],)
        ).fetchone()


def set_worker_pid(attempt_id: str, pid: int) -> None:
    get_database().execute(
        "UPDATE run_attempts SET worker_pid = ? WHERE attempt_id = ?", (pid, attempt_id)
    )


def end_attempt(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    attempt_id: str,
    state: str,
    failure: RunFailure | None = None,
) -> int | None:
    """Record the end of an active Attempt inside the caller's transaction.

    Returns the Run-state event id, or None when the Attempt already ended (the
    worker, the supervisor, and startup reconciliation may race to finish it).
    Unless the Attempt completed, its active nodes return to pending.
    """
    attempt = connection.execute(
        "SELECT attempt_number, state FROM run_attempts WHERE attempt_id = ? AND run_id = ?",
        (attempt_id, run_id),
    ).fetchone()
    if attempt is None or attempt["state"] not in ACTIVE_ATTEMPT_STATES:
        return None
    now = now_iso()
    record_event(
        connection,
        run_id,
        "attempt",
        {
            "attempt_number": attempt["attempt_number"],
            "state": state,
            "failure": failure.model_dump() if failure is not None else None,
        },
    )
    reason = failure.message if failure is not None else STOPPED_REASON if state == "stopped" else None
    connection.execute(
        """
        UPDATE run_attempts SET state = ?, ended_at = ?, failure_json = ?, failure_reason = ?
        WHERE attempt_id = ?
        """,
        (
            state,
            now,
            failure.model_dump_json() if failure is not None else None,
            reason,
            attempt_id,
        ),
    )
    connection.execute(
        "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?", (state, now, run_id)
    )
    if state != "completed":
        reset_active_nodes(connection, run_id, record_event)
    return record_run_state(connection, run_id, state)


def stored_stage(run_id: str) -> str | None:
    """The stage of a Run's last progress snapshot, for failures recorded outside the worker."""
    row = get_database().fetchone("SELECT progress_json FROM runs WHERE run_id = ?", (run_id,))
    if row is None or not row["progress_json"]:
        return None
    try:
        stage = json.loads(row["progress_json"]).get("stage")
    except (ValueError, AttributeError):
        return None
    return stage if isinstance(stage, str) else None


def application_stopped_failure(stage: str | None = None) -> RunFailure:
    return RunFailure(
        code="application_stopped",
        message="The application stopped during this Run.",
        stage=stage,
        hint="Resume to continue.",
    )


def worker_crashed_failure(exit_code: int, stage: str | None = None) -> RunFailure:
    return RunFailure(
        code="worker_crashed",
        message=f"The worker stopped unexpectedly (exit code {exit_code}).",
        stage=stage,
        hint="Check the server log, then Resume.",
    )
