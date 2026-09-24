"""Startup reconciliation of Runs a previous application process left active.

No Run resumes on its own. A Run whose summary publication the checkpoint
manifest witnesses completed; a Run with a Stop in flight is stopped; every
other active Run is interrupted with RunFailure `application_stopped`. A worker
that outlived its application is terminated first, but only when it still
holds the Run's worker lock, which proves the PID is that worker's (PIDs are
reused, and the kernel releases the lock when the worker dies).
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import time
from collections.abc import Collection

from summarizer.checkpoint import RunManifest
from summarizer.finalization import PublicationError, read_published_summary
from summarizer_web.db.connection import get_database
from summarizer_web.services.events_service import (
    application_stopped_failure,
    checkpoint_manifest_path,
    end_attempt,
    latest_attempt,
    record_run_state,
    run_directory,
    stored_stage,
    worker_lock_path,
)
from summarizer_web.services.progress_service import now_iso

_ORPHAN_EXIT_WAIT_SECONDS = 2.0


def live_worker_pid(run_id: str) -> int | None:
    """PID of this Run's worker if one is still alive, read from the lock it holds."""
    try:
        descriptor = os.open(worker_lock_path(run_id), os.O_RDONLY)
    except FileNotFoundError:
        return None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            recorded = os.read(descriptor, 32).decode("ascii", errors="ignore").strip()
            return int(recorded) if recorded.isdigit() else None
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return None
    finally:
        os.close(descriptor)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_orphan(pid: int) -> None:
    """SIGTERM, then SIGKILL if the process is still alive after a short wait."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + _ORPHAN_EXIT_WAIT_SECONDS
    while time.monotonic() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def has_published_summary(run_id: str) -> bool:
    """True when the manifest witnesses both summary.txt and audit.json byte for byte."""
    manifest_path = checkpoint_manifest_path(run_id)
    try:
        manifest = RunManifest.model_validate(json.loads(manifest_path.read_bytes()))
    except (OSError, ValueError):
        return False
    run_dir = run_directory(run_id)
    try:
        read_published_summary(run_dir / "summary.txt", run_dir / "audit.json", manifest)
    except PublicationError:
        return False
    return True


def reconcile_on_startup(exclude_run_ids: Collection[str] = ()) -> None:
    db = get_database()
    excluded = set(exclude_run_ids)
    rows = db.fetchall(
        "SELECT run_id, state FROM runs WHERE state IN ('queued', 'running', 'stopping')"
    )
    for row in rows:
        run_id = row["run_id"]
        if run_id in excluded:
            continue
        orphan = live_worker_pid(run_id)
        if orphan is not None:
            terminate_orphan(orphan)
        if has_published_summary(run_id):
            state, failure = "completed", None
        elif row["state"] == "stopping":
            state, failure = "stopped", None
        else:
            state, failure = "interrupted", application_stopped_failure(stored_stage(run_id))
        with db.transaction() as connection:
            attempt = latest_attempt(connection, run_id)
            if attempt is not None and end_attempt(
                connection,
                run_id=run_id,
                attempt_id=attempt["attempt_id"],
                state=state,
                failure=failure,
            ) is not None:
                continue
            # No active Attempt row (legacy data): settle the Run row alone.
            changed = connection.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ? AND state = ?",
                (state, now_iso(), run_id, row["state"]),
            ).rowcount
            if changed:
                record_run_state(connection, run_id, state)
