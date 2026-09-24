"""Run lifecycle: create, read, Stop, Resume, delete, and app-wide activity.

One Run is active (queued, running, or stopping) at a time. State changes run
in one transaction with their Attempt event (see events_service); the
supervisor executes queued Attempts. Reads run inside one read transaction,
so a response never mixes two database states. Stored configurations load
with `RunConfig.model_construct`, so rows written by older versions still
render.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import get_args

from summarizer_web.db.connection import get_database
from summarizer_web.errors import ApiError
from summarizer_web.models.api import (
    ActiveRunResponse,
    ActivityResponse,
    AttemptInfo,
    CreateRunRequest,
    NodeTreeItem,
    RunConfig,
    RunFailure,
    RunListResponse,
    RunProgress,
    RunResponse,
    RunState,
)
from summarizer_web.services.documents_service import latest_revision, list_importing_documents
from summarizer_web.services.events_service import (
    checkpoint_manifest_path,
    end_attempt,
    latest_attempt,
    record_event,
    record_run_state,
    run_directory,
)
from summarizer_web.services.progress_service import ACTIVE_RUN_STATES, now_iso, read_progress
from summarizer_web.worker.projections import tree_items
from summarizer_web.worker.supervisor import get_supervisor

RESUMABLE_STATES = frozenset({"stopped", "failed", "interrupted"})
STOPPABLE_STATES = frozenset({"queued", "running"})
_RUN_STATES = frozenset(get_args(RunState))
_STRATEGIES = frozenset({"auto", "direct", "hierarchical"})

_RUN_SELECT = """
    SELECT r.*, d.title AS document_title,
           (SELECT COUNT(*) FROM run_attempts AS a WHERE a.run_id = r.run_id) AS attempt_count
    FROM runs AS r JOIN documents AS d ON d.document_id = r.document_id
"""
_ACTIVE_FILTER = " WHERE r.state IN ('queued', 'running', 'stopping')"


@contextmanager
def _reading() -> Iterator[sqlite3.Connection]:
    """A read transaction: every query inside sees the same database state."""
    connection = get_database().connect()
    try:
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()


# --- Responses -----------------------------------------------------------------


def _stored_config(config_json: str) -> RunConfig:
    try:
        values = json.loads(config_json)
    except ValueError:
        values = {}
    return RunConfig.model_construct(**(values if isinstance(values, dict) else {}))


def _attempt_failure(attempt: sqlite3.Row | None) -> RunFailure | None:
    if attempt is None or attempt["state"] not in ("failed", "interrupted"):
        return None
    if attempt["failure_json"]:
        try:
            return RunFailure.model_validate_json(attempt["failure_json"])
        except ValueError:
            pass
    if attempt["failure_reason"]:
        return RunFailure(code="legacy", message=attempt["failure_reason"])
    return None


def _to_response(
    row: sqlite3.Row,
    attempt: sqlite3.Row | None,
    *,
    any_active: bool,
    progress: RunProgress | None = None,
) -> RunResponse:
    config = _stored_config(row["config_json"])
    state = row["state"]
    failure = _attempt_failure(attempt)
    attempt_info = None
    if attempt is not None:
        attempt_info = AttemptInfo(
            attempt_id=attempt["attempt_id"],
            attempt_number=attempt["attempt_number"],
            state=attempt["state"] if attempt["state"] in _RUN_STATES else state,
            started_at=attempt["started_at"],
            ended_at=attempt["ended_at"],
            failure=failure,
        )
    return RunResponse(
        run_id=row["run_id"],
        document_id=row["document_id"],
        document_title=row["document_title"],
        state=state,
        requested_strategy=config.strategy if config.strategy in _STRATEGIES else "auto",
        selected_strategy=(
            row["selected_strategy"] if row["selected_strategy"] in ("direct", "hierarchical") else None
        ),
        config=config,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        attempt=attempt_info,
        attempt_count=row["attempt_count"],
        failure=failure,
        can_stop=state in STOPPABLE_STATES,
        can_resume=state in RESUMABLE_STATES and not any_active,
        progress=progress,
    )


def _latest_attempts(connection: sqlite3.Connection, run_ids: list[str]) -> dict[str, sqlite3.Row]:
    if not run_ids:
        return {}
    placeholders = ", ".join("?" * len(run_ids))
    rows = connection.execute(
        f"""
        SELECT a.* FROM run_attempts AS a
        JOIN (
            SELECT run_id, MAX(attempt_number) AS attempt_number FROM run_attempts
            WHERE run_id IN ({placeholders}) GROUP BY run_id
        ) AS latest USING (run_id, attempt_number)
        """,
        tuple(run_ids),
    ).fetchall()
    return {row["run_id"]: row for row in rows}


def _active_in(connection: sqlite3.Connection, *, excluding: str | None = None) -> sqlite3.Row | None:
    return connection.execute(
        _RUN_SELECT + _ACTIVE_FILTER + " AND r.run_id != ? ORDER BY r.created_at DESC LIMIT 1",
        (excluding or "",),
    ).fetchone()


def _run_active_error(active: sqlite3.Row) -> ApiError:
    return ApiError(
        409,
        "run_active",
        f"Another Run is active on “{active['document_title']}”. Stop it or wait for it to finish.",
        details={
            "run_id": active["run_id"],
            "document_id": active["document_id"],
            "document_title": active["document_title"],
            "state": active["state"],
        },
    )


def _run_with_progress(connection: sqlite3.Connection, row: sqlite3.Row, *, any_active: bool) -> RunResponse:
    attempt = _latest_attempts(connection, [row["run_id"]]).get(row["run_id"])
    return _to_response(row, attempt, any_active=any_active, progress=read_progress(row, attempt))


def get_run(run_id: str) -> RunResponse:
    with _reading() as connection:
        row = connection.execute(_RUN_SELECT + " WHERE r.run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ApiError(404, "run_not_found", "Run not found.")
        any_active = row["state"] in ACTIVE_RUN_STATES or _active_in(connection) is not None
        return _run_with_progress(connection, row, any_active=any_active)


def list_runs(document_id: str) -> RunListResponse:
    with _reading() as connection:
        document = connection.execute(
            "SELECT 1 FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if document is None:
            raise ApiError(404, "document_not_found", "Document not found.")
        rows = connection.execute(
            _RUN_SELECT + " WHERE r.document_id = ? ORDER BY r.created_at DESC, r.rowid DESC",
            (document_id,),
        ).fetchall()
        attempts = _latest_attempts(connection, [row["run_id"] for row in rows])
        any_active = _active_in(connection) is not None
    return RunListResponse(
        runs=[_to_response(row, attempts.get(row["run_id"]), any_active=any_active) for row in rows]
    )


def get_active_run() -> ActiveRunResponse:
    with _reading() as connection:
        row = _active_in(connection)
        return ActiveRunResponse(
            run=_run_with_progress(connection, row, any_active=True) if row is not None else None
        )


# --- Activity ---------------------------------------------------------------------


@dataclass(frozen=True)
class ActivityPoll:
    """One consistent read of what the activity stream sends, at `activity.cursor`."""

    activity: ActivityResponse
    # Changed tree rows per Run since the previous global cursor.
    nodes: list[tuple[str, list[NodeTreeItem]]]
    # Runs whose state changed after the previous cursor, in order of change.
    runs: list[RunResponse]


def poll_activity(after: int, *, changes: bool = True, replay: bool = False) -> ActivityPoll:
    """Read one activity snapshot and all node/state changes after the cursor.

    Projections, state events, and the active Run share one read transaction,
    so every event is reflected by the returned global cursor. On reconnect,
    state changes are collapsed to the latest RunResponse for each changed Run.
    """
    with _reading() as connection:
        cursor = int(
            connection.execute("SELECT COALESCE(MAX(event_id), 0) FROM run_events").fetchone()[0]
        )
        active = _active_in(connection)
        active_run = _run_with_progress(connection, active, any_active=True) if active else None
        nodes: list[tuple[str, list[NodeTreeItem]]] = []
        runs: list[RunResponse] = []
        if changes:
            # Scan the indexed event range, not every archived node projection.
            changed_nodes = connection.execute(
                """
                SELECT run_id, MIN(event_id) AS first_event
                FROM run_events
                WHERE event_id > ? AND event_id <= ? AND event_type = 'item'
                GROUP BY run_id ORDER BY first_event, run_id
                """,
                (after, cursor),
            ).fetchall()
            for row in changed_nodes:
                run_id = row["run_id"]
                changed = tree_items(
                    run_id, after_event_id=after, through_event_id=cursor, connection=connection
                )
                if changed:
                    nodes.append((run_id, changed))

            state_events = connection.execute(
                """
                SELECT event_id, run_id FROM run_events
                WHERE event_id > ? AND event_id <= ? AND event_type = 'run_state'
                ORDER BY event_id
                """,
                (after, cursor),
            ).fetchall()
            if replay:
                latest_by_run = {row["run_id"]: row for row in state_events}
                state_events = sorted(latest_by_run.values(), key=lambda row: row["event_id"])
            changed_runs = list(dict.fromkeys(row["run_id"] for row in state_events))
            if changed_runs:
                placeholders = ", ".join("?" * len(changed_runs))
                rows_by_id = {
                    row["run_id"]: row
                    for row in connection.execute(
                        _RUN_SELECT + f" WHERE r.run_id IN ({placeholders})", tuple(changed_runs)
                    ).fetchall()
                }
                attempts = _latest_attempts(connection, list(rows_by_id))
                runs = [
                    _to_response(
                        rows_by_id[row["run_id"]],
                        attempts.get(row["run_id"]),
                        any_active=active is not None,
                    )
                    for row in state_events
                    if row["run_id"] in rows_by_id
                ]
    activity = ActivityResponse(
        active_run=active_run, importing=list_importing_documents(), cursor=cursor
    )
    return ActivityPoll(activity=activity, nodes=nodes, runs=runs)


def get_activity() -> ActivityResponse:
    return poll_activity(0, changes=False).activity


# --- Lifecycle -------------------------------------------------------------------


def _validate_config(config: RunConfig) -> None:
    if not config.model.strip():
        raise ApiError(400, "model_required", "Choose a model before starting a Run.")
    window = config.context_window
    if window is not None and config.max_output_tokens >= window:
        raise ApiError(
            422,
            "invalid_request",
            "config.max_output_tokens must be smaller than config.context_window.",
        )
    if window is not None and config.chunk_tokens is not None and config.chunk_tokens >= window:
        raise ApiError(
            422,
            "invalid_request",
            "config.chunk_tokens must be smaller than config.context_window.",
        )


def _run_for_key(connection: sqlite3.Connection, key: str | None) -> str | None:
    if key is None:
        return None
    row = connection.execute("SELECT run_id FROM runs WHERE idempotency_key = ?", (key,)).fetchone()
    return row["run_id"] if row is not None else None


def create_run(payload: CreateRunRequest, idempotency_key: str | None) -> tuple[RunResponse, bool]:
    """Queue a new Run; returns it and whether it was created (False for a repeated key)."""
    key = (idempotency_key or "").strip() or None
    db = get_database()
    if key is not None:
        row = db.fetchone("SELECT run_id FROM runs WHERE idempotency_key = ?", (key,))
        if row is not None:
            return get_run(row["run_id"]), False
    _validate_config(payload.config)
    document = db.fetchone(
        "SELECT document_id, title, import_state FROM documents WHERE document_id = ?",
        (payload.document_id,),
    )
    if document is None:
        raise ApiError(404, "document_not_found", "Document not found.")
    revision = latest_revision(payload.document_id) if document["import_state"] == "ready" else None
    if revision is None:
        message = (
            "The Document's import failed; it cannot be summarized."
            if document["import_state"] == "failed"
            else "The Document is still importing. Start the Run once its import finishes."
        )
        raise ApiError(
            409,
            "document_not_ready",
            message,
            details={"document_id": payload.document_id, "import_state": document["import_state"]},
        )
    now = now_iso()
    with db.transaction() as connection:
        # Checked again under the write lock: two submissions may race.
        repeated = _run_for_key(connection, key)
        if repeated is None:
            active = _active_in(connection)
            if active is not None:
                raise _run_active_error(active)
            run_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, document_id, revision_id, state, strategy, config_json,
                    idempotency_key, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    payload.document_id,
                    revision["revision_id"],
                    payload.config.strategy,
                    payload.config.model_dump_json(),
                    key,
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO run_attempts (attempt_id, run_id, attempt_number, state) "
                "VALUES (?, ?, 1, 'queued')",
                (str(uuid.uuid4()), run_id),
            )
            record_event(connection, run_id, "attempt", {"attempt_number": 1, "state": "queued"})
            record_run_state(connection, run_id, "queued")
    if repeated is not None:
        return get_run(repeated), False
    get_supervisor().enqueue(run_id)
    return get_run(run_id), True


def stop_run(run_id: str) -> RunResponse:
    """Queued Runs stop at once; running Runs go stopping and the worker is signalled."""
    with get_database().transaction() as connection:
        run = connection.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            raise ApiError(404, "run_not_found", "Run not found.")
        attempt = latest_attempt(connection, run_id)
        state = run["state"]
        if state == "queued" and attempt is not None:
            end_attempt(connection, run_id=run_id, attempt_id=attempt["attempt_id"], state="stopped")
        elif state == "running" and attempt is not None:
            now = now_iso()
            record_event(
                connection,
                run_id,
                "attempt",
                {"attempt_number": attempt["attempt_number"], "state": "stopping"},
            )
            connection.execute(
                "UPDATE run_attempts SET state = 'stopping' WHERE attempt_id = ?",
                (attempt["attempt_id"],),
            )
            connection.execute(
                "UPDATE runs SET state = 'stopping', updated_at = ? WHERE run_id = ?",
                (now, run_id),
            )
            record_run_state(connection, run_id, "stopping")
        elif state != "stopping":
            raise ApiError(
                409,
                "run_not_stoppable",
                f"Only queued or running Runs can be stopped; this Run is {state}.",
                details={"state": state},
            )
    if state == "queued":
        get_supervisor().discard(run_id)
    elif state == "running":
        get_supervisor().request_stop(run_id)
    return get_run(run_id)


def resume_run(run_id: str) -> RunResponse:
    """Queue a new Attempt that redoes only the unfinished work of a stopped, failed, or interrupted Run."""
    with get_database().transaction() as connection:
        run = connection.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            raise ApiError(404, "run_not_found", "Run not found.")
        if run["state"] not in RESUMABLE_STATES:
            raise ApiError(
                409,
                "run_not_resumable",
                f"Only stopped, failed, or interrupted Runs can be resumed; this Run is {run['state']}.",
                details={"state": run["state"]},
            )
        active = _active_in(connection, excluding=run_id)
        if active is not None:
            raise _run_active_error(active)
        attempt = latest_attempt(connection, run_id)
        number = (attempt["attempt_number"] if attempt is not None else 0) + 1
        connection.execute(
            "INSERT INTO run_attempts (attempt_id, run_id, attempt_number, state) VALUES (?, ?, ?, 'queued')",
            (str(uuid.uuid4()), run_id, number),
        )
        connection.execute(
            "UPDATE runs SET state = 'queued', progress_json = NULL, updated_at = ? WHERE run_id = ?",
            (now_iso(), run_id),
        )
        record_event(connection, run_id, "attempt", {"attempt_number": number, "state": "queued"})
        record_run_state(connection, run_id, "queued")
    get_supervisor().enqueue(run_id)
    return get_run(run_id)


def delete_run(run_id: str) -> None:
    """Remove a finished Run: rows (attempts, events, nodes, segments cascade) and files."""
    with get_database().transaction() as connection:
        row = connection.execute(_RUN_SELECT + " WHERE r.run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ApiError(404, "run_not_found", "Run not found.")
        if row["state"] in ACTIVE_RUN_STATES:
            raise _run_active_error(row)
        connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    shutil.rmtree(run_directory(run_id), ignore_errors=True)
    manifest = checkpoint_manifest_path(run_id)
    manifest.unlink(missing_ok=True)
    manifest.with_suffix(".lock").unlink(missing_ok=True)
