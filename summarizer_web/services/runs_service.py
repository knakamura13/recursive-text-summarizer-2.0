"""Run lifecycle management."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database
from summarizer_web.models.api import (
    CreateRunRequest,
    FinalSummaryResponse,
    NodeDetailResponse,
    NodeTreeItem,
    NodeTreeResponse,
    RunConfig,
    RunListResponse,
    RunResponse,
)
from summarizer_web.services.documents_service import _latest_revision, get_document
from summarizer_web.worker.supervisor import get_supervisor

_ACTIVE_STATES = {"queued", "running", "cancelling"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_run(row, attempt_row=None) -> RunResponse:
    return RunResponse(
        run_id=row["run_id"],
        document_id=row["document_id"],
        state=row["state"],
        strategy=row["strategy"],
        config=RunConfig.model_validate_json(row["config_json"]),
        attempt_id=attempt_row["attempt_id"] if attempt_row is not None else None,
        attempt_state=attempt_row["state"] if attempt_row is not None else None,
        failure_reason=attempt_row["failure_reason"] if attempt_row is not None else None,
    )


def _latest_attempt(run_id: str):
    return get_database().fetchone(
        """
        SELECT * FROM run_attempts
        WHERE run_id = ?
        ORDER BY attempt_number DESC
        LIMIT 1
        """,
        (run_id,),
    )


def list_runs(document_id: str) -> RunListResponse:
    rows = get_database().fetchall(
        "SELECT * FROM runs WHERE document_id = ? ORDER BY created_at DESC",
        (document_id,),
    )
    runs = [_row_to_run(row, _latest_attempt(row["run_id"])) for row in rows]
    return RunListResponse(runs=runs)


def get_run(run_id: str) -> RunResponse:
    row = get_database().fetchone("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _row_to_run(row, _latest_attempt(run_id))


def create_run(payload: CreateRunRequest, idempotency_key: str | None) -> RunResponse:
    if not payload.config.model.strip():
        raise HTTPException(status_code=400, detail="Model selection is required")
    document = get_document(payload.document_id)
    if document.import_state != "ready":
        raise HTTPException(status_code=400, detail="Document is not ready to summarize")
    revision = _latest_revision(payload.document_id)
    if revision is None:
        raise HTTPException(status_code=400, detail="Missing source revision")
    db = get_database()
    if idempotency_key:
        existing = db.fetchone("SELECT run_id FROM runs WHERE idempotency_key = ?", (idempotency_key,))
        if existing is not None:
            return get_run(existing["run_id"])
    active = db.fetchone(
        "SELECT run_id FROM runs WHERE state IN ('queued', 'running', 'cancelling') LIMIT 1"
    )
    if active is not None:
        raise HTTPException(status_code=409, detail="Another run is already active")
    run_id = str(uuid.uuid4())
    now = _now()
    db.execute(
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
            idempotency_key,
            now,
            now,
        ),
    )
    attempt_id = str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO run_attempts (attempt_id, run_id, attempt_number, state, started_at)
        VALUES (?, ?, 1, 'queued', ?)
        """,
        (attempt_id, run_id, now),
    )
    get_supervisor().enqueue(run_id)
    return get_run(run_id)


def cancel_run(run_id: str) -> RunResponse:
    row = get_database().fetchone("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if row["state"] not in _ACTIVE_STATES:
        raise HTTPException(status_code=409, detail="Run is not active")
    get_database().execute(
        "UPDATE runs SET state = 'cancelling', updated_at = ? WHERE run_id = ?",
        (_now(), run_id),
    )
    get_supervisor().request_cancel(run_id)
    return get_run(run_id)


def resume_run(run_id: str) -> RunResponse:
    row = get_database().fetchone("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if row["state"] not in {"failed", "cancelled", "interrupted"}:
        raise HTTPException(status_code=409, detail="Run cannot be resumed")
    active = get_database().fetchone(
        "SELECT run_id FROM runs WHERE state IN ('queued', 'running', 'cancelling') LIMIT 1"
    )
    if active is not None:
        raise HTTPException(status_code=409, detail="Another run is already active")
    attempt_number = get_database().fetchone(
        "SELECT MAX(attempt_number) AS max_attempt FROM run_attempts WHERE run_id = ?",
        (run_id,),
    )["max_attempt"] + 1
    attempt_id = str(uuid.uuid4())
    now = _now()
    get_database().execute(
        "UPDATE runs SET state = 'queued', updated_at = ? WHERE run_id = ?",
        (now, run_id),
    )
    get_database().execute(
        """
        INSERT INTO run_attempts (attempt_id, run_id, attempt_number, state, started_at)
        VALUES (?, ?, ?, 'queued', ?)
        """,
        (attempt_id, run_id, attempt_number, now),
    )
    get_supervisor().enqueue(run_id, resume=True)
    return get_run(run_id)


def get_node_tree(run_id: str) -> NodeTreeResponse:
    rows = get_database().fetchall(
        """
        SELECT node_id, parent_id, level, order_index, label, provisional, summary_text
        FROM node_projections
        WHERE run_id = ?
        ORDER BY level, order_index
        """,
        (run_id,),
    )
    nodes = [
        NodeTreeItem(
            node_id=row["node_id"],
            parent_id=row["parent_id"],
            level=row["level"],
            order=row["order_index"],
            label=row["label"],
            provisional=bool(row["provisional"]),
            state="completed" if row["summary_text"] is not None else "pending",
        )
        for row in rows
    ]
    return NodeTreeResponse(nodes=nodes)


def get_node_detail(run_id: str, node_id: str) -> NodeDetailResponse:
    row = get_database().fetchone(
        """
        SELECT * FROM node_projections
        WHERE run_id = ? AND node_id = ?
        """,
        (run_id, node_id),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return NodeDetailResponse(
        node_id=row["node_id"],
        label=row["label"],
        summary_text=row["summary_text"],
        provisional=bool(row["provisional"]),
        source_passage=None,
        evidence_refs=json.loads(row["evidence_refs_json"] or "[]"),
        covered_segment_ids=json.loads(row["covered_segment_ids_json"] or "[]"),
    )


def get_final_summary(run_id: str) -> FinalSummaryResponse:
    summary_path = load_paths().runs / run_id / "summary.txt"
    verification_path = load_paths().runs / run_id / "verification.json"
    row = get_database().fetchone("SELECT state FROM runs WHERE run_id = ?", (run_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if row["state"] != "completed" or not summary_path.exists():
        verification_state = "not_run"
        if row["state"] == "running":
            verification_state = "in_progress"
        return FinalSummaryResponse(available=False, verification_state=verification_state)
    text = summary_path.read_text(encoding="utf-8")
    verification = None
    verification_state = "not_run"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        verification_state = verification.get("state", "completed")
    return FinalSummaryResponse(
        available=True,
        text=text,
        verification_state=verification_state,
        verification=verification,
    )

