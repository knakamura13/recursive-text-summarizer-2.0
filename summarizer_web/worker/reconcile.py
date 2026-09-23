"""Startup reconciliation for interrupted runs."""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime, timezone

from summarizer.finalization import read_published_summary
from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reconcile_on_startup(exclude_run_ids: Collection[str] = ()) -> None:
    db = get_database()
    excluded = set(exclude_run_ids)
    rows = db.fetchall(
        "SELECT run_id, state FROM runs WHERE state IN ('running', 'cancelling', 'queued')"
    )
    for row in rows:
        run_id = row["run_id"]
        if run_id in excluded:
            continue
        summary_path = load_paths().runs / run_id / "summary.txt"
        if summary_path.exists() and summary_path.read_text(encoding="utf-8").strip():
            db.execute(
                "UPDATE runs SET state = 'completed', updated_at = ? WHERE run_id = ?",
                (_now(), run_id),
            )
            continue
        db.execute(
            "UPDATE runs SET state = 'interrupted', updated_at = ? WHERE run_id = ?",
            (_now(), run_id),
        )
        db.execute(
            """
            UPDATE run_attempts
            SET state = 'interrupted', ended_at = ?, failure_reason = 'Application restarted'
            WHERE run_id = ? AND state IN ('running', 'queued', 'cancelling')
            """,
            (_now(), run_id),
        )
