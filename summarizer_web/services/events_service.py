"""Persist run events for SSE replay."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from summarizer_web.db.connection import get_database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_event(run_id: str, event_type: str, payload: dict) -> int:
    cursor = get_database().execute(
        """
        INSERT INTO run_events (run_id, event_type, payload_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, event_type, json.dumps(payload), _now()),
    )
    return int(cursor.lastrowid)
