"""Seed Documents, Runs, node projections, segments, and Run artifacts for Run view tests.

Rows are written straight into the migrated schema, so read endpoints can be
exercised against the exact states the worker and legacy versions leave behind.
`web_client` starts the app with its Run supervisor and ImportManager stubbed,
so no test here can launch a child process.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database
from summarizer_web.main import create_app
from summarizer_web.models.api import RunConfig
from summarizer_web.worker.imports import ImportManager
from summarizer_web.worker.supervisor import RunSupervisor

NOW = "2026-09-23T12:00:00+00:00"


@contextmanager
def web_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("SUMMARIZER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(RunSupervisor, "start", lambda self: None)
    monkeypatch.setattr(ImportManager, "start", lambda self: None)
    with TestClient(create_app()) as client:
        yield client


@dataclass(frozen=True)
class SeededDocument:
    document_id: str
    revision_id: str | None
    text: str
    source_sha256: str
    page_map: list[dict[str, int]] | None

    def offset(self, fragment: str) -> tuple[int, int]:
        start = self.text.index(fragment)
        return start, start + len(fragment)


def csrf_headers(client: TestClient) -> dict[str, str]:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    return {"X-CSRF-Token": response.headers["X-CSRF-Token"]}


def paged_text(pages: Sequence[str]) -> tuple[str, list[dict[str, int]]]:
    """Canonical text with page markers, and its page map (bodies only, end exclusive)."""
    parts: list[str] = []
    page_map: list[dict[str, int]] = []
    offset = 0
    for number, body in enumerate(pages, start=1):
        marker = f"{'' if number == 1 else chr(10) * 2}--- Page {number} ---\n"
        parts.extend((marker, body))
        offset += len(marker)
        page_map.append({"page": number, "start": offset, "end": offset + len(body)})
        offset += len(body)
    return "".join(parts), page_map


def seed_document(
    *,
    document_id: str = "doc-1",
    title: str = "Bridge Report",
    text: str | None = "The bridge deck spans 200 feet across the river.",
    page_map: list[dict[str, int]] | None = None,
    import_state: str = "ready",
    import_error: str | None = None,
) -> SeededDocument:
    """A Document row, plus its revision and canonical text once it is ready."""
    db = get_database()
    body = text or ""
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    db.execute(
        """
        INSERT INTO documents (
            document_id, title, filename, format, size_bytes, import_state, created_at,
            updated_at, origin, import_error, original_sha256, char_count, page_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'upload', ?, ?, ?, ?)
        """,
        (
            document_id,
            title,
            f"{document_id}.{'pdf' if page_map else 'txt'}",
            "pdf" if page_map else "txt",
            len(body.encode("utf-8")),
            import_state,
            NOW,
            NOW,
            import_error,
            sha,
            len(body) if import_state == "ready" else None,
            len(page_map) if page_map else None,
        ),
    )
    if import_state != "ready":
        return SeededDocument(document_id, None, body, sha, page_map)
    document_dir = load_paths().documents / document_id
    document_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = document_dir / "canonical.txt"
    canonical_path.write_text(body, encoding="utf-8")
    revision_id = f"rev-{document_id}"
    db.execute(
        """
        INSERT INTO source_revisions (
            revision_id, document_id, source_sha256, extraction_version, canonical_path,
            page_map_json, blank_pages_json, confirmed_at, created_at
        ) VALUES (?, ?, ?, 'text/1', ?, ?, '[]', ?, ?)
        """,
        (
            revision_id,
            document_id,
            sha,
            str(canonical_path),
            json.dumps(page_map) if page_map else None,
            NOW,
            NOW,
        ),
    )
    return SeededDocument(document_id, revision_id, body, sha, page_map)


def seed_run(
    document: SeededDocument,
    *,
    run_id: str = "run-1",
    state: str = "completed",
    config: RunConfig | None = None,
    summary: str | None = None,
    audit: dict | None = None,
) -> Path:
    """A Run row and its directory with optional summary.txt and audit.json."""
    config = config or RunConfig(model="llama3.2:3b", target_words=40)
    get_database().execute(
        """
        INSERT INTO runs (
            run_id, document_id, revision_id, state, strategy, config_json,
            idempotency_key, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            document.document_id,
            document.revision_id,
            state,
            config.strategy,
            config.model_dump_json(),
            run_id,
            NOW,
            NOW,
        ),
    )
    run_dir = load_paths().runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
    if audit is not None:
        (run_dir / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return run_dir


def seed_segments(
    run_id: str,
    segments: Sequence[tuple[str, int, int]],
    page_map: Sequence[dict[str, int]] | None = None,
) -> None:
    """Segments as (segment_id, core_start, core_end); context equals core.

    Pages are the first and last page the core overlaps, as the worker stores them.
    """

    def pages(start: int, end: int) -> tuple[int | None, int | None]:
        hits = [
            entry["page"] for entry in page_map or () if entry["start"] < end and entry["end"] > start
        ]
        return (hits[0], hits[-1]) if hits else (None, None)

    get_database().executemany(
        """
        INSERT INTO run_segments (
            run_id, segment_id, order_index, start_offset, end_offset, core_start, core_end,
            page_start, page_end
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (run_id, segment_id, order, start, end, start, end, *pages(start, end))
            for order, (segment_id, start, end) in enumerate(segments)
        ],
    )


def seed_event(run_id: str, event_type: str = "item") -> int:
    cursor = get_database().execute(
        "INSERT INTO run_events (run_id, event_type, payload_json, created_at) VALUES (?, ?, '{}', ?)",
        (run_id, event_type, NOW),
    )
    return int(cursor.lastrowid)


def seed_node(
    run_id: str,
    node_id: str,
    *,
    level: int,
    order: int,
    label: str,
    parent_id: str | None = None,
    kind: str | None = None,
    state: str | None = None,
    summary: dict | None = None,
    covered: Sequence[str] = (),
    child_ids: Sequence[str] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    error: str | None = None,
    updated_event_id: int = 0,
    legacy_evidence: Sequence[dict] | None = None,
) -> None:
    """A node_projections row; `summary` is a SummaryNode dict as the worker stores it.

    Without `kind`/`state` the row keeps the migration defaults of a legacy
    row; `legacy_evidence` stores only the flattened evidence, as legacy rows do.
    """
    columns: dict[str, object] = {
        "projection_id": f"{run_id}:{node_id}",
        "run_id": run_id,
        "node_id": node_id,
        "parent_id": parent_id,
        "level": level,
        "order_index": order,
        "label": label,
        "summary_text": summary["summary"] if summary else None,
        "provisional": 0,
        "covered_segment_ids_json": json.dumps(list(covered)),
        "evidence_refs_json": json.dumps(list(legacy_evidence or [])),
        "started_at": started_at,
        "completed_at": completed_at,
        "error": error,
        "updated_event_id": updated_event_id,
    }
    if kind is not None:
        columns["kind"] = kind
    if state is not None:
        columns["state"] = state
    if summary is not None and legacy_evidence is None:
        columns["content_units_json"] = json.dumps(summary.get("content_units", []))
        columns["detail_json"] = json.dumps(
            {
                key: summary.get(key, [])
                for key in ("entities", "qualifications", "contradictions", "quotations")
            }
        )
    if child_ids is not None:
        columns["child_ids_json"] = json.dumps(list(child_ids))
    names = ", ".join(columns)
    marks = ", ".join("?" for _ in columns)
    get_database().execute(
        f"INSERT INTO node_projections ({names}) VALUES ({marks})", tuple(columns.values())
    )


def audit_segment(segment_id: str, order: int, start: int, end: int, source_id: str) -> dict:
    """An audit.json `source_segments` entry whose context equals its core."""
    return {
        "segment_id": segment_id,
        "source_id": source_id,
        "order": order,
        "core_start": start,
        "core_end": end,
        "context_start": start,
        "context_end": end,
        "core_token_count": end - start,
        "token_count": end - start,
        "leading_overlap_tokens": 0,
        "trailing_overlap_tokens": 0,
        "boundary_kind": "paragraph",
    }
