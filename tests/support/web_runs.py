"""Seed data and scripted providers for web Run tests (no network, no Ollama)."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database, init_database

NOW = "2026-09-23T00:00:00+00:00"
TEXT = " ".join(
    f"Sentence {index} reports that the harbor moved {index} tonnes of cargo." for index in range(1, 25)
)


def use_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUMMARIZER_DATA_DIR", str(tmp_path))
    init_database(load_paths())


def page_map(text: str, page_chars: int) -> list[dict[str, int]]:
    return [
        {"page": index + 1, "start": start, "end": min(start + page_chars, len(text))}
        for index, start in enumerate(range(0, len(text), page_chars))
    ]


def seed_document(
    text: str = TEXT,
    *,
    title: str = "Harbor report",
    import_state: str = "ready",
    pages: list[dict[str, int]] | None = None,
) -> tuple[str, str]:
    """Insert a Document with one source revision; returns (document_id, revision_id)."""
    document_id = str(uuid.uuid4())
    revision_id = str(uuid.uuid4())
    directory = load_paths().documents / document_id
    directory.mkdir(parents=True)
    canonical = directory / "canonical.txt"
    canonical.write_text(text, encoding="utf-8")
    db = get_database()
    db.execute(
        """
        INSERT INTO documents (
            document_id, title, filename, format, size_bytes, import_state, created_at, updated_at
        ) VALUES (?, ?, 'report.txt', 'txt', ?, ?, ?, ?)
        """,
        (document_id, title, len(text.encode()), import_state, NOW, NOW),
    )
    db.execute(
        """
        INSERT INTO source_revisions (
            revision_id, document_id, source_sha256, extraction_version, canonical_path,
            page_map_json, created_at
        ) VALUES (?, ?, ?, 'text/1', ?, ?, ?)
        """,
        (
            revision_id,
            document_id,
            hashlib.sha256(text.encode()).hexdigest(),
            str(canonical),
            json.dumps(pages) if pages is not None else None,
            NOW,
        ),
    )
    return document_id, revision_id


def seed_run(
    document_id: str,
    revision_id: str,
    config: dict[str, object] | None = None,
    *,
    state: str = "queued",
    attempt_state: str | None = None,
    created_at: str = NOW,
    worker_pid: int | None = None,
    progress_json: str | None = None,
) -> str:
    run_id = str(uuid.uuid4())
    values = {"model": "scripted-model", **(config or {})}
    db = get_database()
    db.execute(
        """
        INSERT INTO runs (
            run_id, document_id, revision_id, state, strategy, config_json,
            idempotency_key, created_at, updated_at, progress_json
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
        """,
        (
            run_id,
            document_id,
            revision_id,
            state,
            values.get("strategy", "auto"),
            json.dumps(values),
            created_at,
            created_at,
            progress_json,
        ),
    )
    db.execute(
        """
        INSERT INTO run_attempts (attempt_id, run_id, attempt_number, state, started_at, worker_pid)
        VALUES (?, ?, 1, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            run_id,
            attempt_state or state,
            None if state == "queued" else created_at,
            worker_pid,
        ),
    )
    return run_id


def summary_payload(level: int, provenance: list[str]) -> dict[str, object]:
    return {
        "summary": f"The harbor summary at level {level}.",
        "content_units": [],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": provenance,
        "level": level,
    }


class ScriptedProvider:
    """Answers leaf, merge, and editorial requests like a cooperative model.

    `on_request(request)` runs first and may raise or return a replacement
    GenerationResult; `context_window` is what Ollama would report.
    """

    def __init__(
        self,
        *,
        context_window: int = 16_384,
        on_request: Callable[[GenerationRequest], GenerationResult | None] | None = None,
        on_configure: Callable[[], None] | None = None,
    ) -> None:
        self.context_window = context_window
        self.on_request = on_request
        self.on_configure = on_configure
        self.requests: list[GenerationRequest] = []

    def configure_context_window(
        self, model: str, requested: int | None, *, timeout_seconds: float
    ) -> int:
        if self.on_configure is not None:
            self.on_configure()
        return requested or self.context_window

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if self.on_request is not None:
            replacement = self.on_request(request)
            if replacement is not None:
                return replacement
        operation = request.operation_id or ""
        if operation == "editorial-final":
            payload: dict[str, object] = {"text": "The harbor moved record cargo this year."}
        elif operation.startswith(("S", "D")):
            payload = summary_payload(0, [operation])
        else:
            level = int(operation.rsplit("L", 1)[1])
            identifiers = re.findall(r'"segment_id":"([SD]\d+)"', request.input_text)
            payload = summary_payload(level, [identifiers[-1]])
        return GenerationResult(json.dumps(payload), "scripted", request.model, 1, 1, "stop")

    def operations(self) -> list[str]:
        return [request.audit_work_id or request.operation_id or "" for request in self.requests]
