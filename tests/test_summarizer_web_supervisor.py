import os
from pathlib import Path

import pytest

from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database, init_database
from summarizer_web.worker.supervisor import RunSupervisor


def _prepare_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUMMARIZER_DATA_DIR", str(tmp_path))
    init_database(load_paths())
    now = "2026-09-23T00:00:00+00:00"
    db = get_database()
    db.execute(
        """
        INSERT INTO documents (
            document_id, title, filename, format, size_bytes, import_state, created_at, updated_at
        ) VALUES ('doc-1', 'Sample', 'sample.txt', 'txt', 10, 'ready', ?, ?)
        """,
        (now, now),
    )
    db.execute(
        """
        INSERT INTO source_revisions (
            revision_id, document_id, source_sha256, extraction_version, canonical_path, created_at
        ) VALUES ('rev-1', 'doc-1', ?, 'text/1', 'canonical.txt', ?)
        """,
        ("a" * 64, now),
    )


def _insert_run(run_id: str, state: str) -> None:
    now = "2026-09-23T00:00:00+00:00"
    db = get_database()
    db.execute(
        """
        INSERT INTO runs (
            run_id, document_id, revision_id, state, strategy, config_json,
            idempotency_key, created_at, updated_at
        ) VALUES (?, 'doc-1', 'rev-1', ?, 'direct', '{}', ?, ?, ?)
        """,
        (run_id, state, run_id, now, now),
    )
    db.execute(
        """
        INSERT INTO run_attempts (attempt_id, run_id, attempt_number, state, started_at)
        VALUES (?, ?, 1, ?, ?)
        """,
        (f"attempt-{run_id}", run_id, state, now),
    )


def test_enqueue_keeps_a_new_run_active_after_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_database(tmp_path, monkeypatch)
    supervisor = RunSupervisor()
    monkeypatch.setattr(supervisor, "_start_process", lambda run_id, resume=False: None)
    supervisor.start()
    _insert_run("run-new", "queued")

    supervisor.enqueue("run-new")

    run = get_database().fetchone("SELECT state FROM runs WHERE run_id = 'run-new'")
    attempt = get_database().fetchone(
        "SELECT state, failure_reason FROM run_attempts WHERE run_id = 'run-new'"
    )
    assert run["state"] == "queued"
    assert attempt["state"] == "queued"
    assert attempt["failure_reason"] is None


def test_enqueue_that_starts_the_supervisor_does_not_interrupt_its_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_database(tmp_path, monkeypatch)
    _insert_run("run-new", "queued")
    _insert_run("run-old", "running")
    supervisor = RunSupervisor()
    monkeypatch.setattr(supervisor, "_start_process", lambda run_id, resume=False: None)

    supervisor.enqueue("run-new")

    new_run = get_database().fetchone("SELECT state FROM runs WHERE run_id = 'run-new'")
    old_run = get_database().fetchone("SELECT state FROM runs WHERE run_id = 'run-old'")
    assert new_run["state"] == "queued"
    assert old_run["state"] == "interrupted"


def test_startup_still_interrupts_an_orphaned_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_database(tmp_path, monkeypatch)
    _insert_run("run-old", "running")
    supervisor = RunSupervisor()
    monkeypatch.setattr(supervisor, "_start_process", lambda run_id, resume=False: None)

    supervisor.start()

    run = get_database().fetchone("SELECT state FROM runs WHERE run_id = 'run-old'")
    attempt = get_database().fetchone(
        "SELECT state, failure_reason FROM run_attempts WHERE run_id = 'run-old'"
    )
    assert run["state"] == "interrupted"
    assert attempt["state"] == "interrupted"
    assert attempt["failure_reason"] == "Application restarted"
