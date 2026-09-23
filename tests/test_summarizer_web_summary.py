import json
import os
from pathlib import Path

import pytest

from summarizer_web.db.connection import get_database, init_database
from summarizer_web.models.api import RunConfig
from summarizer_web.services.runs_service import get_final_summary


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SUMMARIZER_DATA_DIR", str(tmp_path))
    init_database()
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
    config = RunConfig(target_words=80, model="test")
    db.execute(
        """
        INSERT INTO runs (
            run_id, document_id, revision_id, state, strategy, config_json,
            idempotency_key, created_at, updated_at
        ) VALUES ('run-1', 'doc-1', 'rev-1', 'completed', 'direct', ?, 'run-1', ?, ?)
        """,
        (config.model_dump_json(), now, now),
    )
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.txt").write_text(
        "One two three four five.\n\nSources: S000001",
        encoding="utf-8",
    )
    (run_dir / "audit.json").write_text(
        json.dumps({"warnings": ["verified_content_unit_fallback"]}),
        encoding="utf-8",
    )
    return tmp_path


def test_get_final_summary_reports_short_of_target(data_dir: Path) -> None:
    del data_dir
    response = get_final_summary("run-1")
    assert response.available is True
    assert response.word_count == 5
    assert response.target_words == 80
    assert response.short_of_target is True
    assert response.audit_warnings == ["verified_content_unit_fallback"]
