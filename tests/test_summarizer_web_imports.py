from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database
from summarizer_web.main import create_app
from summarizer_web.worker.imports import import_document


class _ImportQueue:
    def __init__(self) -> None:
        self.enqueued: list[str] = []
        self.cancelled: list[str] = []

    def enqueue(self, document_id: str) -> None:
        self.enqueued.append(document_id)

    def cancel(self, document_id: str) -> None:
        self.cancelled.append(document_id)


@pytest.fixture()
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SUMMARIZER_DATA_DIR", str(tmp_path))
    queue = _ImportQueue()
    monkeypatch.setattr("summarizer_web.services.documents_service.get_import_manager", lambda: queue)
    with TestClient(create_app()) as client:
        health = client.get("/api/v1/health")
        headers = {"X-CSRF-Token": health.headers["X-CSRF-Token"]}
        yield client, headers, queue


def _upload(client: TestClient, headers: dict[str, str], filename: str, data: bytes):
    return client.post(
        "/api/v1/documents",
        files={"file": (filename, data, "application/octet-stream")},
        headers=headers,
    )


def test_upload_is_queued_then_imported_with_dedup_and_source_slices(app_client) -> None:
    client, headers, queue = app_client
    original = ("Each paragraph contains enough readable document text.\n" * 24_000).encode()
    created = _upload(client, headers, "large.txt", original)

    assert created.status_code == 201
    document = created.json()["document"]
    document_id = document["document_id"]
    assert document["import_state"] == "importing"
    assert document["import_progress"]["phase"] == "queued"
    assert queue.enqueued == [document_id]

    upload_path = load_paths().documents / document_id / "original.txt"
    assert upload_path.stat().st_size == len(original)
    import_document(get_database(), load_paths(), document_id)

    detail = client.get(f"/api/v1/documents/{document_id}").json()
    assert detail["import_state"] == "ready"
    assert detail["import_report"]["extraction_version"] == "import/3" and detail["import_report"]["char_count"] > 1_000_000
    start, limit = 31_000, 113
    source = client.get(
        f"/api/v1/documents/{document_id}/source", params={"offset": start, "limit": limit}
    )
    assert source.status_code == 200
    canonical = (load_paths().documents / document_id / "canonical.txt").read_text()
    assert source.json()["text"] == canonical[start : start + limit]

    duplicate = _upload(client, headers, "renamed.txt", original)
    assert duplicate.status_code == 200
    assert duplicate.json()["already_imported"] is True
    assert duplicate.json()["document"]["document_id"] == document_id
    assert queue.enqueued == [document_id]


def test_deleting_importing_document_stops_import_and_removes_upload(app_client) -> None:
    client, headers, queue = app_client
    created = _upload(client, headers, "cancel-me.txt", b"Import this text in the background.")
    document_id = created.json()["document"]["document_id"]
    original = load_paths().documents / document_id / "original.txt"
    assert original.exists()

    deleted = client.delete(f"/api/v1/documents/{document_id}", headers=headers)

    assert deleted.status_code == 204
    assert queue.cancelled == [document_id]
    assert not original.exists()
    assert client.get(f"/api/v1/documents/{document_id}").json()["code"] == "document_not_found"


def test_import_with_no_usable_text_fails_with_reason(app_client) -> None:
    client, headers, queue = app_client
    created = _upload(client, headers, "blank.txt", b"  \n\t  ")
    document_id = created.json()["document"]["document_id"]
    assert created.status_code == 201
    assert queue.enqueued == [document_id]

    import_document(get_database(), load_paths(), document_id)

    failed = client.get(f"/api/v1/documents/{document_id}").json()
    assert failed["import_state"] == "failed"
    assert "no text" in failed["import_error"].lower()


def test_empty_and_unsupported_uploads_report_actionable_errors(app_client) -> None:
    client, headers, _queue = app_client
    empty = _upload(client, headers, "empty.txt", b"")
    unsupported = _upload(client, headers, "unknown.bin", bytes((0, 1, 2)))

    assert (empty.status_code, empty.json()["code"]) == (400, "empty_file")
    assert (unsupported.status_code, unsupported.json()["code"]) == (415, "unsupported_format")
    assert ".pdf" in unsupported.json()["details"]["supported"]
    assert "Supported formats" in unsupported.json()["message"]


def test_upload_limit_is_enforced_while_file_bytes_arrive(app_client, monkeypatch) -> None:
    from summarizer_web.services import documents_service

    client, headers, _queue = app_client
    monkeypatch.setattr(documents_service, "MAX_UPLOAD_BYTES", 32)

    response = _upload(client, headers, "large.txt", b"x" * 100)

    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"
    assert response.json()["details"]["limit_bytes"] == 32
