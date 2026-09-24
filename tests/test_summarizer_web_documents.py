from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database
from summarizer_web.main import create_app
from summarizer_web.worker.imports import import_document

FIXTURE = Path(__file__).parent / "fixtures" / "article.txt"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SUMMARIZER_DATA_DIR", str(tmp_path))
    queue = SimpleNamespace(enqueue=lambda _document_id: None, cancel=lambda _document_id: None)
    monkeypatch.setattr("summarizer_web.services.documents_service.get_import_manager", lambda: queue)
    with TestClient(create_app()) as test_client:
        yield test_client


def _session_headers(client: TestClient) -> dict[str, str]:
    health = client.get("/api/v1/health")
    token = health.headers.get("X-CSRF-Token", "")
    assert token
    return {"X-CSRF-Token": token}


def test_upload_list_and_source_slice(client: TestClient):
    headers = _session_headers(client)
    with FIXTURE.open("rb") as handle:
        response = client.post(
            "/api/v1/documents",
            files={"file": ("article.txt", handle, "text/plain")},
            headers=headers,
        )
    assert response.status_code == 201
    document = response.json()["document"]
    document_id = document["document_id"]
    assert document["import_state"] == "importing"

    import_document(get_database(), load_paths(), document_id)

    listed = client.get("/api/v1/documents")
    assert listed.status_code == 200
    assert any(item["document_id"] == document_id for item in listed.json()["documents"])

    canonical = (load_paths().documents / document_id / "canonical.txt").read_text()
    source = client.get(
        f"/api/v1/documents/{document_id}/source", params={"offset": 0, "limit": 64}
    )
    assert source.status_code == 200
    assert source.json()["total_length"] == len(canonical)
    assert source.json()["text"] == canonical[:64]
