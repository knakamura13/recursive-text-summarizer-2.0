import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from summarizer_web.main import create_app

FIXTURE = Path(__file__).parent / "fixtures" / "article.txt"


@pytest.fixture()
def client(tmp_path: Path):
    os.environ["SUMMARIZER_DATA_DIR"] = str(tmp_path)
    app = create_app()
    with TestClient(app) as test_client:
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
    assert response.status_code == 200
    document = response.json()["documents"][0]
    document_id = document["document_id"]

    listed = client.get("/api/v1/documents")
    assert listed.status_code == 200
    assert any(item["document_id"] == document_id for item in listed.json()["documents"])

    source = client.get(f"/api/v1/documents/{document_id}/source")
    assert source.status_code == 200
    assert source.json()["total_length"] > 0
