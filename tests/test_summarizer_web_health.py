import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from summarizer_web.config import load_paths
from summarizer_web.main import create_app


@pytest.fixture()
def client(tmp_path: Path):
    os.environ["SUMMARIZER_DATA_DIR"] = str(tmp_path)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "summarizer_session" in response.cookies
