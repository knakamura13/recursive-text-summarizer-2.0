"""GET/PATCH /settings: default run config merge patches and Ollama host validation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from summarizer_web.db.connection import get_database
from summarizer_web.models.api import RunConfig
from tests.support.web_views import csrf_headers, web_client


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with web_client(tmp_path, monkeypatch) as test_client:
        yield test_client


def _patch(client: TestClient, body: dict):
    return client.patch("/api/v1/settings", json=body, headers=csrf_headers(client))


def test_fresh_install_reports_the_local_host_and_default_config(client: TestClient) -> None:
    response = client.get("/api/v1/settings")

    assert response.status_code == 200
    assert response.json() == {
        "ollama_host": "http://localhost:11434",
        "defaults": RunConfig().model_dump(mode="json"),
    }


def test_patches_merge_into_stored_defaults(client: TestClient) -> None:
    first = _patch(client, {"defaults": {"model": "llama3.2:3b", "target_words": 500, "context_window": 16384}})
    second = _patch(client, {"defaults": {"verify": False, "model": None}})

    assert first.status_code == 200 and second.status_code == 200
    defaults = second.json()["defaults"]
    assert (defaults["model"], defaults["target_words"], defaults["context_window"], defaults["verify"]) == (
        "llama3.2:3b",
        500,
        16384,
        False,
    )
    assert client.get("/api/v1/settings").json() == second.json()


def test_clear_resets_nullable_fields(client: TestClient) -> None:
    _patch(client, {"defaults": {"context_window": 16384, "chunk_tokens": 2048, "max_merge_children": 8}})

    response = _patch(client, {"defaults": {"clear": ["context_window", "max_merge_children"]}})

    defaults = response.json()["defaults"]
    assert (defaults["context_window"], defaults["chunk_tokens"], defaults["max_merge_children"]) == (
        None,
        2048,
        None,
    )


def test_setting_and_clearing_the_same_field_is_rejected(client: TestClient) -> None:
    response = _patch(client, {"defaults": {"context_window": 16384, "clear": ["context_window"]}})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert client.get("/api/v1/settings").json()["defaults"]["context_window"] is None


def test_a_merge_that_breaks_cross_field_rules_stores_nothing(client: TestClient) -> None:
    _patch(client, {"defaults": {"chunk_tokens": 1024, "overlap_tokens": 200}})

    response = _patch(client, {"defaults": {"chunk_tokens": 200, "model": "qwen3:8b"}})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_request"
    assert "overlap_tokens" in body["message"]
    stored = client.get("/api/v1/settings").json()["defaults"]
    assert (stored["chunk_tokens"], stored["model"]) == (1024, "")


def test_out_of_range_patch_values_are_invalid_requests(client: TestClient) -> None:
    response = _patch(client, {"defaults": {"target_words": 5}})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


@pytest.mark.parametrize(
    "host",
    ["localhost:11434", "ftp://localhost:11434", "http://", "http://local host:11434", "http://localhost:99999", ""],
)
def test_ollama_host_must_be_an_http_url(client: TestClient, host: str) -> None:
    response = _patch(client, {"ollama_host": host})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert client.get("/api/v1/settings").json()["ollama_host"] == "http://localhost:11434"


def test_ollama_host_is_stored_without_a_trailing_slash(client: TestClient) -> None:
    response = _patch(client, {"ollama_host": " https://gpu-box.local:11434/ "})

    assert response.status_code == 200
    assert response.json()["ollama_host"] == "https://gpu-box.local:11434"
    assert client.get("/api/v1/settings").json()["ollama_host"] == "https://gpu-box.local:11434"


def test_stored_defaults_from_older_versions_keep_their_valid_fields(client: TestClient) -> None:
    get_database().execute(
        "INSERT INTO settings (key, value_json) VALUES ('defaults', ?)",
        (json.dumps({"model": "llama3.2:3b", "target_words": 10, "include_citations": True, "verify": False}),),
    )

    defaults = client.get("/api/v1/settings").json()["defaults"]

    assert (defaults["model"], defaults["verify"]) == ("llama3.2:3b", False)
    assert defaults["target_words"] == RunConfig().target_words
