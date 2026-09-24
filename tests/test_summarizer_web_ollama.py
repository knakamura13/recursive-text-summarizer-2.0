"""GET /ollama/health and /ollama/models against a scripted Ollama API."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from summarizer_web.services.ollama_service import OllamaError, model_context_length
from tests.support.fake_ollama import FakeOllama
from tests.support.web_views import csrf_headers, web_client


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with web_client(tmp_path, monkeypatch) as test_client:
        yield test_client


def test_health_reports_the_version_of_the_configured_host(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOllama(version="0.12.3").install(monkeypatch)
    client.patch(
        "/api/v1/settings", json={"ollama_host": "http://gpu-box:11434"}, headers=csrf_headers(client)
    )

    body = client.get("/api/v1/ollama/health").json()

    assert body["connected"] is True
    assert body["version"] == "0.12.3"
    assert body["host"] == "http://gpu-box:11434"
    assert "0.12.3" in body["message"]
    assert [host for host, _ in fake.paths("/api/version")] == ["http://gpu-box:11434"]



def test_health_checks_a_validated_unsaved_host_without_changing_settings(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOllama(version="0.12.3").install(monkeypatch)

    response = client.get("/api/v1/ollama/health", params={"host": "http://unsaved-box:11434/"})

    assert response.status_code == 200
    assert response.json()["connected"] is True
    assert response.json()["host"] == "http://unsaved-box:11434"
    assert fake.paths("/api/version") == [("http://unsaved-box:11434", None)]
    assert client.get("/api/v1/settings").json()["ollama_host"] == "http://localhost:11434"


def test_health_rejects_an_invalid_unsaved_host_with_api_error_shape(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOllama().install(monkeypatch)

    response = client.get("/api/v1/ollama/health", params={"host": "ftp://unsaved-box:11434"})

    assert response.status_code == 422
    assert response.json() == {
        "code": "invalid_request",
        "message": "The Ollama host must be an http:// or https:// URL, for example http://localhost:11434.",
        "details": {"field": "ollama_host"},
        "retryable": False,
    }
    assert fake.requests == []
def test_health_explains_an_unreachable_host(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOllama().install(monkeypatch)
    fake.unreachable = True

    response = client.get("/api/v1/ollama/health")

    assert response.status_code == 200
    body = response.json()
    assert (body["connected"], body["version"], body["host"]) == (False, None, "http://localhost:11434")
    assert "http://localhost:11434" in body["message"] and "ollama serve" in body["message"]


def test_models_list_details_sorted_by_name(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOllama(
        [
            {
                "name": "qwen3:8b",
                "model": "qwen3:8b",
                "size": 5_200_000_000,
                "modified_at": "2026-09-01T10:00:00Z",
                "details": {"family": "qwen3", "parameter_size": "8.2B", "quantization_level": "Q4_K_M"},
            },
            {"name": "llama3.2:3b", "model": "llama3.2:3b", "size": 2_019_393_189, "details": {}},
            {"model": ""},
        ]
    ).install(monkeypatch)

    response = client.get("/api/v1/ollama/models")

    assert response.status_code == 200
    assert response.json()["models"] == [
        {
            "name": "llama3.2:3b",
            "size_bytes": 2_019_393_189,
            "parameter_size": None,
            "family": None,
            "quantization": None,
            "modified_at": None,
        },
        {
            "name": "qwen3:8b",
            "size_bytes": 5_200_000_000,
            "parameter_size": "8.2B",
            "family": "qwen3",
            "quantization": "Q4_K_M",
            "modified_at": "2026-09-01T10:00:00Z",
        },
    ]


@pytest.mark.parametrize("failure", ["unreachable", "server_error"])
def test_models_fail_with_ollama_unreachable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    fake = FakeOllama(["llama3.2:3b"]).install(monkeypatch)
    if failure == "unreachable":
        fake.unreachable = True
    else:
        fake.status_override = 500

    response = client.get("/api/v1/ollama/models")

    assert response.status_code == 503
    body = response.json()
    assert (body["code"], body["retryable"]) == ("ollama_unreachable", True)
    assert body["details"] == {"host": "http://localhost:11434"}


def test_context_lengths_are_cached_per_host_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOllama(context_lengths={"llama3.2:3b": 131_072, "qwen3:8b": 40_960}).install(monkeypatch)

    assert model_context_length("http://a:11434", "llama3.2:3b") == 131_072
    assert model_context_length("http://a:11434", "llama3.2:3b") == 131_072
    assert model_context_length("http://a:11434", "qwen3:8b") == 40_960
    assert model_context_length("http://b:11434", "llama3.2:3b") == 131_072

    assert [(host, body["model"]) for host, body in fake.paths("/api/show")] == [
        ("http://a:11434", "llama3.2:3b"),
        ("http://a:11434", "qwen3:8b"),
        ("http://b:11434", "llama3.2:3b"),
    ]


def test_failed_context_lookups_are_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOllama(context_lengths={"llama3.2:3b": None}).install(monkeypatch)

    with pytest.raises(OllamaError, match="context length"):
        model_context_length("http://a:11434", "llama3.2:3b")
    fake.context_lengths["llama3.2:3b"] = 8192

    assert model_context_length("http://a:11434", "llama3.2:3b") == 8192
    assert len(fake.paths("/api/show")) == 2
