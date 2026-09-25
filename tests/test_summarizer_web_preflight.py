"""POST /preflight: coded errors instead of rejections, the worker's context window, and estimates."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from summarizer.ingestion import SourceDocument
from summarizer.segmentation import SegmentationConfig, segment_document
from summarizer.tokenization import resolve_token_counter
from tests.support.fake_ollama import FakeOllama
from tests.support.web_views import csrf_headers, seed_document, web_client

MODEL = "llama3.2:3b"
SHORT_TEXT = "The bridge deck spans 200 feet across the river. Engineers inspected it in 1998."
# About 33 KB: more than a 32768-token window can take in one direct request.
LONG_TEXT = "\n\n".join(
    f"Paragraph {index}. The inspection team measured the north pier and recorded cracks "
    f"along the eastern face in section {index}. Repairs were scheduled for the following spring."
    for index in range(200)
)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with web_client(tmp_path, monkeypatch) as test_client:
        yield test_client


@pytest.fixture()
def ollama(monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
    return FakeOllama([MODEL], context_lengths={MODEL: 131_072}).install(monkeypatch)


def _preflight(client: TestClient, document_id: str = "doc-1", **config: object) -> dict:
    response = client.post(
        "/api/v1/preflight",
        json={"document_id": document_id, "config": {"model": MODEL, **config}},
        headers=csrf_headers(client),
    )
    assert response.status_code == 200
    return response.json()


def _codes(notices: list[dict]) -> list[str]:
    return [notice["code"] for notice in notices]


def test_short_document_goes_direct_with_the_model_window(
    client: TestClient, ollama: FakeOllama
) -> None:
    seed_document(text=SHORT_TEXT)

    plain = _preflight(client, verify=False)
    verified = _preflight(client, verify=True)

    assert plain["ok"] is True and plain["errors"] == [] and plain["warnings"] == []
    assert plain["selected_strategy"] == "direct"
    # Worker rule: min(model maximum, 32768) when no window is configured.
    assert (plain["context_window_tokens"], plain["context_window_source"]) == (32_768, "model")
    assert plain["model_installed"] is True
    assert plain["document_tokens"] == len(SHORT_TEXT.encode("utf-8"))
    assert plain["usable_input_capacity"] > plain["document_tokens"]
    # One direct summary plus the editorial call.
    assert (plain["estimated_leaf_count"], plain["estimated_model_calls"]) == (1, 2)
    assert verified["estimated_model_calls"] > plain["estimated_model_calls"]


def test_long_document_estimates_follow_the_pipeline_segmentation(
    client: TestClient, ollama: FakeOllama
) -> None:
    document = seed_document(text=LONG_TEXT)
    source = SourceDocument(text=document.text, source_id=document.source_sha256)
    counter = resolve_token_counter(provider="ollama", model=MODEL)

    fine = _preflight(client, verify=False, chunk_tokens=2000)
    coarse = _preflight(client, verify=False, chunk_tokens=4000)

    assert fine["ok"] is True and fine["selected_strategy"] == "hierarchical"
    assert fine["estimated_leaf_count"] == len(
        segment_document(source, counter, SegmentationConfig(max_tokens=2000))
    )
    assert coarse["estimated_leaf_count"] < fine["estimated_leaf_count"]
    for result in (fine, coarse):
        leaves = result["estimated_leaf_count"]
        # Leaves, at least one and at most leaves - 1 merges, and the editorial call.
        assert leaves + 2 <= result["estimated_model_calls"] <= 2 * leaves


def test_invalid_config_comes_back_as_coded_errors(client: TestClient, ollama: FakeOllama) -> None:
    seed_document(text=SHORT_TEXT)

    too_short = _preflight(client, target_words=5)
    overlap = _preflight(client, chunk_tokens=256, overlap_tokens=300)
    unknown = _preflight(client, bogus=True)

    assert too_short["ok"] is False
    assert _codes(too_short["errors"]) == ["invalid_config"]
    assert "Target words" in too_short["errors"][0]["message"]
    assert too_short["model_installed"] is True
    assert "overlap_tokens must be smaller than chunk_tokens" in overlap["errors"][0]["message"]
    assert _codes(unknown["errors"]) == ["invalid_config"]
    assert "bogus" in unknown["errors"][0]["message"]


def test_a_missing_model_is_required_before_anything_is_checked(
    client: TestClient, ollama: FakeOllama
) -> None:
    seed_document(text=SHORT_TEXT)

    result = _preflight(client, model="  ")

    assert result["ok"] is False
    assert _codes(result["errors"]) == ["model_required"]
    assert ollama.requests == []


def test_models_that_are_not_installed_block_the_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOllama(["qwen3:8b", "llama3.2:latest"], context_lengths={"llama3.2": 8192}).install(
        monkeypatch
    )
    seed_document(text=SHORT_TEXT)

    missing = _preflight(client, model=MODEL)
    untagged = _preflight(client, model="llama3.2")

    assert missing["ok"] is False
    assert _codes(missing["errors"]) == ["model_not_installed"]
    assert f"ollama pull {MODEL}" in missing["errors"][0]["message"]
    assert missing["model_installed"] is False
    assert missing["context_window_source"] == "assumed"
    # An untagged name means :latest, as Ollama resolves it.
    assert (untagged["ok"], untagged["model_installed"]) == (True, True)
    assert [body["model"] for _, body in fake.paths("/api/show")] == ["llama3.2"]


def test_unreachable_ollama_is_a_warning_with_the_assumed_window(
    client: TestClient, ollama: FakeOllama
) -> None:
    seed_document(text=SHORT_TEXT)
    ollama.unreachable = True

    result = _preflight(client)

    assert result["ok"] is True
    assert _codes(result["warnings"]) == ["ollama_unreachable"]
    assert result["warnings"][0]["severity"] == "warning"
    assert result["model_installed"] is None
    assert (result["context_window_tokens"], result["context_window_source"]) == (8192, "assumed")
    # An assumed window is never trusted for a direct request.
    assert result["selected_strategy"] == "hierarchical"
    assert len(ollama.requests) == 1


def test_a_model_without_context_length_requires_an_explicit_window(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeOllama([MODEL], context_lengths={MODEL: None}).install(monkeypatch)
    seed_document(text=SHORT_TEXT)

    unknown = _preflight(client)
    configured = _preflight(client, context_window=8192)

    assert unknown["ok"] is False
    assert _codes(unknown["errors"]) == ["context_window_unknown"]
    assert unknown["errors"][0]["severity"] == "error"
    assert "Set Context window in Advanced" in unknown["errors"][0]["message"]
    assert unknown["selected_strategy"] is None
    assert unknown["estimated_model_calls"] is None
    assert configured["ok"] is True
    assert configured["context_window_source"] == "configured"
    assert configured["context_window_tokens"] == 8192


def test_document_readiness_is_checked(client: TestClient, ollama: FakeOllama) -> None:
    seed_document(document_id="doc-importing", text=None, import_state="importing")
    seed_document(
        document_id="doc-failed",
        text=None,
        import_state="failed",
        import_error="The PDF has no text layer and Tesseract is not installed.",
    )

    missing = _preflight(client, document_id="doc-missing")
    importing = _preflight(client, document_id="doc-importing")
    failed = _preflight(client, document_id="doc-failed")

    assert _codes(missing["errors"]) == ["document_not_found"]
    assert _codes(importing["errors"]) == ["document_not_ready"]
    assert _codes(failed["errors"]) == ["document_not_ready"]
    assert "Tesseract is not installed" in failed["errors"][0]["message"]
    assert not any(result["ok"] for result in (missing, importing, failed))


def test_a_configured_window_above_the_model_maximum_is_invalid(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeOllama([MODEL], context_lengths={MODEL: 8192}).install(monkeypatch)
    seed_document(text=SHORT_TEXT)

    too_large = _preflight(client, context_window=16_384)
    fits = _preflight(client, context_window=8192)

    assert too_large["ok"] is False
    assert _codes(too_large["errors"]) == ["invalid_config"]
    assert "8192" in too_large["errors"][0]["message"]
    assert too_large["context_window_source"] == "configured"
    assert (fits["ok"], fits["context_window_tokens"], fits["context_window_source"]) == (
        True,
        8192,
        "configured",
    )


def test_budget_failures_block_the_run(client: TestClient, ollama: FakeOllama) -> None:
    seed_document(text=LONG_TEXT)

    direct = _preflight(client, strategy="direct")
    tiny_window = _preflight(client, context_window=8192)

    assert direct["ok"] is False
    assert _codes(direct["errors"]) == ["budget"]
    assert direct["selected_strategy"] is None
    # 8192 tokens leave leaves room but not the merge instructions.
    assert _codes(tiny_window["errors"]) == ["budget"]
    assert tiny_window["selected_strategy"] == "hierarchical"
    assert tiny_window["estimated_leaf_count"] is None


def test_context_length_is_fetched_once_per_host_and_model(
    client: TestClient, ollama: FakeOllama
) -> None:
    seed_document(text=SHORT_TEXT)

    _preflight(client, target_words=200)
    _preflight(client, target_words=250)
    client.patch(
        "/api/v1/settings", json={"ollama_host": "http://gpu-box:11434"}, headers=csrf_headers(client)
    )
    moved = _preflight(client)

    assert [host for host, _ in ollama.paths("/api/show")] == [
        "http://localhost:11434",
        "http://gpu-box:11434",
    ]
    assert moved["context_window_source"] == "model"


def test_a_request_without_a_document_id_is_invalid(client: TestClient) -> None:
    response = client.post(
        "/api/v1/preflight", json={"config": {"model": MODEL}}, headers=csrf_headers(client)
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
