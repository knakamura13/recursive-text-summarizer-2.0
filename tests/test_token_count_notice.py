"""A fallback to the byte counter is never silent (#190).

The byte counter counts one token per byte, several times what the model
counts, so budgets shrink and strategy and capacity verdicts move. Preflight,
the Run and the command line each say so, and the reason is stated where there
is room for it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import summarizer_web.worker.runner as runner
from summarizer.cli import main
from summarizer.tokenization import (
    ConservativeUtf8TokenCounter,
    TokenizerJsonCounter,
    estimate_notice,
)
from summarizer_web.config import load_paths
from summarizer_web.services import preflight_service
from summarizer_web.services.events_service import start_attempt
from tests.support.fake_ollama import FakeOllama
from tests.support.web_runs import TEXT, ScriptedProvider, seed_document as seed_run_document, seed_run
from tests.support.web_views import csrf_headers, seed_document, web_client
from tests.test_cli import CharacterCounter, RecordingProvider
from tests.test_tokenizer_json_counter import _tensor_model

MODEL = "llama3.2:3b"
TENSOR_MODEL = "big:31b"
CODE = "token_counts_estimated"
SHORT_TEXT = "The bridge deck spans 200 feet across the river. Engineers inspected it in 1998."
LONG_TEXT = "\n\n".join(
    f"Paragraph {index}. The inspection team measured the north pier and recorded cracks "
    f"along the eastern face in section {index}. Repairs were scheduled for the following spring."
    for index in range(200)
)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with web_client(tmp_path, monkeypatch) as test_client:
        yield test_client


def _preflight(client: TestClient, model: str = MODEL, **config: object) -> dict:
    response = client.post(
        "/api/v1/preflight",
        json={"document_id": "doc-1", "config": {"model": model, **config}},
        headers=csrf_headers(client),
    )
    assert response.status_code == 200
    return response.json()


def _notices(result: dict, code: str = CODE) -> list[dict]:
    return [notice for notice in result["warnings"] if notice["code"] == code]


# --- preflight -------------------------------------------------------------


def test_preflight_warns_when_the_byte_counter_is_used(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeOllama([MODEL], context_lengths={MODEL: 131_072}).install(monkeypatch)
    seed_document(text=SHORT_TEXT)

    result = _preflight(client, verify=False)

    [notice] = _notices(result)
    assert result["ok"] is True
    assert "estimate:utf8-bytes" in notice["message"]
    assert "segments may be smaller than the context window requires" in notice["message"]
    # No model file on this machine for the reason to name.
    assert "no local Ollama model file" in notice["message"]
    # The estimate still comes back; the warning does not replace it.
    assert result["selected_strategy"] == "direct" and result["document_tokens"] is not None


def test_preflight_keeps_the_warning_when_the_budget_refuses_the_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The refusal is where an overcount hurts most, so the warning must survive it.
    FakeOllama([MODEL], context_lengths={MODEL: 131_072}).install(monkeypatch)
    seed_document(text=LONG_TEXT)

    result = _preflight(client, strategy="direct")

    assert result["ok"] is False
    assert [error["code"] for error in result["errors"]] == ["budget"]
    assert len(_notices(result)) == 1


def test_preflight_names_an_unverified_tensor_layout_tokenizer(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeOllama([TENSOR_MODEL], context_lengths={TENSOR_MODEL: 131_072}).install(monkeypatch)
    models = tmp_path / "ollama"
    monkeypatch.setenv("OLLAMA_MODELS", str(models))
    _tensor_model(models)
    fingerprint = TokenizerJsonCounter.for_ollama_model(
        TENSOR_MODEL, allow_unverified=True
    ).fingerprint
    seed_document(text=SHORT_TEXT)

    [notice] = _notices(_preflight(client, TENSOR_MODEL, verify=False))

    assert "has not been checked against Ollama" in notice["message"]
    assert fingerprint[:12] in notice["message"]


def test_preflight_names_a_host_whose_model_files_it_cannot_read(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeOllama([MODEL], context_lengths={MODEL: 131_072}).install(monkeypatch)
    client.patch(
        "/api/v1/settings", json={"ollama_host": "http://gpu-box:11434"}, headers=csrf_headers(client)
    )
    seed_document(text=SHORT_TEXT)

    [notice] = _notices(_preflight(client, verify=False))

    assert "not on this machine" in notice["message"]


def test_preflight_has_no_warning_when_the_counter_is_exact(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeOllama([MODEL], context_lengths={MODEL: 131_072}).install(monkeypatch)
    monkeypatch.setattr(
        preflight_service, "resolve_token_counter", lambda **_: CharacterCounter()
    )
    seed_document(text=SHORT_TEXT)

    result = _preflight(client, verify=False)

    assert result["ok"] is True and result["warnings"] == []


# --- the Run ---------------------------------------------------------------


def _complete_run(monkeypatch: pytest.MonkeyPatch, *, exact: bool) -> str:
    if exact:
        monkeypatch.setattr(runner, "build_counter", lambda app: CharacterCounter())
    monkeypatch.setattr(runner, "build_provider", lambda app: ScriptedProvider())
    document_id, revision_id = seed_run_document(TEXT)
    run_id = seed_run(
        document_id,
        revision_id,
        {"strategy": "hierarchical", "chunk_tokens": 300, "max_merge_children": 2,
         "verify": False, "citations": False, "target_words": 40},
    )
    assert start_attempt(run_id) is not None
    assert runner.run_job(run_id) == "completed"
    return run_id


def _audit_warnings(run_id: str) -> list[str]:
    audit = json.loads((load_paths().runs / run_id / "audit.json").read_text(encoding="utf-8"))
    return audit["warnings"]


def test_a_run_on_the_byte_counter_records_and_shows_the_notice(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = _complete_run(monkeypatch, exact=False)

    assert CODE in _audit_warnings(run_id)
    summary = client.get(f"/api/v1/runs/{run_id}/summary").json()
    [notice] = [item for item in summary["notices"] if item["code"] == CODE]
    assert notice["severity"] == "info"
    assert "estimates" in notice["message"]
    assert "segments may be smaller than the context window requires" in notice["message"]


def test_a_run_on_an_exact_counter_carries_no_notice(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = _complete_run(monkeypatch, exact=True)

    assert CODE not in _audit_warnings(run_id)
    summary = client.get(f"/api/v1/runs/{run_id}/summary").json()
    assert all(item["code"] != CODE for item in summary["notices"])


# --- the command line ------------------------------------------------------


def _write_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("Source.", encoding="utf-8")


def test_the_command_line_prints_the_notice_to_stderr_not_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_input(tmp_path, monkeypatch)
    counter = ConservativeUtf8TokenCounter(reason="the host is elsewhere")

    assert main(["--dry-run"], counter_factory=lambda _config: counter) == 0

    captured = capsys.readouterr()
    notice = estimate_notice(counter)
    assert notice is not None and notice in captured.err
    assert "estimates" not in captured.out


def test_a_real_run_prints_the_notice_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_input(tmp_path, monkeypatch)

    exit_code = main(
        [],
        provider_factory=lambda _config: RecordingProvider("Concise summary."),
        counter_factory=lambda _config: ConservativeUtf8TokenCounter(),
    )

    assert exit_code == 0
    assert capsys.readouterr().err.count("Token counts are estimates") == 1


def test_an_exact_counter_prints_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_input(tmp_path, monkeypatch)

    exit_code = main(
        [],
        provider_factory=lambda _config: RecordingProvider("Concise summary."),
        counter_factory=lambda _config: CharacterCounter(),
    )

    assert exit_code == 0
    assert capsys.readouterr().err == ""
