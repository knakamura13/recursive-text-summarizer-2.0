"""The Run worker in-process: real pipeline, scripted provider, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import summarizer_web.worker.runner as runner
from summarizer.budget import BudgetError
from summarizer.finalization import FinalizationVerificationError
from summarizer.providers.base import (
    GenerationResult,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderRetriesExhaustedError,
    ProviderServerError,
    ProviderTimeoutError,
    RetryAttempt,
    RetryErrorCategory,
)
from summarizer.runtime.observers import ItemFailedError, StageEvent, StageName
from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database
from summarizer_web.services.events_service import request_stop_flag, start_attempt
from tests.support.web_runs import TEXT, ScriptedProvider, page_map, seed_document, seed_run, use_data_dir

HIERARCHICAL = {
    "strategy": "hierarchical",
    "chunk_tokens": 300,
    "max_merge_children": 2,
    "verify": False,
    "citations": False,
    "target_words": 40,
}


@pytest.fixture
def paged_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    use_data_dir(tmp_path, monkeypatch)
    document_id, revision_id = seed_document(pages=page_map(TEXT, 400))
    return seed_run(document_id, revision_id, HIERARCHICAL)


def _use_provider(monkeypatch: pytest.MonkeyPatch, provider: ScriptedProvider) -> None:
    monkeypatch.setattr(runner, "build_provider", lambda app: provider)


def _execute(run_id: str) -> str | None:
    assert start_attempt(run_id) is not None
    return runner.run_job(run_id)


def _nodes(run_id: str) -> dict[str, dict]:
    rows = get_database().fetchall(
        "SELECT * FROM node_projections WHERE run_id = ? ORDER BY level, order_index", (run_id,)
    )
    return {row["node_id"]: dict(row) for row in rows}


def _item_events(run_id: str, work_id: str) -> list[tuple[int, str]]:
    rows = get_database().fetchall(
        "SELECT event_id, payload_json FROM run_events WHERE run_id = ? AND event_type = 'item' "
        "ORDER BY event_id",
        (run_id,),
    )
    events = [(row["event_id"], json.loads(row["payload_json"])) for row in rows]
    return [(event_id, payload["state"]) for event_id, payload in events if payload["work_id"] == work_id]


def _attempts(run_id: str) -> list[dict]:
    return [
        dict(row)
        for row in get_database().fetchall(
            "SELECT * FROM run_attempts WHERE run_id = ? ORDER BY attempt_number", (run_id,)
        )
    ]


def _run(run_id: str) -> dict:
    return dict(get_database().fetchone("SELECT * FROM runs WHERE run_id = ?", (run_id,)))


def test_completed_run_records_events_nodes_segments_and_progress(
    paged_run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = ScriptedProvider()
    _use_provider(monkeypatch, provider)

    assert _execute(paged_run) == "completed"

    run = _run(paged_run)
    assert run["state"] == "completed"
    assert run["selected_strategy"] == "hierarchical"
    [attempt] = _attempts(paged_run)
    assert attempt["state"] == "completed"
    assert attempt["failure_json"] is None

    segments = get_database().fetchall(
        "SELECT * FROM run_segments WHERE run_id = ? ORDER BY order_index", (paged_run,)
    )
    leaf_count = sum(1 for operation in provider.operations() if operation.startswith("S"))
    assert len(segments) == leaf_count > 2
    assert (segments[0]["page_start"], segments[-1]["page_end"]) == (1, 4)
    assert all(segment["core_start"] < segment["core_end"] for segment in segments)

    nodes = _nodes(paged_run)
    assert {node["state"] for node in nodes.values()} == {"completed"}
    leaf = nodes["L0N0001"]
    assert leaf["label"] == "Segment 1 · p. 1"
    assert leaf["summary_text"] == "The harbor summary at level 0."
    assert json.loads(leaf["covered_segment_ids_json"]) == ["S000001"]
    assert nodes["L0N0002"]["label"] == "Segment 2 · pp. 1–2"
    top_level = max(node["level"] for node in nodes.values())
    [root] = [node for node in nodes.values() if node["level"] == top_level]
    assert root["label"] == f"Root · Level {top_level}"
    assert root["parent_id"] is None
    assert all(
        get_database().fetchone(
            "SELECT event_type FROM run_events WHERE event_id = ?", (node["updated_event_id"],)
        )["event_type"]
        == "item"
        for node in nodes.values()
    )
    for node in nodes.values():
        for child_id in json.loads(node["child_ids_json"]):
            assert nodes[child_id]["parent_id"] == node["node_id"]
    assert all(node["parent_id"] for node in nodes.values() if node is not root)

    history = _item_events(paged_run, "L0N0001")
    assert [state for _, state in history] == ["planned", "active", "completed"]
    assert leaf["updated_event_id"] >= history[-1][0]

    progress = json.loads(run["progress_json"])
    assert progress["leaves"] == {"done": leaf_count, "total": leaf_count, "reused": 0, "failed": 0}
    assert progress["merges"]["done"] == progress["merges"]["total"] == len(nodes) - leaf_count
    stages = {stage["stage"]: stage for stage in progress["stages"]}
    assert stages["preparing"]["detail"] == "hierarchical"
    assert stages["summarizing"]["completed"] == leaf_count
    assert stages["publishing"]["state"] == "completed"
    assert progress["current_items"] == []

    run_dir = load_paths().runs / paged_run
    assert (run_dir / "summary.txt").read_text(encoding="utf-8").strip()
    assert (run_dir / "audit.json").exists()
    assert not (run_dir / "verification.json").exists()


def test_unreachable_ollama_fails_in_preparing_with_the_host(
    paged_run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> None:
        raise ProviderConnectionError("Ollama connection failed; confirm the service is running")

    _use_provider(monkeypatch, ScriptedProvider(on_configure=refuse))

    assert _execute(paged_run) == "failed"

    [attempt] = _attempts(paged_run)
    failure = json.loads(attempt["failure_json"])
    assert failure["code"] == "ollama_unreachable"
    assert failure["message"] == "Ollama is not reachable at http://localhost:11434."
    assert failure["stage"] == "preparing"
    assert "ollama serve" in failure["hint"]
    assert "Ollama connection failed" in failure["detail"]
    assert attempt["failure_reason"] == failure["message"]
    assert _run(paged_run)["state"] == "failed"
    written = json.loads((load_paths().runs / paged_run / "failure.json").read_text())
    assert written["code"] == "ollama_unreachable"


def test_missing_model_names_the_model_and_the_item(
    paged_run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(request):
        raise ProviderRequestError("Ollama model was not found; pull it before retrying")

    _use_provider(monkeypatch, ScriptedProvider(on_request=missing))

    assert _execute(paged_run) == "failed"

    failure = json.loads(_attempts(paged_run)[0]["failure_json"])
    assert failure["code"] == "model_not_found"
    assert failure["message"] == "Model scripted-model is not installed."
    assert failure["stage"] == "summarizing"
    assert failure["item"] == "Segment 1 · p. 1"
    assert "ollama pull scripted-model" in failure["hint"]
    nodes = _nodes(paged_run)
    assert nodes["L0N0001"]["state"] == "pending"
    assert "active" not in {node["state"] for node in nodes.values()}


def test_invalid_output_after_reasks_fails_naming_the_item(
    paged_run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid_second_leaf(request):
        if request.operation_id == "S000002":
            return GenerationResult("not json", "scripted", request.model)
        return None

    provider = ScriptedProvider(on_request=invalid_second_leaf)
    _use_provider(monkeypatch, provider)

    assert _execute(paged_run) == "failed"

    failure = json.loads(_attempts(paged_run)[0]["failure_json"])
    assert failure["code"] == "item_invalid_output"
    assert failure["item"] == "Segment 2 · pp. 1–2"
    assert failure["stage"] == "summarizing"
    assert failure["message"].startswith("Segment 2 · pp. 1–2 produced invalid output after 3 tries: ")
    assert provider.operations().count("S000002") == 3
    node = _nodes(paged_run)["L0N0002"]
    assert node["state"] == "failed"
    assert node["error"]
    assert [state for _, state in _item_events(paged_run, "L0N0002")][-1] == "failed"


def test_verification_failure_is_classified(paged_run: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_pipeline(document, provider, counter, *, app, strategy, config, observer):
        observer.emit(StageEvent(StageName.VERIFYING, "active", detail="Checking the editorial draft"))
        raise FinalizationVerificationError("verification closed without a safe final summary")

    _use_provider(monkeypatch, ScriptedProvider())
    monkeypatch.setattr(runner, "run_pipeline", failing_pipeline)

    assert _execute(paged_run) == "failed"

    failure = json.loads(_attempts(paged_run)[0]["failure_json"])
    assert failure["code"] == "verification_failed"
    assert failure["message"] == "Verification could not confirm any sentence of the summary."
    assert failure["stage"] == "verifying"


def test_stop_flag_ends_stopped_keeping_finished_nodes_and_resume_completes(
    paged_run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from summarizer_web.services import runs_service

    def stop_during_second_leaf(request):
        if request.operation_id == "S000002":
            request_stop_flag(paged_run)
        return None

    first = ScriptedProvider(on_request=stop_during_second_leaf)
    _use_provider(monkeypatch, first)

    assert _execute(paged_run) == "stopped"

    run = _run(paged_run)
    assert run["state"] == "stopped"
    [attempt] = _attempts(paged_run)
    assert attempt["state"] == "stopped"
    assert attempt["failure_json"] is None
    nodes = _nodes(paged_run)
    assert nodes["L0N0001"]["state"] == nodes["L0N0002"]["state"] == "completed"
    pending = [nodes[node_id] for node_id in nodes if node_id not in ("L0N0001", "L0N0002")]
    assert {node["state"] for node in pending} == {"pending"}
    assert all(
        get_database().fetchone(
            "SELECT event_type FROM run_events WHERE event_id = ?", (node["updated_event_id"],)
        )["event_type"]
        == "item"
        for node in pending
    )
    assert json.loads(run["progress_json"])["leaves"]["done"] == 2

    class QueueOnly:
        def enqueue(self, run_id: str) -> None:
            pass

    monkeypatch.setattr(runs_service, "get_supervisor", lambda: QueueOnly())
    resumed = runs_service.resume_run(paged_run)
    assert (resumed.state, resumed.attempt_count, resumed.attempt.attempt_number) == ("queued", 2, 2)

    second = ScriptedProvider()
    _use_provider(monkeypatch, second)

    assert _execute(paged_run) == "completed"

    assert _run(paged_run)["state"] == "completed"
    assert [attempt["state"] for attempt in _attempts(paged_run)] == ["stopped", "completed"]
    assert "S000001" not in second.operations()
    assert "S000002" not in second.operations()
    nodes = _nodes(paged_run)
    assert {node["state"] for node in nodes.values()} == {"completed"}
    assert nodes["L0N0001"]["started_at"] is not None


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (ConnectionRefusedError(61, "Connection refused"), "ollama_unreachable", "Ollama is not reachable at http://ollama.test:11434."),
        (
            ProviderRetriesExhaustedError(
                5,
                "Ollama request timed out",
                (RetryAttempt(5, RetryErrorCategory.TIMEOUT, None, True, 0.0),),
            ),
            "provider_timeout",
            "The model did not answer within 90 s.",
        ),
        (ProviderTimeoutError("Ollama request timed out"), "provider_timeout", "The model did not answer within 90 s."),
        (
            ProviderRetriesExhaustedError(
                2,
                "Ollama connection failed",
                (RetryAttempt(2, RetryErrorCategory.CONNECTION, None, True, 0.0),),
            ),
            "ollama_unreachable",
            "Ollama is not reachable at http://ollama.test:11434.",
        ),
        (ProviderServerError("Ollama server request failed"), "provider_error", "Ollama server request failed"),
        (BudgetError("no usable input capacity"), "budget_error", "no usable input capacity"),
        (KeyError("boom"), "internal_error", "KeyError: 'boom'"),
    ],
)
def test_failure_classification(error: BaseException, code: str, message: str) -> None:
    failure = runner.classify_failure(
        error, host="http://ollama.test:11434", model="m", timeout_seconds=90.0, stage="merging", item=None
    )
    assert (failure.code, failure.message, failure.stage) == (code, message, "merging")
    assert failure.hint


def test_item_failure_classification_uses_the_item_label() -> None:
    error = ItemFailedError(
        "the summary field was blank",
        stage=StageName.MERGING,
        kind="merge",
        work_id="L1N0002",
        covered_segment_ids=("S000003", "S000004"),
    )
    failure = runner.classify_failure(
        error, host="h", model="m", timeout_seconds=90.0, stage="merging", item="Level 1 · Group 2 · pp. 2–3"
    )
    assert failure.code == "item_invalid_output"
    assert failure.message == (
        "Level 1 · Group 2 · pp. 2–3 produced invalid output after 3 tries: the summary field was blank"
    )
    assert failure.stage == "merging"


def test_failure_detail_is_bounded_to_4_kb() -> None:
    failure = runner.classify_failure(
        RuntimeError("é" * 5000), host="h", model="m", timeout_seconds=1.0, stage=None, item=None
    )
    assert len(failure.detail.encode("utf-8")) <= 4096
