"""Run state replay and the browser tab's single activity stream."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import summarizer_web.api.v1.activity as activity_api
from summarizer.runtime.observers import ItemEvent, StageName
from summarizer_web.api.v1.activity import activity_events
from summarizer_web.db.connection import get_database
from summarizer_web.models.api import CreateRunRequest, RunConfig
from summarizer_web.services import runs_service
from summarizer_web.services.events_service import end_attempt, record_event, start_attempt
from summarizer_web.services.progress_service import now_iso
from summarizer_web.worker.projections import NodeLabeler, PageMap, apply_item_event
from tests.support.web_runs import seed_document, use_data_dir


class _QueueOnlySupervisor:
    def enqueue(self, _run_id: str) -> None:
        pass

    def discard(self, _run_id: str) -> None:
        pass

    def request_stop(self, _run_id: str) -> None:
        pass


def _create_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    use_data_dir(tmp_path, monkeypatch)
    document_id, _ = seed_document()
    monkeypatch.setattr(runs_service, "get_supervisor", _QueueOnlySupervisor)
    run, created = runs_service.create_run(
        CreateRunRequest(
            document_id=document_id,
            config=RunConfig(model="scripted-model", strategy="direct", verify=False),
        ),
        idempotency_key=None,
    )
    assert created
    return run.run_id


def _run_states(run_id: str) -> list[str]:
    rows = get_database().fetchall(
        "SELECT payload_json FROM run_events WHERE run_id = ? AND event_type = 'run_state' ORDER BY event_id",
        (run_id,),
    )
    return [json.loads(row["payload_json"])["state"] for row in rows]


def test_each_run_transition_is_durable_in_the_state_cursor(tmp_path, monkeypatch) -> None:
    run_id = _create_run(tmp_path, monkeypatch)
    assert _run_states(run_id) == ["queued"]

    attempt = start_attempt(run_id)
    assert attempt is not None
    stopping = runs_service.stop_run(run_id)
    assert (stopping.state, stopping.can_stop) == ("stopping", False)
    with get_database().transaction() as connection:
        end_attempt(
            connection,
            run_id=run_id,
            attempt_id=attempt["attempt_id"],
            state="stopped",
        )
    assert runs_service.resume_run(run_id).state == "queued"

    resumed = start_attempt(run_id)
    assert resumed is not None
    with get_database().transaction() as connection:
        end_attempt(
            connection,
            run_id=run_id,
            attempt_id=resumed["attempt_id"],
            state="completed",
        )

    assert _run_states(run_id) == [
        "queued",
        "running",
        "stopping",
        "stopped",
        "queued",
        "running",
        "completed",
    ]


def test_reconnect_replays_final_nodes_before_the_terminal_run(tmp_path, monkeypatch) -> None:
    run_id = _create_run(tmp_path, monkeypatch)
    attempt = start_attempt(run_id)
    assert attempt is not None
    db = get_database()
    after = db.fetchone("SELECT COALESCE(MAX(event_id), 0) AS cursor FROM run_events")["cursor"]

    summary = {
        "summary": "The source supports this conclusion.",
        "content_units": [],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": ["D000001"],
        "level": 0,
    }
    item = ItemEvent(
        kind="leaf",
        work_id="L0N0001",
        state="completed",
        stage=StageName.SUMMARIZING,
        level=0,
        order=0,
        total=1,
        covered_segment_ids=("D000001",),
        summary=summary,
    )
    label = NodeLabeler(PageMap()).label_event(item)
    with db.transaction() as connection:
        item_event_id = record_event(
            connection,
            run_id,
            "item",
            {
                "work_id": item.work_id,
                "kind": item.kind,
                "state": item.state,
                "stage": item.stage.value,
                "label": label,
            },
        )
        apply_item_event(connection, run_id, item_event_id, item, label, now_iso())
        end_attempt(
            connection,
            run_id=run_id,
            attempt_id=attempt["attempt_id"],
            state="completed",
        )

    async def first_events() -> list[dict[str, str]]:
        stream = activity_events(after)
        try:
            return [await stream.__anext__() for _ in range(3)]
        finally:
            await stream.aclose()

    events = asyncio.run(first_events())
    assert [event["event"] for event in events] == ["nodes", "run", "activity"]
    node_event, run_event, activity_event = events
    node_payload = json.loads(node_event["data"])
    run_payload = json.loads(run_event["data"])
    activity_payload = json.loads(activity_event["data"])
    assert node_payload["run_id"] == run_id
    assert [(node["node_id"], node["state"]) for node in node_payload["nodes"]] == [
        ("L0N0001", "completed")
    ]
    assert (run_payload["run_id"], run_payload["document_id"], run_payload["state"]) == (
        run_id,
        runs_service.get_run(run_id).document_id,
        "completed",
    )
    assert run_payload["progress"] is None
    assert activity_payload["active_run"] is None
    assert node_event["id"] == run_event["id"] == str(after)
    # The payload reflects the changed nodes even while the SSE id stays at the
    # prior cursor until the whole batch has been delivered.
    assert node_payload["cursor"] == activity_payload["cursor"] > after
    assert activity_event["id"] == str(activity_payload["cursor"])
    assert activity_payload["cursor"] > after

    async def reconnect_after_nodes() -> list[dict[str, str]]:
        stream = activity_events(int(node_event["id"]))
        try:
            return [await stream.__anext__() for _ in range(3)]
        finally:
            await stream.aclose()

    replayed = asyncio.run(reconnect_after_nodes())
    assert [event["event"] for event in replayed] == ["nodes", "run", "activity"]
    assert json.loads(replayed[1]["data"])["state"] == "completed"


def test_live_stream_emits_each_run_state_transition(tmp_path, monkeypatch) -> None:
    run_id = _create_run(tmp_path, monkeypatch)
    attempt = start_attempt(run_id)
    assert attempt is not None
    db = get_database()
    after = db.fetchone("SELECT COALESCE(MAX(event_id), 0) AS cursor FROM run_events")["cursor"]

    async def stop_between_polls(_seconds: float) -> None:
        assert runs_service.stop_run(run_id).state == "stopping"
        with db.transaction() as connection:
            end_attempt(
                connection,
                run_id=run_id,
                attempt_id=attempt["attempt_id"],
                state="stopped",
            )

    monkeypatch.setattr(activity_api.anyio, "sleep", stop_between_polls)

    async def poll_twice() -> list[dict[str, str]]:
        stream = activity_events(after)
        try:
            events = [await stream.__anext__()]
            events.extend([await stream.__anext__() for _ in range(3)])
            return events
        finally:
            await stream.aclose()

    events = asyncio.run(poll_twice())
    assert [event["event"] for event in events] == ["activity", "run", "run", "activity"]
    run_payloads = [json.loads(event["data"]) for event in events[1:3]]
    assert [payload["run_id"] for payload in run_payloads] == [run_id, run_id]
    assert [payload["state"] for payload in run_payloads] == ["stopped", "stopped"]
    assert all(payload["progress"] is None for payload in run_payloads)
    assert _run_states(run_id)[-2:] == ["stopping", "stopped"]
    assert json.loads(events[-1]["data"])["active_run"] is None

def test_stream_without_a_cursor_sends_only_the_activity_snapshot(tmp_path, monkeypatch) -> None:
    use_data_dir(tmp_path, monkeypatch)

    async def first_event() -> dict[str, str]:
        stream = activity_events(0)
        try:
            return await stream.__anext__()
        finally:
            await stream.aclose()

    event = asyncio.run(first_event())
    assert event["event"] == "activity"
    assert json.loads(event["data"])["cursor"] == 0
