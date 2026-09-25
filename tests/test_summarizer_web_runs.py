"""Run lifecycle HTTP contracts: idempotency and the single-active invariant."""

from __future__ import annotations

from fastapi.testclient import TestClient

from summarizer_web.main import create_app
from summarizer_web.services import runs_service
from tests.support.web_runs import seed_document, use_data_dir


class _QueueOnlySupervisor:
    def enqueue(self, _run_id: str) -> None:
        pass


def test_run_create_is_idempotent_and_rejects_a_second_active_run(tmp_path, monkeypatch) -> None:
    use_data_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(runs_service, "get_supervisor", _QueueOnlySupervisor)
    first_document, _ = seed_document(title="First report")
    second_document, _ = seed_document(title="Second report", text="A distinct report for another Run.")
    payload = {
        "config": {"model": "scripted-model", "strategy": "direct", "verify": False},
        "document_id": first_document,
    }

    with TestClient(create_app()) as client:
        csrf = client.get("/api/v1/health").headers["x-csrf-token"]
        headers = {"Idempotency-Key": "request-1", "X-CSRF-Token": csrf}
        created = client.post("/api/v1/runs", json=payload, headers=headers)
        repeated = client.post("/api/v1/runs", json=payload, headers=headers)

        assert created.status_code == 201
        assert repeated.status_code == 200
        assert repeated.json()["run_id"] == created.json()["run_id"]
        assert repeated.json()["attempt_count"] == 1

        conflict = client.post(
            "/api/v1/runs",
            json={**payload, "document_id": second_document},
            headers={"Idempotency-Key": "request-2", "X-CSRF-Token": csrf},
        )

    assert conflict.status_code == 409
    assert conflict.json()["code"] == "run_active"
    assert conflict.json()["details"] == {
        "run_id": created.json()["run_id"],
        "document_id": first_document,
        "document_title": "First report",
        "state": "queued",
    }
