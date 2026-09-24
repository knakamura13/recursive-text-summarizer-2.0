"""The Run supervisor with real spawned workers, and startup reconciliation.

Workers here are stand-ins (module-level targets run by the spawn context);
none of them runs the pipeline, so no test can reach a live Ollama.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database
from summarizer_web.services.events_service import failure_path, stop_flag_path
from summarizer_web.worker.reconcile import reconcile_on_startup
from summarizer_web.worker.runner import hold_worker_lock, watch_parent
from summarizer_web.worker.supervisor import RunSupervisor
from tests.support.web_runs import seed_document, seed_run, use_data_dir

SPAWN = multiprocessing.get_context("spawn")


# --- Stand-in workers (spawn pickles targets by name, so they live here) ---------------


def hung_worker(run_id: str, parent_pid: int) -> None:
    """Stuck in a model call and deaf to SIGTERM: only SIGKILL ends it."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready = stop_flag_path(run_id).with_name("hung.ready")
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text("ready", encoding="utf-8")
    time.sleep(60)


def sleeping_worker(run_id: str, parent_pid: int) -> None:
    time.sleep(60)


def crashing_worker(run_id: str, parent_pid: int) -> None:
    os._exit(3)


def reporting_worker(run_id: str, parent_pid: int) -> None:
    """Writes its RunFailure to failure.json, then dies before recording it."""
    path = failure_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"code": "budget_error", "message": "No room left."}))
    os._exit(1)


def locked_orphan(run_id: str) -> None:
    """A worker that outlived its application: it still holds the Run's lock."""
    hold_worker_lock(run_id)
    time.sleep(60)


def unrelated_process(run_id: str) -> None:
    time.sleep(60)


def watching_grandchild(parent_pid: int, pid_file: str) -> None:
    Path(pid_file).write_text(str(os.getpid()))
    watch_parent(parent_pid, interval_seconds=0.05)
    time.sleep(60)


def short_lived_parent(pid_file: str) -> None:
    child = SPAWN.Process(target=watching_grandchild, args=(os.getpid(), pid_file))
    child.start()
    time.sleep(60)


# --- Helpers ---------------------------------------------------------------------------


def _supervisor(target: Callable[[str, int], None]) -> RunSupervisor:
    return RunSupervisor(
        stop_grace_seconds=0.3, kill_grace_seconds=0.3, poll_seconds=0.02, worker_target=target
    )


def _wait_for(predicate: Callable[[], bool], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached in time")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _run_state(run_id: str) -> str:
    return get_database().fetchone("SELECT state FROM runs WHERE run_id = ?", (run_id,))["state"]


def _attempt(run_id: str) -> dict:
    return dict(
        get_database().fetchone(
            "SELECT * FROM run_attempts WHERE run_id = ? ORDER BY attempt_number DESC LIMIT 1",
            (run_id,),
        )
    )


def _failure(run_id: str) -> dict:
    return json.loads(_attempt(run_id)["failure_json"])


@pytest.fixture
def document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    use_data_dir(tmp_path, monkeypatch)
    return seed_document()


@pytest.fixture
def supervisors():
    started: list[RunSupervisor] = []
    yield started
    for supervisor in started:
        supervisor.shutdown()


def _running(supervisors: list[RunSupervisor], run_id: str, target) -> tuple[RunSupervisor, int]:
    supervisor = _supervisor(target)
    supervisors.append(supervisor)
    supervisor.enqueue(run_id)
    _wait_for(lambda: _attempt(run_id)["worker_pid"] is not None and _run_state(run_id) == "running")
    return supervisor, _attempt(run_id)["worker_pid"]


# --- Supervisor --------------------------------------------------------------------------


def test_stop_returns_at_once_and_a_hung_worker_is_killed_after_the_grace_periods(
    document, supervisors, monkeypatch: pytest.MonkeyPatch
) -> None:
    from summarizer_web.services import runs_service

    run_id = seed_run(*document)
    supervisor, pid = _running(supervisors, run_id, hung_worker)
    ready = stop_flag_path(run_id).with_name("hung.ready")
    _wait_for(ready.exists)
    monkeypatch.setattr(runs_service, "get_supervisor", lambda: supervisor)

    started = time.monotonic()
    response = runs_service.stop_run(run_id)
    assert time.monotonic() - started < 0.25
    assert (response.state, response.can_stop) == ("stopping", False)

    _wait_for(lambda: _run_state(run_id) == "stopped", timeout=5)
    # SIGTERM after the stop grace, then SIGKILL after the kill grace.
    assert time.monotonic() - started >= 0.6
    assert not _alive(pid)
    attempt = _attempt(run_id)
    assert (attempt["state"], attempt["failure_json"]) == ("stopped", None)


def test_stopping_a_queued_run_never_starts_its_worker(
    document, supervisors, monkeypatch: pytest.MonkeyPatch
) -> None:
    from summarizer_web.services import runs_service

    active = seed_run(*document)
    supervisor, _ = _running(supervisors, active, sleeping_worker)
    # Insert this queued Run after supervisor startup; D14 reconciles stale
    # queued rows from the prior application process as interrupted.
    waiting = seed_run(*document)
    supervisor.enqueue(waiting)
    monkeypatch.setattr(runs_service, "get_supervisor", lambda: supervisor)

    assert runs_service.stop_run(waiting).state == "stopped"

    supervisor.shutdown()
    attempt = _attempt(waiting)
    assert (attempt["state"], attempt["worker_pid"]) == ("stopped", None)
    assert _run_state(active) == "interrupted"


def test_a_worker_that_dies_without_reporting_fails_as_crashed(document, supervisors) -> None:
    run_id = seed_run(*document)
    supervisor = _supervisor(crashing_worker)
    supervisors.append(supervisor)

    supervisor.enqueue(run_id)

    _wait_for(lambda: _run_state(run_id) == "failed")
    failure = _failure(run_id)
    assert failure["code"] == "worker_crashed"
    assert failure["message"] == "The worker stopped unexpectedly (exit code 3)."


def test_the_failure_file_is_used_when_the_worker_could_not_record_it(document, supervisors) -> None:
    run_id = seed_run(*document)
    supervisor = _supervisor(reporting_worker)
    supervisors.append(supervisor)

    supervisor.enqueue(run_id)

    _wait_for(lambda: _run_state(run_id) == "failed")
    assert (_failure(run_id)["code"], _failure(run_id)["message"]) == ("budget_error", "No room left.")


def test_shutdown_terminates_the_worker_and_interrupts_the_run(document, supervisors) -> None:
    run_id = seed_run(*document)
    supervisor, pid = _running(supervisors, run_id, hung_worker)
    _wait_for(stop_flag_path(run_id).with_name("hung.ready").exists)

    supervisor.shutdown()

    assert not _alive(pid)
    assert _run_state(run_id) == "interrupted"
    assert _attempt(run_id)["state"] == "interrupted"
    assert _failure(run_id)["code"] == "application_stopped"


def test_attempt_start_clears_the_flags_of_an_earlier_attempt(document, supervisors) -> None:
    run_id = seed_run(*document)
    for path in (stop_flag_path(run_id), failure_path(run_id)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")

    _running(supervisors, run_id, sleeping_worker)
    time.sleep(0.5)

    assert _run_state(run_id) == "running"
    assert not stop_flag_path(run_id).exists()
    assert not failure_path(run_id).exists()


def test_enqueue_that_starts_the_supervisor_only_settles_other_runs(document) -> None:
    orphaned = seed_run(*document, state="running")
    new = seed_run(*document)
    supervisor = RunSupervisor()
    launched: list[str] = []
    supervisor._launch = launched.append  # type: ignore[method-assign]

    supervisor.enqueue(new)
    _wait_for(lambda: launched == [new])
    supervisor.shutdown()

    assert _run_state(orphaned) == "interrupted"
    assert _failure(orphaned)["code"] == "application_stopped"


def test_a_worker_exits_when_its_parent_dies(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    parent = SPAWN.Process(target=short_lived_parent, args=(str(pid_file),))
    parent.start()
    _wait_for(lambda: pid_file.exists() and pid_file.read_text().isdigit())
    grandchild = int(pid_file.read_text())
    try:
        parent.kill()
        parent.join()
        _wait_for(lambda: not _alive(grandchild), timeout=10)
    finally:
        if _alive(grandchild):
            os.kill(grandchild, signal.SIGKILL)


# --- Startup reconciliation ------------------------------------------------------------


def _publish(run_id: str, summary: str) -> None:
    run_dir = load_paths().runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    audit = b'{"schema_version": "audit/4"}'
    (run_dir / "audit.json").write_bytes(audit)
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
    manifest = {
        "format_version": "run/1",
        "run_id": f"run-{run_id}",
        "descriptor_sha256": "0" * 64,
        "source_sha256": "1" * 64,
        "work_ids": ["D000001", "editorial-final"],
        "publication": "complete",
        "audit_sha256": hashlib.sha256(audit).hexdigest(),
        "summary_sha256": hashlib.sha256(summary.encode()).hexdigest(),
    }
    path = load_paths().cache / "runs" / f"run-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_startup_completes_a_run_whose_publication_the_manifest_witnesses(document) -> None:
    published = seed_run(*document, state="running")
    _publish(published, "The harbor moved record cargo.")
    tampered = seed_run(*document, state="running")
    _publish(tampered, "The harbor moved record cargo.")
    (load_paths().runs / tampered / "summary.txt").write_text("Edited afterwards.", encoding="utf-8")

    reconcile_on_startup()

    assert _run_state(published) == "completed"
    assert _attempt(published)["failure_json"] is None
    assert _run_state(tampered) == "interrupted"


def test_startup_interrupts_active_runs_and_returns_active_nodes_to_pending(document) -> None:
    running = seed_run(*document, state="running", progress_json=json.dumps({"stage": "merging"}))
    queued = seed_run(*document)
    stopping = seed_run(*document, state="stopping")
    db = get_database()
    for node_id, state in (("L0N0001", "completed"), ("L0N0002", "active")):
        db.execute(
            """
            INSERT INTO node_projections (
                projection_id, run_id, node_id, level, order_index, label,
                covered_segment_ids_json, evidence_refs_json, kind, state, started_at, updated_event_id
            ) VALUES (?, ?, ?, 0, 0, 'Segment', '[]', '[]', 'leaf', ?, '2026-09-23T00:00:00+00:00', 1)
            """,
            (f"{running}-{node_id}", running, node_id, state),
        )

    reconcile_on_startup()

    assert _run_state(running) == "interrupted"
    failure = _failure(running)
    assert (failure["code"], failure["stage"]) == ("application_stopped", "merging")
    assert failure["message"] == "The application stopped during this Run."
    assert _attempt(running)["ended_at"] is not None
    nodes = {
        row["node_id"]: row
        for row in db.fetchall("SELECT * FROM node_projections WHERE run_id = ?", (running,))
    }
    assert nodes["L0N0001"]["state"] == "completed"
    assert (nodes["L0N0002"]["state"], nodes["L0N0002"]["started_at"]) == ("pending", None)
    assert nodes["L0N0002"]["updated_event_id"] > 1
    assert _run_state(queued) == "interrupted"
    assert _run_state(stopping) == "stopped"


def test_startup_terminates_an_orphan_worker_only_when_it_holds_the_runs_lock(document) -> None:
    ours_run = seed_run(*document, state="running")
    other_run = seed_run(*document, state="running")
    ours = SPAWN.Process(target=locked_orphan, args=(ours_run,))
    unrelated = SPAWN.Process(target=unrelated_process, args=(other_run,))
    ours.start()
    unrelated.start()
    try:
        _wait_for(lambda: (load_paths().runs / ours_run / "worker.lock").exists())
        _wait_for(lambda: (load_paths().runs / ours_run / "worker.lock").read_text().isdigit())
        db = get_database()
        db.execute("UPDATE run_attempts SET worker_pid = ? WHERE run_id = ?", (ours.pid, ours_run))
        db.execute("UPDATE run_attempts SET worker_pid = ? WHERE run_id = ?", (unrelated.pid, other_run))

        reconcile_on_startup()

        ours.join(timeout=5)
        assert ours.exitcode == -signal.SIGTERM
        assert unrelated.is_alive()
        assert _run_state(ours_run) == _run_state(other_run) == "interrupted"
    finally:
        for process in (ours, unrelated):
            if process.is_alive():
                process.kill()
            process.join()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_default_grace_periods_escalate_within_the_stop_budget() -> None:
    supervisor = RunSupervisor()
    assert supervisor.stop_grace_seconds + supervisor.kill_grace_seconds <= 4.0
