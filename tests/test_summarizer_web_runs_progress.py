"""Run progress: the ETA formula, the live tracker, and snapshot reads."""

from __future__ import annotations

import sqlite3

import pytest

from summarizer.runtime.observers import ItemEvent, SegmentInfo, StageEvent, StageName
from summarizer_web.models.api import CurrentItem, RunProgress, StageProgress
from summarizer_web.services.progress_service import (
    STAGES,
    ProgressTracker,
    RemainingWork,
    estimate_eta,
    estimate_merge_calls,
    estimate_verification_calls,
    read_progress,
    remaining_model_calls,
)


@pytest.mark.parametrize(
    ("leaves", "fanout", "calls"),
    [(1, 8, 0), (2, 8, 1), (8, 2, 7), (3, 2, 2), (10, 4, 4), (10, 8, 3)],
)
def test_merge_call_estimate_counts_only_real_merges(leaves: int, fanout: int, calls: int) -> None:
    assert estimate_merge_calls(leaves, fanout) == calls


def test_verification_call_estimate_scales_with_target_words() -> None:
    assert estimate_verification_calls(40) == 6
    assert estimate_verification_calls(300) == 39


def test_remaining_calls_before_merges_are_planned_estimate_the_merge_levels() -> None:
    work = RemainingWork(strategy="hierarchical", leaves_total=10, leaves_finished=4)
    # 6 leaves + 3 estimated merges (fan-in 8: 10 -> 2 -> 1) + editorial.
    assert remaining_model_calls(work) == (10, True)
    with_verify = RemainingWork(
        strategy="hierarchical", leaves_total=10, leaves_finished=4, verify=True, target_words=40
    )
    assert remaining_model_calls(with_verify) == (16, True)


def test_remaining_calls_use_the_observed_fan_in_of_planned_levels() -> None:
    work = RemainingWork(
        strategy="hierarchical",
        leaves_total=10,
        leaves_finished=10,
        merge_level_sizes=(3,),
        merges_unfinished=2,
    )
    # Fan-in ceil(10 / 3) = 4 leaves one more merge above the 3 planned nodes.
    assert remaining_model_calls(work) == (2 + 1 + 1, True)
    at_root = RemainingWork(
        strategy="hierarchical",
        leaves_total=10,
        leaves_finished=10,
        merge_level_sizes=(3, 1),
        merges_unfinished=1,
        editorial_finished=False,
    )
    assert remaining_model_calls(at_root) == (2, False)


def test_remaining_calls_count_claims_of_the_current_pass_exactly() -> None:
    work = RemainingWork(
        strategy="direct",
        leaves_total=1,
        leaves_finished=1,
        editorial_finished=True,
        verify=True,
        claims_total=12,
        claims_finished=5,
    )
    assert remaining_model_calls(work) == (7, False)
    assert remaining_model_calls(RemainingWork(strategy=None, leaves_total=None)) is None


def test_eta_waits_for_two_completed_calls() -> None:
    work = RemainingWork(strategy="direct", leaves_total=1)
    eta, basis = estimate_eta(work, completed_calls=1, calling_seconds=30.0)
    assert eta is None
    assert "2 model calls" in basis


def test_eta_divides_remaining_calls_by_the_attempts_throughput() -> None:
    work = RemainingWork(
        strategy="hierarchical",
        leaves_total=8,
        leaves_finished=4,
        merge_level_sizes=(4, 2, 1),
        merges_unfinished=1,
    )
    eta, basis = estimate_eta(work, completed_calls=4, calling_seconds=40.0)
    # 4 leaves + 1 planned merge + editorial = 6 calls at one per 10 s.
    assert eta == 60.0
    assert basis == (
        "6 model calls left at one per 10.0 s, the pace of the 4 calls finished in this Attempt."
    )
    done = RemainingWork(strategy="direct", leaves_total=1, leaves_finished=1, editorial_finished=True)
    assert estimate_eta(done, completed_calls=2, calling_seconds=4.0) == (0.0, "No model calls left.")


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _leaf(order: int, state: str, *, total: int = 3, **extra) -> ItemEvent:
    return ItemEvent(
        kind="leaf",
        work_id=f"L0N{order + 1:04d}",
        state=state,
        stage=StageName.SUMMARIZING,
        level=0,
        order=order,
        total=total,
        covered_segment_ids=(f"S{order + 1:06d}",),
        **extra,
    )


def _six_leaf_tracker(clock: FakeClock) -> ProgressTracker:
    tracker = ProgressTracker(
        attempt_number=1,
        started_at="2026-09-23T00:00:00+00:00",
        verify=False,
        target_words=40,
        max_merge_children=6,
        clock=clock,
    )
    tracker.on_stage(StageEvent(StageName.PREPARING, "completed", detail="hierarchical"))
    tracker.on_stage(StageEvent(StageName.SUMMARIZING, "active", total=6))
    for order in range(6):
        tracker.on_item(_leaf(order, "planned", total=6), f"Segment {order + 1}")
    return tracker


def test_eta_with_two_concurrent_calls_uses_throughput_not_call_duration() -> None:
    clock = FakeClock()
    tracker = _six_leaf_tracker(clock)
    tracker.on_item(_leaf(0, "active", total=6), "Segment 1")
    tracker.on_item(_leaf(1, "active", total=6), "Segment 2")
    clock.now = 110.0
    tracker.on_item(_leaf(0, "completed", total=6, summary={}), "Segment 1")
    tracker.on_item(_leaf(1, "completed", total=6, summary={}), "Segment 2")

    # Each call took 10 s, but two ran at once: one finished per 5 s. Left:
    # 4 leaves, 1 merge (all 6 leaves fit one group), and the editorial.
    assert tracker.snapshot(cursor=1).eta_seconds == 30.0


def test_eta_after_resume_ignores_reused_items() -> None:
    clock = FakeClock()
    tracker = _six_leaf_tracker(clock)
    for order in range(4):
        tracker.on_item(_leaf(order, "reused", total=6, summary={}), f"Segment {order + 1}")
    tracker.on_item(_leaf(4, "active", total=6), "Segment 5")
    clock.now = 108.0
    tracker.on_item(_leaf(4, "completed", total=6, summary={}), "Segment 5")
    tracker.on_item(_leaf(5, "active", total=6), "Segment 6")
    clock.now = 116.0
    tracker.on_item(_leaf(5, "completed", total=6, summary={}), "Segment 6")

    progress = tracker.snapshot(cursor=1)
    assert (progress.leaves.done, progress.leaves.reused) == (6, 4)
    # Two computed calls in 16 s; the merge and the editorial are left.
    assert progress.eta_seconds == 16.0


def test_tracker_counts_items_and_estimates_from_completed_calls() -> None:
    clock = FakeClock()
    tracker = ProgressTracker(
        attempt_number=2,
        started_at="2026-09-23T00:00:00+00:00",
        verify=False,
        target_words=40,
        max_merge_children=2,
        clock=clock,
    )
    assert tracker.on_stage(StageEvent(StageName.PREPARING, "completed", detail="hierarchical"))
    tracker.on_segments(
        tuple(SegmentInfo(f"S{index:06d}", index - 1, 0, 10, 0, 10) for index in range(1, 4))
    )
    tracker.on_stage(StageEvent(StageName.SUMMARIZING, "active", total=3))
    for order in range(3):
        tracker.on_item(_leaf(order, "planned"), f"Segment {order + 1}")
    tracker.on_item(_leaf(0, "reused", summary={}), "Segment 1")
    tracker.on_item(_leaf(1, "active"), "Segment 2")
    clock.now = 110.0
    tracker.on_item(_leaf(1, "completed", summary={}), "Segment 2")
    tracker.on_item(_leaf(2, "active"), "Segment 3")
    tracker.on_item(_leaf(2, "retrying", attempt=1, message="summary was blank"), "Segment 3")

    progress = tracker.snapshot(cursor=42)
    assert progress.cursor == 42
    assert progress.attempt_number == 2
    assert progress.stage == "summarizing"
    assert (progress.leaves.done, progress.leaves.total, progress.leaves.reused) == (2, 3, 1)
    summarizing = next(stage for stage in progress.stages if stage.stage == "summarizing")
    assert (summarizing.state, summarizing.completed, summarizing.total) == ("active", 2, 3)
    [current] = progress.current_items
    assert (current.work_id, current.label, current.attempt) == ("L0N0003", "Segment 3", 1)
    # One computed call so far: the reused leaf is not a model call.
    assert progress.eta_seconds is None

    clock.now = 130.0
    tracker.on_item(_leaf(2, "completed", summary={}), "Segment 3")
    progress = tracker.snapshot(cursor=43)
    # Two calls in 30 s since the first started: 15 s each. Left: 2 merges
    # (fan-in 2: 3 -> 2 -> 1) and the editorial.
    assert progress.eta_seconds == 45.0
    assert progress.current_items == []


def _row(**values) -> sqlite3.Row:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    columns = ", ".join(f"? AS {name}" for name in values)
    return connection.execute(f"SELECT {columns}", tuple(values.values())).fetchone()


def _stored(attempt_number: int) -> str:
    return RunProgress(
        cursor=9,
        attempt_number=attempt_number,
        started_at="2026-09-23T00:00:00+00:00",
        elapsed_seconds=5.0,
        stage="summarizing",
        stages=[StageProgress(stage=stage, state="pending") for stage in STAGES],
        current_items=[CurrentItem(kind="leaf", work_id="L0N0001", label="Segment 1")],
        eta_seconds=12.0,
        eta_basis="basis",
    ).model_dump_json()


def test_read_progress_ticks_elapsed_for_active_runs_only() -> None:
    attempt = _row(attempt_number=1, started_at="2026-09-23T00:00:00+00:00", ended_at=None)
    active = read_progress(_row(state="running", progress_json=_stored(1)), attempt)
    assert active.elapsed_seconds > 5.0
    assert active.current_items and active.eta_seconds == 12.0

    ended = _row(
        attempt_number=1,
        started_at="2026-09-23T00:00:00+00:00",
        ended_at="2026-09-23T00:01:30+00:00",
    )
    stopped = read_progress(_row(state="stopped", progress_json=_stored(1)), ended)
    assert stopped.elapsed_seconds == 90.0
    assert (stopped.current_items, stopped.eta_seconds, stopped.eta_basis) == ([], None, None)


def test_read_progress_ignores_a_snapshot_of_an_earlier_attempt() -> None:
    attempt = _row(attempt_number=2, started_at=None, ended_at=None)
    progress = read_progress(_row(state="queued", progress_json=_stored(1)), attempt)
    assert progress.attempt_number == 2
    assert progress.leaves.done == 0
    assert {stage.state for stage in progress.stages} == {"pending"}


def test_read_progress_of_a_legacy_completed_run_shows_stages_completed() -> None:
    attempt = _row(attempt_number=1, started_at=None, ended_at=None)
    progress = read_progress(_row(state="completed", progress_json=None), attempt)
    assert {stage.state for stage in progress.stages} == {"completed"}
