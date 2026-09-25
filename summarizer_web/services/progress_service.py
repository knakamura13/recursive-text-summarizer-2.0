"""Run progress: the counters a worker keeps from pipeline observer callbacks,
the ETA estimate, and the snapshot that readers load from `runs.progress_json`.

The worker owns a `ProgressTracker` per Attempt and persists `snapshot()`
(throttled, on stage changes, and at the end); readers never scan run_events.
`elapsed_seconds` is recomputed at read time for active Runs.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic

from pydantic import ValidationError

from summarizer.hierarchy import group_children
from summarizer.runtime.observers import ItemEvent, SegmentInfo, StageEvent, StageName
from summarizer_web.models.api import (
    CountProgress,
    CurrentItem,
    LevelProgress,
    RunProgress,
    StageProgress,
)

STAGES: tuple[str, ...] = tuple(stage.value for stage in StageName)
ACTIVE_RUN_STATES = frozenset({"queued", "running", "stopping"})
# Merge fan-in before any merge level is planned and without a configured
# ceiling. The pipeline measures the real fan-in per level from summary sizes.
DEFAULT_ASSUMED_FANOUT = 8
MIN_CALLS_FOR_ETA = 2
_WORDS_PER_SENTENCE = 20
_CLAIMS_PER_SENTENCE = 2.5
_MAX_CURRENT_ITEMS = 20


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_since(started_at: str | None, until: str | None = None) -> float | None:
    if not started_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(until) if until else datetime.now(timezone.utc)
    except ValueError:
        return None
    return round(max(0.0, (end - start).total_seconds()), 1)


# --- Estimates shared with preflight -------------------------------------------


def estimate_merge_calls(leaf_count: int, fanout: int) -> int:
    """Model calls to reduce `leaf_count` nodes to one root; single-child groups pass through."""
    calls = 0
    count = leaf_count
    while count > 1:
        groups = group_children(count, max(2, fanout))
        calls += sum(1 for group in groups if len(group) > 1)
        count = len(groups)
    return calls


def estimate_verification_calls(target_words: int) -> int:
    """One decomposition call plus about one classification call per claim of one pass."""
    sentences = max(1, target_words) / _WORDS_PER_SENTENCE
    return 1 + math.ceil(_CLAIMS_PER_SENTENCE * sentences)


# --- ETA ------------------------------------------------------------------------


@dataclass(frozen=True)
class RemainingWork:
    """What the ETA needs to know about the work an Attempt has left."""

    strategy: str | None
    leaves_total: int | None
    leaves_finished: int = 0
    # Node counts of the planned merge levels, lowest level first.
    merge_level_sizes: tuple[int, ...] = ()
    merges_unfinished: int = 0
    max_merge_children: int | None = None
    editorial_finished: bool = False
    verify: bool = False
    verification_finished: bool = False
    claims_total: int | None = None
    claims_finished: int = 0
    target_words: int = 300


def remaining_model_calls(work: RemainingWork) -> tuple[int, bool] | None:
    """Model calls left and whether part of the count is estimated; None before planning."""
    estimated = False
    if work.strategy == "direct":
        leaves = 0 if work.leaves_finished else 1
        merges = 0
    elif work.strategy == "hierarchical" and work.leaves_total is not None:
        leaves = max(0, work.leaves_total - work.leaves_finished)
        if work.merge_level_sizes:
            top = work.merge_level_sizes[-1]
            below = work.merge_level_sizes[-2] if len(work.merge_level_sizes) > 1 else work.leaves_total
            fanout = max(2, math.ceil(below / top)) if top else 2
        else:
            top = work.leaves_total
            fanout = work.max_merge_children or DEFAULT_ASSUMED_FANOUT
        unplanned = estimate_merge_calls(top, fanout) if top > 1 else 0
        estimated = unplanned > 0
        merges = work.merges_unfinished + unplanned
    else:
        return None
    editorial = 0 if work.editorial_finished else 1
    verification = 0
    if work.verify and not work.verification_finished:
        if work.claims_total is not None:
            verification = max(0, work.claims_total - work.claims_finished)
        else:
            verification = estimate_verification_calls(work.target_words)
            estimated = True
    return leaves + merges + editorial + verification, estimated


def estimate_eta(
    work: RemainingWork, *, completed_calls: int, calling_seconds: float
) -> tuple[float | None, str]:
    """Remaining model calls divided by this Attempt's throughput.

    `completed_calls` counts non-reused model calls and `calling_seconds` is
    the wall-clock time since the first of them started. Throughput rather
    than mean call duration, because with concurrency a call's duration also
    includes time spent queued behind others.
    """
    if completed_calls < MIN_CALLS_FOR_ETA or calling_seconds <= 0:
        return None, f"The estimate starts after {MIN_CALLS_FOR_ETA} model calls finish."
    remaining = remaining_model_calls(work)
    if remaining is None:
        return None, "The estimate starts once the work is planned."
    calls, estimated = remaining
    if calls == 0:
        return 0.0, "No model calls left."
    seconds_per_call = calling_seconds / completed_calls
    noun = "call" if calls == 1 else "calls"
    return round(calls * seconds_per_call, 1), (
        f"{'About ' if estimated else ''}{calls} model {noun} left at one per "
        f"{seconds_per_call:.1f} s, the pace of the {completed_calls} calls finished in this Attempt."
    )


# --- Live tracker -----------------------------------------------------------------


@dataclass
class _Tracked:
    kind: str
    level: int
    state: str  # pending | active | completed | reused | failed


class ProgressTracker:
    """Counters for one Attempt, fed by observer callbacks.

    Not thread-safe: the worker serializes callbacks before applying them.
    """

    def __init__(
        self,
        *,
        attempt_number: int,
        started_at: str,
        verify: bool,
        target_words: int,
        max_merge_children: int | None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.attempt_number = attempt_number
        self.started_at = started_at
        self._verify = verify
        self._target_words = target_words
        self._max_merge_children = max_merge_children
        self._clock = clock
        self._stages: dict[str, StageProgress] = {
            stage: StageProgress(stage=stage, state="pending") for stage in STAGES
        }
        self._stage: str | None = None
        self.strategy: str | None = None
        self._leaf_total: int | None = None
        self._leaves: dict[str, _Tracked] = {}
        self._merges: dict[str, _Tracked] = {}
        self._editorial_finished = False
        self._claims_seen = False
        self._claims_total: int | None = None
        self._claims_done: set[str] = set()
        self._current: dict[str, CurrentItem] = {}
        self._call_started: dict[str, float] = {}
        self._first_call_started: float | None = None
        self._completed_calls = 0

    @property
    def stage(self) -> str | None:
        return self._stage

    def current_label(self) -> str | None:
        """Label of the only item in progress, or None when zero or several are."""
        if len(self._current) != 1:
            return None
        return next(iter(self._current.values())).label

    def on_stage(self, event: StageEvent) -> bool:
        """Apply a stage event; True when the stage or its state changed."""
        name = event.stage.value
        previous = self._stages[name]
        self._stages[name] = StageProgress(
            stage=name,
            state=event.state,
            completed=event.completed,
            total=event.total,
            detail=event.detail,
        )
        if event.state == "active" or self._stage is None:
            changed_stage = self._stage != name
            self._stage = name
        else:
            changed_stage = False
        if event.stage is StageName.PREPARING and event.state == "completed":
            if event.detail in ("direct", "hierarchical"):
                self.strategy = event.detail
        if event.stage is StageName.SUMMARIZING and event.total is not None:
            self._leaf_total = event.total
        if event.state != "active" and event.stage.value == self._stage:
            self._drop_current_of_stage(name)
        return changed_stage or previous.state != event.state

    def on_segments(self, segments: tuple[SegmentInfo, ...]) -> None:
        if segments and self._leaf_total is None:
            self._leaf_total = len(segments)

    def on_item(self, event: ItemEvent, label: str) -> None:
        now = self._clock()
        if event.kind == "claim":
            self._on_claim(event, label, now)
            return
        if event.kind == "editorial":
            if event.state in ("completed", "reused"):
                self._editorial_finished = True
        elif event.kind == "leaf":
            if event.total is not None:
                self._leaf_total = event.total
            self._track(self._leaves, event)
        else:
            self._track(self._merges, event)
        self._apply_call(event, label, now, is_call=event.kind != "passthrough")

    def _on_claim(self, event: ItemEvent, label: str, now: float) -> None:
        # Claim ids (`V01C000003`) restart in every verification pass, and a
        # pass activates its claims in order, so claim 0 starts a new pass.
        if event.state == "active" and event.order == 0:
            self._claims_total = None
            self._claims_done = set()
        self._claims_seen = True
        if event.total is not None:
            self._claims_total = event.total
        if event.state == "completed":
            self._claims_done.add(event.work_id)
        self._apply_call(event, label, now, is_call=True)

    def _track(self, items: dict[str, _Tracked], event: ItemEvent) -> None:
        state = "pending" if event.state == "planned" else "active" if event.state == "retrying" else event.state
        known = items.get(event.work_id)
        if known is not None and event.state == "planned" and known.state in ("completed", "reused"):
            return
        level = event.level if event.level is not None else (known.level if known else 0)
        items[event.work_id] = _Tracked(kind=event.kind, level=level, state=state)

    def _apply_call(self, event: ItemEvent, label: str, now: float, *, is_call: bool) -> None:
        if event.state in ("active", "retrying"):
            existing = self._current.get(event.work_id)
            self._current[event.work_id] = CurrentItem(
                kind=event.kind,
                work_id=event.work_id,
                label=label,
                started_at=existing.started_at if existing else now_iso(),
                attempt=event.attempt if event.state == "retrying" else (existing.attempt if existing else None),
            )
            if is_call and event.work_id not in self._call_started:
                self._call_started[event.work_id] = now
                if self._first_call_started is None:
                    self._first_call_started = now
            return
        if event.state in ("completed", "reused", "failed"):
            self._current.pop(event.work_id, None)
            started = self._call_started.pop(event.work_id, None)
            if event.state == "completed" and is_call and started is not None:
                self._completed_calls += 1

    def _drop_current_of_stage(self, stage: str) -> None:
        kinds = {
            "summarizing": {"leaf"},
            "merging": {"merge", "passthrough"},
            "writing": {"editorial"},
            "verifying": {"claim"},
        }.get(stage, set())
        for work_id in [key for key, item in self._current.items() if item.kind in kinds]:
            self._current.pop(work_id)
            self._call_started.pop(work_id, None)

    def clear_current(self) -> None:
        """An Attempt ended: nothing is in progress anymore."""
        self._current.clear()
        self._call_started.clear()

    def remaining_work(self) -> RemainingWork:
        levels: dict[int, int] = {}
        for item in self._merges.values():
            levels[item.level] = levels.get(item.level, 0) + 1
        return RemainingWork(
            strategy=self.strategy,
            leaves_total=self._leaf_total,
            leaves_finished=sum(
                1 for item in self._leaves.values() if item.state in ("completed", "reused")
            ),
            merge_level_sizes=tuple(levels[level] for level in sorted(levels)),
            merges_unfinished=sum(
                1
                for item in self._merges.values()
                if item.kind == "merge" and item.state not in ("completed", "reused")
            ),
            max_merge_children=self._max_merge_children,
            editorial_finished=self._editorial_finished,
            verify=self._verify,
            verification_finished=self._stages["verifying"].state == "completed",
            claims_total=self._claims_total,
            claims_finished=len(self._claims_done),
            target_words=self._target_words,
        )

    def snapshot(self, cursor: int) -> RunProgress:
        leaves = _counts(self._leaves.values(), self._leaf_total)
        merges = _counts(self._merges.values(), len(self._merges) if self._merges else None)
        levels: dict[int, list[int]] = {}
        for item in self._merges.values():
            done_total = levels.setdefault(item.level, [0, 0])
            done_total[1] += 1
            if item.state in ("completed", "reused"):
                done_total[0] += 1
        claims = CountProgress(done=len(self._claims_done), total=self._claims_total)
        stages = []
        for name in STAGES:
            stage = self._stages[name]
            if name == "summarizing" and self._leaves:
                stage = stage.model_copy(update={"completed": leaves.done, "total": leaves.total})
            elif name == "merging" and self._merges:
                stage = stage.model_copy(update={"completed": merges.done, "total": merges.total})
            elif name == "verifying" and self._claims_seen:
                stage = stage.model_copy(update={"completed": claims.done, "total": claims.total})
            stages.append(stage)
        calling = (
            self._clock() - self._first_call_started if self._first_call_started is not None else 0.0
        )
        eta, basis = estimate_eta(
            self.remaining_work(), completed_calls=self._completed_calls, calling_seconds=calling
        )
        current = sorted(self._current.values(), key=lambda item: item.started_at or "")
        return RunProgress(
            cursor=cursor,
            attempt_number=self.attempt_number,
            started_at=self.started_at,
            elapsed_seconds=elapsed_since(self.started_at),
            stage=self._stage,
            stages=stages,
            leaves=leaves,
            merges=merges,
            merge_levels=[
                LevelProgress(level=level, done=done, total=total)
                for level, (done, total) in sorted(levels.items())
            ],
            claims=claims,
            current_items=current[:_MAX_CURRENT_ITEMS],
            eta_seconds=eta,
            eta_basis=basis,
        )


def _counts(items, total: int | None) -> CountProgress:
    done = reused = failed = 0
    for item in items:
        if item.state in ("completed", "reused"):
            done += 1
            reused += item.state == "reused"
        elif item.state == "failed":
            failed += 1
    return CountProgress(done=done, total=total, reused=reused, failed=failed)


# --- Reading snapshots ---------------------------------------------------------------


def empty_progress(*, attempt_number: int | None, completed: bool = False) -> RunProgress:
    state = "completed" if completed else "pending"
    return RunProgress(
        cursor=0,
        attempt_number=attempt_number,
        stages=[StageProgress(stage=stage, state=state) for stage in STAGES],
    )


def read_progress(run: sqlite3.Row, attempt: sqlite3.Row | None) -> RunProgress:
    """The stored snapshot, normalized for the Run's current state.

    Active Runs get a fresh `elapsed_seconds`; finished Runs show no items in
    progress and no ETA, and legacy completed Runs without a snapshot show
    every stage completed.
    """
    attempt_number = attempt["attempt_number"] if attempt is not None else None
    progress: RunProgress | None = None
    if run["progress_json"]:
        try:
            progress = RunProgress.model_validate_json(run["progress_json"])
        except ValidationError:
            progress = None
    if progress is not None and progress.attempt_number != attempt_number:
        progress = None
    if progress is None:
        progress = empty_progress(
            attempt_number=attempt_number, completed=run["state"] == "completed"
        )
    started_at = attempt["started_at"] if attempt is not None else progress.started_at
    if run["state"] in ACTIVE_RUN_STATES:
        return progress.model_copy(
            update={"started_at": started_at, "elapsed_seconds": elapsed_since(started_at)}
        )
    ended_at = attempt["ended_at"] if attempt is not None else None
    return progress.model_copy(
        update={
            "started_at": started_at,
            "elapsed_seconds": (
                elapsed_since(started_at, ended_at) if ended_at else progress.elapsed_seconds
            ),
            "current_items": [],
            "eta_seconds": None,
            "eta_basis": None,
        }
    )
