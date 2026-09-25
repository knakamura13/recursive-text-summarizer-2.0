"""Optional progress and stop hooks for pipeline execution.

Callbacks may run on scheduler worker threads, so implementations must be
thread-safe. The pipeline does not guard callbacks; they must not raise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class StageName(str, Enum):
    PREPARING = "preparing"
    SEGMENTING = "segmenting"
    SUMMARIZING = "summarizing"
    MERGING = "merging"
    WRITING = "writing"
    VERIFYING = "verifying"
    PUBLISHING = "publishing"


StageState = Literal["active", "completed", "skipped"]


@dataclass(frozen=True)
class StageEvent:
    stage: StageName
    state: StageState
    completed: int | None = None
    total: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class SegmentInfo:
    """One source segment's position in the document text, in code points."""

    segment_id: str
    order: int
    start: int
    end: int
    core_start: int
    core_end: int
    token_count: int | None = None


ItemKind = Literal["leaf", "merge", "passthrough", "editorial", "claim"]
ItemState = Literal["planned", "active", "retrying", "completed", "reused", "failed"]


@dataclass(frozen=True)
class ItemEvent:
    """Progress of one unit of pipeline work.

    For leaf, merge, and passthrough items `work_id` is the node id that the
    final `PipelineResult.nodes` uses, for example `L0N0001` or `L1N0003`.
    `order` is zero-based within the item's level (tree items) or pass
    (claims), and `total` counts the items in that level or pass. `summary` is
    the validated `SummaryNode` as JSON for completed and reused tree items.
    `attempt` numbers re-asks for `retrying`, and `message` carries the
    validation error for `retrying` and `failed` or the verdict for a
    completed claim.
    """

    kind: ItemKind
    work_id: str
    state: ItemState
    stage: StageName
    level: int | None = None
    order: int | None = None
    total: int | None = None
    child_ids: tuple[str, ...] = ()
    covered_segment_ids: tuple[str, ...] = ()
    summary: Mapping[str, Any] | None = None
    attempt: int | None = None
    message: str | None = None


class PipelineStopped(RuntimeError):
    """The observer asked the pipeline to stop at an item boundary."""


class ItemFailedError(RuntimeError):
    """One unit of work stayed invalid after its re-asks."""

    def __init__(
        self,
        message: str,
        *,
        stage: StageName,
        kind: ItemKind,
        work_id: str,
        covered_segment_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.kind = kind
        self.work_id = work_id
        self.covered_segment_ids = covered_segment_ids


@dataclass(frozen=True)
class RuntimeObserver:
    on_stage: Callable[[StageEvent], None] | None = None
    on_segments: Callable[[tuple[SegmentInfo, ...]], None] | None = None
    on_item: Callable[[ItemEvent], None] | None = None
    should_stop: Callable[[], bool] | None = None

    def emit(self, event: StageEvent) -> None:
        if self.on_stage is not None:
            self.on_stage(event)

    def emit_segments(self, segments: tuple[SegmentInfo, ...]) -> None:
        if self.on_segments is not None:
            self.on_segments(segments)

    def emit_item(self, event: ItemEvent) -> None:
        if self.on_item is not None:
            self.on_item(event)

    def stop_requested(self) -> bool:
        return self.should_stop is not None and self.should_stop()

    def raise_if_stopped(self, where: str) -> None:
        if self.stop_requested():
            raise PipelineStopped(f"stopped {where}")


_DEFAULT_OBSERVER = RuntimeObserver()


def get_observer(observer: RuntimeObserver | None) -> RuntimeObserver:
    return observer if observer is not None else _DEFAULT_OBSERVER
