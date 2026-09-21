"""Optional progress and cancellation hooks for pipeline execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal


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
class RuntimeObserver:
    on_stage: Callable[[StageEvent], None] | None = None
    should_cancel: Callable[[], bool] | None = None

    def emit(self, event: StageEvent) -> None:
        if self.on_stage is not None:
            self.on_stage(event)

    def cancelled(self) -> bool:
        return self.should_cancel is not None and self.should_cancel()


_DEFAULT_OBSERVER = RuntimeObserver()


def get_observer(observer: RuntimeObserver | None) -> RuntimeObserver:
    return observer if observer is not None else _DEFAULT_OBSERVER
