"""Observed execution of one unit of pipeline work."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ModelProvider,
)
from summarizer.reask import (
    DEFAULT_REASK_ATTEMPTS,
    INVALID_OUTPUT_ERRORS,
    generate_validated,
    rejection_reason,
)
from summarizer.runtime.observers import (
    ItemEvent,
    ItemFailedError,
    ItemKind,
    ItemState,
    RuntimeObserver,
    StageName,
)

_Parsed = TypeVar("_Parsed")


def tree_node_id(level: int, order: int) -> str:
    """Return the node id of the tree node at zero-based `order` in `level`.

    Leaf, merge, and passthrough events use the id the final tree gives the
    node, so both take it from here rather than formatting it separately.
    """
    return f"L{level}N{order + 1:04d}"


@dataclass(frozen=True)
class ObservedItem:
    """The identity every event about one unit of work carries.

    Repeating the whole identity on each event, not only on `planned`, lets a
    consumer create or update its record from whichever event reaches it.
    """

    kind: ItemKind
    work_id: str
    stage: StageName
    level: int | None = None
    order: int | None = None
    total: int | None = None
    child_ids: tuple[str, ...] = ()
    covered_segment_ids: tuple[str, ...] = ()

    def event(
        self,
        state: ItemState,
        *,
        summary: Mapping[str, Any] | None = None,
        attempt: int | None = None,
        message: str | None = None,
    ) -> ItemEvent:
        return ItemEvent(
            kind=self.kind,
            work_id=self.work_id,
            state=state,
            stage=self.stage,
            level=self.level,
            order=self.order,
            total=self.total,
            child_ids=self.child_ids,
            covered_segment_ids=self.covered_segment_ids,
            summary=summary,
            attempt=attempt,
            message=message,
        )


def generate_observed(
    item: ObservedItem,
    provider: ModelProvider,
    request: GenerationRequest,
    parse: Callable[[GenerationResult], _Parsed],
    *,
    observer: RuntimeObserver,
    reask_attempts: int = DEFAULT_REASK_ATTEMPTS,
) -> _Parsed:
    """Make one item's model call, reporting it and re-asking invalid output.

    Emits `active` before the first call and `retrying` before each re-ask. A
    stop requested in the meantime raises `PipelineStopped` instead of making
    another call or reporting the item failed. When every re-ask is rejected,
    emits `failed` and raises `ItemFailedError` naming the item, chained to the
    last validation error.
    """
    observer.emit_item(item.event("active"))

    def on_retry(attempt: int, reason: str) -> None:
        observer.raise_if_stopped(f"before re-asking {item.work_id}")
        observer.emit_item(item.event("retrying", attempt=attempt, message=reason))

    try:
        return generate_validated(
            provider,
            request,
            parse,
            on_retry=on_retry,
            reask_attempts=reask_attempts,
        )
    except INVALID_OUTPUT_ERRORS as error:
        observer.raise_if_stopped(f"after {item.work_id} was rejected")
        reason = rejection_reason(error)
        observer.emit_item(item.event("failed", message=reason))
        raise ItemFailedError(
            reason,
            stage=item.stage,
            kind=item.kind,
            work_id=item.work_id,
            covered_segment_ids=item.covered_segment_ids,
        ) from error
