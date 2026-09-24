"""App-wide activity and the tab's single event stream.

Browsers share about six HTTP/1.1 connections per host across all tabs, so
each tab opens exactly one stream: this one. Only one Run can be active, so it
carries everything that changes: the activity snapshot (active Run with
progress, importing Documents), changed tree rows, and state changes of Runs.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from functools import partial

import anyio
from fastapi import APIRouter, Header, Query
from sse_starlette.sse import EventSourceResponse

from summarizer_web.models.api import ActivityResponse
from summarizer_web.services.runs_service import get_activity, poll_activity

router = APIRouter(tags=["activity"])

ACTIVITY_POLL_SECONDS = 1.0
PING_SECONDS = 15


@router.get("/activity", response_model=ActivityResponse)
def read_activity() -> ActivityResponse:
    return get_activity()


def _change_key(activity: ActivityResponse) -> str:
    """The snapshot without the cursor and elapsed time: clients tick elapsed themselves."""
    run = activity.active_run
    if run is not None and run.progress is not None:
        run = run.model_copy(
            update={"progress": run.progress.model_copy(update={"elapsed_seconds": None})}
        )
    return activity.model_copy(update={"active_run": run, "cursor": 0}).model_dump_json()


async def activity_events(after: int) -> AsyncIterator[dict[str, str]]:
    """Poll activity, replaying changed nodes and Run states after a cursor.

    A reconnect emits nodes before Run state so a terminal Run's final tree is
    present before its state. The cursor advances only on the batch's last
    frame, allowing an interrupted batch to replay. Reads run in worker threads;
    the stream stays open.
    """
    cursor = after
    last_key: str | None = None
    changes = after > 0
    replay = after > 0
    while True:
        poll = await anyio.to_thread.run_sync(
            partial(poll_activity, cursor, changes=changes, replay=replay)
        )
        key = _change_key(poll.activity)
        send_activity = key != last_key
        message_count = len(poll.nodes) + len(poll.runs) + int(send_activity)
        message_index = 0

        for run_id, nodes in poll.nodes:
            message_index += 1
            event_cursor = poll.activity.cursor if message_index == message_count else cursor
            payload = {
                "run_id": run_id,
                "nodes": [node.model_dump(mode="json") for node in nodes],
                "cursor": poll.activity.cursor,
            }
            yield {
                "event": "nodes",
                "id": str(event_cursor),
                "data": json.dumps(payload, separators=(",", ":")),
            }
        for run in poll.runs:
            message_index += 1
            event_cursor = poll.activity.cursor if message_index == message_count else cursor
            yield {"event": "run", "id": str(event_cursor), "data": run.model_dump_json()}
        if send_activity:
            last_key = key
            event_cursor = poll.activity.cursor
            yield {
                "event": "activity",
                "id": str(event_cursor),
                "data": poll.activity.model_dump_json(),
            }
        cursor = poll.activity.cursor
        changes = True
        replay = False
        await anyio.sleep(ACTIVITY_POLL_SECONDS)


def _cursor(after: int | None, last_event_id: str | None) -> int:
    if after is not None:
        return after
    try:
        return max(0, int(last_event_id or 0))
    except ValueError:
        return 0


@router.get("/activity/stream")
async def activity_stream(
    after: int | None = Query(default=None, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> EventSourceResponse:
    return EventSourceResponse(
        activity_events(_cursor(after, last_event_id)), ping=PING_SECONDS
    )
