import json

from fastapi import APIRouter, Header, Request
from sse_starlette.sse import EventSourceResponse

from summarizer_web.db.connection import get_database
from summarizer_web.models.api import (
    CreateRunRequest,
    FinalSummaryResponse,
    NodeDetailResponse,
    NodeTreeResponse,
    RunListResponse,
    RunResponse,
)
from summarizer_web.services.runs_service import (
    cancel_run,
    create_run,
    get_final_summary,
    get_node_detail,
    get_node_tree,
    get_run,
    list_runs,
    resume_run,
)

router = APIRouter(tags=["runs"])


@router.post("/runs", response_model=RunResponse)
def post_run(
    payload: CreateRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> RunResponse:
    return create_run(payload, idempotency_key)


@router.get("/runs/{run_id}", response_model=RunResponse)
def read_run(run_id: str) -> RunResponse:
    return get_run(run_id)


@router.get("/documents/{document_id}/runs", response_model=RunListResponse)
def document_runs(document_id: str) -> RunListResponse:
    return list_runs(document_id)


@router.post("/runs/{run_id}/cancel", response_model=RunResponse)
def post_cancel(run_id: str) -> RunResponse:
    return cancel_run(run_id)


@router.post("/runs/{run_id}/resume", response_model=RunResponse)
def post_resume(run_id: str) -> RunResponse:
    return resume_run(run_id)


@router.get("/runs/{run_id}/tree", response_model=NodeTreeResponse)
def run_tree(run_id: str) -> NodeTreeResponse:
    return get_node_tree(run_id)


@router.get("/runs/{run_id}/nodes/{node_id}", response_model=NodeDetailResponse)
def run_node(run_id: str, node_id: str) -> NodeDetailResponse:
    return get_node_detail(run_id, node_id)


@router.get("/runs/{run_id}/summary", response_model=FinalSummaryResponse)
def run_summary(run_id: str) -> FinalSummaryResponse:
    return get_final_summary(run_id)


@router.get("/runs/{run_id}/events")
async def run_events(request: Request, run_id: str):
    last_event_id = request.headers.get("Last-Event-ID")
    start_id = int(last_event_id) if last_event_id else 0

    async def event_generator():
        seen = start_id
        while True:
            rows = get_database().fetchall(
                """
                SELECT event_id, event_type, payload_json
                FROM run_events
                WHERE run_id = ? AND event_id > ?
                ORDER BY event_id ASC
                """,
                (run_id, seen),
            )
            for row in rows:
                seen = row["event_id"]
                yield {
                    "id": str(row["event_id"]),
                    "event": row["event_type"],
                    "data": row["payload_json"],
                }
            if await request.is_disconnected():
                break
            run = get_database().fetchone("SELECT state FROM runs WHERE run_id = ?", (run_id,))
            if run is not None and run["state"] in {"completed", "failed", "cancelled", "interrupted"}:
                terminal = json.dumps({"state": run["state"]})
                yield {"event": "terminal", "data": terminal}
                break
            import asyncio

            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())
