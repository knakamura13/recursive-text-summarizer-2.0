"""Run lifecycle routes. Live updates travel on the activity stream."""

from __future__ import annotations

from fastapi import APIRouter, Header, Response

from summarizer_web.models.api import (
    ActiveRunResponse,
    CreateRunRequest,
    RunListResponse,
    RunResponse,
)
from summarizer_web.services.runs_service import (
    create_run,
    delete_run,
    get_active_run,
    get_run,
    list_runs,
    resume_run,
    stop_run,
)

router = APIRouter(tags=["runs"])


@router.post("/runs", response_model=RunResponse, status_code=201)
def post_run(
    payload: CreateRunRequest,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> RunResponse:
    run, created = create_run(payload, idempotency_key)
    response.status_code = 201 if created else 200
    return run


@router.get("/runs/active", response_model=ActiveRunResponse)
def read_active_run() -> ActiveRunResponse:
    return get_active_run()


@router.get("/runs/{run_id}", response_model=RunResponse)
def read_run(run_id: str) -> RunResponse:
    return get_run(run_id)


@router.get("/documents/{document_id}/runs", response_model=RunListResponse)
def document_runs(document_id: str) -> RunListResponse:
    return list_runs(document_id)


@router.post("/runs/{run_id}/stop", response_model=RunResponse, status_code=202)
def post_stop(run_id: str) -> RunResponse:
    return stop_run(run_id)


@router.post("/runs/{run_id}/resume", response_model=RunResponse, status_code=202)
def post_resume(run_id: str) -> RunResponse:
    return resume_run(run_id)


@router.delete("/runs/{run_id}", status_code=204)
def remove_run(run_id: str) -> Response:
    delete_run(run_id)
    return Response(status_code=204)
