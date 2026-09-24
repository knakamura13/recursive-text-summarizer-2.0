from typing import Any

from fastapi import APIRouter, Body

from summarizer_web.models.api import PreflightResponse
from summarizer_web.services.preflight_service import run_preflight

router = APIRouter(tags=["preflight"])


@router.post("/preflight", response_model=PreflightResponse)
def preflight(payload: dict[str, Any] = Body(...)) -> PreflightResponse:
    """Body is a PreflightRequest. It is validated by the service, so an invalid
    `config` yields `ok: false` with `invalid_config` errors instead of a 422."""
    return run_preflight(payload)
