from fastapi import APIRouter

from summarizer_web.models.api import PreflightRequest, PreflightResponse
from summarizer_web.services.preflight_service import run_preflight

router = APIRouter(tags=["preflight"])


@router.post("/preflight", response_model=PreflightResponse)
def preflight(payload: PreflightRequest) -> PreflightResponse:
    return run_preflight(payload)
