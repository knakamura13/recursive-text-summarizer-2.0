from fastapi import APIRouter

from summarizer_web.models.api import OpenRouterStatusResponse
from summarizer_web.services.openrouter_service import key_status

router = APIRouter(prefix="/openrouter", tags=["openrouter"])


@router.get("/health", response_model=OpenRouterStatusResponse)
def openrouter_health() -> OpenRouterStatusResponse:
    return key_status()
