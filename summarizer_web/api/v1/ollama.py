from fastapi import APIRouter, Query

from summarizer_web.models.api import ErrorResponse, OllamaHealthResponse, OllamaModelsResponse
from summarizer_web.services.ollama_service import check_health, list_models
from summarizer_web.services.settings_service import normalize_ollama_host

router = APIRouter(prefix="/ollama", tags=["ollama"])


@router.get("/health", response_model=OllamaHealthResponse)
def ollama_health(host: str | None = Query(default=None)) -> OllamaHealthResponse:
    return check_health(None if host is None else normalize_ollama_host(host))

@router.get(
    "/models",
    response_model=OllamaModelsResponse,
    responses={503: {"model": ErrorResponse, "description": "ollama_unreachable"}},
)
def ollama_models() -> OllamaModelsResponse:
    return list_models()
