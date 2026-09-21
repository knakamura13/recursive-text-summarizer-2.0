from fastapi import APIRouter, HTTPException

from summarizer_web.models.api import OllamaHealthResponse, OllamaModelsResponse
from summarizer_web.services.ollama_service import check_health, list_models

router = APIRouter(prefix="/ollama", tags=["ollama"])


@router.get("/health", response_model=OllamaHealthResponse)
def ollama_health() -> OllamaHealthResponse:
    return check_health()


@router.get("/models", response_model=OllamaModelsResponse)
def ollama_models() -> OllamaModelsResponse:
    try:
        return list_models()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
