"""Ollama connectivity helpers."""

from __future__ import annotations

import httpx

from summarizer_web.models.api import OllamaHealthResponse, OllamaModel, OllamaModelsResponse
from summarizer_web.services.settings_service import get_settings


def check_health() -> OllamaHealthResponse:
    host = get_settings().ollama_host.rstrip("/")
    try:
        response = httpx.get(f"{host}/api/tags", timeout=5.0)
        response.raise_for_status()
        return OllamaHealthResponse(connected=True, message="Connected to Ollama")
    except httpx.HTTPError as error:
        return OllamaHealthResponse(
            connected=False,
            message=f"Could not reach Ollama at {host}: {error}",
        )


def list_models() -> OllamaModelsResponse:
    host = get_settings().ollama_host.rstrip("/")
    response = httpx.get(f"{host}/api/tags", timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    models = [OllamaModel(name=item["name"]) for item in payload.get("models", [])]
    return OllamaModelsResponse(models=models)
