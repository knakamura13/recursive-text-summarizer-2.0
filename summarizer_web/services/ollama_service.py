"""HTTP access to the configured Ollama server: health, installed models, and
model context lengths.

Every call uses a 5 s timeout. ``model_context_length`` caches successful
``/api/show`` answers per (host, model) for five minutes, so a debounced
preflight does not ask Ollama again on every keystroke while a newly pulled
model is still picked up.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

import httpx

from summarizer_web.errors import ApiError
from summarizer_web.models.api import OllamaHealthResponse, OllamaModel, OllamaModelsResponse
from summarizer_web.services.settings_service import get_settings

OLLAMA_TIMEOUT_SECONDS = 5.0
_CONTEXT_CACHE_SECONDS = 300.0
_START_HINT = "Start Ollama with `ollama serve` or fix the host in Settings."

_context_lengths: dict[tuple[str, str], tuple[float, int]] = {}
_context_lock = Lock()


class OllamaError(Exception):
    """Ollama could not answer a request; `message` is reader-facing."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class OllamaUnreachableError(OllamaError):
    """No HTTP answer: connection refused, DNS failure, or timeout."""


class OllamaContextLengthUnknownError(OllamaError):
    """Ollama answered /api/show but did not report an architectural window."""


class OllamaModelNotFoundError(OllamaError):
    """Ollama answered 404 for a model."""


def _client(host: str) -> httpx.Client:
    return httpx.Client(base_url=host, timeout=OLLAMA_TIMEOUT_SECONDS)


def _request_json(host: str, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        with _client(host) as client:
            response = client.request(method, path, json=body)
    except httpx.TimeoutException as error:
        raise OllamaUnreachableError(
            f"Ollama at {host} did not answer within {OLLAMA_TIMEOUT_SECONDS:g} s."
        ) from error
    except (httpx.HTTPError, ValueError) as error:
        # ValueError covers hosts httpx cannot even form a request for.
        raise OllamaUnreachableError(f"Ollama is not reachable at {host}.") from error
    if response.status_code == 404 and path == "/api/show":
        raise OllamaModelNotFoundError(f"Model {body and body.get('model')} is not installed.")
    if response.status_code >= 400:
        raise OllamaError(f"Ollama at {host} answered {path} with HTTP {response.status_code}.")
    try:
        payload = response.json()
    except ValueError as error:
        raise OllamaError(f"Ollama at {host} returned a malformed response to {path}.") from error
    if not isinstance(payload, dict):
        raise OllamaError(f"Ollama at {host} returned a malformed response to {path}.")
    return payload


def check_health(host: str | None = None) -> OllamaHealthResponse:
    host = get_settings().ollama_host if host is None else host
    try:
        payload = _request_json(host, "GET", "/api/version")
    except OllamaError as error:
        return OllamaHealthResponse(
            connected=False, message=f"{error.message} {_START_HINT}", host=host
        )
    version = payload.get("version")
    version = version if isinstance(version, str) and version.strip() else None
    label = f"Ollama {version}" if version else "Ollama"
    return OllamaHealthResponse(
        connected=True, message=f"Connected to {label} at {host}.", host=host, version=version
    )


def installed_models(host: str) -> list[OllamaModel]:
    """Models from /api/tags, sorted by name; raises OllamaError."""
    payload = _request_json(host, "GET", "/api/tags")
    entries = payload.get("models")
    if not isinstance(entries, list):
        raise OllamaError(f"Ollama at {host} returned a malformed model list.")
    models = [_model(entry) for entry in entries if isinstance(entry, dict)]
    return sorted(
        (model for model in models if model is not None), key=lambda item: item.name.casefold()
    )


def list_models() -> OllamaModelsResponse:
    host = get_settings().ollama_host
    try:
        return OllamaModelsResponse(models=installed_models(host))
    except OllamaError as error:
        raise ApiError(
            503,
            "ollama_unreachable",
            f"{error.message} {_START_HINT}",
            details={"host": host},
            retryable=True,
        ) from error


def is_installed(model: str, models: list[OllamaModel]) -> bool:
    """Match Ollama's own name resolution: a name without a tag means `:latest`."""
    wanted = _with_tag(model.strip())
    return any(_with_tag(item.name) == wanted for item in models)


def model_context_length(host: str, model: str) -> int:
    """The model's architectural context length from /api/show; raises OllamaError.

    Reads ``model_info["<general.architecture>.context_length"]``, the value
    the pipeline's provider enforces as the model maximum.
    """
    key = (host, model)
    now = time.monotonic()
    with _context_lock:
        cached = _context_lengths.get(key)
    if cached is not None and cached[0] > now:
        return cached[1]
    payload = _request_json(host, "POST", "/api/show", {"model": model})
    info = payload.get("model_info")
    architecture = info.get("general.architecture") if isinstance(info, dict) else None
    maximum = info.get(f"{architecture}.context_length") if isinstance(architecture, str) else None
    if type(maximum) is not int or maximum <= 0:
        raise OllamaContextLengthUnknownError(
            f"Ollama did not report a context length for {model}."
        )
    with _context_lock:
        _context_lengths[key] = (now + _CONTEXT_CACHE_SECONDS, maximum)
    return maximum


def _with_tag(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def _model(entry: dict[str, Any]) -> OllamaModel | None:
    name = entry.get("name") or entry.get("model")
    if not isinstance(name, str) or not name.strip():
        return None
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    size = entry.get("size")
    return OllamaModel(
        name=name,
        size_bytes=size if type(size) is int and size >= 0 else None,
        parameter_size=_text(details.get("parameter_size")),
        family=_text(details.get("family")),
        quantization=_text(details.get("quantization_level")),
        modified_at=_text(entry.get("modified_at")),
    )


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
