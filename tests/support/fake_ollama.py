"""A scripted Ollama HTTP API served through httpx.MockTransport (no sockets)."""

from __future__ import annotations

import json
from collections.abc import Iterable

import httpx
import pytest

from summarizer_web.services import ollama_service


class FakeOllama:
    def __init__(
        self,
        models: Iterable[str | dict] = (),
        *,
        context_lengths: dict[str, int | None] | None = None,
        version: str = "0.12.3",
    ) -> None:
        self.models = [
            item if isinstance(item, dict) else {"name": item, "model": item} for item in models
        ]
        self.context_lengths = dict(context_lengths or {})
        self.version = version
        self.unreachable = False
        self.status_override: int | None = None
        self.requests: list[tuple[str, str, dict | None]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
        transport = httpx.MockTransport(self._handle)
        monkeypatch.setattr(
            ollama_service,
            "_client",
            lambda host: httpx.Client(base_url=host, transport=transport),
        )
        monkeypatch.setattr(ollama_service, "_context_lengths", {})
        return self

    def paths(self, path: str) -> list[tuple[str, dict | None]]:
        return [(host, body) for host, requested, body in self.requests if requested == path]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        self.requests.append((host, request.url.path, body))
        if self.unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if self.status_override is not None:
            return httpx.Response(self.status_override, text="unavailable")
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": self.models})
        if request.url.path == "/api/show":
            model = (body or {}).get("model")
            if model not in self.context_lengths:
                return httpx.Response(404, json={"error": f"model '{model}' not found"})
            info: dict[str, object] = {"general.architecture": "llama"}
            if self.context_lengths[model] is not None:
                info["llama.context_length"] = self.context_lengths[model]
            return httpx.Response(200, json={"model_info": info, "details": {"family": "llama"}})
        return httpx.Response(404, json={"error": "not found"})
