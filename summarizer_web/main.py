"""FastAPI application entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from summarizer_web.api.v1.router import api_router
from summarizer_web.config import load_paths
from summarizer_web.db.connection import init_database
from summarizer_web.errors import error_body, install_error_handlers
from summarizer_web.security import SecurityMiddleware
from summarizer_web.worker.imports import get_import_manager
from summarizer_web.worker.supervisor import get_supervisor

_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"


class _ImmutableAssets(StaticFiles):
    """Serve SvelteKit's content-hashed bundles with a long cache lifetime."""

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = _IMMUTABLE_CACHE
        return response


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    supervisor = get_supervisor()
    imports = get_import_manager()
    supervisor.start()
    imports.start()
    try:
        yield
    finally:
        imports.shutdown()
        supervisor.shutdown()


def _frontend_dir() -> Path | None:
    build_dir = Path(__file__).resolve().parents[1] / "frontend" / "build"
    static_dir = build_dir if build_dir.exists() else load_paths().static
    return static_dir if (static_dir / "index.html").exists() else None


def create_app() -> FastAPI:
    init_database(load_paths())

    app = FastAPI(title="Recursive Summarizer", version="2.0.0", lifespan=_lifespan)
    install_error_handlers(app)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.include_router(api_router)

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def unknown_api_route(rest: str) -> JSONResponse:
        return JSONResponse(error_body("not_found", "Unknown API route."), status_code=404)

    static_dir = _frontend_dir()
    if static_dir is not None:
        root = static_dir.resolve()
        immutable_dir = root / "_app" / "immutable"
        if immutable_dir.exists():
            app.mount("/_app/immutable", _ImmutableAssets(directory=immutable_dir), name="immutable")
        index = root / "index.html"

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa_fallback(full_path: str) -> FileResponse:
            requested = (root / full_path).resolve()
            if full_path and requested.is_relative_to(root) and requested.is_file():
                return FileResponse(requested)
            return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
