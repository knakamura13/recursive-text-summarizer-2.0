"""FastAPI application entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from summarizer_web.api.v1.router import api_router
from summarizer_web.config import load_paths
from summarizer_web.db.connection import init_database
from summarizer_web.security import SecurityMiddleware
from summarizer_web.worker.supervisor import get_supervisor


def create_app() -> FastAPI:
    paths = load_paths()
    init_database(paths)
    get_supervisor().start()

    app = FastAPI(title="Recursive Summarizer", version="1.0.0")
    app.add_middleware(SecurityMiddleware)
    app.include_router(api_router)

    build_dir = Path(__file__).resolve().parents[1] / "frontend" / "build"
    static_dir = build_dir if build_dir.exists() else paths.static
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    if static_dir.exists() and (static_dir / "index.html").exists():

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str):
            if full_path.startswith("api/"):
                return {"detail": "Not Found"}
            requested = static_dir / full_path
            if requested.is_file():
                return FileResponse(requested)
            return FileResponse(static_dir / "index.html")

    return app


app = create_app()
