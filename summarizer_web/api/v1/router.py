from fastapi import APIRouter

from summarizer_web.api.v1 import documents, exports, health, ollama, preflight, runs, settings, sources

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(settings.router)
api_router.include_router(ollama.router)
api_router.include_router(documents.router)
api_router.include_router(sources.router)
api_router.include_router(preflight.router)
api_router.include_router(runs.router)
api_router.include_router(exports.router)
