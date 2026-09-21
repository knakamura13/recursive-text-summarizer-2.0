from fastapi import APIRouter
from fastapi.responses import FileResponse

from summarizer_web.models.api import SourceLocationsResponse, SourceSliceResponse
from summarizer_web.services.documents_service import (
    get_original_path,
    get_source_locations,
    get_source_slice,
)

router = APIRouter(prefix="/documents", tags=["sources"])


@router.get("/{document_id}/source", response_model=SourceSliceResponse)
def source_slice(document_id: str, offset: int = 0, limit: int = 4000) -> SourceSliceResponse:
    return get_source_slice(document_id, offset=offset, limit=limit)


@router.get("/{document_id}/source/locations", response_model=SourceLocationsResponse)
def source_locations(document_id: str) -> SourceLocationsResponse:
    return get_source_locations(document_id)


@router.get("/{document_id}/original")
def original_artifact(document_id: str) -> FileResponse:
    path = get_original_path(document_id)
    return FileResponse(path)
