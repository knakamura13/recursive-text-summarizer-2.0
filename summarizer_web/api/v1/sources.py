from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from summarizer_web.models.api import SourcePagesResponse, SourceSliceResponse
from summarizer_web.services.documents_service import (
    attachment_disposition,
    get_original,
    get_source_pages,
    get_source_slice,
)

router = APIRouter(prefix="/documents", tags=["sources"])


@router.get("/{document_id}/source", response_model=SourceSliceResponse)
def source_slice(
    document_id: str,
    offset: int = Query(0, ge=0, description="Start, in code points."),
    limit: int = Query(16384, ge=1, le=65536, description="Length, in code points."),
) -> SourceSliceResponse:
    return get_source_slice(document_id, offset=offset, limit=limit)


@router.get("/{document_id}/pages", response_model=SourcePagesResponse)
def source_pages(document_id: str) -> SourcePagesResponse:
    return get_source_pages(document_id)


@router.get("/{document_id}/original", response_class=FileResponse)
def original_file(document_id: str) -> FileResponse:
    original = get_original(document_id)
    return FileResponse(
        original.path,
        media_type=original.media_type,
        headers={
            "Content-Disposition": attachment_disposition(original.filename),
            "X-Content-Type-Options": "nosniff",
        },
    )
