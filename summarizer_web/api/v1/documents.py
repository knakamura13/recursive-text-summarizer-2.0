from fastapi import APIRouter, Request, Response

from summarizer_web.models.api import (
    DocumentCreatedResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentRenameRequest,
    PasteTextRequest,
)
from summarizer_web.services.documents_service import (
    create_pasted_document,
    delete_document,
    get_document,
    list_documents,
    rename_document,
    upload_document,
)

router = APIRouter(prefix="/documents", tags=["documents"])

# The upload is parsed from the raw stream (see upload_document), so the
# multipart body is described here for the OpenAPI schema.
_UPLOAD_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {"file": {"type": "string", "format": "binary"}},
                }
            }
        },
    }
}


@router.get("", response_model=DocumentListResponse)
def get_documents(search: str | None = None) -> DocumentListResponse:
    return list_documents(search)


@router.post(
    "",
    response_model=DocumentCreatedResponse,
    status_code=201,
    openapi_extra=_UPLOAD_BODY,
    responses={200: {"model": DocumentCreatedResponse, "description": "Already imported"}},
)
async def post_document(request: Request, response: Response) -> DocumentCreatedResponse:
    created = await upload_document(request)
    if created.already_imported:
        response.status_code = 200
    return created


@router.post("/text", response_model=DocumentCreatedResponse, status_code=201)
def post_text(payload: PasteTextRequest) -> DocumentCreatedResponse:
    return create_pasted_document(payload)


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document_detail(document_id: str) -> DocumentDetailResponse:
    return get_document(document_id)


@router.patch("/{document_id}", response_model=DocumentDetailResponse)
def patch_document(document_id: str, payload: DocumentRenameRequest) -> DocumentDetailResponse:
    return rename_document(document_id, payload.title)


@router.delete("/{document_id}", status_code=204)
def remove_document(document_id: str) -> Response:
    delete_document(document_id)
    return Response(status_code=204)
