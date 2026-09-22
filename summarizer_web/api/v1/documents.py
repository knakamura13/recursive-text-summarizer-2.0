from fastapi import APIRouter, File, UploadFile

from summarizer_web.models.api import (
    ConfirmExtractionResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentRenameRequest,
)
from summarizer_web.services.documents_service import (
    confirm_extraction,
    delete_document,
    get_document,
    list_documents,
    rename_document,
    upload_document,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
def get_documents(search: str | None = None) -> DocumentListResponse:
    return list_documents(search)


@router.post("", response_model=DocumentListResponse)
async def post_document(file: UploadFile = File(...)) -> DocumentListResponse:
    return await upload_document(file)


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document_detail(document_id: str) -> DocumentDetailResponse:
    return get_document(document_id)


@router.patch("/{document_id}", response_model=DocumentDetailResponse)
def patch_document(document_id: str, payload: DocumentRenameRequest) -> DocumentDetailResponse:
    return rename_document(document_id, payload.title)


@router.delete("/{document_id}", status_code=204)
def remove_document(document_id: str) -> None:
    delete_document(document_id)


@router.post("/{document_id}/confirm-extraction", response_model=ConfirmExtractionResponse)
def post_confirm_extraction(document_id: str) -> ConfirmExtractionResponse:
    return confirm_extraction(document_id)
