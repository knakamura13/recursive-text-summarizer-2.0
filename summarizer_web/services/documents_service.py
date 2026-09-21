"""Document library operations."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile

from summarizer.ingestion import (
    EmptySourceError,
    SourceDecodeError,
    ingest_text,
    read_source,
)
from summarizer_web.config import EXTRACTION_VERSION, MAX_UPLOAD_BYTES, load_paths
from summarizer_web.db.connection import get_database
from summarizer_web.ingestion.pdf import PdfImportError, extract_pdf, serialize_page_map
from summarizer_web.models.api import (
    ConfirmExtractionResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    SourceLocationsResponse,
    SourceSliceResponse,
)

_ACTIVE_RUN_STATES = {"queued", "running", "cancelling"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_from_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".md":
        return "md"
    if suffix == ".pdf":
        return "pdf"
    return "txt"


def _document_dir(document_id: str) -> Path:
    return load_paths().documents / document_id


def _latest_revision(document_id: str):
    return get_database().fetchone(
        """
        SELECT * FROM source_revisions
        WHERE document_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (document_id,),
    )


def _latest_run_state(document_id: str) -> str | None:
    row = get_database().fetchone(
        "SELECT state FROM runs WHERE document_id = ? ORDER BY created_at DESC LIMIT 1",
        (document_id,),
    )
    return row["state"] if row is not None else None


def _to_summary(row) -> DocumentSummary:
    return DocumentSummary(
        document_id=row["document_id"],
        title=row["title"],
        filename=row["filename"],
        format=row["format"],
        size_bytes=row["size_bytes"],
        import_state=row["import_state"],
        latest_run_state=_latest_run_state(row["document_id"]),
    )


def list_documents(search: str | None = None) -> DocumentListResponse:
    db = get_database()
    if search:
        pattern = f"%{search.strip()}%"
        rows = db.fetchall(
            """
            SELECT * FROM documents
            WHERE title LIKE ? OR filename LIKE ?
            ORDER BY updated_at DESC
            """,
            (pattern, pattern),
        )
    else:
        rows = db.fetchall("SELECT * FROM documents ORDER BY updated_at DESC")
    return DocumentListResponse(documents=[_to_summary(row) for row in rows])


def get_document(document_id: str) -> DocumentDetailResponse:
    row = get_database().fetchone("SELECT * FROM documents WHERE document_id = ?", (document_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    revision = _latest_revision(document_id)
    blank_pages: list[int] = []
    page_count = None
    revision_id = None
    source_sha256 = None
    latest_run_id = None
    if revision is not None:
        revision_id = revision["revision_id"]
        source_sha256 = revision["source_sha256"]
        if revision["blank_pages_json"]:
            blank_pages = json.loads(revision["blank_pages_json"])
        if revision["page_map_json"]:
            page_count = len(json.loads(revision["page_map_json"]))
    run_row = get_database().fetchone(
        "SELECT run_id FROM runs WHERE document_id = ? ORDER BY created_at DESC LIMIT 1",
        (document_id,),
    )
    if run_row is not None:
        latest_run_id = run_row["run_id"]
    summary = _to_summary(row)
    return DocumentDetailResponse(
        **summary.model_dump(),
        revision_id=revision_id,
        source_sha256=source_sha256,
        page_count=page_count,
        blank_pages=blank_pages,
        latest_run_id=latest_run_id,
    )


def _store_revision(
    *,
    document_id: str,
    canonical_text: str,
    source_sha256: str,
    original_path: Path | None,
    page_map: list[dict[str, int]] | None,
    blank_pages: list[int],
    import_state: str,
) -> str:
    revision_id = str(uuid.uuid4())
    doc_dir = _document_dir(document_id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = doc_dir / "canonical.txt"
    canonical_path.write_text(canonical_text, encoding="utf-8")
    db = get_database()
    db.execute(
        """
        INSERT INTO source_revisions (
            revision_id, document_id, source_sha256, extraction_version,
            canonical_path, original_path, page_map_json, blank_pages_json,
            confirmed_at, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            revision_id,
            document_id,
            source_sha256,
            EXTRACTION_VERSION,
            str(canonical_path),
            str(original_path) if original_path is not None else None,
            serialize_page_map(page_map) if page_map is not None else None,
            json.dumps(blank_pages),
            _now() if import_state == "ready" else None,
            _now(),
        ),
    )
    return revision_id


def _find_duplicate(source_sha256: str, extraction_version: str):
    return get_database().fetchone(
        """
        SELECT d.* FROM documents d
        JOIN source_revisions sr ON sr.document_id = d.document_id
        WHERE sr.source_sha256 = ? AND sr.extraction_version = ?
        ORDER BY d.created_at ASC
        LIMIT 1
        """,
        (source_sha256, extraction_version),
    )


async def upload_document(file: UploadFile) -> DocumentListResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 50 MiB upload limit")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    format_name = _format_from_filename(file.filename)
    doc_format = format_name
    import_state = "ready"
    page_map = None
    blank_pages: list[int] = []
    original_path: Path | None = None

    if format_name == "pdf":
        temp_dir = load_paths().documents / f".tmp-{uuid.uuid4()}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        original_path = temp_dir / "original.pdf"
        original_path.write_bytes(content)
        try:
            extracted = extract_pdf(original_path)
        except PdfImportError as error:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(error)) from error
        canonical_text = extracted.canonical_text
        source_sha256 = extracted.source_id
        page_map = extracted.page_map
        blank_pages = extracted.blank_pages
        import_state = "pending_confirmation"
    else:
        try:
            text = content.decode("utf-8")
            document = ingest_text(text)
        except UnicodeDecodeError as error:
            raise HTTPException(status_code=400, detail="File must be valid UTF-8") from error
        except EmptySourceError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        canonical_text = document.text
        source_sha256 = document.source_id
        original_path = None

    duplicate = _find_duplicate(source_sha256, EXTRACTION_VERSION)
    if duplicate is not None:
        return DocumentListResponse(
            documents=[_to_summary(duplicate)],
            already_imported=True,
        )

    document_id = str(uuid.uuid4())
    title = Path(file.filename).stem
    now = _now()
    db = get_database()
    db.execute(
        """
        INSERT INTO documents (
            document_id, title, filename, format, size_bytes, import_state, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            title,
            file.filename,
            doc_format,
            len(content),
            import_state,
            now,
            now,
        ),
    )
    doc_dir = _document_dir(document_id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    if format_name == "pdf":
        final_original = doc_dir / "original.pdf"
        if original_path is not None:
            shutil.move(str(original_path), final_original)
            shutil.rmtree(original_path.parent, ignore_errors=True)
        original_path = final_original
    else:
        original_path = doc_dir / f"original.{format_name}"
        original_path.write_bytes(content)

    _store_revision(
        document_id=document_id,
        canonical_text=canonical_text,
        source_sha256=source_sha256,
        original_path=original_path,
        page_map=page_map,
        blank_pages=blank_pages,
        import_state=import_state,
    )
    row = db.fetchone("SELECT * FROM documents WHERE document_id = ?", (document_id,))
    return DocumentListResponse(documents=[_to_summary(row)])


def rename_document(document_id: str, title: str) -> DocumentDetailResponse:
    title = title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title must not be empty")
    db = get_database()
    row = db.fetchone("SELECT document_id FROM documents WHERE document_id = ?", (document_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    db.execute(
        "UPDATE documents SET title = ?, updated_at = ? WHERE document_id = ?",
        (title, _now(), document_id),
    )
    return get_document(document_id)


def delete_document(document_id: str) -> None:
    active = get_database().fetchone(
        """
        SELECT run_id FROM runs
        WHERE document_id = ? AND state IN ('queued', 'running', 'cancelling')
        LIMIT 1
        """,
        (document_id,),
    )
    if active is not None:
        raise HTTPException(status_code=409, detail="Cannot delete document with an active run")
    row = get_database().fetchone("SELECT document_id FROM documents WHERE document_id = ?", (document_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    get_database().execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
    shutil.rmtree(_document_dir(document_id), ignore_errors=True)


def confirm_extraction(document_id: str) -> ConfirmExtractionResponse:
    row = get_database().fetchone("SELECT * FROM documents WHERE document_id = ?", (document_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if row["format"] != "pdf":
        raise HTTPException(status_code=400, detail="Only PDF imports require confirmation")
    revision = _latest_revision(document_id)
    if revision is None:
        raise HTTPException(status_code=400, detail="Missing source revision")
    get_database().execute(
        "UPDATE documents SET import_state = 'ready', updated_at = ? WHERE document_id = ?",
        (_now(), document_id),
    )
    get_database().execute(
        "UPDATE source_revisions SET confirmed_at = ? WHERE revision_id = ?",
        (_now(), revision["revision_id"]),
    )
    return ConfirmExtractionResponse(document_id=document_id, import_state="ready")


def get_source_slice(document_id: str, offset: int = 0, limit: int = 4000) -> SourceSliceResponse:
    revision = _latest_revision(document_id)
    if revision is None:
        raise HTTPException(status_code=404, detail="Source revision not found")
    text = Path(revision["canonical_path"]).read_text(encoding="utf-8")
    total = len(text)
    offset = max(0, min(offset, total))
    end = min(total, offset + max(1, limit))
    slice_text = text[offset:end]
    return SourceSliceResponse(
        text=slice_text,
        offset=offset,
        total_length=total,
        has_more=end < total,
    )


def get_source_locations(document_id: str) -> SourceLocationsResponse:
    revision = _latest_revision(document_id)
    if revision is None:
        raise HTTPException(status_code=404, detail="Source revision not found")
    page_map = json.loads(revision["page_map_json"] or "[]")
    text = Path(revision["canonical_path"]).read_text(encoding="utf-8")
    segments = []
    for entry in page_map:
        segments.append(
            {
                "page": entry["page"],
                "start": entry["start"],
                "end": entry["end"],
                "preview": text[entry["start"] : min(entry["end"], entry["start"] + 120)],
            }
        )
    if not segments:
        segments.append({"segment_id": "document", "start": 0, "end": len(text)})
    return SourceLocationsResponse(segments=segments)


def get_original_path(document_id: str) -> Path:
    revision = _latest_revision(document_id)
    if revision is None or revision["original_path"] is None:
        raise HTTPException(status_code=404, detail="Original artifact not found")
    path = Path(revision["original_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Original artifact missing")
    return path
