"""Document library: streamed uploads and pasted text, listing, detail, rename,
delete with cascade, and reads of a Document's canonical text.

Public helpers for other services: `latest_revision`, `get_document`,
`list_importing_documents`, `canonical_text`, `revision_pages`, and
`attachment_disposition`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import sqlite3
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import anyio
from fastapi import Request
from python_multipart.multipart import MultipartParser, parse_options_header

from summarizer_web.config import MAX_PASTE_BYTES, MAX_UPLOAD_BYTES, load_paths
from summarizer_web.db.connection import get_database
from summarizer_web.errors import ApiError
from summarizer_web.ingestion.common import Extraction, ImportFailure
from summarizer_web.ingestion.detect import SUPPORTED_EXTENSIONS, detect_format
from summarizer_web.ingestion.extract import assemble, build_report, count_words, preview_text
from summarizer_web.models.api import (
    DocumentCreatedResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    ImportProgress,
    ImportReport,
    PasteTextRequest,
    RunBrief,
    SourcePage,
    SourcePagesResponse,
    SourceSliceResponse,
)
from summarizer_web.worker.imports import (
    QUEUED_PROGRESS_JSON,
    get_import_manager,
    insert_revision,
    write_canonical_text,
)

_ACTIVE_RUN_STATES = ("queued", "running", "stopping")
_MIB = 1024 * 1024
# Multipart framing (boundaries, part headers) around the file bytes.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024
_FLUSH_BYTES = _MIB
_MAX_FIELD_BYTES = 64 * 1024
_MAX_PARTS = 16
_TITLE_FROM_TEXT_CHARS = 80
_MAX_TITLE_CHARS = 300

_ORIGINAL_SUFFIX: dict[str, str] = {
    "txt": "txt", "md": "md", "srt": "srt", "vtt": "vtt", "html": "html", "rtf": "rtf",
    "pdf": "pdf", "docx": "docx", "odt": "odt", "epub": "epub",
    "png": "png", "jpeg": "jpg", "tiff": "tiff",
}
_MEDIA_TYPES: dict[str, str] = {
    "txt": "text/plain",
    "md": "text/markdown",
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "html": "text/html",
    "rtf": "application/rtf",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "odt": "application/vnd.oasis.opendocument.text",
    "epub": "application/epub+zip",
    "png": "image/png",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
}

_SUMMARY_QUERY = """
    SELECT d.*,
           r.run_id AS latest_run_id,
           r.state AS latest_run_state,
           r.created_at AS latest_run_created_at,
           r.updated_at AS latest_run_updated_at
    FROM documents d
    LEFT JOIN runs r ON r.run_id = (
        SELECT run_id FROM runs WHERE document_id = d.document_id
        ORDER BY created_at DESC, rowid DESC LIMIT 1
    )
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _not_found() -> ApiError:
    return ApiError(404, "document_not_found", "The Document does not exist.")


# --- Reading ------------------------------------------------------------------


def latest_revision(document_id: str) -> sqlite3.Row | None:
    """The newest source revision of a Document; None until it is ready."""
    return get_database().fetchone(
        "SELECT * FROM source_revisions WHERE document_id = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (document_id,),
    )


@functools.lru_cache(maxsize=4)
def _read_text(path: str, modified_ns: int, size: int) -> str:
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def canonical_text(path: str | Path) -> str:
    """A revision's canonical text, cached by path, modification time, and size
    so large texts are not re-read for every slice. Raises OSError if missing."""
    status = os.stat(path)
    return _read_text(str(path), status.st_mtime_ns, status.st_size)


def revision_pages(revision: sqlite3.Row) -> list[SourcePage]:
    """Page offsets of a revision (code points into the canonical text, end
    exclusive); empty for formats without pages."""
    if not revision["page_map_json"]:
        return []
    legacy_blank = set(json.loads(revision["blank_pages_json"] or "[]"))
    return [
        SourcePage(
            page=entry["page"],
            start=entry["start"],
            end=entry["end"],
            ocr=bool(entry.get("ocr", False)),
            blank=bool(entry.get("blank", entry["page"] in legacy_blank)),
        )
        for entry in json.loads(revision["page_map_json"])
    ]


def _to_summary(row: sqlite3.Row) -> DocumentSummary:
    progress = row["import_progress_json"]
    latest = None
    if row["latest_run_id"] is not None:
        latest = RunBrief(
            run_id=row["latest_run_id"],
            state=row["latest_run_state"],
            created_at=row["latest_run_created_at"],
            updated_at=row["latest_run_updated_at"],
        )
    return DocumentSummary(
        document_id=row["document_id"],
        title=row["title"],
        filename=row["filename"],
        format=row["format"],
        origin=row["origin"],
        size_bytes=row["size_bytes"],
        import_state=row["import_state"],
        import_progress=(
            ImportProgress.model_validate_json(progress)
            if progress and row["import_state"] == "importing"
            else None
        ),
        import_error=row["import_error"],
        char_count=row["char_count"],
        page_count=row["page_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        latest_run=latest,
    )


def _summary(document_id: str) -> DocumentSummary:
    row = get_database().fetchone(_SUMMARY_QUERY + " WHERE d.document_id = ?", (document_id,))
    if row is None:
        raise _not_found()
    return _to_summary(row)


def list_documents(search: str | None = None) -> DocumentListResponse:
    """Documents, most recently updated first; `search` matches title or
    filename, case-insensitively."""
    rows = get_database().fetchall(_SUMMARY_QUERY + " ORDER BY d.updated_at DESC")
    needle = (search or "").strip().casefold()
    return DocumentListResponse(
        documents=[
            _to_summary(row)
            for row in rows
            if not needle
            or needle in row["title"].casefold()
            or needle in row["filename"].casefold()
        ]
    )


def list_importing_documents() -> list[DocumentSummary]:
    """Documents whose Import is queued or running, oldest first."""
    rows = get_database().fetchall(
        _SUMMARY_QUERY + " WHERE d.import_state = 'importing' ORDER BY d.created_at"
    )
    return [_to_summary(row) for row in rows]


def _legacy_report(summary: DocumentSummary, revision: sqlite3.Row) -> ImportReport | None:
    """An Import report for Documents imported before reports were stored."""
    try:
        text = canonical_text(revision["canonical_path"])
    except OSError:
        return None
    pages = revision_pages(revision)
    return ImportReport(
        detected_format=summary.format,
        page_count=len(pages) if revision["page_map_json"] else None,
        blank_pages=[page.page for page in pages if page.blank],
        char_count=len(text),
        word_count=count_words(text),
        preview=preview_text(text),
        extraction_version=revision["extraction_version"],
    )


def get_document(document_id: str) -> DocumentDetailResponse:
    """Raises ApiError(404, "document_not_found") for unknown ids."""
    summary = _summary(document_id)
    revision = latest_revision(document_id)
    report = None
    if revision is not None:
        report = (
            ImportReport.model_validate_json(revision["report_json"])
            if revision["report_json"]
            else _legacy_report(summary, revision)
        )
    return DocumentDetailResponse(
        **summary.model_dump(),
        source_sha256=revision["source_sha256"] if revision is not None else None,
        import_report=report,
    )


def _ready_revision(document_id: str) -> sqlite3.Row:
    row = get_database().fetchone(
        "SELECT import_state FROM documents WHERE document_id = ?", (document_id,)
    )
    if row is None:
        raise _not_found()
    if row["import_state"] == "importing":
        raise ApiError(409, "document_not_ready", "The Document is still importing.")
    if row["import_state"] == "failed":
        raise ApiError(409, "document_not_ready", "The import of this Document failed; it has no text.")
    revision = latest_revision(document_id)
    if revision is None:
        raise ApiError(404, "source_not_found", "The Document has no extracted text.")
    return revision


def get_source_slice(document_id: str, offset: int, limit: int) -> SourceSliceResponse:
    revision = _ready_revision(document_id)
    try:
        text = canonical_text(revision["canonical_path"])
    except OSError as error:
        raise ApiError(404, "source_not_found", "The text file of this Document is missing.") from error
    total = len(text)
    start = min(offset, total)
    end = min(total, start + limit)
    return SourceSliceResponse(text=text[start:end], offset=start, total_length=total, has_more=end < total)


def get_source_pages(document_id: str) -> SourcePagesResponse:
    return SourcePagesResponse(pages=revision_pages(_ready_revision(document_id)))


@dataclass(frozen=True)
class OriginalFile:
    path: Path
    filename: str
    media_type: str


def get_original(document_id: str) -> OriginalFile:
    row = get_database().fetchone(
        "SELECT filename, format, upload_path FROM documents WHERE document_id = ?", (document_id,)
    )
    if row is None:
        raise _not_found()
    location = row["upload_path"]
    if not location:
        revision = latest_revision(document_id)
        location = revision["original_path"] if revision is not None else None
    if not location or not Path(location).is_file():
        raise ApiError(404, "original_not_found", "The original file of this Document is not available.")
    return OriginalFile(
        Path(location), row["filename"], _MEDIA_TYPES.get(row["format"], "application/octet-stream")
    )


def attachment_disposition(filename: str) -> str:
    """Content-Disposition for a download: an ASCII fallback plus the exact
    UTF-8 name (RFC 6266 / RFC 5987)."""
    fallback = "".join(
        char if 0x20 <= ord(char) < 0x7F and char not in '"\\' else "_"
        for char in unicodedata.normalize("NFKD", filename)
        if not unicodedata.combining(char)
    ).strip() or "download"
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"


# --- Creating -----------------------------------------------------------------


def _safe_filename(name: str) -> str:
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    base = "".join(char for char in base if unicodedata.category(char) != "Cc").strip()
    if len(base) > 255:
        stem, dot, suffix = base.rpartition(".")
        base = (stem[: 250 - len(suffix)] + dot + suffix) if dot and len(suffix) <= 10 else base[:255]
    return base or "document"


def _title_from_filename(filename: str) -> str:
    stem = Path(filename).stem.strip() or filename
    return " ".join(stem.split())[:_MAX_TITLE_CHARS]


def _title_from_text(text: str) -> str:
    first_line = text.lstrip().split("\n", 1)[0]
    line = " ".join(first_line.split())
    line = line.lstrip("#>*-\u2022 ").strip() or line
    if not line:
        return "Pasted text"
    if len(line) <= _TITLE_FROM_TEXT_CHARS:
        return line
    cut = line[: _TITLE_FROM_TEXT_CHARS - 1]
    space = cut.rfind(" ")
    if space >= _TITLE_FROM_TEXT_CHARS // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-") + "\u2026"


def _file_too_large() -> ApiError:
    return ApiError(
        413,
        "file_too_large",
        f"The file is larger than the {MAX_UPLOAD_BYTES // _MIB} MiB upload limit.",
        details={"limit_bytes": MAX_UPLOAD_BYTES},
    )


def _invalid_upload(message: str) -> ApiError:
    return ApiError(400, "invalid_upload", message)


class _UploadTooLarge(Exception):
    pass


class _UploadReceiver:
    """Parses a multipart/form-data body chunk by chunk and streams the `file`
    part to disk while hashing it, enforcing the size limit as bytes arrive."""

    def __init__(self, boundary: bytes, destination: Path, limit: int) -> None:
        self._destination = destination
        self._limit = limit
        self._handle = None
        self._buffer = bytearray()
        self._digest = hashlib.sha256()
        self._parts = 0
        self._header_name = bytearray()
        self._header_value = bytearray()
        self._disposition = b""
        self._in_file = False
        self._field_bytes = 0
        self.filename: str | None = None
        self.size = 0
        self.complete = False
        self._parser = MultipartParser(
            boundary,
            callbacks={
                "on_part_begin": self._on_part_begin,
                "on_part_data": self._on_part_data,
                "on_part_end": self._on_part_end,
                "on_header_field": self._on_header_field,
                "on_header_value": self._on_header_value,
                "on_header_end": self._on_header_end,
                "on_headers_finished": self._on_headers_finished,
                "on_end": self._on_end,
            },
        )

    # Parser callbacks.

    def _on_part_begin(self) -> None:
        self._parts += 1
        if self._parts > _MAX_PARTS:
            raise _invalid_upload("The upload has too many form fields.")
        self._disposition = b""
        self._in_file = False
        self._field_bytes = 0

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name += data[start:end]

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _on_header_end(self) -> None:
        if bytes(self._header_name).lower() == b"content-disposition":
            self._disposition = bytes(self._header_value)
        self._header_name.clear()
        self._header_value.clear()

    def _on_headers_finished(self) -> None:
        _, options = parse_options_header(self._disposition)
        if options.get(b"name") == b"file" and b"filename" in options and self.filename is None:
            raw = options[b"filename"]
            try:
                name = raw.decode("utf-8")
            except UnicodeDecodeError:
                name = raw.decode("latin-1")
            self.filename = _safe_filename(name)
            self._in_file = True

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._in_file:
            self.size += end - start
            if self.size > self._limit:
                raise _UploadTooLarge
            self._buffer += data[start:end]
        else:
            self._field_bytes += end - start
            if self._field_bytes > _MAX_FIELD_BYTES:
                raise _invalid_upload("A form field of the upload is too large.")

    def _on_part_end(self) -> None:
        self._in_file = False

    def _on_end(self) -> None:
        self.complete = True

    # Feeding and writing.

    def feed(self, chunk: bytes) -> None:
        self._parser.write(chunk)

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    def flush(self) -> None:
        """Write buffered file bytes; runs on a worker thread."""
        if not self._buffer:
            return
        data, self._buffer = self._buffer, bytearray()
        if self._handle is None:
            self._handle = self._destination.open("wb")
        self._handle.write(data)
        self._digest.update(data)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        elif self.filename is not None:
            self._destination.touch()

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _multipart_boundary(content_type: str | None) -> bytes:
    kind, options = parse_options_header(content_type or "")
    boundary = options.get(b"boundary")
    if kind != b"multipart/form-data" or not boundary:
        raise _invalid_upload('Upload the file as multipart/form-data in the field "file".')
    return boundary


def _check_declared_length(value: str | None) -> None:
    if value is None:
        return
    try:
        declared = int(value)
    except ValueError as error:
        raise _invalid_upload("The Content-Length header is not a number.") from error
    if declared > MAX_UPLOAD_BYTES + _MULTIPART_OVERHEAD_BYTES:
        raise _file_too_large()


def _accept_upload(
    document_id: str, directory: Path, partial: Path, filename: str, size: int, sha256: str
) -> DocumentCreatedResponse:
    """Sniff the stored upload, deduplicate it, and create the Document row.
    Runs on a worker thread."""
    if size == 0:
        raise ApiError(400, "empty_file", "The file is empty.")
    detection = detect_format(partial, filename)
    if detection.format is None:
        raise ApiError(
            415,
            "unsupported_format",
            f"{detection.reason} Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}.",
            details={"supported": list(SUPPORTED_EXTENSIONS), "filename": filename},
        )
    original = directory / f"original.{_ORIGINAL_SUFFIX[detection.format]}"
    partial.rename(original)
    now = _now()
    with get_database().transaction() as connection:
        existing = connection.execute(
            "SELECT document_id FROM documents "
            "WHERE original_sha256 = ? AND import_state != 'failed' "
            "ORDER BY created_at LIMIT 1",
            (sha256,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, title, filename, format, size_bytes, import_state, created_at,
                    updated_at, origin, import_progress_json, original_sha256, upload_path
                ) VALUES (?, ?, ?, ?, ?, 'importing', ?, ?, 'upload', ?, ?, ?)
                """,
                (
                    document_id,
                    _title_from_filename(filename),
                    filename,
                    detection.format,
                    size,
                    now,
                    now,
                    QUEUED_PROGRESS_JSON,
                    sha256,
                    str(original),
                ),
            )
    if existing is not None:
        shutil.rmtree(directory, ignore_errors=True)
        return DocumentCreatedResponse(document=_summary(existing["document_id"]), already_imported=True)
    get_import_manager().enqueue(document_id)
    return DocumentCreatedResponse(document=_summary(document_id))


async def upload_document(request: Request) -> DocumentCreatedResponse:
    """Stream a multipart upload (field `file`) into documents/<id>/ and queue
    its Import. Oversized bodies are refused from Content-Length before any
    byte is read, and otherwise as soon as the file part passes the limit."""
    boundary = _multipart_boundary(request.headers.get("content-type"))
    _check_declared_length(request.headers.get("content-length"))
    document_id = str(uuid.uuid4())
    directory = load_paths().documents / document_id
    directory.mkdir(mode=0o700)
    partial = directory / "upload.part"
    receiver = _UploadReceiver(boundary, partial, MAX_UPLOAD_BYTES)
    try:
        try:
            async for chunk in request.stream():
                receiver.feed(chunk)
                if receiver.buffered >= _FLUSH_BYTES:
                    await anyio.to_thread.run_sync(receiver.flush)
            await anyio.to_thread.run_sync(receiver.flush)
        except _UploadTooLarge:
            raise _file_too_large() from None
        finally:
            receiver.close()
        if not receiver.complete:
            raise _invalid_upload("The upload ended before the file was complete.")
        if receiver.filename is None:
            raise ApiError(422, "invalid_request", 'Attach the document in the multipart field "file".')
        return await anyio.to_thread.run_sync(
            _accept_upload,
            document_id,
            directory,
            partial,
            receiver.filename,
            receiver.size,
            receiver.hexdigest(),
        )
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def create_pasted_document(payload: PasteTextRequest) -> DocumentCreatedResponse:
    """Store pasted text as a ready Document (origin `paste`, format `txt`)."""
    started = time.monotonic()
    encoded = payload.text.encode("utf-8", errors="replace")
    if len(encoded) > MAX_PASTE_BYTES:
        raise ApiError(
            413,
            "text_too_large",
            f"Pasted text is limited to {MAX_PASTE_BYTES // _MIB} MiB.",
            details={"limit_bytes": MAX_PASTE_BYTES},
        )
    extraction = Extraction("txt", text=encoded.decode("utf-8"), encoding="utf-8")
    try:
        imported = assemble(extraction)
    except ImportFailure as error:
        raise ApiError(400, "empty_text", "The pasted text contains no words.") from error
    title = " ".join((payload.title or "").split())[:_MAX_TITLE_CHARS] or _title_from_text(imported.text)
    report = build_report(extraction, imported, duration_seconds=time.monotonic() - started)
    document_id = str(uuid.uuid4())
    directory = load_paths().documents / document_id
    directory.mkdir(mode=0o700)
    try:
        original = directory / "original.txt"
        original.write_bytes(encoded)
        canonical = write_canonical_text(directory, imported.text)
        now = _now()
        with get_database().transaction() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, title, filename, format, size_bytes, import_state, created_at,
                    updated_at, origin, upload_path, char_count, page_count
                ) VALUES (?, ?, ?, 'txt', ?, 'ready', ?, ?, 'paste', ?, ?, NULL)
                """,
                (
                    document_id,
                    title,
                    _safe_filename(f"{title.rstrip(chr(0x2026))}.txt"),
                    len(encoded),
                    now,
                    now,
                    str(original),
                    report.char_count,
                ),
            )
            insert_revision(
                connection,
                document_id=document_id,
                imported=imported,
                report=report,
                canonical=canonical,
                original=original,
                created_at=now,
            )
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return DocumentCreatedResponse(document=_summary(document_id))


# --- Changing -----------------------------------------------------------------


def rename_document(document_id: str, title: str) -> DocumentDetailResponse:
    cleaned = " ".join(title.split())
    if not cleaned:
        raise ApiError(422, "invalid_request", "The title must not be empty.")
    cursor = get_database().execute(
        "UPDATE documents SET title = ?, updated_at = ? WHERE document_id = ?",
        (cleaned, _now(), document_id),
    )
    if cursor.rowcount == 0:
        raise _not_found()
    return get_document(document_id)


def delete_document(document_id: str) -> None:
    """Delete a Document with its Runs (attempts, events, node projections,
    segments) in one transaction, then its files and Run directories. An
    Import in progress is stopped first; an active Run blocks deletion."""
    database = get_database()
    row = database.fetchone("SELECT import_state FROM documents WHERE document_id = ?", (document_id,))
    if row is None:
        raise _not_found()
    if row["import_state"] == "importing":
        get_import_manager().cancel(document_id)
    placeholders = ", ".join("?" for _ in _ACTIVE_RUN_STATES)
    with database.transaction() as connection:
        document = connection.execute(
            "SELECT title FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if document is None:
            raise _not_found()
        active = connection.execute(
            f"SELECT run_id, state FROM runs WHERE document_id = ? AND state IN ({placeholders}) "
            "ORDER BY created_at DESC LIMIT 1",
            (document_id, *_ACTIVE_RUN_STATES),
        ).fetchone()
        if active is not None:
            raise ApiError(
                409,
                "run_active",
                f"A Run of this Document is {active['state']}. Stop it before deleting the Document.",
                details={
                    "run_id": active["run_id"],
                    "document_id": document_id,
                    "document_title": document["title"],
                    "state": active["state"],
                },
            )
        run_ids = [
            run["run_id"]
            for run in connection.execute("SELECT run_id FROM runs WHERE document_id = ?", (document_id,))
        ]
        # Attempts, events, node projections, and segments cascade from runs;
        # source revisions cascade from the document.
        connection.execute("DELETE FROM runs WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
    paths = load_paths()
    shutil.rmtree(paths.documents / document_id, ignore_errors=True)
    for run_id in run_ids:
        shutil.rmtree(paths.runs / run_id, ignore_errors=True)
