"""Background Imports (D5).

The web process owns one ImportManager. It runs one Import at a time, each in
a spawned child process that extracts the text (with OCR where needed),
writes its progress to the Document row at most ~4 times per second, and
records the outcome: the Document becomes `ready` with a source revision and
Import report, or `failed` with a reason. Deleting a Document stops its Import
by terminating the child. On startup, Documents left `importing` are queued
again when their upload is still on disk and fail otherwise.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import shutil
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing.connection import wait as wait_for_exit
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Protocol

from summarizer_web.config import EXTRACTION_VERSION, IMPORT_PROGRESS_INTERVAL_SECONDS, AppPaths
from summarizer_web.db.connection import Database, get_database, init_database
from summarizer_web.ingestion.common import ImportFailure, ProgressUnit
from summarizer_web.ingestion.extract import ImportedText, assemble, build_report, extract_document
from summarizer_web.ingestion.ocr import Tesseract, kill_running_ocr
from summarizer_web.models.api import ImportPhase, ImportProgress, ImportReport

logger = logging.getLogger(__name__)

QUEUED_PROGRESS_JSON = ImportProgress(phase="queued").model_dump_json()
_MISSING_UPLOAD = (
    "The uploaded file is missing, so the import cannot continue. "
    "Delete this Document and upload the file again."
)
_EXIT_GRACE_SECONDS = 5.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_failed(database: Database, document_id: str, reason: str) -> None:
    database.execute(
        """
        UPDATE documents
        SET import_state = 'failed', import_error = ?, import_progress_json = NULL, updated_at = ?
        WHERE document_id = ? AND import_state = 'importing'
        """,
        (reason, _now(), document_id),
    )


# --- Child process --------------------------------------------------------------


class _ProgressWriter:
    """Writes progress to the Document row: on every phase change, when a phase
    finishes, and otherwise at most every IMPORT_PROGRESS_INTERVAL_SECONDS."""

    def __init__(self, database: Database, document_id: str) -> None:
        self._database = database
        self._document_id = document_id
        self._phase: ImportPhase | None = None
        self._written_at = 0.0

    def __call__(
        self,
        phase: ImportPhase,
        done: int = 0,
        total: int | None = None,
        *,
        unit: ProgressUnit | None = None,
        message: str | None = None,
    ) -> None:
        now = time.monotonic()
        finished = total is not None and done >= total
        if (
            phase == self._phase
            and not finished
            and now - self._written_at < IMPORT_PROGRESS_INTERVAL_SECONDS
        ):
            return
        self._phase = phase
        self._written_at = now
        progress = ImportProgress(phase=phase, done=done, total=total, unit=unit, message=message)
        self._database.execute(
            "UPDATE documents SET import_progress_json = ? "
            "WHERE document_id = ? AND import_state = 'importing'",
            (progress.model_dump_json(), self._document_id),
        )


def write_canonical_text(directory: Path, text: str) -> Path:
    """Write a Document's canonical text atomically; returns its path."""
    canonical = directory / "canonical.txt"
    partial = directory / "canonical.txt.part"
    with partial.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(partial, canonical)
    return canonical


def insert_revision(
    connection: sqlite3.Connection,
    *,
    document_id: str,
    imported: ImportedText,
    report: ImportReport,
    canonical: Path,
    original: Path,
    created_at: str,
) -> None:
    """Record the source revision (canonical text, page map, Import report)."""
    pages = imported.pages
    connection.execute(
        """
        INSERT INTO source_revisions (
            revision_id, document_id, source_sha256, extraction_version, canonical_path,
            original_path, page_map_json, blank_pages_json, ocr_pages_json, report_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            document_id,
            imported.source_id,
            EXTRACTION_VERSION,
            str(canonical),
            str(original),
            json.dumps([span.as_json() for span in pages]) if pages is not None else None,
            json.dumps(report.blank_pages) if pages is not None else None,
            json.dumps(report.ocr_pages) if pages is not None else None,
            report.model_dump_json(),
            created_at,
        ),
    )


def _record_ready(
    database: Database,
    paths: AppPaths,
    document_id: str,
    upload: Path,
    imported: ImportedText,
    report: ImportReport,
) -> None:
    canonical = write_canonical_text(paths.documents / document_id, imported.text)
    now = _now()
    with database.transaction() as connection:
        current = connection.execute(
            "SELECT import_state FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if current is None or current["import_state"] != "importing":
            return
        insert_revision(
            connection,
            document_id=document_id,
            imported=imported,
            report=report,
            canonical=canonical,
            original=upload,
            created_at=now,
        )
        connection.execute(
            """
            UPDATE documents
            SET import_state = 'ready', import_progress_json = NULL, import_error = NULL,
                char_count = ?, page_count = ?, updated_at = ?
            WHERE document_id = ?
            """,
            (report.char_count, report.page_count, now, document_id),
        )


def import_document(database: Database, paths: AppPaths, document_id: str) -> None:
    """Import one Document and record the outcome on its row."""
    row = database.fetchone("SELECT * FROM documents WHERE document_id = ?", (document_id,))
    if row is None or row["import_state"] != "importing":
        return
    started = time.monotonic()
    progress = _ProgressWriter(database, document_id)
    try:
        upload = Path(row["upload_path"]) if row["upload_path"] else None
        if upload is None or not upload.is_file():
            raise ImportFailure(_MISSING_UPLOAD)
        extraction = extract_document(upload, row["format"], progress, tesseract=Tesseract.find())
        progress("finalizing")
        imported = assemble(extraction)
        report = build_report(extraction, imported, duration_seconds=time.monotonic() - started)
        _record_ready(database, paths, document_id, upload, imported, report)
    except ImportFailure as failure:
        _record_failed(database, document_id, failure.message)
    except Exception as error:
        logger.exception("Import of Document %s failed", document_id)
        _record_failed(
            database,
            document_id,
            f"The import failed unexpectedly ({type(error).__name__}: {error}). Check the server log.",
        )


def _exit_with_parent() -> None:
    """Exit when the web process dies, even if it could not terminate us."""
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        wait_for_exit([parent.sentinel])
        kill_running_ocr()
        os._exit(1)

    threading.Thread(target=watch, name="import-parent-watch", daemon=True).start()


def _exit_on_terminate(signum: int, _frame: object) -> None:
    kill_running_ocr()
    os._exit(128 + signum)


def run_import(document_id: str, data_root: str) -> None:
    """Entry point of the Import child process."""
    _exit_with_parent()
    signal.signal(signal.SIGTERM, _exit_on_terminate)
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    paths = AppPaths.from_root(Path(data_root))
    import_document(init_database(paths), paths, document_id)


# --- Web process ----------------------------------------------------------------


class ImportProcess(Protocol):
    """A launched Import. `wait` blocks until it ends and returns its exit
    code; `stop` ends it early and returns once it has exited."""

    def wait(self) -> int | None: ...

    def stop(self) -> None: ...


class _SpawnedImport:
    def __init__(self, process: BaseProcess) -> None:
        self._process = process

    def wait(self) -> int | None:
        self._process.join()
        return self._process.exitcode

    def stop(self) -> None:
        # Only wait() reaps the child, so this watches its sentinel instead.
        self._process.terminate()
        if not wait_for_exit([self._process.sentinel], _EXIT_GRACE_SECONDS):
            self._process.kill()
            wait_for_exit([self._process.sentinel], _EXIT_GRACE_SECONDS)


@dataclass
class _Running:
    document_id: str
    handle: ImportProcess | None = None
    stop_requested: bool = False
    launched: threading.Event = field(default_factory=threading.Event)

    def stop(self) -> None:
        self.launched.wait(timeout=2 * _EXIT_GRACE_SECONDS)
        if self.handle is not None:
            self.handle.stop()


class ImportManager:
    """Runs queued Imports one at a time, each in a spawned child process."""

    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._queue: deque[str] = deque()
        self._wakeup = threading.Event()
        self._thread: threading.Thread | None = None
        self._running: _Running | None = None
        self._stopping = False

    def start(self) -> None:
        """Start the loop and queue Imports interrupted by the last shutdown.
        Runs before the app serves requests, so no upload is in flight."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._queue.clear()
        _remove_orphaned_uploads(get_database())
        self._requeue_interrupted()
        thread = threading.Thread(target=self._loop, name="import-manager", daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()

    def shutdown(self) -> None:
        """Stop the loop and terminate a running Import. Its Document stays
        `importing` and is queued again on the next start."""
        with self._lock:
            self._stopping = True
            self._queue.clear()
            running = self._running
            if running is not None:
                running.stop_requested = True
            thread = self._thread
        self._wakeup.set()
        if running is not None:
            running.stop()
        if thread is not None:
            thread.join(timeout=2 * _EXIT_GRACE_SECONDS)

    def enqueue(self, document_id: str) -> None:
        with self._lock:
            active = self._running.document_id if self._running is not None else None
            if document_id != active and document_id not in self._queue:
                self._queue.append(document_id)
        self._wakeup.set()

    def cancel(self, document_id: str) -> None:
        """Stop the Import of a Document: drop it from the queue or terminate its
        child process. Returns once the process has exited."""
        with self._lock:
            if document_id in self._queue:
                self._queue.remove(document_id)
            running = self._running
            if running is None or running.document_id != document_id:
                return
            running.stop_requested = True
        running.stop()

    def _launch(self, document_id: str, data_root: str) -> ImportProcess:
        """Start the child process of one Import."""
        process = self._context.Process(
            target=run_import,
            args=(document_id, data_root),
            name=f"import-{document_id[:8]}",
            daemon=True,
        )
        process.start()
        return _SpawnedImport(process)

    def _requeue_interrupted(self) -> None:
        database = get_database()
        rows = database.fetchall(
            "SELECT document_id, upload_path FROM documents "
            "WHERE import_state = 'importing' ORDER BY created_at"
        )
        for row in rows:
            upload = row["upload_path"]
            if upload and Path(upload).is_file():
                database.execute(
                    "UPDATE documents SET import_progress_json = ? WHERE document_id = ?",
                    (QUEUED_PROGRESS_JSON, row["document_id"]),
                )
                with self._lock:
                    self._queue.append(row["document_id"])
            else:
                _record_failed(database, row["document_id"], _MISSING_UPLOAD)
        if rows:
            self._wakeup.set()

    def _loop(self) -> None:
        _backfill_counts(get_database())
        while True:
            self._wakeup.wait(timeout=1.0)
            with self._lock:
                if self._stopping:
                    return
                if not self._queue:
                    self._wakeup.clear()
                    continue
                running = _Running(self._queue.popleft())
                self._running = running
            database = get_database()
            try:
                running.handle = self._launch(running.document_id, str(database.path.parent))
            except Exception as error:
                logger.exception("Could not start the Import of Document %s", running.document_id)
                _record_failed(
                    database, running.document_id, f"The import could not be started ({error})."
                )
            finally:
                running.launched.set()
            exit_code = running.handle.wait() if running.handle is not None else 0
            with self._lock:
                self._running = None
            if not running.stop_requested and exit_code != 0:
                _record_failed(database, running.document_id, _crash_reason(exit_code))


def _crash_reason(exit_code: int | None) -> str:
    if exit_code is not None and exit_code < 0:
        return (
            f"The import process was killed (signal {-exit_code}); the file may need more "
            "memory than is available."
        )
    return f"The import process stopped unexpectedly (exit code {exit_code}). Check the server log."


def _backfill_counts(database: Database) -> None:
    """Fill char_count and page_count of Documents imported before they existed."""
    rows = database.fetchall(
        """
        SELECT d.document_id, r.canonical_path, r.page_map_json
        FROM documents d
        JOIN source_revisions r ON r.revision_id = (
            SELECT revision_id FROM source_revisions
            WHERE document_id = d.document_id ORDER BY created_at DESC LIMIT 1
        )
        WHERE d.import_state = 'ready' AND d.char_count IS NULL
        """
    )
    for row in rows:
        try:
            with open(row["canonical_path"], encoding="utf-8", newline="") as handle:
                char_count = len(handle.read())
        except OSError:
            continue
        pages = json.loads(row["page_map_json"]) if row["page_map_json"] else None
        database.execute(
            "UPDATE documents SET char_count = ?, page_count = ? WHERE document_id = ?",
            (char_count, len(pages) if pages is not None else None, row["document_id"]),
        )


def _remove_orphaned_uploads(database: Database) -> None:
    """Delete Document directories without a Document row: uploads cut off by
    a server stop before their row was written."""
    root = database.path.parent / "documents"
    if not root.is_dir():
        return
    known = {row["document_id"] for row in database.fetchall("SELECT document_id FROM documents")}
    for directory in root.iterdir():
        if directory.name in known or not directory.is_dir():
            continue
        try:
            uuid.UUID(directory.name)
        except ValueError:
            continue
        shutil.rmtree(directory, ignore_errors=True)


_manager: ImportManager | None = None


def get_import_manager() -> ImportManager:
    global _manager
    if _manager is None:
        _manager = ImportManager()
    return _manager
