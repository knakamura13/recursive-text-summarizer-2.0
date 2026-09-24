"""Ordered schema migrations tracked with ``PRAGMA user_version``.

``schema.sql`` is the version-1 baseline. Each entry upgrades the database to
its version inside one transaction, so a failed step leaves the prior version
intact. Every process that opens the database applies pending steps, and the
version check runs inside the write lock, so concurrent openers are safe.
"""

from __future__ import annotations

import sqlite3

MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        2,
        (
            # Documents: background imports, paste origin, upload dedup.
            "ALTER TABLE documents ADD COLUMN origin TEXT NOT NULL DEFAULT 'upload'",
            "ALTER TABLE documents ADD COLUMN import_progress_json TEXT",
            "ALTER TABLE documents ADD COLUMN import_error TEXT",
            "ALTER TABLE documents ADD COLUMN original_sha256 TEXT",
            "ALTER TABLE documents ADD COLUMN upload_path TEXT",
            "ALTER TABLE documents ADD COLUMN char_count INTEGER",
            "ALTER TABLE documents ADD COLUMN page_count INTEGER",
            "UPDATE documents SET import_state = 'ready' "
            "WHERE import_state = 'pending_confirmation'",
            "CREATE INDEX IF NOT EXISTS idx_documents_original_sha "
            "ON documents(original_sha256)",
            "CREATE INDEX IF NOT EXISTS idx_documents_import_state "
            "ON documents(import_state, created_at)",
            # Revisions: import report and OCR provenance.
            "ALTER TABLE source_revisions ADD COLUMN report_json TEXT",
            "ALTER TABLE source_revisions ADD COLUMN ocr_pages_json TEXT",
            # Runs: Stop replaces cancel everywhere.
            "UPDATE runs SET state = 'stopped' WHERE state = 'cancelled'",
            "UPDATE runs SET state = 'stopping' WHERE state = 'cancelling'",
            "UPDATE run_attempts SET state = 'stopped' WHERE state = 'cancelled'",
            "UPDATE run_attempts SET state = 'stopping' WHERE state = 'cancelling'",
            "UPDATE run_attempts SET failure_reason = 'Stopped by user' "
            "WHERE failure_reason = 'Cancelled by user'",
            "ALTER TABLE runs ADD COLUMN selected_strategy TEXT",
            "ALTER TABLE runs ADD COLUMN progress_json TEXT",
            "ALTER TABLE run_attempts ADD COLUMN failure_json TEXT",
            "CREATE INDEX IF NOT EXISTS idx_runs_state ON runs(state)",
            # Live node projections.
            "ALTER TABLE node_projections ADD COLUMN kind TEXT NOT NULL DEFAULT 'leaf'",
            "ALTER TABLE node_projections ADD COLUMN state TEXT NOT NULL DEFAULT 'completed'",
            "ALTER TABLE node_projections ADD COLUMN content_units_json TEXT",
            "ALTER TABLE node_projections ADD COLUMN detail_json TEXT",
            "ALTER TABLE node_projections ADD COLUMN child_ids_json TEXT",
            "ALTER TABLE node_projections ADD COLUMN started_at TEXT",
            "ALTER TABLE node_projections ADD COLUMN completed_at TEXT",
            "ALTER TABLE node_projections ADD COLUMN error TEXT",
            "ALTER TABLE node_projections ADD COLUMN updated_event_id INTEGER NOT NULL DEFAULT 0",
            "UPDATE node_projections SET kind = 'merge' WHERE level > 0",
            "CREATE INDEX IF NOT EXISTS idx_node_projections_updated "
            "ON node_projections(run_id, updated_event_id)",
            # Segment offsets, known once segmentation finishes.
            """
            CREATE TABLE IF NOT EXISTS run_segments (
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                segment_id TEXT NOT NULL,
                order_index INTEGER NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                core_start INTEGER NOT NULL,
                core_end INTEGER NOT NULL,
                token_count INTEGER,
                page_start INTEGER,
                page_end INTEGER,
                PRIMARY KEY (run_id, segment_id)
            )
            """,
        ),
    ),
)

LATEST_VERSION = MIGRATIONS[-1][0]


def apply_migrations(connection: sqlite3.Connection) -> None:
    previous_isolation = connection.isolation_level
    connection.isolation_level = None
    try:
        for version, statements in MIGRATIONS:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = connection.execute("PRAGMA user_version").fetchone()[0]
                if current < version:
                    for statement in statements:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA user_version = {version}")
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
    finally:
        connection.isolation_level = previous_isolation
