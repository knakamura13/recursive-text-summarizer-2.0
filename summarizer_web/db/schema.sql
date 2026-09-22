PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    csrf_token TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    format TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    import_state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_revisions (
    revision_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    source_sha256 TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    canonical_path TEXT NOT NULL,
    original_path TEXT,
    page_map_json TEXT,
    blank_pages_json TEXT,
    confirmed_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(document_id, source_sha256, extraction_version)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    revision_id TEXT NOT NULL REFERENCES source_revisions(revision_id),
    state TEXT NOT NULL,
    strategy TEXT,
    config_json TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_attempts (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    state TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    failure_reason TEXT,
    worker_pid INTEGER,
    UNIQUE(run_id, attempt_number)
);

CREATE TABLE IF NOT EXISTS run_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS node_projections (
    projection_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    parent_id TEXT,
    level INTEGER NOT NULL,
    order_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    summary_text TEXT,
    provisional INTEGER NOT NULL DEFAULT 1,
    covered_segment_ids_json TEXT NOT NULL,
    evidence_refs_json TEXT NOT NULL,
    UNIQUE(run_id, node_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_runs_document ON runs(document_id);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, event_id);
CREATE INDEX IF NOT EXISTS idx_node_projections_run ON node_projections(run_id);
