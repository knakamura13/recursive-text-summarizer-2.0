"""SQLite connection, schema initialization, and migrations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from summarizer_web.config import AppPaths, load_paths
from summarizer_web.db.migrations import apply_migrations

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
# The web process, the run worker, and the import worker write concurrently.
_BUSY_TIMEOUT_SECONDS = 15.0
_db: Database | None = None
_init_lock = Lock()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path, check_same_thread=False, timeout=_BUSY_TIMEOUT_SECONDS
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def execute(self, query: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            connection = self.connect()
            try:
                cursor = connection.execute(query, params)
                connection.commit()
                return cursor
            finally:
                connection.close()

    def executemany(self, query: str, rows: Iterable[tuple[object, ...]]) -> None:
        with self._lock:
            connection = self.connect()
            try:
                connection.executemany(query, rows)
                connection.commit()
            finally:
                connection.close()

    def fetchone(self, query: str, params: tuple[object, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            connection = self.connect()
            try:
                return connection.execute(query, params).fetchone()
            finally:
                connection.close()

    def fetchall(self, query: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            connection = self.connect()
            try:
                return connection.execute(query, params).fetchall()
            finally:
                connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run several statements atomically under one write lock."""
        with self._lock:
            connection = self.connect()
            connection.isolation_level = None
            try:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    yield connection
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
                connection.execute("COMMIT")
            finally:
                connection.close()


def init_database(paths: AppPaths | None = None) -> Database:
    global _db
    app_paths = paths or load_paths()
    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    with _init_lock:
        connection = sqlite3.connect(app_paths.database, timeout=_BUSY_TIMEOUT_SECONDS)
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(schema)
            connection.commit()
            apply_migrations(connection)
        finally:
            connection.close()
        _db = Database(app_paths.database)
        return _db


def get_database() -> Database:
    global _db
    if _db is None:
        return init_database()
    return _db
