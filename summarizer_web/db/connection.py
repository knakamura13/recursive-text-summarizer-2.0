"""SQLite connection and schema initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock

from summarizer_web.config import AppPaths, load_paths

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_db: Database | None = None
_init_lock = Lock()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
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


def init_database(paths: AppPaths | None = None) -> Database:
    global _db
    app_paths = paths or load_paths()
    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    with _init_lock:
        connection = sqlite3.connect(app_paths.database)
        try:
            connection.executescript(schema)
            connection.commit()
        finally:
            connection.close()
        _db = Database(app_paths.database)
        return _db


def get_database() -> Database:
    global _db
    if _db is None:
        return init_database()
    return _db
