"""Application configuration and data directory resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "recursive-text-summarizer"
EXTRACTION_VERSION = "pdf/1"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 2000
PDF_EXTRACTION_TIMEOUT_SECONDS = 120


def resolve_data_dir() -> Path:
    override = os.environ.get("SUMMARIZER_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(user_data_dir(APP_NAME, appauthor=False))


@dataclass(frozen=True)
class AppPaths:
    root: Path
    database: Path
    documents: Path
    runs: Path
    cache: Path
    static: Path

    @classmethod
    def from_root(cls, root: Path) -> AppPaths:
        return cls(
            root=root,
            database=root / "app.db",
            documents=root / "documents",
            runs=root / "runs",
            cache=root / "cache",
            static=Path(__file__).resolve().parent / "static",
        )


def load_paths() -> AppPaths:
    paths = AppPaths.from_root(resolve_data_dir())
    for directory in (paths.root, paths.documents, paths.runs, paths.cache):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    return paths
