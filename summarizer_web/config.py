"""Application configuration and data directory resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "recursive-text-summarizer"

# Imports (D3-D5). The version is stored on every source revision so the text
# of a Document can be traced to the extraction rules that produced it.
EXTRACTION_VERSION = "import/3"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_PASTE_BYTES = 20 * 1024 * 1024
MAX_IMPORT_PAGES = 5_000
IMPORT_PROGRESS_INTERVAL_SECONDS = 0.25
OCR_LANGUAGE = "eng"
OCR_DPI = 300
OCR_PAGE_TIMEOUT_SECONDS = 120.0
# Tesseract runs single-threaded per page; pages are recognized in parallel.
OCR_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))


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
