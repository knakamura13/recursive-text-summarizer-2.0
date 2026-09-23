from pathlib import Path

from summarizer_web.config import load_paths


def test_load_paths_restricts_existing_data_directories(monkeypatch, tmp_path: Path):
    root = tmp_path / "app-data"
    cache = root / "cache"
    cache.mkdir(parents=True)
    root.chmod(0o755)
    cache.chmod(0o755)
    monkeypatch.setenv("SUMMARIZER_DATA_DIR", str(root))

    paths = load_paths()

    for directory in (paths.root, paths.documents, paths.runs, paths.cache):
        assert directory.stat().st_mode & 0o777 == 0o700
