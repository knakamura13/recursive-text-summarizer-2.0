import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.support import legacy_loader
from tests.support.legacy_loader import FakeHarness, load_legacy_main


_root_logging_snapshot: tuple[list[logging.Handler], int] | None = None


def _legacy_module_names() -> set[str]:
    return {name for name in sys.modules if name.startswith("_legacy_main_")}


def test_import_uses_test_doubles_without_credentials_or_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    harness = FakeHarness()

    module = load_legacy_main(harness)

    assert module.client is harness.client
    assert harness.api_keys == [None]
    assert harness.downloads == ["punkt"]
    assert harness.dotenv_load_count == 1


def test_import_time_logging_configuration_is_scoped_to_its_test() -> None:
    global _root_logging_snapshot

    root_logger = logging.getLogger()
    _root_logging_snapshot = (list(root_logger.handlers), root_logger.level)
    root_logger.handlers.clear()
    root_logger.setLevel(logging.NOTSET)

    load_legacy_main(FakeHarness())

    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0], logging.FileHandler)
    assert root_logger.level == logging.INFO


def test_root_logging_state_is_restored_after_legacy_import() -> None:
    assert _root_logging_snapshot is not None
    handlers, level = _root_logging_snapshot
    root_logger = logging.getLogger()

    assert root_logger.handlers == handlers
    assert root_logger.level == level


def test_preexisting_external_modules_are_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_names = (
        "dotenv",
        "nltk",
        "nltk.tokenize",
        "openai",
        "requests",
        "tqdm",
    )
    sentinels = {
        name: ModuleType(f"sentinel_{name.replace('.', '_')}")
        for name in module_names
    }
    for name, sentinel in sentinels.items():
        monkeypatch.setitem(sys.modules, name, sentinel)

    load_legacy_main(FakeHarness())

    for name, sentinel in sentinels.items():
        assert sys.modules[name] is sentinel


def test_successful_import_removes_temporary_module() -> None:
    original_names = _legacy_module_names()

    load_legacy_main(FakeHarness())

    assert _legacy_module_names() == original_names


def test_failed_import_removes_temporary_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_names = _legacy_module_names()
    (tmp_path / "main.py").write_text(
        "raise RuntimeError('expected import failure')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(legacy_loader, "PROJECT_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="expected import failure"):
        load_legacy_main(FakeHarness())

    assert _legacy_module_names() == original_names
