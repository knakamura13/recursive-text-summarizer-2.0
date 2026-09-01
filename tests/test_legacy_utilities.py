from pathlib import Path
from types import ModuleType

import pytest

from tests.support.legacy_loader import FakeHarness, load_legacy_main


@pytest.fixture
def legacy_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> ModuleType:
    monkeypatch.chdir(tmp_path)
    return load_legacy_main(FakeHarness())


def test_open_file_reads_exact_utf8_content(
    legacy_module: ModuleType,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Résumé, 東京", encoding="utf-8")

    assert legacy_module.open_file(str(source)) == "Résumé, 東京"


def test_save_file_creates_parent_directories_and_writes_exact_utf8_content(
    legacy_module: ModuleType,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "output.txt"

    legacy_module.save_file("Résumé, 東京", str(destination))

    assert destination.read_text(encoding="utf-8") == "Résumé, 東京"


def test_save_file_rejects_parentless_relative_path(
    legacy_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        legacy_module.save_file("summary", "output.txt")


def test_remove_extra_whitespace_collapses_all_whitespace(
    legacy_module: ModuleType,
) -> None:
    text = "  Alpha  \n\t Beta     Gamma  "

    assert legacy_module.remove_extra_whitespace(text) == "Alpha Beta Gamma"


def test_chunk_text_approximately_packs_sentences_by_character_count(
    legacy_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        legacy_module,
        "sent_tokenize",
        lambda _text: ["Alpha.", "Beta.", "Gamma."],
    )

    assert legacy_module.chunk_text_by_sentences("ignored", 12) == [
        " Alpha. Beta.",
        "Gamma.",
    ]


def test_chunk_text_keeps_oversized_first_sentence_unsplit(
    legacy_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        legacy_module,
        "sent_tokenize",
        lambda _text: ["Oversized sentence."],
    )

    assert legacy_module.chunk_text_by_sentences("ignored", 4) == [
        " Oversized sentence."
    ]
