import logging
from pathlib import Path

import pytest

from tests.support.legacy_loader import FakeHarness, run_legacy_main


def test_default_workflow_summarizes_chunks_in_source_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    first_sentence = "A" * 600 + "."
    second_sentence = "B" * 600 + "."
    (tmp_path / "input.txt").write_text(
        f"{first_sentence} {second_sentence}",
        encoding="utf-8",
    )
    harness = FakeHarness(sentences=[first_sentence, second_sentence])
    harness.queue("first summary", "second summary")

    namespace = run_legacy_main(harness)

    assert (tmp_path / "output.txt").read_text(
        encoding="utf-8"
    ) == "first summary\nsecond summary"
    assert [call["model"] for call in harness.calls] == [
        "gpt-4-1106-preview",
        "gpt-4-1106-preview",
    ]
    provider_chunks = []
    for call in harness.calls:
        messages = call["messages"]
        assert isinstance(messages, list)
        provider_chunks.append(messages[1]["content"].split('\n"""\n')[1])
    assert provider_chunks == [f" {first_sentence}", second_sentence]
    assert namespace["MAX_CHUNKS"] == -1


def test_missing_input_logs_fatal_error_without_writing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.CRITICAL)

    run_legacy_main(FakeHarness())

    assert not (tmp_path / "output.txt").exists()
    assert any(
        record.levelno == logging.CRITICAL
        and "Fatal error in main application" in record.getMessage()
        and "input.txt" in record.getMessage()
        for record in caplog.records
    )
