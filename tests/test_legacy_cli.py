from __future__ import annotations

import inspect
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType

import pytest

from tests.support.legacy_loader import (
    PROJECT_ROOT,
    FakeHarness,
    run_legacy_main,
)


SYSTEM_PROMPT = (
    "You are a writing assistant, skilled in revising and summarizing "
    "complex technical writing with accuracy and precision."
)
USER_PROMPT_PREFIX = (
    "Provide an executive summary of the following text (delimited by "
    "triple quotes). Present the key ideas and findings directly, without "
    "bullet points, as if for a busy professional who needs to grasp the "
    "essential points quickly. Ignore complete sentences and grammatical "
    "correctness. Abbreviate long and repetitive words. "
)
MAIN_PATH = PROJECT_ROOT / "main.py"
MAIN_ENTRY_LINE = next(
    line_number
    for line_number, line in enumerate(
        MAIN_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    )
    if line == 'if __name__ == "__main__":'
)


def _expected_messages(chunk: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f'{USER_PROMPT_PREFIX}\n"""\n{chunk}\n"""\n',
        },
    ]


def _set_max_chunks_at_entry(max_chunks: int) -> Callable:
    def configure_max_chunks(
        frame: FrameType,
        event: str,
        _arg: object,
    ) -> Callable | None:
        if (
            event == "line"
            and Path(frame.f_code.co_filename) == MAIN_PATH
            and frame.f_code.co_name == "<module>"
            and frame.f_lineno == MAIN_ENTRY_LINE
        ):
            frame.f_globals["MAX_CHUNKS"] = max_chunks
        return configure_max_chunks

    return configure_max_chunks


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
    real_time = time.time
    timestamps = iter((101.0, 202.0))

    def deterministic_provider_time() -> float:
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_name == "summarize_with_gpt":
            return next(timestamps)
        return real_time()

    monkeypatch.setattr(time, "time", deterministic_provider_time)

    namespace = run_legacy_main(harness)

    assert (tmp_path / "output.txt").read_text(
        encoding="utf-8"
    ) == "first summary\nsecond summary"
    expected_chunks = [f" {first_sentence}", second_sentence]
    assert harness.calls == [
        {
            "model": "gpt-4-1106-preview",
            "messages": _expected_messages(expected_chunks[0]),
            "timeout": 180,
        },
        {
            "model": "gpt-4-1106-preview",
            "messages": _expected_messages(expected_chunks[1]),
            "timeout": 180,
        },
    ]
    assert (tmp_path / "gpt_logs" / "101.0_gpt.txt").read_text(
        encoding="utf-8"
    ) == (
        f"PROMPT:\n\n{expected_chunks[0]}\n\n==========\n\n"
        "RESPONSE:\n\nfirst summary"
    )
    assert (tmp_path / "gpt_logs" / "202.0_gpt.txt").read_text(
        encoding="utf-8"
    ) == (
        f"PROMPT:\n\n{expected_chunks[1]}\n\n==========\n\n"
        "RESPONSE:\n\nsecond summary"
    )
    assert namespace["MAX_CHUNKS"] == -1


def test_positive_max_chunks_processes_only_the_source_prefix(
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
    harness.queue("first summary", "must remain unused")
    previous_trace = sys.gettrace()
    sys.settrace(_set_max_chunks_at_entry(1))
    try:
        namespace = run_legacy_main(harness)
    finally:
        sys.settrace(previous_trace)

    assert namespace["MAX_CHUNKS"] == 1
    assert (tmp_path / "output.txt").read_text(
        encoding="utf-8"
    ) == "first summary"
    assert len(harness.calls) == 1
    assert harness.calls[0]["messages"] == _expected_messages(
        f" {first_sentence}"
    )
    assert list(harness.scripted_results) == ["must remain unused"]


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
