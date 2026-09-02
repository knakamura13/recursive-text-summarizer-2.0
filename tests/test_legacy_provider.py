from pathlib import Path

import pytest

from tests.support.legacy_loader import FakeHarness, load_legacy_main


def test_successful_summary_records_prompt_model_timeout_and_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    harness = FakeHarness()
    harness.queue("  concise\n summary  ")
    module = load_legacy_main(harness)
    monkeypatch.setattr(module, "time", lambda: 123.0)

    result = module.summarize_with_gpt("Source text", _model="gpt-4")

    assert result == "concise summary"
    assert len(harness.calls) == 1
    assert harness.calls[0] == {
        "model": "gpt-4-1106-preview",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a writing assistant, skilled in revising and "
                    "summarizing complex technical writing with accuracy and "
                    "precision."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Provide an executive summary of the following text "
                    "(delimited by triple quotes). Present the key ideas and "
                    "findings directly, without bullet points, as if for a busy "
                    "professional who needs to grasp the essential points "
                    "quickly. Ignore complete sentences and grammatical "
                    "correctness. Abbreviate long and repetitive words. "
                    '\n"""\nSource text\n"""\n'
                ),
            },
        ],
        "timeout": 180,
    }
    assert (tmp_path / "gpt_logs" / "123.0_gpt.txt").read_text(
        encoding="utf-8"
    ) == (
        "PROMPT:\n\nSource text\n\n==========\n\n"
        "RESPONSE:\n\nconcise summary"
    )


def test_default_model_alias_uses_gpt_35_turbo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    harness = FakeHarness()
    harness.queue("summary")
    module = load_legacy_main(harness)

    module.summarize_with_gpt("Source")

    assert harness.calls[0]["model"] == "gpt-3.5-turbo"


def test_dry_run_collapses_source_without_calling_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    harness = FakeHarness()
    module = load_legacy_main(harness)
    module.DRY_RUN = True

    assert module.summarize_with_gpt("  Alpha\n Beta  ") == "Alpha Beta"
    assert harness.calls == []
