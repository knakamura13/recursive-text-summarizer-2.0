"""S6 repro for traceability row 208 (#37): a run does not dirty the tree.

Runs `summarizer.cli.main()` end-to-end with a fully offline, deterministic
fake provider and a fake token counter (the `provider_factory=`/
`counter_factory=` seam documented in tests/test_cli.py), against a /tmp git
copy of the repo at 301cc4d — never the real checkout. After the run, checks
`git status --short --untracked-files=all` inside that /tmp copy.

Usage: run with the target repo copy's directory as argv[1]. Must be executed
with that directory's `summarizer` package importable (this script inserts
argv[1] onto sys.path[0]).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

repo_dir = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo_dir))
os.chdir(repo_dir)

from summarizer.cli import main  # noqa: E402
from summarizer.config import AppConfig  # noqa: E402
from summarizer.providers.base import GenerationRequest, GenerationResult  # noqa: E402


class CharacterCounter:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if request.operation_id == "editorial-final":
            text = json.dumps({"text": "a deterministic offline summary"})
        elif request.operation_id == "D000001":
            text = json.dumps(
                {
                    "summary": "a deterministic offline summary",
                    "content_units": [],
                    "entities": [],
                    "qualifications": [],
                    "contradictions": [],
                    "quotations": [],
                    "provenance": ["D000001"],
                    "level": 0,
                }
            )
        else:
            identifiers = re.findall(r'"segment_id":"([SD]\d+)"', request.input_text)
            text = json.dumps(
                {
                    "summary": "a deterministic offline summary",
                    "content_units": [],
                    "entities": [],
                    "qualifications": [],
                    "contradictions": [],
                    "quotations": [],
                    "provenance": identifiers[-1:] or ["S000001"],
                    "level": 0,
                }
            )
        return GenerationResult(text, "fake", request.model)


def counter_factory(_config: AppConfig):
    return CharacterCounter()


def provider_factory(_config: AppConfig):
    return RecordingProvider()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True
    )
    return result.stdout


def main_repro() -> int:
    before = git("status", "--short", "--untracked-files=all")
    print("=== git status before run ===")
    print(before or "(clean)")

    exit_code = main([], provider_factory=provider_factory, counter_factory=counter_factory)
    print(f"=== summarizer.cli.main() exit code: {exit_code} ===")

    after = git("status", "--short", "--untracked-files=all")
    print("=== git status after run (--untracked-files=all) ===")
    print(after or "(clean)")

    # summarizer.log must never appear, tracked or untracked, ignored or not.
    log_hits = git("status", "--short", "--ignored", "--untracked-files=all")
    log_lines = [line for line in log_hits.splitlines() if "summarizer.log" in line]
    print("=== summarizer.log-related status lines (including ignored) ===")
    print("\n".join(log_lines) or "(none)")

    # Report exactly which paths changed vs. the M/A/?? expected set.
    changed_paths = {line[3:] for line in after.splitlines()}
    expected = {"output.txt"}
    unexpected = changed_paths - expected
    print(f"=== changed paths: {sorted(changed_paths)} ===")
    print(f"=== unexpected paths beyond {expected}: {sorted(unexpected)} ===")

    return 0 if exit_code == 0 and not unexpected and not log_lines else 1


if __name__ == "__main__":
    raise SystemExit(main_repro())
