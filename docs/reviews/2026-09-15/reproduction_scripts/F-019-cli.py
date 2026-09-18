#!/usr/bin/env python3
"""F-019 CLI-level repro: run the *actual* `summarizer.cli.main()` entry point,
the same function `python main.py ...` invokes, with real CLI argv
(`--strategy hierarchical --audit <path>`), against a real temp input file
containing a fenced code block, using the same offline provider-injection seam
tests/test_cli.py uses (`provider_factory=`, `counter_factory=`) so no network
or Ollama call happens.

This exists to answer, concretely, at the CLI boundary: does the crash lose
only the audit artifact, or does it also lose the user's summary output file?

Run with:
    uv run --with-requirements requirements-dev.txt python .review/repro/F-019-cli.py
"""

import json
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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
    """Same shape as tests/test_cli.py's RecordingProvider fake."""

    def __init__(self) -> None:
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if request.operation_id == "editorial-final":
            text = json.dumps({"text": "A concise, coherent final summary."})
        elif request.operation_id == "D000001":
            text = json.dumps(self._node("D000001"))
        else:
            identifiers = re.findall(r'"segment_id":"([SD]\d+)"', request.input_text)
            text = json.dumps(self._node(identifiers[-1] if identifiers else "S000001"))
        return GenerationResult(text, "fake", request.model)

    @staticmethod
    def _node(identifier: str) -> dict:
        return {
            "summary": "Grounded summary.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [identifier],
            "level": 0,
        }


def counter_factory(_config: AppConfig) -> CharacterCounter:
    return CharacterCounter()


DOCUMENT_WITH_TRAILING_CODE_FENCE = (
    "This report walks through the deployment script used to roll out the "
    "nightly batch job, and the snippet below is exactly what operators run "
    "by hand today when the scheduler is down.\n\n"
    "```python\n"
    "def deploy():\n"
    "    print('rolling out nightly batch job')\n"
    "```\n"
)


def main_repro() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.txt"
        output_path = tmp_path / "output.txt"
        audit_path = tmp_path / "audit.json"
        input_path.write_text(DOCUMENT_WITH_TRAILING_CODE_FENCE, encoding="utf-8")

        argv = [
            "--input", str(input_path),
            "--output", str(output_path),
            "--strategy", "hierarchical",
            "--audit", str(audit_path),
            "--context-window", "100000",
            "--max-output-tokens", "1",
            "--safety-margin-tokens", "0",
            "--safety-margin-fraction", "0",
        ]

        print(f"=== Invoking the real summarizer.cli.main() with argv: {argv} ===")
        exit_code = main(
            argv,
            provider_factory=lambda _config: RecordingProvider(),
            counter_factory=counter_factory,
        )

        print(f"\nexit_code = {exit_code}")
        print(f"output file exists?  {output_path.exists()}")
        print(f"audit file exists?   {audit_path.exists()}")
        if output_path.exists():
            print(f"output file contents: {output_path.read_text()!r}")
        return exit_code


if __name__ == "__main__":
    main_repro()
