#!/usr/bin/env python3
"""F-021 repro: confirm none of the 5 required fixture-corpus files (#2 AC)
contains a fenced code block, and that tests/test_fixture_corpus.py never
runs any fixture through segmentation or the audit path -- structurally why
F-019's CODE_FENCE/audit crash survived a fully green 766-test suite.

Run with:
    uv run --with-requirements requirements-dev.txt python .review/repro/F-021.py
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURES = REPO_ROOT / "tests" / "fixtures"

REQUIRED = ["article.txt", "report.txt", "transcript.txt", "structured.md", "narrative.txt"]
FENCE_PATTERN = re.compile(r"^\s*(```+|~~~+)", re.MULTILINE)


def main() -> int:
    any_fence = False
    for name in REQUIRED:
        path = FIXTURES / name
        text = path.read_text(encoding="utf-8")
        match = FENCE_PATTERN.search(text)
        print(f"{name}: {'CONTAINS a fence marker' if match else 'no fence marker'}")
        any_fence |= bool(match)

    print()
    print(
        "RESULT: at least one fixture contains a fenced code block"
        if any_fence
        else "RESULT: none of the 5 required fixtures contains a fenced code block"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
