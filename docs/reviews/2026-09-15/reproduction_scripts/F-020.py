#!/usr/bin/env python3
"""F-020 repro: confirm two independent facts.

(1) No test in the repo exercises "e.g." for the sentence-boundary abbreviation
    guard, even though issue #32's acceptance criterion literally names it
    alongside "Dr." and "U.S.".
(2) The guard itself works correctly for "e.g." at runtime -- this is what
    determines whether this is a Tier B coverage gap (guard works, untested)
    or a Tier A live defect (guard is broken).

Run with:
    uv run --with-requirements requirements-dev.txt python .review/repro/F-020.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from summarizer.segmentation import _SENTENCE_TOKENIZER, _SENTENCE_ABBREVS  # noqa: E402


def main() -> int:
    print(f"'e.g' present in _SENTENCE_ABBREVS seed set: {'e.g' in _SENTENCE_ABBREVS}")

    cases = [
        ("For various reasons (e.g. cost, time) the project stalled. Next sentence.", 2),
        ("Use a linter (e.g. Ruff) before committing. It catches typos.", 2),
        ("Common tools include linters, e.g. Ruff, and formatters. Both help.", 2),
    ]
    all_ok = True
    for text, expected in cases:
        spans = list(_SENTENCE_TOKENIZER.span_tokenize(text))
        ok = len(spans) == expected
        all_ok &= ok
        print(
            f"text={text!r}\n  spans={[text[s:e] for s, e in spans]}\n"
            f"  expected {expected} sentence(s), got {len(spans)} -- {'OK' if ok else 'WRONG'}"
        )

    print()
    print("RESULT: guard behaves correctly for 'e.g.' at runtime" if all_ok else
          "RESULT: guard is BROKEN for 'e.g.' -- this would NOT be a mere coverage gap")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
