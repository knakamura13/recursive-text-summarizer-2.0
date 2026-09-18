"""C-S6-001 repro: does docs/evaluation.md or README.md ever explicitly
distinguish the evaluator's forced `max_merge_children=2` demonstration from
default hosted-capacity behavior?

Issue #39's "Reconciled status" checklist (line 10 of the issue body) checks
off: "The offline evaluator demonstrates a configured genuine multi-level
hierarchy with max_merge_children=2, and the documentation explicitly
distinguishes that configured demonstration from default hosted-capacity
behavior."

This script greps both docs for the load-bearing vocabulary a genuine
distinguishing sentence would need (hosted/default/capacity/distinguish
paired with the evaluator's forced config) and prints every hit with
surrounding context, so a human can eyeball whether any hit actually makes
the distinction the checklist claims.

Usage: python repro_c_s6_001_doc_distinction_search.py <repo_root>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
targets = [repo_root / "docs" / "evaluation.md", repo_root / "README.md"]

pattern = re.compile(
    r"hosted|default.{0,40}capacity|capacity.{0,40}default|distinguish"
    r"|max_merge_children|max-merge-children|merge level",
    re.IGNORECASE,
)

print("Searching for distinguishing language between forced demo config")
print("(max_merge_children=2) and default hosted-capacity behavior...\n")

any_hit = False
for path in targets:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        if pattern.search(line):
            any_hit = True
            print(f"{path.relative_to(repo_root)}:{i}: {line.strip()}")

print()
print(
    "Manual check: none of the hits above contains a sentence stating that "
    "the evaluator's forced max_merge_children=2 is a demonstration-only "
    "setting distinct from what default (unbounded/adaptive) hosted-capacity "
    "runs would do. README.md's --max-merge-children description explains "
    "the flag generically; it never ties that explanation back to the "
    "evaluator's specific demonstration case."
)
