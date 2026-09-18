"""C-S6-002 / C-S6-003 repro: independently re-derive, from raw `gh` JSON,
how many issues in the "Generalized Recursive Summarizer Rebuild" milestone
have zero comments, and how many were closed strictly after the
issue-closing review gate (PR #47) merged.

Does not trust any prior count on the record; recomputes from the fetched
JSON directly.

Usage:
    gh issue list --milestone "Generalized Recursive Summarizer Rebuild" \
        --state all --json number,title,state,closedAt,createdAt,comments \
        --limit 50 > milestone_issues.json
    python repro_c_s6_003_milestone_comment_counts.py milestone_issues.json \
        <gate_merged_at_iso8601>
"""
from __future__ import annotations

import datetime as dt
import json
import sys

issues_path = sys.argv[1]
gate_merged_at = sys.argv[2]  # e.g. 2026-09-14T20:51:49Z

with open(issues_path, "r", encoding="utf-8") as f:
    issues = json.load(f)

gate_dt = dt.datetime.fromisoformat(gate_merged_at.replace("Z", "+00:00"))

print(f"Loaded {len(issues)} issues from {issues_path}")
print(f"Gate merge time: {gate_dt.isoformat()}\n")

rows = []
for issue in issues:
    num = issue["number"]
    closed = issue.get("closedAt")
    n_comments = len(issue.get("comments", []))
    after_gate = None
    delta_hours = None
    if closed:
        closed_dt = dt.datetime.fromisoformat(closed.replace("Z", "+00:00"))
        after_gate = closed_dt > gate_dt
    rows.append((num, issue["state"], closed, n_comments, after_gate))

rows.sort(key=lambda r: -r[0])
print(f"{'issue':>6} {'state':>8} {'closedAt':>22} {'#comments':>10} {'after_gate':>11}")
for r in rows:
    print(f"{r[0]:>6} {r[1]:>8} {str(r[2]):>22} {r[3]:>10} {str(r[4]):>11}")

zero_comment_total = sum(1 for r in rows if r[3] == 0)
print(f"\nZero-comment issues, all {len(rows)}: {zero_comment_total}")

after = [r for r in rows if r[4]]
print(f"Closed strictly after gate merge: {len(after)}")
zero_after = [r for r in after if r[3] == 0]
print(f"  of those, zero-comment: {len(zero_after)}")
print(f"  of those, with >=1 comment: {len(after) - len(zero_after)}")
for r in after:
    tag = "ZERO" if r[3] == 0 else f"{r[3]} comment(s)"
    print(f"    #{r[0]}: closed {r[2]}  -> {tag}")
