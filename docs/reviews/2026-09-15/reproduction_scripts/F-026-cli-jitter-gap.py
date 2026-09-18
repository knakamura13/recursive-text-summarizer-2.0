"""S6 independent-verifier repro for C-S5-002: does any CLI-driven invocation
ever produce a non-zero RetryPolicy.jitter_fraction?

Read-only against the repo; does not modify any tracked file. Confirms:
  1. summarizer.cli's argparse parser exposes no --jitter flag (or any flag
     that reaches RetryPolicy.jitter_fraction).
  2. parse_args() with a representative set of CLI flags (including
     --max-retries, since that is the only retry-related flag that exists)
     always yields RetryPolicy(jitter_fraction=0) -- the dataclass default.

Run with (from a checkout at 301cc4d):
  UV_OFFLINE=1 uv run --with-requirements requirements-dev.txt python \
    .review/wip/v-misc/C-S5-002-cli-jitter-gap.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from summarizer.cli import _parser, parse_args

parser = _parser()
jitter_actions = [
    action
    for action in parser._actions  # noqa: SLF001 - introspection only, read-only
    if "jitter" in (action.dest or "") or any("jitter" in s for s in action.option_strings)
]
print("argparse actions mentioning 'jitter':", jitter_actions)

# Exercise parse_args with every retry-adjacent flag argparse actually exposes.
parsed = parse_args(["--max-retries", "9", "--dry-run"])
print("parsed.retry               =", parsed.retry)
print("parsed.retry.jitter_fraction =", parsed.retry.jitter_fraction)

# Also confirm the parser accepts nothing named --jitter at all.
try:
    parse_args(["--jitter", "0.5", "--dry-run"])
    print("RESULT: --jitter was ACCEPTED (would contradict the claim)")
except SystemExit:
    print("RESULT: --jitter is not a recognized flag (argparse rejects it via SystemExit)")
