#!/usr/bin/env bash
# C-S6-005 repro: falsify test_help_lists_every_documented_flag by injecting
# misleading text into a CLI help= string, on a /tmp-only copy of the repo.
# Never touches the git worktree or the primary checkout.
#
# Usage: repro_c_s6_005_help_text_falsification.sh <repo_root> <throwaway_venv_python>
set -euo pipefail

REPO_ROOT="$(cd "$1" && pwd)"
VENV_PY="$2"
WORKDIR="$(mktemp -d /tmp/v-s6-005-repro.XXXXXX)"

echo "Copying repo to $WORKDIR (excluding .git/.venv/.review)..."
rsync -a --exclude='.git' --exclude='.venv' --exclude='.review' "$REPO_ROOT"/ "$WORKDIR"/

echo
echo "=== Baseline: unmodified copy ==="
"$VENV_PY" -m pytest "$WORKDIR/tests/test_documented_cli.py::test_help_lists_every_documented_flag" -q

echo
echo "=== Patching --strategy help= string with a misleading sentence ==="
python3 - "$WORKDIR/summarizer/cli.py" <<'PYEOF'
import sys
path = sys.argv[1]
content = open(path, encoding="utf-8").read()
old = (
    '"how to execute: auto picks direct when the document provably "\n'
    '            "fits, direct requires that it fits, hierarchical always splits"'
)
new = (
    '"how to execute: auto picks direct when the document provably "\n'
    '            "fits, direct requires that it fits, hierarchical always splits; "\n'
    '            "also runs the legacy chunk workflow"'
)
assert old in content, "anchor text not found; cli.py help string changed"
open(path, "w", encoding="utf-8").write(content.replace(old, new))
print("patched", path)
PYEOF

echo
echo "=== Confirming the misleading sentence reaches --help output ==="
"$VENV_PY" "$WORKDIR/main.py" --help | tr -s ' \n' ' ' | grep -o "how to execute[^-]*" || true

echo
echo "=== Re-running the test against the mutated copy ==="
"$VENV_PY" -m pytest "$WORKDIR/tests/test_documented_cli.py::test_help_lists_every_documented_flag" -q

echo
echo "Result: if the second pytest run above shows '1 passed', the test is"
echo "confirmed to pin only presence of documented substrings, not absence"
echo "of injected/misleading text."

rm -rf "$WORKDIR"
