#!/usr/bin/env bash
set -euo pipefail

primary=/Users/kylenakamura/documents-local/development-local/side-projects/recursive-text-summarizer
expected=301cc4d56d6326b5b0449da059b3b35f484cc5ca
copy=$(mktemp -d /tmp/rts-f002.XXXXXX)
trap 'rm -rf "$copy"' EXIT

test "$(git -C "$primary" rev-parse HEAD)" = "$expected"
cp -a "$primary/." "$copy/"
rm -rf "$copy/.review"
rm -rf "$copy/.venv" "$copy/.pytest_cache"
find "$copy" -type d -name __pycache__ -prune -exec rm -rf {} +

cd "$copy"
export UV_OFFLINE=1

probe='from summarizer.budget import BudgetError, select_strategy
from summarizer.config import StrategyConfig
from summarizer.ingestion import ingest_text
from summarizer.tokenization import ConservativeUtf8TokenCounter
try:
    result = select_strategy(ingest_text("x"), ConservativeUtf8TokenCounter(), provider="ollama", model="unknown", config=StrategyConfig(strategy="direct", max_output_tokens=1, safety_margin_tokens=0, safety_margin_fraction=0))
except BudgetError as exc:
    print("REJECT", "not known" in str(exc))
else:
    print("ALLOW", result.strategy, result.context_window_assumed)'

echo 'BASELINE_BEHAVIOR'
uv run --with-requirements requirements-dev.txt python -c "$probe"

echo 'BASELINE_SCOPED_SUITE'
uv run --with-requirements requirements-dev.txt python -m pytest -q \
  tests/test_pipeline.py tests/test_cli.py tests/test_documented_cli.py \
  tests/test_entrypoint.py tests/test_context_windows.py \
  tests/test_strategy_config.py tests/test_strategy_selection.py \
  tests/providers/test_ollama.py

echo 'BASELINE_EXISTING_GUARD_TEST'
uv run --with-requirements requirements-dev.txt python -m pytest -q \
  tests/test_direct.py::test_explicit_direct_refuses_an_assumed_context_window

python - <<'PY'
from pathlib import Path
path = Path("summarizer/budget.py")
text = path.read_text()
old = '    if config.strategy == "direct" and window.assumed:\n'
assert text.count(old) == 1
path.write_text(text.replace(old, '    if False:  # F-002 mutation\n'))
PY

echo 'MUTANT_BEHAVIOR'
uv run --with-requirements requirements-dev.txt python -c "$probe"

echo 'MUTANT_SCOPED_SUITE'
uv run --with-requirements requirements-dev.txt python -m pytest -q \
  tests/test_pipeline.py tests/test_cli.py tests/test_documented_cli.py \
  tests/test_entrypoint.py tests/test_context_windows.py \
  tests/test_strategy_config.py tests/test_strategy_selection.py \
  tests/providers/test_ollama.py

echo 'MUTANT_EXISTING_GUARD_TEST_EXPECTED_FAILURE'
set +e
uv run --with-requirements requirements-dev.txt python -m pytest -q \
  tests/test_direct.py::test_explicit_direct_refuses_an_assumed_context_window
status=$?
set -e
test "$status" -ne 0
echo "EXISTING_GUARD_TEST_KILLED_MUTANT exit=$status"
