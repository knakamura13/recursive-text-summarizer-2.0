# S1 tests-and-mutation review

Reviewed primary checkout:
`/Users/kylenakamura/documents-local/development-local/side-projects/recursive-text-summarizer`

Commit: `301cc4d56d6326b5b0449da059b3b35f484cc5ca`.

Graph discovery limitation: codebase-memory MCP graph and coverage tools were
not exposed to this reviewer session. Exact, bounded source and test reads were
used instead; no coverage claim is made.

## Primary baseline

```sh
UV_OFFLINE=1 uv run --with-requirements requirements-dev.txt python -m pytest -q tests/test_pipeline.py tests/test_cli.py tests/test_documented_cli.py tests/test_entrypoint.py tests/test_context_windows.py tests/test_strategy_config.py tests/test_strategy_selection.py tests/providers/test_ollama.py
```

Result: `91 passed in 5.81s`.

## Mutations (each in a distinct full disposable copy under /tmp)

| mutation | result | interpretation |
| --- | --- | --- |
| `budget.py:286`, replace the forced-direct assumed-window guard with `if False:` | `91 passed in 5.75s` against a fresh full copy of the primary checkout | survivor; no listed test covers forced direct with an assumed context window |
| `budget.py:351`, replace auto assumed-window guard with `if False:` | `1 failed, 90 passed` | killed by `test_an_assumed_window_routes_auto_to_hierarchical` (earlier identical-HEAD worktree pass) |
| `budget.py:279`, replace `<=` with `<` | `1 failed, 90 passed` | killed by `test_selects_direct_exactly_at_capacity` (earlier identical-HEAD worktree pass) |
| `ollama.py:93`, replace terminal `done` guard with `if False:` | `2 failed, 19 passed` in `tests/providers/test_ollama.py` | killed by `test_rejects_nonterminal_or_missing_content` (earlier identical-HEAD worktree pass) |

The primary source hashes match the earlier worktree: `summarizer/budget.py`
`36425dcc6f92d628113c7ddf52a06c5ff41144b8`; `tests/test_strategy_selection.py`
`7d46bd370715c5dc4c6e9b150054e3bbdb303817`.

The survivor's behavior in its separate copy was confirmed with:

```sh
UV_OFFLINE=1 uv run --with-requirements requirements-dev.txt python -c 'from summarizer.budget import select_strategy; from summarizer.config import StrategyConfig; from summarizer.ingestion import ingest_text; from summarizer.tokenization import ConservativeUtf8TokenCounter; r=select_strategy(ingest_text("x"), ConservativeUtf8TokenCounter(), provider="ollama", model="unknown", config=StrategyConfig(strategy="direct", max_output_tokens=1, safety_margin_tokens=0, safety_margin_fraction=0)); print(r.strategy, r.context_window_assumed)'
```

Output: `direct True`. The primary checkout's tracked status was empty before
this excluded evidence file was created.
