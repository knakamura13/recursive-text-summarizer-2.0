# Automatic Strategy Selection and Direct Summarization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compute usable input capacity from real measurements, select `direct` or `hierarchical` safely, and complete the direct path by reusing the leaf machinery through a document-spanning segment.

**Architecture:** A new `summarizer/budget.py` owns the context table, overhead measurement, capacity arithmetic, and selection. A new `summarizer/direct.py` owns the document-spanning segment and the direct stage. `StrategyConfig` joins `ParsedConfig` rather than `AppConfig`. The legacy CLI workflow is untouched.

**Tech Stack:** Python 3, frozen dataclasses, `tiktoken`, the existing provider boundary, pytest

**Test command:** `uv run --with-requirements requirements-dev.txt python -m pytest -q`

Use exactly that form. `pytest` is absent from `requirements.txt`, so `uv run --with-requirements requirements.txt pytest` resolves the bare command off `PATH` to an interpreter with no project dependencies, and every third-party import then fails in a way that looks like broken source rather than a wrong interpreter.

---

### Task 1: Resolve a model's context window

**Files:**
- Create: `summarizer/budget.py`
- Create: `tests/test_context_windows.py`

**Step 1: Write failing resolution tests**

Cover an exact table hit, a prefix hit (`gpt-4o-mini` must resolve the way `tiktoken`'s two-tier lookup makes it resolve), an explicit override winning over the table, an unknown OpenAI model, and an arbitrary Ollama tag. Assert the `assumed` flag distinguishes a table hit from a fallback:

```python
known = resolve_context_window(provider="openai", model="gpt-4o-mini")
assert known.assumed is False
unknown = resolve_context_window(provider="ollama", model="qwen3.8")
assert unknown.assumed is True
```

**Step 2: Run the tests and verify they fail**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest tests/test_context_windows.py -q`

Expected: FAIL because `summarizer.budget` does not exist.

**Step 3: Implement the table and resolution**

Add `ContextWindow` (tokens plus `assumed`), `_MODEL_CONTEXT_WINDOWS`, `_MODEL_PREFIX_CONTEXT_WINDOWS`, and `resolve_context_window(*, provider, model, explicit=None)`. Two-tier lookup, exact then longest matching prefix. Record in a comment that neither `openai` 3.7.0 nor `tiktoken` exposes a window, so the table is the only offline source and must be maintained by hand.

**Step 4: Run focused and full tests**

Run the focused file, then `uv run --with-requirements requirements-dev.txt python -m pytest -q`.

Expected: 253 existing tests plus the new ones pass.

**Step 5: Commit**

```bash
git add summarizer/budget.py tests/test_context_windows.py
git commit -m "feat(budget): resolve model context windows"
```

### Task 2: Measure overhead and compute usable capacity

**Files:**
- Modify: `summarizer/budget.py`
- Create: `tests/test_budget.py`

**Step 1: Write failing measurement and arithmetic tests**

Pin the measured overhead so that editing the instructions or the record fails loudly. Use a deterministic character counter for the arithmetic, and one real-encoding case guarded by the established skip pattern.

Cover: overhead rising when overlap is configured; capacity equal to `window − overhead − reserve − margin`; the margin taking the larger of its fixed and fractional terms; and a non-positive capacity raising `BudgetError` whose message contains every term.

```python
with pytest.raises(BudgetError) as error:
    usable_input_capacity(window=ContextWindow(4096, assumed=True), ...)
for term in ("4096", "overhead", "reserve", "margin"):
    assert term in str(error.value)
```

**Step 2: Run the tests and verify they fail**

Expected: FAIL because `measure_overhead` and `usable_input_capacity` do not exist.

**Step 3: Implement measurement and arithmetic**

Add `OverheadMeasurement`, `measure_overhead(counter, *, with_overlap)`, `safety_margin(window, config)`, `usable_input_capacity(...)`, and `BudgetError`.

Measure by counting the filled instruction template, the compactly-serialized schema, and the fencing — never a hard-coded constant, because an acceptance criterion covers overhead changes. Take the overlap-carrying variant when overlap is configured, since it is the larger.

**Step 4: Run focused and full tests**

**Step 5: Commit**

```bash
git add summarizer/budget.py tests/test_budget.py
git commit -m "feat(budget): measure prompt overhead and usable capacity"
```

### Task 3: Configure strategy

**Files:**
- Modify: `summarizer/config.py`
- Create: `tests/test_strategy_config.py`

**Step 1: Write failing validation tests**

Cover the default strategy, rejection of an unknown strategy name, rejection of a non-positive explicit window, a negative reserve, a fractional margin outside `[0, 1)`, and a non-positive direct cap. Assert `AppConfig` positional construction still works, since a commit exists in this repository's history solely to repair that.

**Step 2: Run the tests and verify they fail**

**Step 3: Implement `StrategyConfig`**

Add `StrategyName = Literal["auto", "direct", "hierarchical"]` and a frozen `StrategyConfig` validating in `__post_init__`, mirroring how `AppConfig` validates `provider`.

**Step 4: Run focused and full tests**

**Step 5: Commit**

```bash
git add summarizer/config.py tests/test_strategy_config.py
git commit -m "feat(config): add strategy and budget configuration"
```

### Task 4: Select a strategy and report why

**Files:**
- Modify: `summarizer/budget.py`
- Create: `tests/test_strategy_selection.py`

**Step 1: Write failing selection tests**

Cover a document one token under capacity selecting `direct` and one token over selecting `hierarchical`; explicit `hierarchical` never measuring; explicit `direct` over capacity raising `BudgetError` with its arithmetic; an unknown window routing `auto` to hierarchical even when the document fits; the direct cap forcing hierarchical; and determinism of the report across runs.

Assert the report carries the counter identity and exactness, every overhead term, and a machine-readable reason.

**Step 2: Run the tests and verify they fail**

**Step 3: Implement selection**

Add `BudgetReport` and `select_strategy(document, counter, *, provider, model, config)`. Measure the document with `counter.count(document.text)` — the text a direct request actually sends, not a sum of segment counts.

**Step 4: Run focused and full tests**

**Step 5: Commit**

```bash
git add summarizer/budget.py tests/test_strategy_selection.py
git commit -m "feat(budget): select a strategy from a measured budget"
```

### Task 5: Summarize a whole document directly

**Files:**
- Modify: `summarizer/segmentation.py`
- Modify: `summarizer/leaf.py`
- Create: `summarizer/direct.py`
- Create: `tests/test_direct.py`
- Modify: `tests/test_leaf_prompt.py`

**Step 1: Write failing direct tests**

Cover a document-spanning segment reconstructing the whole canonical text with `BoundaryKind.DOCUMENT`; the prompt not describing the input as a fragment; a successful direct summary returning one `SummaryNode` at level zero whose provenance is the single identifier; a quotation from anywhere in the document validating; a malformed response failing with the document's identifier; exactly one provider call; and determinism.

**Step 2: Run the tests and verify they fail**

**Step 3: Implement the direct path**

Add `BoundaryKind.DOCUMENT`. A unit that is the entire document has no boundary *within* a document, and reporting `HARD` or `PARAGRAPH` would misstate provenance.

In `summarizer/leaf.py`, select the prompt's framing sentence on whether the unit is the whole document, and bump `LEAF_PROMPT_VERSION`. Telling a model it is reading "one region of a longer document" when it is reading everything invites it to hedge about absent context, which is the opposite of the cohesive result required.

In `summarizer/direct.py`, add `whole_document_segment(document, counter)` and `summarize_direct(document, provider, counter, *, model, timeout_seconds)`, reusing `build_leaf_request` and `parse_leaf_summary` unchanged.

**Step 4: Run focused and full tests**

**Step 5: Commit**

```bash
git add summarizer/segmentation.py summarizer/leaf.py summarizer/direct.py tests/test_direct.py tests/test_leaf_prompt.py
git commit -m "feat(direct): summarize a whole document in one call"
```

### Task 6: Expose strategy on the command line

**Files:**
- Modify: `summarizer/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_documented_cli.py`

**Step 1: Write failing CLI tests**

Cover the default strategy on a bare invocation, each strategy accepted, an unknown strategy exiting 2, each budget flag reaching `StrategyConfig`, and a rejected value producing a usage error rather than a traceback.

Validation must surface through the argument parser. `main()` catches only `OSError` and `ProviderError`, so a `ValueError` escaping into it would print a traceback — which an existing CLI test already asserts against for other paths.

**Step 2: Run the tests and verify they fail**

**Step 3: Add the flags**

Add `--strategy`, `--context-window`, `--max-output-tokens`, `--safety-margin-tokens`, `--safety-margin-fraction`, and `--max-direct-tokens`, with `help=` text, and add `strategy` to `ParsedConfig`. Keep the existing `try/except ValueError → parser.error` pattern so a bad value exits 2.

Do not rewire `main()` onto the new pipeline. Issue #12 owns the end-to-end demonstration, and #4 and #5 both deliberately left the command line on the legacy path.

**Step 4: Run focused and full tests**

**Step 5: Commit**

```bash
git add summarizer/cli.py tests/test_cli.py tests/test_documented_cli.py
git commit -m "feat(cli): configure summarization strategy"
```

### Task 7: Document the boundary and validate

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-09-03-strategy-selection-design.md`

**Step 1: Document the budget and the strategy**

State how usable capacity is derived, that overhead is measured rather than assumed, that an unknown window routes to hierarchical rather than guessing, that the output reserve is accounted for but not enforced, and that the command line accepts strategy configuration without yet running the new pipeline.

Record the local-truncation hazard explicitly: OpenAI rejects an oversized request, while Ollama silently truncates the prompt and returns plausible output, so budget arithmetic is the only defence on that path.

Preserve every literal string asserted by `tests/test_documented_cli.py`.

**Step 2: Run the byte-compile check**

Run: `uv run --with-requirements requirements-dev.txt python -m compileall -q summarizer tests`

**Step 3: Run the complete suite through both import modes**

Run it plain, then with `--import-mode=importlib`.

**Step 4: Verify repository hygiene**

Run: `git status --short`

Expected: only intended issue #6 changes; no cache, log, output, or credential files.

**Step 5: Commit**

```bash
git add README.md docs/plans/2026-09-03-strategy-selection-design.md
git commit -m "chore: document strategy selection and budgets"
```

**Step 6: Record acceptance evidence on issue #6**

Map each criterion to its tests and implementation paths. Note that no live provider call is covered, and that the output reserve is notional until an output cap crosses the provider boundary. Do not close the issue until its PR merges.
