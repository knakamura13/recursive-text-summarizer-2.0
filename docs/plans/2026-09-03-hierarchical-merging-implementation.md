# Adaptive Multi-Level Hierarchical Merging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce ordered leaf nodes to a single root through as many measured merge levels as the budget requires, preserving order, provenance, qualifications, and disagreements.

**Architecture:** `summarizer/merge.py` owns the merge prompt, request, and parsing. `summarizer/hierarchy.py` owns the tree records, grouping, and recursion. `_validate_provenance` in `summarizer/leaf.py` is generalized to a multi-segment legal set. Provenance is computed locally and excluded from merge payloads.

**Tech Stack:** Python 3, frozen dataclasses, pydantic records from #5, the #6 budget calculator, pytest

**Test command:** `uv run --with-requirements requirements-dev.txt python -m pytest -q`

Use exactly that form. `pytest` is absent from `requirements.txt`, so the bare-command form resolves an interpreter with no project dependencies and every third-party import fails in a way that looks like broken source.

---

### Task 1: Generalize provenance validation to many segments

**Files:**
- Modify: `summarizer/leaf.py`
- Create: `tests/test_provenance_validation.py`

**Step 1: Write failing tests**

Cover a node citing two of three legal identifiers; a node citing an identifier outside the set; a node citing nothing at all; a quotation matched against the core of the segment it cites; and a quotation that occurs in a *different* legal segment than the one cited, which must fail — a concatenated check would wrongly accept it.

Assert the existing single-segment behaviour is unchanged by running the existing leaf parsing tests.

**Step 2: Run and verify they fail**

Expected: FAIL because the validator takes one segment.

**Step 3: Generalize**

Change `_validate_provenance` to take a mapping from identifier to attributable text plus a subject label for messages. Have `parse_leaf_summary` build a one-entry mapping. Keep `_sanitize` and `_describe` unchanged, and keep the message bounds the existing tests assert.

The legal set must still never come from the payload.

**Step 4: Run focused and full tests**

**Step 5: Commit**

```bash
git add summarizer/leaf.py tests/test_provenance_validation.py
git commit -m "refactor(leaf): validate provenance against a legal set"
```

### Task 2: Build the merge prompt and parse a merged node

**Files:**
- Create: `summarizer/merge.py`
- Create: `tests/test_merge_prompt.py`

**Step 1: Write failing tests**

Cover: children appear only in the input slot, never the instructions; the prompt forbids inventing causal or temporal links and says adjacency is not evidence; it requires deduplication that keeps all supporting evidence; it requires contradictions preserved rather than reconciled; it is genre-neutral; the target level is stated; child provenance is absent from the payload; per-child fences are derived and a child cannot forge one; requests are deterministic; and the legal identifier set is not enumerated in the prompt.

**Step 2: Run and verify they fail**

**Step 3: Implement**

Add `MERGE_PROMPT_VERSION`, `MERGE_SCHEMA_NAME`, the instruction template, `build_merge_request(children, *, level, model, timeout_seconds, source_id)`, and `parse_merged_summary(text, *, legal, subject, level)`.

Serialize each child *without* its provenance. Reuse `leaf_summary_schema()`, which needs no change.

**Step 4: Run focused and full tests**

**Step 5: Commit**

```bash
git add summarizer/merge.py tests/test_merge_prompt.py
git commit -m "feat(merge): add a genre-neutral merge prompt"
```

### Task 3: Model the tree and group children by measurement

**Files:**
- Create: `summarizer/hierarchy.py`
- Create: `tests/test_grouping.py`

**Step 1: Write failing tests**

Cover: fanout derived from measured child size, so larger children produce smaller groups; balanced groups rather than a ragged tail; a configured ceiling clamping the measured fanout; a single child too large to fit raising `BudgetError` with the capacity, the child's size, and which child; groups preserving order under shuffled input; and a trailing group of one passing through.

**Step 2: Run and verify they fail**

**Step 3: Implement**

Add `TreeNode`, `HierarchyReport`, `measure_child_tokens`, `merge_fanout`, and `group_children`. Balance groups by splitting into near-equal sizes rather than filling greedily.

**Step 4: Run focused and full tests**

**Step 5: Commit**

```bash
git add summarizer/hierarchy.py tests/test_grouping.py
git commit -m "feat(hierarchy): group children by measured size"
```

### Task 4: Reduce leaves to a root

**Files:**
- Modify: `summarizer/hierarchy.py`
- Create: `tests/test_hierarchy.py`

**Step 1: Write failing tests**

Cover: leaves reduced to exactly one root; at least three levels forced with an injected small counter and a ceiling; every level strictly reducing the node count; a level that fails to shrink raising rather than looping; provenance on each node equalling exactly the union of its children's covered segments; the model's own provenance being discarded; deterministic results and identical request sequences across runs; source order preserved at every level; a provider failure propagating; and one provider call per merged group with none for a pass-through.

**Step 2: Run and verify they fail**

**Step 3: Implement**

Add `build_hierarchy(segments_or_leaves, provider, counter, *, model, timeout_seconds, config)` returning the root and a `HierarchyReport`. Assert the node count strictly decreases each level. Compute provenance by union and replace whatever the model emitted.

**Step 4: Run focused and full tests**

**Step 5: Commit**

```bash
git add summarizer/hierarchy.py tests/test_hierarchy.py
git commit -m "feat(hierarchy): reduce leaves to a root through measured levels"
```

### Task 5: Document and validate

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-09-03-hierarchical-merging-design.md`

**Step 1: Document**

Record that provenance is computed rather than taken from the response and is excluded from merge payloads, why that is what makes the recursion terminate, that groups are balanced, that a single oversized child fails with its arithmetic, and that three levels are unreachable at hosted capacity so multi-level behaviour is exercised by configuration.

Preserve every literal string `tests/test_documented_cli.py` asserts.

**Step 2: Byte-compile, then run the suite in both import modes**

**Step 3: Verify hygiene with `git status --short`**

**Step 4: Commit**

```bash
git add README.md docs/plans/2026-09-03-hierarchical-merging-design.md
git commit -m "chore: document hierarchical merging"
```

**Step 5: Record acceptance evidence on issue #7**

Map each criterion to its tests. Note that no live provider call is covered and that three-level behaviour is configuration-forced. Do not close the issue until its PR merges.
