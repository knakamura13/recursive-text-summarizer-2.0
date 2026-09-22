# Issue #65 Audit Durability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every audit artifact use the same durable replacement primitive and document the distinction from cache-backed paired publication.

**Architecture:** Move the durable temporary-file replacement code from `summarizer.finalization` into `summarizer.audit`, where `write_audit()` delegates to it. `finalization` imports the same helper under the existing `_atomic_replace` name, keeping publication and CLI imports stable. Tests independently pin file-then-directory fsync behavior for each exposed route.

**Tech Stack:** Python 3, pytest, pathlib, tempfile, POSIX-style file descriptors.

---

### Task 1: Add failing durability regression tests

**Files:**
- Modify: `tests/test_audit.py`
- Modify: `tests/test_publication.py`

**Step 1: Write the standalone audit test**

Add `os` and `stat` imports. Reuse `fixture()` to create the artifact, replace
`summarizer.audit.os.fsync` with a no-op spy that classifies each open descriptor
using `os.fstat()`, then assert a successful `write_audit()` yields
`["file", "directory"]` and canonical output bytes.

**Step 2: Write the publication-helper test**

Import `_atomic_replace`, `os`, and `stat` in `tests/test_publication.py`. Apply
the same no-op descriptor-classifying spy to `summarizer.finalization.os.fsync`,
write `b"payload"` through `_atomic_replace()`, and assert both the payload and
the exact `file`, then `directory`, sync sequence.

**Step 3: Verify the tests fail on the baseline**

Run:

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q \
  tests/test_audit.py tests/test_publication.py
```

Expected: the new direct-audit durability assertion fails because the current
writer does not call `fsync`.

### Task 2: Converge the two production write paths

**Files:**
- Modify: `summarizer/audit.py:1-14,1432-1445`
- Modify: `summarizer/finalization.py:5-25,70-90`

**Step 1: Move the helper**

Move the current `_atomic_replace(path: Path, payload: bytes)` implementation
verbatim from `finalization.py` into `audit.py`, adding the `os` import there.
Keep its temporary-file cleanup and error propagation behavior intact.

**Step 2: Delegate the direct writer**

Reduce `write_audit()` to canonical serialization followed by
`_atomic_replace(path, payload)`. Its validation behavior remains unchanged.

**Step 3: Preserve finalization's internal API**

Import `_atomic_replace` from `summarizer.audit` in `finalization.py` alongside
the existing audit imports, then remove its duplicated definition and now-unused
`os`/`NamedTemporaryFile` imports. Leave `publish_final_output`'s default
argument and `cli.py` import unchanged.

**Step 4: Run the focused tests**

Run the Task 1 command again. Expected: PASS.

**Step 5: Commit**

```bash
git add summarizer/audit.py summarizer/finalization.py tests/test_audit.py tests/test_publication.py
git commit -m "fix: unify audit artifact durability"
```

### Task 3: State the user-facing durability contract

**Files:**
- Modify: `README.md:130-150`

**Step 1: Clarify the two guarantees**

Add one sentence stating that all `--audit` artifacts use durable
same-directory replacement, while `--cache-dir` additionally provides the
manifest-witnessed audit/summary publication protocol. Do not describe the pair
as cross-file atomic.

**Step 2: Check formatting and targeted behavior**

Run:

```bash
git diff --check
uv run --with-requirements requirements.txt --with pytest python -m pytest -q \
  tests/test_audit.py tests/test_publication.py
```

Expected: no whitespace errors and all focused tests pass.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: clarify audit durability guarantees"
```

### Task 4: Validate the complete change

**Files:**
- Verify only

**Step 1: Run the full test suite**

```bash
uv run --with-requirements requirements.txt --with pytest python -m pytest -q
```

Expected: all tests pass.

**Step 2: Inspect the final diff**

Run:

```bash
git diff origin/main...HEAD --check
git diff origin/main...HEAD
git status --short --branch
```

Expected: only the approved design/plan, durability implementation, tests, and
README clarification are present; frozen review evidence is unchanged.
