# Source Grounding and Provenance Propagation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ground every recursive merge in selected original passages and retain
only validated source references that support its retained output.

**Architecture:** A narrow `GroundedAnnotation` schema makes retained
qualifications and contradictions evidence-linkable. A new merge-local selector
derives deterministic candidate source IDs from those annotations and child
evidence, then serializes bounded complete source passages independently from
generated child summaries. The hierarchy reserves source capacity before
fanout; parsing validates output against selected passages and canonicalizes
narrowed provenance while the tree retains complete structural coverage.

**Tech Stack:** Python 3.12, Pydantic records, injectable `TokenCounter`,
pytest fakes.

---

### Task 1: Specify and test bounded source selection

**Files:**
- Modify: `summarizer/summaries.py`
- Modify: `summarizer/leaf.py`
- Create: `summarizer/grounding.py`
- Create: `tests/test_grounding.py`
- Modify: `tests/test_summaries.py`
- Modify: `tests/test_provenance_validation.py`

**Step 1: Write failing tests**

Cover source-order candidate priority from grounded contradictions,
qualifications, uncertainty, quotations, ordinary content evidence, and
fallback provenance. Require content units and grounded annotations to have
evidence. Add failure cases where a mandatory source passage cannot fit or
where evidence cites an unknown source.

**Step 2: Run the focused test file**

Run: `.venv/bin/python -m pytest tests/test_grounding.py -q`

Expected: FAIL because the grounding module does not exist.

**Step 3: Implement the smallest selector**

Add immutable `GroundingPolicy`, `SourcePassage`, and selection records plus
candidate collection and token-verified whole-passage packing. Add
`GroundedAnnotation`, validate all direct evidence in the shared validator,
and derive narrowed provenance from validated model output in source order.

**Step 4: Re-run the focused test file**

Run: `.venv/bin/python -m pytest tests/test_grounding.py -q`

Expected: PASS.

### Task 2: Add a separately fenced authoritative merge block

**Files:**
- Modify: `summarizer/merge.py`
- Modify: `tests/test_merge_prompt.py`

**Step 1: Write failing request tests**

Require complete selected source passages in `build_merge_request`, assert source text is absent
from instructions, present only in distinct source fences, and that the prompt
names source passages authoritative and child summaries provisional.

**Step 2: Run the focused test file**

Run: `.venv/bin/python -m pytest tests/test_merge_prompt.py -q`

Expected: FAIL because merge requests have no source block.

**Step 3: Implement the request shape**

Add deterministic source fences and compact serialization. Bump the merge
prompt version and include its static framing in overhead measurement.

**Step 4: Re-run the focused test file**

Run: `.venv/bin/python -m pytest tests/test_merge_prompt.py -q`

Expected: PASS.

### Task 3: Reserve grounding capacity and narrow merge provenance

**Files:**
- Modify: `summarizer/hierarchy.py`
- Modify: `summarizer/merge.py`
- Modify: `tests/test_hierarchy.py`
- Modify: `tests/test_merge_prompt.py`

**Step 1: Write failing hierarchy tests**

Assert that actual requests remain within the usable budget, selected passages
are only from the group's covered segments, a misleading child receives the
contrary original passage, invalid references fail, full
`covered_segments` remains available, and output provenance narrows in source
order.

**Step 2: Run focused hierarchy and merge tests**

Run: `.venv/bin/python -m pytest tests/test_hierarchy.py tests/test_merge_prompt.py -q`

Expected: FAIL because hierarchy neither reserves source capacity nor passes
selected authoritative text to merge construction.

**Step 3: Implement the minimal integration**

Reserve a fixed `GroundingPolicy.max_tokens` before fanout; use that bounded
reserve for selection and assert exact final request accounting. Validate every
incoming leaf against its covered source mapping, pass only selected passages
as the merge legal map, require merge provenance, and canonicalize it to
selected source order instead of replacing it with the child union.

**Step 4: Re-run focused tests**

Run: `.venv/bin/python -m pytest tests/test_grounding.py tests/test_hierarchy.py tests/test_merge_prompt.py -q`

Expected: PASS.

### Task 4: Document and verify the issue boundary

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-09-03-source-grounding-provenance-design.md`

**Step 1: Add the pipeline-stage documentation**

Describe source-grounded merges, the difference between structural coverage
and narrowed claim provenance, and offline verification limits without
claiming factual perfection.

**Step 2: Run all checks**

Run: `.venv/bin/python -m pytest -q`

Expected: PASS with no network calls.

**Step 3: Review the issue acceptance criteria**

Map each #8 criterion to code and tests, inspect `git diff --check`, then
commit the issue-sized change.
