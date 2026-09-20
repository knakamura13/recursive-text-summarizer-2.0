# Generalized Ingestion and Token-Aware Segmentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build reusable, offline ingestion and token-aware segmentation that preserves document structure and exact source provenance for later hierarchical summarization.

**Architecture:** Add immutable source models, a narrow canonicalization layer, an injectable token-counter boundary, and a deterministic structure-first segmenter. Keep the legacy CLI workflow untouched; issue #5 will consume the new `SourceSegment` objects.

**Tech Stack:** Python 3, frozen dataclasses, `typing.Protocol`, NLTK Punkt sentence boundaries, `tiktoken`, pytest

---

### Task 1: Add canonical source ingestion

**Files:**
- Create: `summarizer/ingestion.py`
- Create: `tests/test_ingestion.py`

**Step 1: Write failing normalization and validation tests**

Cover UTF-8 file reading, a leading BOM, CRLF and CR normalization, removal of trailing horizontal whitespace, preservation of headings/list indentation/blank lines, Unicode text, whitespace-only rejection, and decode/read errors that mention the path and encoding.

Use invariants such as:

```python
document = ingest_text("\ufeff# H\r\n\r\n  - café  \r\n")
assert document.text == "# H\n\n  - café"
assert document.source_id == ingest_text(document.text).source_id
```

**Step 2: Run the tests and verify they fail**

Run: `uv run --with-requirements requirements.txt pytest tests/test_ingestion.py -q`

Expected: FAIL because `summarizer.ingestion` does not exist.

**Step 3: Implement the minimal immutable model and ingestion functions**

Add `SourceDocument`, `SourceReadError`, `SourceDecodeError`, `EmptySourceError`, `normalize_source_text`, `ingest_text`, and `read_source`. Compute `source_id` as a SHA-256 digest of canonical UTF-8 bytes. Catch only relevant filesystem and Unicode exceptions and preserve their cause.

**Step 4: Run focused and full tests**

Run: `uv run --with-requirements requirements.txt pytest tests/test_ingestion.py -q`

Expected: PASS.

Run: `uv run --with-requirements requirements.txt pytest -q`

Expected: 111 existing tests plus the new ingestion tests pass.

**Step 5: Commit**

```bash
git add summarizer/ingestion.py tests/test_ingestion.py
git commit -m "feat(ingestion): add canonical source documents"
```

### Task 2: Add the injectable token-counter boundary

**Files:**
- Create: `summarizer/tokenization.py`
- Create: `tests/test_tokenization.py`
- Modify: `requirements.txt`

**Step 1: Write failing protocol and resolver tests**

Test a deterministic fake through the `TokenCounter` protocol, exact `tiktoken` counting for a known OpenAI model, explicit OpenAI encoding fallback, conservative UTF-8 counting for ASCII and multibyte Unicode, stable counter identities, `exact` metadata, and Ollama/unknown model selection without any network access.

```python
counter = resolve_token_counter(provider="ollama", model="custom/model:tag")
assert counter.exact is False
assert counter.count("é") >= 1
```

**Step 2: Run the tests and verify they fail**

Run: `uv run --with-requirements requirements.txt pytest tests/test_tokenization.py -q`

Expected: FAIL because `summarizer.tokenization` does not exist.

**Step 3: Add the dependency and minimal implementation**

Add a bounded `tiktoken` requirement. Implement `TokenCounter`, `TiktokenCounter`, `ConservativeUtf8TokenCounter`, `TokenAccountingError`, and `resolve_token_counter`. The resolver may inspect provider/model strings but must not construct a provider client or call the network.

The conservative counter uses the UTF-8 byte length as a deliberately cautious upper estimate:

```python
return len(text.encode("utf-8"))
```

Reject negative results from injected counters at the segmentation boundary.

**Step 4: Run focused and full tests**

Run: `uv run --with-requirements requirements.txt pytest tests/test_tokenization.py -q`

Expected: PASS.

Run: `uv run --with-requirements requirements.txt pytest -q`

Expected: all tests pass without provider access.

**Step 5: Commit**

```bash
git add requirements.txt summarizer/tokenization.py tests/test_tokenization.py
git commit -m "feat(tokenization): add provider-aware token counters"
```

### Task 3: Model structural blocks and source segments

**Files:**
- Create: `summarizer/segmentation.py`
- Create: `tests/test_segmentation_models.py`

**Step 1: Write failing model-invariant tests**

Test `BoundaryKind`, `StructuralBlock`, `SegmentationConfig`, and `SourceSegment`. Reject empty IDs, invalid or inverted ranges, nonpositive budgets, negative overlap, and context ranges that do not contain their core ranges.

**Step 2: Run the tests and verify they fail**

Run: `uv run --with-requirements requirements.txt pytest tests/test_segmentation_models.py -q`

Expected: FAIL because the models do not exist.

**Step 3: Implement frozen dataclasses and validation**

Keep models free of provider and prompt concepts. Include `core_start`, `core_end`, `context_start`, `context_end`, `core_token_count`, `token_count`, `leading_overlap_tokens`, `trailing_overlap_tokens`, and `boundary_kind` on `SourceSegment`.

**Step 4: Run focused tests**

Run: `uv run --with-requirements requirements.txt pytest tests/test_segmentation_models.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add summarizer/segmentation.py tests/test_segmentation_models.py
git commit -m "feat(segmentation): define provenance-aware segment models"
```

### Task 4: Detect headings, paragraphs, lists, and sentence boundaries

**Files:**
- Modify: `summarizer/segmentation.py`
- Create: `tests/test_structure_detection.py`

**Step 1: Write failing structure tests**

Cover ATX headings, setext headings, prose paragraphs, contiguous list items, blank-line separators, repeated identical paragraphs, Unicode, and injection-like text. Assert every block resolves to the exact canonical source slice and repeated text receives distinct ranges.

**Step 2: Run the tests and verify they fail**

Run: `uv run --with-requirements requirements.txt pytest tests/test_structure_detection.py -q`

Expected: FAIL because `detect_structural_blocks` does not exist.

**Step 3: Implement offset-preserving detection**

Scan with `re.finditer` and explicit indices. Do not locate repeated blocks with `str.find` from the beginning. Preserve separators in contiguous ranges. Reuse the resource-free Punkt tokenizer only to identify subordinate sentence spans inside oversized blocks.

**Step 4: Run focused tests**

Run: `uv run --with-requirements requirements.txt pytest tests/test_structure_detection.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add summarizer/segmentation.py tests/test_structure_detection.py
git commit -m "feat(segmentation): preserve document structure and offsets"
```

### Task 5: Implement deterministic budgeted segmentation

**Files:**
- Modify: `summarizer/segmentation.py`
- Create: `tests/test_segmentation.py`

**Step 1: Write failing boundary and packing tests**

Use a deterministic counter such as one token per character. Test preference for sections, then paragraphs, then sentences, then hard fallback. Test exact-budget packing, an oversized first unit, Unicode, tiny budgets, reconstruction from core ranges, stable `S000001` identifiers, deterministic reruns, source order, and every emitted segment staying within budget.

**Step 2: Run the tests and verify they fail**

Run: `uv run --with-requirements requirements.txt pytest tests/test_segmentation.py -q`

Expected: FAIL because `segment_document` does not exist.

**Step 3: Implement greedy packing and recursive fallback**

Add a private `largest_fitting_prefix` that performs monotonic binary search and guarantees at least one-character progress or raises `SegmentationError`. Recount final candidate slices rather than summing subunit counts because tokenization is not generally additive.

**Step 4: Add deterministic invariant validation**

Validate ordered disjoint core ranges, exact source slicing, budget compliance, and complete reconstruction at the public boundary. A counter returning a negative number raises `TokenAccountingError`.

**Step 5: Run focused and full tests**

Run: `uv run --with-requirements requirements.txt pytest tests/test_segmentation.py -q`

Expected: PASS.

Run: `uv run --with-requirements requirements.txt pytest -q`

Expected: all tests pass.

**Step 6: Commit**

```bash
git add summarizer/segmentation.py tests/test_segmentation.py
git commit -m "feat(segmentation): add token-budgeted structure-first splitting"
```

### Task 6: Add explicit bounded overlap

**Files:**
- Modify: `summarizer/segmentation.py`
- Modify: `tests/test_segmentation.py`

**Step 1: Write failing overlap tests**

Test zero-overlap defaults, backward context expansion, reduction when core text nearly fills the budget, boundary preference, exact leading-overlap metadata, unchanged core ranges and IDs, no ambiguity in repeated text, and continued budget compliance.

**Step 2: Run the overlap tests and verify they fail**

Run: `uv run --with-requirements requirements.txt pytest tests/test_segmentation.py -q -k overlap`

Expected: FAIL because overlap is not applied.

**Step 3: Implement overlap as context-only expansion**

Expand `context_start` backward after core segments are fixed. Never change `core_start`, `core_end`, or IDs. Prefer known structural boundaries, then use the same safe prefix/suffix search. Reduce overlap until the combined context fits.

**Step 4: Run focused and full tests**

Run: `uv run --with-requirements requirements.txt pytest tests/test_segmentation.py -q`

Expected: PASS.

Run: `uv run --with-requirements requirements.txt pytest -q`

Expected: all tests pass.

**Step 5: Commit**

```bash
git add summarizer/segmentation.py tests/test_segmentation.py
git commit -m "feat(segmentation): add explicit bounded context overlap"
```

### Task 7: Document and validate the issue #4 boundary

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-09-02-ingestion-segmentation-design.md`
- Test: `tests/test_ingestion.py`
- Test: `tests/test_tokenization.py`
- Test: `tests/test_segmentation.py`

**Step 1: Document canonical offsets and counter accuracy**

Explain that offsets resolve against canonical normalized text, overlap is context-only, OpenAI counters can be exact, arbitrary Ollama tags use a conservative offline estimator unless an exact counter is injected, and no provider call occurs during segmentation.

**Step 2: Run formatting/static checks available in the repository**

Run: `uv run --with-requirements requirements.txt python -m compileall -q summarizer tests`

Expected: exit 0.

**Step 3: Run the complete suite twice through normal and importlib modes**

Run: `uv run --with-requirements requirements.txt pytest -q`

Expected: all tests pass.

Run: `uv run --with-requirements requirements.txt pytest -q --import-mode=importlib`

Expected: all tests pass.

**Step 4: Verify repository hygiene**

Run: `git status --short`

Expected: only intended issue #4 documentation changes remain before the final commit; no cache, output, log, credential, or virtual-environment files are present.

**Step 5: Commit**

```bash
git add README.md docs/plans/2026-09-02-ingestion-segmentation-design.md
git commit -m "chore: document token-aware segmentation"
```

**Step 6: Update issue #4 acceptance evidence**

Post the final test counts and a concise mapping from each acceptance criterion to tests and implementation paths. Do not close issue #4 until its PR is merged.
