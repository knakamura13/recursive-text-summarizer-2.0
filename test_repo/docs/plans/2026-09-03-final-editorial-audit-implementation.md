# Final Editorial Synthesis, Citations, and Audit Implementation Plan

**Goal:** Deliver Issue #9 as a library-level finalization stage that works for
both direct and hierarchical roots without moving the legacy CLI boundary.

## Task 1: Write the editorial contract and tests

- Create `summarizer/editorial.py` and `tests/test_editorial.py`.
- Add a strict `FinalDraft` record, deterministic fenced request builder,
  defensive parser, source-ordered citation resolver, and optional formatter.
- Test dedicated calls, requested lengths, genre-neutral/no-unsupported-meaning
  instructions, prompt injection boundaries, deterministic requests, unknown
  citations, and default plain rendering.

## Task 2: Add the audit artifact

- Create `summarizer/audit.py` and `tests/test_audit.py`.
- Model source segment metadata, tree nodes, content/evidence links, sanitized
  configuration, usage, warnings/failures, and citation mappings.
- Validate all identifier links; redact raw secrets; serialize canonically;
  validate bytes before atomically writing them.

## Task 3: Compose the public finalization API

- Create `summarizer/finalization.py` and `tests/test_finalization.py`.
- Compose final writing, stable citations, and optional audit production around
  a direct root or a multi-level hierarchy tree.
- Cover end-to-end deterministic fake-provider flows and provider failure
  propagation. Do not wire CLI flags or replace `LegacyWorkflow`.

## Task 4: Document and verify

- Update `README.md` to distinguish delivered library functionality from the
  still-legacy command line.
- Run the focused tests, full suite, compilation, and whitespace checks.
- Review acceptance criteria, commit, push, create a GitHub PR, and add an
  Issue #9 update. Do not merge without a separate user instruction.
