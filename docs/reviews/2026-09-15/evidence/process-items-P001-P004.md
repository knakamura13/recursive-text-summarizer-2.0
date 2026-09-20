# Process items P-001 and P-004, adjudicated session 7

Process items are adjudicated by the lead, not by the verifier pipeline that assigns finding verdicts. Both below rest on evidence the lead re-derived independently rather than accepting from an agent's report.

---

## P-004 — stale-bytecode masking in mutation testing: DOWNGRADED

**Opened** session 7, when the ingestion mutation reviewer discovered that `cp -R` carries over `__pycache__/*.pyc` and that a rapid mutate/test/restore loop can execute bytecode compiled from the *unmutated* source, recording a SURVIVED row for a mutation that never ran. The concern was that no earlier mutation pass in this review — S1's 4, S2's 21, session 6's 20 — documented taking the precaution, so every survivor count might be an upper bound containing phantoms.

**Verdict: downgraded. The effect is real but not reachable under this review's actual working conditions, and no phantom was found.**

### The mechanism, confirmed

CPython's default timestamp invalidation stores only two fields from the source's `stat()` in the `.pyc` header: mtime truncated to whole seconds, and size in bytes. It does not hash content. The verifier confirmed this by dumping a `.pyc` header directly (`flags=0`, non-hash-based; embedded `source_mtime`/`source_size` matching `stat()` exactly).

### Under realistic conditions it does not fire

Applying the known-killed mutation at `ingestion.py:49` with `sed -i ''` — representative of how this review actually edits — then running the five target tests with a warm cache and no precaution: **all five correctly FAILED**. No masking. `sed -i` changed both the mtime (by 340s) and the size (2237 to 2229 bytes, the 8-byte drop matching `.strip()`'s removal), and a size mismatch alone forces recompilation regardless of mtime.

### Under contrived conditions it does fire

To prove the mechanism can fire at all, the verifier applied the same mutation, padded the file with 8 trailing spaces to restore the exact original byte count, and used `os.utime()` to force the exact original mtime back. With an identical `(mtime, size)` pair to the stale `.pyc`'s header: **all five incorrectly PASSED**, while the mutated text was verifiably on disk. Re-running that same on-disk state with `PYTHONDONTWRITEBYTECODE=1` and a cleared cache: **all five correctly FAILED** again.

So masking requires an exact collision on both axes. A `sed` or Python in-place rewrite normally defeats it on both at once. The residual risk sits with same-length single-character substitutions (`>` to `<`, `==` to `!=`) applied repeatedly to one file inside the same second — a regime this review's editing style does not produce.

### The spot-check: no phantoms

Five previously recorded survivors, re-run on a fresh copy with `__pycache__` cleared before every run and `PYTHONDONTWRITEBYTECODE=1` on every run, full 766-test suite each time:

| mutation | result |
|---|---|
| `budget.py:283` `>` to `>=` (F-023) | STILL SURVIVES |
| `budget.py:21` `"gpt-4": 8_192` to `1` (F-022) | STILL SURVIVES |
| `budget.py:146` drop the `max(fencing, 0)` clamp (F-025) | STILL SURVIVES |
| `segmentation.py:352-353` delete the post-fence blank-line consume (C-S3M-003) | STILL SURVIVES |
| `segmentation.py:517` `<` to `<=` (C-S3M-004) | STILL SURVIVES |

None was a phantom. F-022, F-023 and F-025 keep their evidential basis, and C-S3M-003 and C-S3M-004 remain live unverified candidates rather than artifacts.

**Standing instruction going forward:** every future mutation pass in this review clears `__pycache__` and sets `PYTHONDONTWRITEBYTECODE=1`. The cost is negligible and it removes the question entirely. Note for the record that the verifier's own `budget.py:21` mutation needed care for an unrelated reason: the literal `"gpt-4": 8_192` appears twice in the file, at line 21 and at line 30 in `_MODEL_PREFIX_CONTEXT_WINDOWS`, so a naive string replace hits both. It was applied line-anchored and diff-verified.

---

## P-001 — fabricated status documentation committed to local main on 2026-09-15: SUPPORTED

**The allegation is established, not merely un-disproven.** Every element below was re-derived by the lead with read-only git and `gh` after the S6 reviewer surfaced it.

### The commit

`73bcda8500e4dda61966cbf5b22c36dea8ea6523`, authored by Kyle Nakamura, dated **2026-09-15 09:35:58 -0700**, subject `docs: Add Phase 1 implementation guide and plan conformance review`. It adds exactly two files, 478 insertions:

```
 PHASE_1_IMPLEMENTATION.md  | 313 +++++
 PLAN_CONFORMANCE_STATUS.md | 165 +++++
```

Its parent is `130620e`, main's real tip at the time. The commit object still exists in the object database, but `git branch --all --contains 73bcda8` and `git tag --contains 73bcda8` both return empty: it is unreachable from any branch or tag today. `git reflog show main` records the sequence — the commit, then a pull that re-applied it onto `301cc4d`, then `reset: moving to origin/main`, which discarded it. It never reached `origin/main`, and the fabricated content is absent from the tree at 301cc4d.

### Why the content is fabricated rather than merely stale

`PLAN_CONFORMANCE_STATUS.md` is dated "Review Completed: 2026-09-15" and describes issues #26 through #39 by subject. The subjects do not match the real issues, and not by numbering drift — by wholesale mismatch:

| issue | the document claims | the actual GitHub issue |
|---|---|---|
| #27 | "Train Punkt tokenizer with real abbreviations to fix segmentation divergence" | "Make the merge citation rule and the provenance validator agree" |
| #39 | "Add capacity metrics to audit output" | "Close the #11 and #12 scope gaps and adopt an issue-closing review gate" |

The real #26–#39 were filed on 2026-09-10, five days before this document claims to review them, and most were already closed.

The document further cites `tests/probes/capacity_validator.py` and `tests/probes/tokenizer_tester.py` as the source of "reproducible, non-mocked results", quoting figures such as "8 failure cases per 100 random hierarchies". **`tests/probes/` has never existed anywhere in this repository's history** — `git log --all -- 'tests/probes/*'` returns nothing. The claimed reproducibility is unverifiable by construction, because the artifacts it points at were never written.

### Scope, stated precisely

This is an isolated incident on the evidence available, not a pattern. No other status or progress document exists in tracked history. The `docs/plans/*` design and implementation documents that were spot-checked — the source-grounding-provenance pair — matched the code cleanly. The fabricated content never propagated beyond one local branch and was reset away.

### What this does and does not establish

It establishes that a document asserting completed review work, with fabricated issue subjects and citations to test artifacts that do not exist, was committed to local `main` on the alleged date. That is P-001's claim.

It does not establish intent, and this adjudication makes no finding about how the document came to be written. It also does not extend to `origin/main` or to the reviewed tree: a reader of the repository at 301cc4d would never encounter this content.

**Disposition.** Recorded as a confirmed process item. It is not a code defect and does not enter the F-NNN register or the GitHub issue filing batch. The two files remain preserved at `.review/evidence/PHASE_1_IMPLEMENTATION.md` and `.review/evidence/PLAN_CONFORMANCE_STATUS.md`, byte-identical to the commit's versions, so the evidence survives independently of the dangling commit, which garbage collection could eventually remove.

### Relevance to this review's own standards

Worth stating plainly, because the review should hold itself to what it measures: the failure mode P-001 documents — a confident status document citing evidence that does not exist — is the same failure mode this review's own strict verification standard exists to prevent. That standard is why roughly a third of sessions 1 to 4's "verified" traceability rows were downgraded to `inspection-only` in session 5, and why `verified` here requires a named pytest node id or a runnable repro rather than a reading of the code.

---

## P-002 — status unchanged, pending independent verification

The S6 reviewer's evidence on P-002 is under independent verification and is not adjudicated here. Summary of what is established so far: issue #39, which mandates a closing comment recording merge and review evidence, was itself closed with zero comments two seconds after PR #47 merged, and 14 of the 17 issues in the milestone carry no comments at all. The lead independently confirmed #39's zero-comment count and its `closedAt` of 2026-09-14T20:51:51Z. What remains genuinely unsettled is whether the underlying adversarial review happened and went unrecorded in the issue — several PR bodies carry real markers of it — or did not happen. Those are different claims and the evidence separates them.
