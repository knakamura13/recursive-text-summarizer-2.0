# H9 — the unmerged `issue-7-hierarchical-merging` branch: five alleged defects

Opened in session 7 out of H4's Q4. The lead did not read the branch's code; the hypothesis was raised from commit-message prose alone and handed to a dedicated investigator with the instruction that the branch is nearly irrelevant and the question is whether each defect exists in **main at 301cc4d**.

## Headline: the branch's fix is already on main, verbatim

`da77a67` and `3359e74` — same subject, same author, same AuthorDate, committed two minutes apart — produce a **byte-identical patch** to `summarizer/hierarchy.py`, `summarizer/merge.py` and `tests/test_hierarchy.py`. `3359e74` is an ancestor of `301cc4d`.

This is not "fixed by another route." It is the same patch committed twice under two hashes, one of which was merged and one of which was not. Independently re-confirmed by the lead: `git merge-base --is-ancestor 3359e74 301cc4d` succeeds, and diffing the two commits' patches (headers stripped) produces no output.

So claims 1 through 4 are refuted for the simplest possible reason.

## Claim-by-claim

| # | claim | present in main at 301cc4d? | evidence |
|---|---|---|---|
| 1 | `merge_fanout` floors at 2, letting an oversized request through silently | **absent** | `hierarchy.py:188-194@301cc4d` raises `BudgetError` when `measured < 2` instead of `max(measured, 2)` |
| 2 | Hardcoded rather than measured per-child delimiter cost; merge stage trusts a leaf-sized capacity | **absent** | `merge.py:172-179@301cc4d` `child_fence_tokens()` calls `counter.count(...)` on the real fence text; no `_CHILD_FENCE_TOKENS` constant exists. `hierarchy.py:324-337@301cc4d` subtracts `measure_merge_overhead()` per level |
| 3 | Pass-through node reports its child's stale level | **absent** | `hierarchy.py:370@301cc4d` `summary=only.summary.model_copy(update={"level": level})` |
| 4 | Coverage computed by concatenation rather than union | **absent from the code** | `hierarchy.py:502-512@301cc4d` `covered = tuple(dict.fromkeys(...))`, a document-order union |
| 5 | Test holes kept the suite green | **partially present, narrowed to one instance** | see below |

## Claim 5, which is the only real result

The investigator did not take "the code is fixed" as the end of the question. It built a mutation harness that reverts each of main's four fixes in an isolated `git archive` export and reruns the real suite. That directly answers the generalizing question: if this defect regressed tomorrow, would anything catch it?

```
1_fanout_floor              -> CAUGHT (2 failed, 65 passed)
2a_hardcoded_fence          -> CAUGHT (2 failed, 65 passed)
2b_trust_leaf_capacity      -> CAUGHT (1 failed, 66 passed)
3_stale_passthrough_level   -> CAUGHT (1 failed, 66 passed)
4_coverage_concatenation    -> NOT CAUGHT (766 passed, full suite)
```

Three of four are actively guarded. One is not. Reproduced twice in independent scratch copies.

The branch's own broader claim — that dropping content units, contradictions, qualifications, quotations, entities or level from a merge payload left the suite green — is closed in main: `tests/test_hierarchy.py::test_child_payloads_carry_the_content_the_prompt_rules_operate_on` and `::test_child_payloads_are_compact_and_key_ordered` assert against `serialize_child(...)`'s output rather than against prompt substrings.

## Candidate

### C-S2H-001 — the coverage union is never exercised on an input where it does anything

Removing the `dict.fromkeys(...)` wrapper at `hierarchy.py:506-512@301cc4d` and reverting to plain concatenation leaves all 766 tests green.

The mechanism is precise: every fixture reaching `_prepare_merge`, and the sole real caller, construct `covered_segments` from disjoint per-leaf singleton segment IDs. On every input the suite or the pipeline ever produces, the union and the concatenation are mathematically equal, so the dedup never does work and nothing can observe its absence.

**Reachability is the decisive weakness, and the investigator said so plainly.** The only real caller is `pipeline.py:361-370@301cc4d`, which builds `covered=[(segment.segment_id,) for segment in segments]` — always disjoint singletons. Segmentation assigns each segment ID to exactly one leaf, so no two siblings at any level can share one. The dedup exists as a documented defensive property of `build_hierarchy`'s own contract; its comment anticipates that "a caller supplying overlapping coverage would otherwise store duplicates for issue #8 to narrow." `audit.py:510`'s `AuditNode` validation would not catch a duplicate either — it checks that `covered_segments` resolve to known IDs by set membership, not that they are unique.

Proposed tier by its author: weaker than minor, on unreachability. **Normalization note: the author proposed "Tier C", which is not in this review's vocabulary (A and B only). Recorded for the verifier to place as B-minor or `none`.**

It would warrant re-rating only if a future change — specifically the issue #8 coverage-narrowing work the comment forecasts — lets two siblings' `covered_segments` overlap without restoring an equivalent dedup. At that point a duplicated identifier could double-count in any coverage or grounding accounting built on `covered_segments` length.

Falsified by: finding a second real caller of `build_hierarchy` supplying overlapping `covered` (only one caller exists), or finding a skipped or environment-gated test that would have caught it (checked with `-rs`; a clean rerun showed 766 passed, 0 skipped).

Repro: `.review/repro/C-S2H-001-coverage-union-hole.py` runs the real `build_hierarchy` / `_prepare_merge` path with deliberately overlapping `covered` input and observes the dedup working. Harness: `.review/repro/C-S2H-001-mutation-harness.py`.

## What this means for the review

H9 closes with one weak candidate and four refutations. The refutations are worth as much as the candidate: five alleged defects, sourced from a commit message, all four code-level ones absent from shipped code, and the reason is a duplicate commit rather than anything subtle.

**The method is the transferable part.** "Revert main's fix and see whether the suite notices" answers a question that reading the code cannot: not "is this correct today" but "is this correctness pinned." It found that four fixes ship with three tests guarding them and one guarding nothing. That is the same shape as F-021 and as the ingestion pass's leading-blank-line gap, and it is now the third independent instance in this review of *correct code whose correctness no test would defend*. That pattern, rather than any single candidate, is the finding worth carrying to the report.

**Process note.** The investigator's writes to the primary checkout were blocked by `worktree-guard` and it wrote into the worktree instead, disclosing the deviation explicitly rather than complying silently with an instruction it could not follow. Both scripts were copied to `.review/repro/` by the lead. This is now the fourth agent this session to hit the same brief defect.
