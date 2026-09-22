# H4 — unmerged branches: superseded duplicates, or lost fixes?

Investigated in session 7 by a dedicated agent restricted to read-only git plumbing (no checkout; branch content reached via `git show <ref>:<path>`). Baseline re-confirmed on main @301cc4d: 766 passed.

**H4 is CLOSED. No finding id proposed.** Neither unmerged local branch contains work that main lacks.

## `fix/critical-blockers-26-27-28`

Points at `301cc4d` itself. It is main. Nothing to investigate.

## Q1 — Branch B, `knakamura/issue-36-quotation-limits` @152c025: superseded duplicate

Both branch and main agree on `LEAF_SCHEMA_VERSION = "leaf/2"` and `MERGE_PROMPT_VERSION = "merge-prompt/4"`. The only difference is style: main expresses the 500-character and 5-quotation caps as shared named constants (`MAX_QUOTE_CHARS`, `MAX_QUOTATIONS_PER_NODE`, introduced by the already-merged `8353189`) referenced from `leaf.py`, `merge.py` and `summaries.py`, while Branch B hardcodes the same two numbers as literals. Both prompts convey the same cap.

Main is a strict superset: it additionally carries `test_oversized_quote_is_rejected_wherever_it_appears`, covering nested `ContentUnit`/`GroundedAnnotation` shapes, which Branch B's suite lacks entirely. Nothing in Branch B is absent from main.

This was checked by content, not by commit message, because the question H4 asks cannot be answered from subject lines.

## Q2 — Branch A, `knakamura/issue-35-verification-diagnostics-batching` @5ff4fb6: contains no unmerged work

Three commits, all superseded, and merging any of them would regress main.

**`8b19190` (quotation limits) is stale, not novel.** Branch A's `leaf.py` and `summaries.py` predate three independent fixes now on main: a `_core_bounds()` arithmetic bug (main computes `core_end - context_start`; the branch computes `core_end - context_end`), a missing `LeafSummaryError` check for `node.level != 0`, and a missing JSON-validity guard in `_top_level_objects()`. Merging this commit would reintroduce all three.

**`8688133` and `5ff4fb6` (#35 diagnostics and batching) were delivered to main by other routes.** Both S4 acceptance criteria at traceability rows 202-203 are already satisfied on main, by commits that are not Branch A's:

- `74546f43` added `AuditGroundingSelection` / `AuditNodeV4` / `AuditArtifactV4` (audit/4), giving per-merge structured visibility into budget-bounded evidence omissions. Absent from Branch A entirely.
- `dd9f08e5` folded `diagnostic_codes` (`"conflicting_evidence"`, `"retrieval_bounded"`) into `AuditVerification.warning_codes` and rewrote the per-claim escalation loop to batch a claim's omitted segments into token-bounded batches with one provider call per batch. It reached main through the **sibling** branch `origin/knakamura/issue-35-verification-diagnostics`, not Branch A. `tests/test_verification_repair.py::test_verify_once_escalates_raw_contradictions_through_omitted_evidence` exercises the acceptance criterion's exact scenario — three omitted segments (S000002-4) in a single escalation request — and passes today.

**Branch A's own batching code carries a bug main's route avoids.** Because a batch's items all share one claim, Branch A's `evidence={item[0].claim_id: item[1] for item in batch}` collapses to a single surviving key, silently dropping all but the last segment's evidence from the request. Main's merged version combines a batch's passages into one bundle before building the request.

**Carried to S4, not treated as an H4 finding:** `AuditGroundingSelection.omission_reason` is `Literal["budget"]` with no `"conflict"` variant, so bounded retrieval gets structured audit detail while conflicting evidence gets only a warning-code string. That may be why row 202 is still `pending`. Branch A does not touch grounding, so merging it would not close that gap either. Note the shape: this is a second closed-`Literal` observation in `audit.py`, the same class as F-019.

## Q3 — the six committed scratch scripts

All six are dated 2026-09-14 18:44:47 and are source-rewriting codegen, not product code:

- `update_audit.py` — regex codegen adding a `diagnostic_codes` field to `AuditVerification` and threading it through `_audit_verification`'s return paths.
- `update_verification.py` — anchor-based line-splice rewrite of `verification.py`'s escalation loop (the buggy batching version above). One hardcoded replacement line contains a real newline where a literal backslash-n was needed to emit `"\n".join(...)`; that single mistake is what broke `verification.py`'s syntax.
- `update_verification_batching.py` — an earlier incomplete draft that trails off into self-correcting inline comments and has no file-write call at all. A no-op as committed.
- `fix_syntax.py` — aborted repair ending in a bare `pass`, never writes the file back. No-op.
- `fix_verification_syntax.py` — working regex repair, writes the file back.
- `fix_verification_syntax_v2.py` — line-based variant of the same repair, writes the file back.

**All six are absent from main at 301cc4d**, confirmed individually via `git show main:<path>`.

**Relation to P-001 and P-002, stated carefully.** Both remain allegations and neither is adjudicated here. Two objective facts are recorded for whoever does adjudicate them. First, these scripts are dated one day before P-001's alleged 2026-09-15 date, and they are source-rewriting scripts rather than status documentation, so they neither confirm nor refute P-001. Second, their absence from main is evidence that some gate kept them out on this occasion; that is evidence against P-002 on this instance only, and not a general finding about whether the #39 review gate was applied.

## Q4 — origin branches

Merged into main, not examined further: `issue-30-default-coverage` (78f300e), `issue-31-repair-lineage` (e5e444f), `issue-35-verification-diagnostics` (dd9f08e, the route that actually delivered Q2's fix), `issue-36-completion-ef992e` (fba8efa, Q1's superseding route), `issue-31-multi-pass-repair-continuation` (abf976d).

Unmerged:

- `issue-35-verification-diagnostics-batching` (8688133), 36 behind / 2 ahead. Byte-identical to Branch A minus its final commit. Fully covered above.
- **`issue-7-hierarchical-merging` (da77a67), 88 behind / 1 ahead.** Outside H4's assigned #9/#10/#31/#35 scope, so its code was NOT read. Its commit message — `fix(hierarchy): refuse a merge that cannot hold a pair` — describes a `merge_fanout` floor letting oversized requests through silently, a hardcoded rather than measured delimiter cost, a mis-reported pass-through node level, coverage computed by concatenation rather than union, and "test holes" that had kept the suite green. **This is commit-message text only. Nothing is verified.** Flagged as its own hypothesis rather than folded into H4, because H4's question is about the two local branches and this is a different question with a different owner (#7 is S2, which is reopened).

## Verdicts

| question | verdict |
|---|---|
| Branch B superseded? | superseded duplicate; main is a strict superset |
| Branch A holds unmerged work? | contains no unmerged work; merging would regress main |
| Six scratch scripts | real as described, absent from main, not adjudicated against P-001/P-002 |
| Origin branches | five merged, one covered by Q2, one out of scope and newly flagged |

Confidence: high on Q1 and Q3 (direct content diffs, full script reads). High on Q2's central claim (two confirmed-merged commits, a passing test matching the criterion's exact scenario, and a concrete bug in Branch A's alternative). The `issue-7-hierarchical-merging` thread is explicitly low-confidence and separate.

## Process note

The agent's tree diff passed on the substance, with one deviation it disclosed: it created its empty writable directory under the review **worktree's** `.review/wip/` rather than the primary checkout's, because the environment names the worktree as the working directory while `.review/` lives in the primary checkout. Harmless, and removed by the lead. Worth noting for future dispatches: briefs must give the primary checkout's absolute path for the writable directory, not a repo-relative one.
