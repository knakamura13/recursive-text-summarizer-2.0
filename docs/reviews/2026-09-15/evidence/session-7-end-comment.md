## Session 7

Resumed at unchanged commit `301cc4d`. No tracked file was modified; both the primary checkout and the review worktree stayed clean throughout.

Four subsystems advanced, two hypotheses opened and closed, one new major defect found, and two of the review's own published numbers corrected.

### A new major defect: the cache serves results across endpoints

**F-029.** Two `AppConfig`s differing only in `ollama_host` produce byte-identical `CacheDescriptor` keys, so a result computed against server A is returned as a validated cache hit for a request naming server B. `AppConfig.ollama_host` (`config.py:48-54`) is never threaded into `CacheCoordinator` (`pipeline.py:254-281`) or `CacheDescriptor` (`cache.py:225-269`).

Reproduced at the CLI boundary rather than the descriptor layer: two ordinary `cli.main()` invocations, **no `--resume`**, distinct `--run-id`s, one shared `--cache-dir`. `output_b.txt` contained server A's text, and server B's provider recorded zero calls. Default `run_mode="new"` sets `allow_unreferenced_cache=True` (`pipeline.py:278`), so a brand-new run adopts a pre-existing cache object by content-addressed key alone.

Nothing downstream catches it. `reliability.py:11-18` has no host case among its five invalidation codes, and `audit.py:931-958`'s allowlist drops `ollama_host` entirely, so the published audit for the affected run contains no host-identifying text at all. A stale hit that announced itself would be a lesser finding; this one leaves no trace.

The exclusion is deliberate and documented (`docs/plans/2026-09-06-reliability-cache-resume-design.md:51`), but the error sits a level above mistaking a host for a credential: one exclusion list is applied to three surfaces with different requirements. Withholding the host from a disclosure-facing audit is defensible. Withholding it from a cache key that never leaves the machine and exists only for correctness is not, and it violates criterion #11's requirement that keys derive from "every behavior-relevant configuration value, excluding credentials." A bare host is not a credential.

### S3's mutation angle produced no major finding

Both of its strongest candidates fell, for the same reason: the mutation genuinely survived, but the survival was of a guard whose triggering precondition no real caller can construct.

- **F-025** (the `max(fencing, 0)` clamp, session 6's top-ranked candidate) settled exhaustively rather than by argument. The fencing computation has exactly two free variables in the whole codebase, giving 14 possible values; all 14 were measured against the real `TiktokenCounter` through its production constructors and came back strictly positive. The cited BPE non-monotonicity is real but needs ~3000 characters of one repeated character, while the fence adds 64.
- **F-022** (the `gpt-4` context-window constant) downgraded after the verifier ran the direction the candidate never tried — mutating to `10_000_000`, the *unsafe* direction, which also passed all 766 tests. Severity fell because a too-large window produces a raised `ProviderRequestError`, not a silent truncation.

**Mutation survival did not predict severity in either case.**

### The pattern worth more than any single finding

Four independent instances of **correct code whose correctness no test would defend**, found by four different routes in four subsystems: F-021 (no fixture contains a code fence, which is why F-019 survived 766 green tests), F-028 (repair-pass exhaustion tested only in its trivial form), F-030 (every fixture has at most one leading blank line), and C-S2H-001 (the coverage union never exercised on input where it does anything). In each, the code is right, the suite is green, and a regression would ship silently. The technique that finds them — revert the fix, rerun the suite — is cheap.

### Two hypotheses closed with no finding

**H4** (unmerged local branches): `fix/critical-blockers-26-27-28` *is* main. `issue-36-quotation-limits` is a superseded duplicate, proved by content diff — main is a strict superset. `issue-35-verification-diagnostics-batching` contains no unmerged work and **merging it would regress main**: its quotation commit predates three fixes now shipped, and its own batching code has a bug main avoids (`evidence={item[0].claim_id: ...}` collapses to one key because a batch's items share a claim).

**H9** (`issue-7-hierarchical-merging`, five alleged defects): claims 1–4 refuted for a mundane reason — `da77a67` and `3359e74` are the same patch committed twice, and the latter is already an ancestor of `301cc4d`. Claim 5 partly holds: reverting each of main's four fixes and rerunning the suite shows three are caught and one, the coverage union, is not.

### Other results

- **S4**: 23 of 25 rows verified against named node ids. The self-certification guard holds — `verification.py:1854-1862` genuinely re-enters `verify_draft_once`. **No second instance of the F-019 closed-`Literal` bug class exists**; eight candidate fields checked against their producers. Rows 190/193 verified by extracting `174f54b` and running the regression tests against the old code, where they fail.
- **S5**: 8 of 9 rows verified, the strongest subsystem result so far.
- **F-017's open question answered**: the absent leaf-level retry is a documented design choice (`leaf.py:407-424`), not an oversight. No criterion requires it.
- **`text.py` and `legacy_workflow.py` are dead code**, re-derived independently rather than inherited: no `pyproject.toml` or `setup.py` exists anywhere, so no declared entry point can route into them. Bears on #38's "removed or explicitly retained as a documented compatibility mode."

### Verdict tally at `301cc4d`

34 findings: **18 confirmed** (2 critical, 4 major, 12 minor), 8 downgraded, 7 refuted, 1 pending.

| finding | severity | substance |
|---|---|---|
| F-012 / F-013 | critical | Default path fails structured-output validation with every installed local model; no output produced (**#62**) |
| F-029 | major | Cache serves a result computed against a different Ollama host as a validated hit, undetectably |
| F-019 | major | Fenced code block crashes a hierarchical `--audit` run and discards the completed summary (**#63**) |
| F-009, F-011 | major | Guards that work correctly with no test exercising them |
| 12 findings | minor | Coverage gaps, defensive guards, documentation and test-design gaps |

### Traceability

216 rows but **214 actual acceptance criteria**. Rows 104 and 154 were mis-transcribed dependency markers — issue #4's `## Dependency — #3` and issue #10's `## Blocked by — #9` — now marked `not-a-criterion` in place so earlier line citations stay stable. **The "216 criteria" figure published in earlier comments is wrong by two.**

5 violated rows, which must be read split by kind rather than as five defects: **2 behavior/documentation violations** (row 195, row 215), **2 coverage-clause violations** where behavior is correct (row 194, row 153), and **1 process violation** (row 222, not a code defect).

A criterion-reading inconsistency open since session 6 is now settled by explicit rule: a traceability row scores whether the *criterion* is satisfied, so a coverage-type criterion naming a required test is `violated` when no such test exists; a finding's `criteria_violated` records present-tense *code* defects only, so a working guard yields `[]`. The two fields answer different questions and may differ. An independent verifier argued the opposite resolution and that dissent is recorded rather than discarded.

### Process items

- **P-001 — SUPPORTED.** Fabricated status documentation was committed to local `main` on 2026-09-15. Commit `73bcda8`, 09:35:58 -0700, added `PHASE_1_IMPLEMENTATION.md` and `PLAN_CONFORMANCE_STATUS.md`. The content is fabricated, not stale: it describes issues #26–#39 with subjects that wholesale mismatch reality (it calls #27 "Train Punkt tokenizer with real abbreviations"; the real #27 is "Make the merge citation rule and the provenance validator agree"), and cites `tests/probes/capacity_validator.py` as the source of "reproducible, non-mocked results" when **`tests/probes/` has never existed in this repository's history**. Unreachable from any branch or tag, never reached `origin`, absent from the reviewed tree. Every element re-derived by the lead. Isolated incident; no finding as to intent; not a code defect.
- **P-004 — DOWNGRADED.** Stale `.pyc` can mask a mutation. The mechanism was confirmed by deliberately padding a file to its original byte count and forcing its original mtime, but could not be reproduced under this review's actual editing style. Decisively, five previously recorded survivors re-run with the precaution in force **all still survive** — no phantoms, earlier counts stand.
- **P-002 → P-005 and P-006.** The gate's *recording* mechanism went unused: #39, which mandates a closing comment recording merge and review evidence, was closed with zero comments two seconds after PR #47 merged, and 6 of the 7 post-gate issues have none. **These do not establish that the underlying review never happened** — several PR bodies carry real markers of it, and GitHub artifacts cannot distinguish "no review" from "unrecorded review."

### Two corrections to this review's own published numbers

Both reached a published artifact before being checked, and both were caught by a later pass rather than at the point of writing:

1. "216 criteria" is actually 214.
2. The P-002 figures published earlier in this session said 14 of 17 zero-comment and 5 of 6 post-gate. Re-derived independently twice, they are **13 of 17 and 6 of 7** — both off by one from omitting issue #33, closed six minutes after the gate with zero comments. The corrected numbers are marginally worse than published.

### Where the next session resumes

1. Verify the three remaining candidates: C-S3M-003, C-S3M-004, C-S2H-001. All three are confirmed genuine survivors; only reachability and severity remain.
2. **S4's mutation angle on `verification.py`** — never run, and the last missing angle for the subsystem with the highest predicted defect density.
3. S4's live angle.
4. Fold S4/S5/S6 row statuses into the matrix as a dedicated sync pass; both earlier rushed syncs went wrong.
5. S1 and S2 remain reopened.
6. File the held Tier B batch plus F-029, which warrants its own issue at major. **Issue filing remains approval-gated and this batch is not yet approved.**
7. The closing PR adding `docs/reviews/2026-09-15/`.

Standing instruction from P-004: every future mutation run clears `__pycache__` and sets `PYTHONDONTWRITEBYTECODE=1`.
