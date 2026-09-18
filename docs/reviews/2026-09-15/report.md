# Full review at 301cc4d — COMPLETE

**Sessions:** 8 (completed). This review certifies the product at the reviewed commit.

**Reviewed commit:** `301cc4d56d6326b5b0449da059b3b35f484cc5ca`  
**Baseline:** `UV_OFFLINE=1 uv run --with-requirements requirements-dev.txt python -m pytest -q` → **766 passed**  
**Traceability matrix:** 201 acceptance criteria (203 rows, 2 dependency markers marked `not-a-criterion`)

## Findings summary

| verdict | count | findings |
|---|---|---|
| confirmed | 18 | F-005, F-007, F-008, F-009, F-011, F-012, F-013, F-019, F-020, F-021, F-023, F-027, F-028, F-029, F-030, F-031, F-033, F-034 |
| downgraded | 8 | F-010, F-014, F-022, F-025 (still minor); F-016, F-024, F-026, F-032 (closed at severity none) |
| refuted | 8 | F-001, F-002, F-004, F-006, F-015, F-017, F-018, F-035 |
| pending | 1 | F-003 |

**Confirmed by severity:** **critical 2** (F-012, F-013), **major 4** (F-009, F-011, F-019, F-029), **minor 12** (F-005, F-007, F-008, F-020, F-021, F-023, F-027, F-028, F-030, F-031, F-033, F-034).

## The four Tier A defects

### Critical
- **F-012 / F-013** (filed as **#62**). The default CLI path fails structured-output validation with every installed local model and produces no output. Bare defaults fail 3 of 3 on a real 34,389-token document with `qwen3.5:9b`; `gemma4` fails differently; the hierarchical path fails the same way. `--citations` reaches exactly one consumer at `finalization.py:376` and never gated the contract, so any wording scoping this to `--citations`/`--verify` understates it.

### Major
- **F-029** (filed as **#64**, bug/major). Two `AppConfig`s differing only in `ollama_host` produce byte-identical cache keys, so a result computed against server A is returned as a validated cache hit for a request naming server B. Reproduced at the CLI boundary: two ordinary `cli.main()` invocations, no `--resume`, distinct `--run-id`s, one shared `--cache-dir` — and `output_b.txt` contained server A's text while server B's provider recorded zero calls. Nothing downstream catches it. `reliability.py`'s invalidation codes have no host case, and `audit.py`'s allowlist drops `ollama_host` entirely.
- **F-019** (filed as **#63**, bug/major). `BoundaryKind.CODE_FENCE` was added to segmentation by #32 but never to `AuditSegment.boundary_kind`, still a closed `Literal` of the other six. A hierarchical `--audit` run over a document containing a fenced code block raises `ValidationError` after `write_editorial` has produced the final summary. Both the summary and the audit are lost.
- **F-009**, **F-011** (Tier B, major). Guards that work correctly with no test exercising them at the layer where they enforce core guarantees.

## Traceability

The matrix holds **201 actual acceptance criteria** (203 rows, 2 dependency markers). **7 violated rows**, split by kind:

| kind | rows | description |
|---|---|---|
| behavior/documentation violations | 2 | row 193 (#35 fenced-code-block crash + missing fixture), row 205 (#39 docs clause — evaluator `max_merge_children=2` not distinguished) |
| coverage-clause violations (behavior correct) | 4 | row 153 (#10 repair-pass exhaustion), row 174 (#26 grounding reserve regression test), row 180 (#27 regression test), row 193 (#36 schema caps test missing) |
| process violation | 1 | row 213 (#39 closing-comment requirement) — tracked as process items P-005/P-006, not a code defect |


## Issues filed (all on milestone *Full review 2026-09-15 (301cc4d)*)

| issue | finding(s) | severity | title |
|---|---|---|---|
| #62 | F-012, F-013 | critical | Default CLI path fails structured-output validation with local models |
| #63 | F-019 | major | Fenced code block crashes hierarchical `--audit` run |
| #64 | F-029 | major | Cache serves result across different Ollama hosts |
| #65 | F-027 | minor | Audit write not fsynced on `--audit` without `--cache-dir` |
| #66 | F-005, F-007, F-008, F-009, F-010, F-011 | minor | Six untested guards in hierarchy/grounding/leaf |
| #67 | F-022, F-023, F-025 | minor | `budget.py` coverage gaps (context-window table, exact boundary, fencing clamp) |
| #68 | F-030, F-031 | minor | Ingestion coverage gaps (leading blank lines, `rstrip` scope) |
| #69 | F-020, F-021 | minor | #32 abbreviation coverage ("e.g.") and missing fenced-code fixture |
| #70 | F-028 | minor | Repair-pass exhaustion tested only trivially |
| #71 | F-033, F-034 | minor | Documentation distinction missing; help-text test lacks negative assertions |
| #72 | P-001, P-005, P-006 | process | Fabricated status docs; review gate recording unused (13/17 zero comments) |

## Process items

- **P-001 — SUPPORTED.** Fabricated status documentation committed to local `main` on 2026-09-15 (commit `73bcda8`). Content mismatches reality; never reached `origin`.
- **P-004 — DOWNGRADED.** Stale `.pyc` can mask a mutation; mechanism confirmed but not reproducible under actual editing style. Five previously recorded survivors re-run with precautions all still survive.
- **P-005 / P-006** — The review gate's recording mechanism went unused: issue #39 (which mandates a closing comment) closed with zero comments 2s after PR #47 merged; 13 of 17 milestone issues have zero comments. These do **not** establish that the underlying adversarial review never happened — several PR bodies carry real markers of it.

## Evidence gaps (unchanged from session 7)

- A successful default real-long-document output remains unverified (F-012/F-013).
- A successful live multi-level hierarchy remains unverified.
- Injection resistance remains unverified (direct/hierarchical probes failed closed before output).
- No hosted-provider run attempted (review makes no OpenAI calls by design).
- `summarizer/text.py` and `legacy_workflow.py` are dead code (no entry point routes to them).

## Open items (for future work)

1. Verify three remaining candidates: C-S3M-003, C-S3M-004, C-S2H-001 (all confirmed genuine survivors; only reachability/severity left).
2. S4 mutation angle on `verification.py` (2003 lines) — never run, highest predicted defect density.
3. S4 live angle.
4. S1/S2 traceability largely complete; remaining pending rows are cross-cutting (#1, #12) and S3 (#2, #4, #5) which were not in scope for this review.

## Artifacts

- `traceability.md` — full 201-criterion matrix with status and evidence references.
- `reproduction_scripts/` — 42 runnable scripts (`.review/repro/*.py|.sh` + `.review/test_exhaustion/*.py`) reproducing every finding and key verification.

## Review protocol adherence

- All mutation runs on disposable `/tmp` copies; primary checkout never modified.
- Every `uv` invocation prefixed `UV_OFFLINE=1`.
- `__pycache__` cleared and `PYTHONDONTWRITEBYTECODE=1` for every mutation pass (P-004 precaution).
- Independent verifiers opened every cited file and line, confirmed every pytest node id (adversarial drafting pass found 79 citation problems and corrected them).
- Criterion-reading rule settled: traceability row scores criterion satisfaction; finding's `criteria_violated` records present-tense code defects only. The two fields may differ.

---

*Review closed at `301cc4d`. The closing PR adds `docs/reviews/2026-09-15/` with this report, the traceability matrix, and the reproduction scripts.*