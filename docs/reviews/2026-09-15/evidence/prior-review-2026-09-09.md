# Prior review, 2026-09-09: outcome and already-refuted findings

Place this file at .review/evidence/prior-review-2026-09-09.md so a lead on any harness can dedupe against it. Source: the project memory written after that review.


On 2026-09-09 a 19-agent workflow (Sonnet 5 reviewers, Opus 5 adversarial verifiers) audited
the merged work for issues #2-#10 (repo tip 174f54b) against the master plan in issue #1.
33 gaps reported, 16 confirmed, 10 downgraded, 7 refuted. #8 (source grounding) was judged
incomplete; #2, #3, #7 complete; the rest complete with minor gaps.

Critical (all verified by reproduction, not by reading alone):
- C1: fixed 1,024-token merge grounding reserve vs full-capacity default segments makes every
  default hierarchical run abort before the first merge (`summarizer/hierarchy.py`,
  `summarizer/pipeline.py`, `summarizer/grounding.py`). The design doc specified a
  proportional quarter-share reserve that was never implemented.
- C2: merge prompt "keep every supporting reference" conflicts with validator `legal=grounded`
  in `summarizer/hierarchy.py`; provenance narrows silently (3 of 5, 2 of 37 segments seen).
- C3: `verify_and_repair` recursion passes the original `draft`, not `repaired`; audit lists
  discarded repairs as applied. Non-default path only (`max_repair_passes` defaults to 1).
Major: M1 segmentation's Punkt tokenizer has no abbreviation set (text.py's does); M2 merged
provenance ordered by grounding priority, not document order (`summarizer/merge.py`).

Scope corrections for the open tickets: #11 lists provider timeouts as new work but #3
shipped them; #12's checklist omits the CLI switchover to `run_pipeline` and the four CLI
flags (target length, verification, citations, audit) that only exist as library fields;
`summarizer.log` is tracked in git and dirtied by every run; a 237,785-token real document
produced one merge level at default budgets, so the "more than one merge level" DoD item is
not demonstrable from size alone.

Filed on 2026-09-10 as GitHub issues #26-#39, all on the "Generalized Recursive Summarizer
Rebuild" milestone: #26 C1 grounding reserve, #27 C2 citation rule vs validator, #28 m16
omissions in audit, #29 M2 provenance order, #30 default-config e2e tests (m17, m10, m11, m8),
#31 C3 repair recursion, #32 segmentation (M1, m1, m2), #33 JSON extractor and leaf level
(m3, m4), #34 window/counter resolution (m6, m15), #35 verification diagnostics and batching
(m12, m13), #36 quotation limit (m5), #37 docs/dead code/inert flags/summarizer.log (m7, m9,
m14), #38 CLI switchover to run_pipeline (blocked by #26, #27, #11), #39 #11/#12 scope gaps
and closing review gate. Each body carries file:line evidence at 174f54b, so the ephemeral
report is no longer needed. PR #25 (open, AI-authored, implements #11) was checked: it fixes
none of C1, C2, C3, M1, M2, leaves summarizer.log tracked, and adds retry jitter.

**Why:** future sessions on #11/#12 should not treat the merged hierarchy as working at
default settings, and should fix C1/C2 before any end-to-end evaluation.

**How to apply:** before #12's evaluation, land C1 and C2 fixes and a default-config
end-to-end test with a real tokenizer. When scoping #11/#12, apply the corrections above.
See [[mocked-suite-hides-contract-bugs]] and [[verify-merged-content-not-ci]].


## Companion lesson: green suite is weak evidence here


A fully green suite here is weak evidence. Repeatedly, an adversarial fresh-eyes
review has found real defects that the whole passing suite sailed past, each time
because every test at a boundary used a hand-built fake or a pinned toy budget:

- **Issue #4 (185 tests green):** `TiktokenCounter.for_model("gpt-4o-mini")` — the
  project's *default* model — crashed on construction, because deriving a value from
  `range(encoding.n_vocab)` hits reserved ids that raise `KeyError` (`o200k_base` has
  19). Every tokenizer test used a fake encoding with `n_vocab=1`. A second finding
  disproved the assumption that BPE non-monotonicity is confined to a
  `_max_token_bytes` window, which made a packing step able to emit over-budget
  segments.
- **Issue #5 (243 tests green):** quotations were matched against `SourceSegment.text`,
  which spans the whole *context* range, so a leaf could quote a neighbouring
  segment's core through the overlap window and claim it. Also: payload-controlled
  text reached error messages unbounded, a leaf with no provenance passed validation,
  a blank quote satisfied the verbatim check, and JSON extraction silently discarded
  everything after the first object in an array-wrapped response.

- **2026-09-09 plan-conformance review (510 tests green, issues #2-#10 merged):** the
  default hierarchical path cannot complete on a document large enough to need it.
  `DEFAULT_GROUNDING_POLICY` in `summarizer/hierarchy.py` reserves a flat 1,024 tokens
  while `run_pipeline` sizes segments at the full usable input capacity, so no segment
  core fits the grounding reserve and `summarizer/grounding.py` raises a bare
  `ValueError` before the first merge. Reproduced with a real tiktoken counter on a
  165k-token document; the break point is near 950-token segments. The only end-to-end
  hierarchical test pins `SegmentationConfig(max_tokens=35)`, a budget no real run
  produces. Same review: the merge prompt says keep every child reference while the
  validator accepts only grounded ids (an obedient response is rejected, a compliant one
  silently narrows provenance), and multi-pass repair recurses on the original draft
  while the audit reports the discarded repair as applied. See
  [[plan-conformance-review-2026-09-09]].

**Why:** the offline discipline that makes this suite fast and hermetic — socket
blocking plus `client_factory` fakes — also means nothing exercises a real tokenizer,
a real provider schema, or the interaction between a record's invariants and its own
docstring's claims. Bugs concentrate exactly where the fake diverges from reality.

**How to apply:** before declaring a backlog issue complete, run a background review
agent over `git diff main...HEAD` with instructions to (1) attack hand-rolled parsers
and boundary searches with fuzzing, (2) check every claim a docstring or README makes
against what the code does — several bugs here were the code contradicting its own
stated contract, (3) mutation-test the new tests and report survivors, (4) verify
library-behaviour claims against the installed package rather than assumptions, and (5)
run at least one end-to-end test with the default `PipelineConfig` and a real tokenizer;
a test that pins toy budgets proves nothing about a default run. Add at
least one test that touches the real dependency (guarded with `pytest.skip` when it
needs network), and prove each regression test fails against the pre-fix code by
reverting the fix in a throwaway copy. See [[uv-pytest-invocation]] for the correct
test command.
