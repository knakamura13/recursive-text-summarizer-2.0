# S3 Mutation-Testing Review — ingestion, segmentation, text, budget arithmetic

Reviewed commit: `301cc4d56d6326b5b0449da059b3b35f484cc5ca` (checked out on `main`, clean tree, verified again at the end of this run).

This document produces **candidates only**. No finding IDs, no verdicts, no adjudicated severities are assigned here — every tier below is *proposed*, for a separate verifier to confirm or refute.

## 1. Method and scope

Disposable copy: `/tmp/s3-mutation-review/repo` (a full `cp -R` of the repo at the reviewed commit). Pristine copies of each touched module were kept at `/tmp/s3-mutation-review/*.orig` and used to restore the file after every mutation before the next one, so no mutation ever compounded onto a prior one. The disposable copy and its `.orig` snapshots were deleted before finishing this review; nothing under `/tmp` remains.

Baseline command (as given):
```
uv run --with-requirements requirements-dev.txt python -m pytest -q <selection>
```
Confirmed baseline: `tests/` at this commit is 766 passed. Two mutations below were checked against the full 766-test selection (noted in the table) specifically because the change was structural enough that a narrower selection felt like it could hide a break in an unrelated consumer (`audit.py`, `hierarchy.py`, etc.); every other row uses a targeted file-level selection to keep iteration fast, per the instruction not to re-run the whole suite repeatedly.

Modules actually reached, in priority order:
1. `summarizer/budget.py` — thorough. Every named area (reserve subtraction, capacity fit/no-fit, direct-vs-hierarchical `_decide`, `resolve_context_window`) was mutated at least once.
2. `summarizer/segmentation.py` — thorough on `_pack_units` and `detect_structural_blocks`; light on `_sentence_units`/`_hard_split` (one mutation each, see table).
3. `summarizer/tokenization.py` — light. One mutation on `TiktokenCounter.fitting_prefix`'s fast-path guard.
4. `summarizer/ingestion.py` and `summarizer/text.py` — inspected but not mutated beyond a reachability check; see §4.

## 2. Mutation table

| # | Module:line@301cc4d | Mutation | Selection run | Result |
|---|---|---|---|---|
| 1 | budget.py:220 | `usable_input_capacity`: `if capacity <= 0` → `if capacity < 0` | test_budget.py, test_strategy_selection.py | KILLED (`test_capacity_of_exactly_zero_is_refused`) |
| 2 | budget.py:279 | `select_strategy`: `fits = document_tokens <= capacity` → `< capacity` | test_strategy_selection.py | KILLED (`test_selects_direct_exactly_at_capacity`) |
| 3 | budget.py:283 | `_decide`'s `capped` predicate: `document_tokens > config.max_direct_tokens` → `>=` | test_strategy_selection.py, test_budget.py, test_direct.py, test_pipeline.py, test_audit_reliability.py, test_publication.py, test_verification_audit.py, test_audit.py, test_finalization.py, test_pipeline_reliability.py, test_cli.py, test_strategy_config.py (186 tests) | **SURVIVED** |
| 4 | budget.py:200 | `safety_margin`: `max(config.safety_margin_tokens, proportional)` → `min(...)` | test_budget.py | KILLED (`test_safety_margin_takes_the_larger_term` + 2 others) |
| 5 | budget.py:217-219 | `usable_input_capacity`: `window.tokens - overhead.total - config.max_output_tokens - margin` → `... + config.max_output_tokens - margin` (sign flip on one term) | test_budget.py, test_strategy_selection.py | KILLED (5 failures) |
| 6 | budget.py:295 | `select_strategy`: `if config.strategy == "direct" and not fits:` → `and fits:` | test_strategy_selection.py, test_direct.py | KILLED (`test_explicit_direct_over_capacity_fails_with_its_arithmetic`) |
| 7 | budget.py:20-24 | `_MODEL_CONTEXT_WINDOWS["gpt-4"]`: `8_192` → `1` (exact-match table entry) | full suite (766 tests) | **SURVIVED** |
| 8 | budget.py:32 | `_MODEL_PREFIX_CONTEXT_WINDOWS["gpt-4o"]`: `128_000` → `1` | test_context_windows.py, test_budget.py, test_strategy_selection.py, test_direct.py, test_pipeline.py, test_cli.py (85 tests) | **SURVIVED** (weaker: relative-equality test still holds since both sides read the same mutated constant) |
| 9 | budget.py:97 | `resolve_context_window`: `longest = max(matches, key=len)` → `min(matches, key=len)` | test_context_windows.py | KILLED (`test_prefers_the_longest_matching_prefix` + 1 other) |
| 10 | budget.py:82 | `resolve_context_window`: `if explicit <= 0:` → `if explicit < 0:` | test_context_windows.py | KILLED (fails via `ContextWindow.__post_init__`'s own guard instead, different message, but still raises — `test_rejects_a_non_positive_explicit_window` fails on message-match) |
| 11 | budget.py:85 | `resolve_context_window`: `if provider.strip().lower() == "openai":` → `if provider.strip() == "openai":` | test_context_windows.py, test_budget.py, test_strategy_selection.py, test_direct.py, test_pipeline.py, test_cli.py, test_audit_reliability.py, test_publication.py, test_verification_audit.py, test_audit.py, test_finalization.py, test_pipeline_reliability.py, test_strategy_config.py (198 tests) | **SURVIVED** |
| 12 | budget.py:139 | `measure_overhead`: `fencing=max(fencing, 0)` → `fencing=fencing` (drop the clamp) | test_budget.py, test_strategy_selection.py, test_direct.py | **SURVIVED** |
| 13 | budget.py:351-359 | `_decide`: swapped the `if assumed:` / `if capped:` block order | test_strategy_selection.py | **SURVIVED** (reason-text/priority ordering only; strategy outcome unaffected in every existing test because no test makes both `assumed` and `capped` true at once) |
| 14 | segmentation.py:616-620 | `_pack_units`: deleted the post-snap recount fallback (`while covered_index > next_index and ... > max_tokens: covered_index -= 1`) | test_segmentation.py | KILLED (`test_packing_recounts_the_boundary_it_snaps_to`) |
| 15 | segmentation.py:594-597 | `_pack_units`: initial "does one more unit fit" check `<= max_tokens` → `< max_tokens` | test_segmentation.py | KILLED (3 failures) |
| 16 | segmentation.py:585-589 | `_pack_units`: run-boundary predicate `is not BoundaryKind.HEADING` → `is BoundaryKind.HEADING` | test_segmentation.py | KILLED (6 failures) |
| 17 | segmentation.py:607-610 | `_pack_units`: `units[covered_index + 1].end <= fitting_end` → `< fitting_end` | test_segmentation.py | KILLED (`test_structural_packing_does_not_recount_growing_document_prefixes`) |
| 18 | segmentation.py:352-353 | `detect_structural_blocks`: deleted `index = _consume_blank_lines(lines, index)` after a closing code fence (trailing blank lines no longer folded into the `CODE_FENCE` block) | test_structure_detection.py, test_segmentation.py, test_segmentation_models.py, then full suite (766 tests) | **SURVIVED** (all 766) |
| 19 | segmentation.py:517-524 | `_hard_split`: whitespace-preference trigger `if next_end < end:` → `if next_end <= end:` | test_segmentation.py, then full suite (766 tests) | **SURVIVED** (all 766) |
| 20 | tokenization.py:119-120 | `TiktokenCounter.fitting_prefix`'s fast path: `if self.count(text[start:end]) <= max_tokens: return end` → `<` | test_tokenization.py, test_segmentation.py | **SURVIVED**, but low-value: see Candidates §, C-S3M-009 |

## 3. Candidates

Each candidate is a falsifiable claim about a gap in test coverage, backed by the surviving mutation named. Tier and confidence are **proposed**, not adjudicated.

---

**C-S3M-001** — No test pins whether the `--max-direct-tokens` cap is inclusive or exclusive of the boundary value itself.

- Claim: a document whose token count exactly equals `config.max_direct_tokens` is routed to `direct` today (`>` in the `capped` predicate), and no test in the 186-test selection distinguishes that from routing it to `hierarchical`.
- Citation: `summarizer/budget.py:283` (`document_tokens > config.max_direct_tokens`) @301cc4d.
- Surviving mutation: row #3 (`>` → `>=`).
- Reachability: real. `max_direct_tokens` is a first-class CLI flag (`summarizer/cli.py:131` `--max-direct-tokens`) threaded straight into `StrategyConfig` and then `select_strategy` (`summarizer/cli.py:167`, `summarizer/pipeline.py:174`); any operator setting this flag can hit the exact-equality boundary with an ordinary document.
- Proposed tier: **minor**. Both branches (`direct` and `hierarchical`) still produce a grounded summary through a validated path; this only affects which pipeline stage handles a document at the cap, not whether the output is trustworthy — an internal policy-boundary question, not a leak.
- Confidence: high that the survival is real (reproduced cleanly against 186 tests); medium-low that it matters, precisely because both outcomes are safe.

---

**C-S3M-002** — `measure_overhead`'s `max(fencing, 0)` clamp is unpinned, and the counter it clamps for is documented as capable of producing exactly the scenario the clamp guards against.

- Claim: no test constructs a counter/template combination where wrapping the probe text in real request fencing yields *fewer* tokens than the bare probe text, so nothing currently exercises the `max(fencing, 0)` clamp doing real work; removing it did not fail any of 36 tests across `test_budget.py` + `test_strategy_selection.py` + `test_direct.py`.
- Citation: `summarizer/budget.py:146` (`fencing=max(fencing, 0)`) @301cc4d.
- Surviving mutation: row #12.
- Why this is more than a hypothetical: `summarizer/tokenization.py:99-100` declares `TiktokenCounter.monotonic -> False`, and the module's own comment (`tokenization.py:53-59`) documents that with the real `cl100k_base` encoding, "`'a' * 3000` at a 12-token budget fits through 96 characters even though 93 characters already exceeds it" — i.e. wrapping text in more characters can *reduce* the token count under real BPE merging. `TiktokenCounter` is exactly the counter used in production for the OpenAI provider (`resolve_token_counter`, `tokenization.py:222`). A negative `fencing` term, if the clamp were absent, lowers `overhead.total`, which raises `usable_input_capacity` (`budget.py:217-219`), which can make an over-budget document read as `fits=True` and get sent `direct` — the exact silent-truncation risk `budget.py:292` already warns about ("a provider that truncates an oversized prompt silently rather than rejecting it").
- Proposed tier: **major**, using the stated rule (a wrongly-passed fit check on a real, non-toy counter can leak a truncated/ungrounded summary as if it were complete).
- Confidence: medium. The survival itself is solid evidence (confirmed against 3 test files). The causal chain to an actual negative fencing value in production is plausible and specifically documented as a property of the real counter, but I did not construct a concrete fence template + tiktoken vocabulary pair that reproduces a negative fencing count — that would be the falsifying/confirming experiment for a verifier: measure `overhead.fencing` from `measure_overhead(TiktokenCounter.for_model(...), with_overlap=False)` and `with_overlap=True` across a range of real models and look for a negative pre-clamp value.

---

**C-S3M-003** — Trailing blank lines after a closing code fence are not pinned to belong to the `CODE_FENCE` block, and this is the same boundary-kind machinery already implicated in a filed defect.

- Claim: whether the blank line(s) immediately following a closing ``` ``` `` fence are folded into the `CODE_FENCE` block (current behavior) or left to start the next block is entirely unasserted — removing the fold produced zero failures across the full 766-test suite, including the dedicated `tests/test_structure_detection.py`.
- Citation: `summarizer/segmentation.py:352-353` (`index = _consume_blank_lines(lines, index)` immediately before the `CODE_FENCE` block is appended) @301cc4d.
- Surviving mutation: row #18.
- Reachability: real and common. `detect_structural_blocks` is the direct input to `_budgeted_units` (`segmentation.py:562`), which is the entry point `segment_document` always uses; any markdown document containing a fenced code block followed by a blank line — an extremely ordinary shape, including in LLM-authored input — exercises this path.
- Relationship to known context: the task names a **confirmed, already-filed** defect in this neighborhood: `BoundaryKind.CODE_FENCE` is produced by segmentation but missing from `AuditSegment.boundary_kind`'s closed `Literal`, crashing hierarchical `--audit`. I did not re-verify or re-report that defect. This candidate is a *different, adjacent* gap: it is about where the `CODE_FENCE` block's boundary sits, not about the closed-set omission itself — but because a packed segment inherits `boundary_kind` from whichever unit is last in its run (`_pack_units`, `segmentation.py:622-628`), moving the fence/non-fence boundary by a few blank lines can change which packed segment carries `CODE_FENCE` at all, i.e. it can shift whether and where the already-known crash is triggered by a given document. I am not claiming this candidate causes or fixes that defect — only that the exact placement of the boundary is unpinned, and it feeds the same mechanism.
- Proposed tier: **minor** on its own (it is an internal segmentation-boundary attribution question — both placements produce contiguous, budget-valid segments) but flagged as **worth cross-checking against the filed `CODE_FENCE`/`AuditSegment` defect** rather than triaged in isolation, since a fix to one may change the reproduction shape of the other.
- Confidence: high that the survival is real (full-suite confirmation). Medium on the practical import, hedged per the backend-caution instinct: I have not executed the specific document shape that would show this changes audit-crash reproducibility either way; that would be the falsifying test for a verifier (a fixture with a fenced block, trailing blank lines, and a packing budget chosen so the fence unit is last in its run, run under both boundary placements).

---

**C-S3M-004** — `_hard_split`'s whitespace-preference re-snap is not guarded against firing on an already-exact-fit chunk, though this appears behaviorally inert today.

- Claim: no test distinguishes `if next_end < end:` from `if next_end <= end:` as the trigger for `_prefer_whitespace_boundary` in `_hard_split` — mutating to `<=` (i.e., also re-snapping when the fitting prefix already reaches the end of the range) passed the full 766-test suite.
- Citation: `summarizer/segmentation.py:517` (`if next_end < end:`) @301cc4d.
- Surviving mutation: row #19.
- Mechanism: with the mutation, a hard-split chunk that already fits the budget exactly (`next_end == end`) would additionally be re-examined for an earlier whitespace boundary, which — per `_prefer_whitespace_boundary`'s own logic (`segmentation.py:488-498`) — can shrink an already-fitting chunk to an earlier space, producing a smaller-than-necessary segment rather than the maximal fitting one. This is reachable any time `_hard_split` is invoked with a range whose true fitting prefix equals its own end (i.e., the remaining text fits the budget exactly) — a normal outcome of the doubling/binary search in `_largest_fitting_prefix`, not an edge case requiring adversarial input.
- Proposed tier: **minor**. Every mutated output remains a valid, contiguous, budget-respecting segment; the only cost is more/smaller segments than optimal, a packing-density regression rather than a correctness or grounding issue.
- Confidence: medium-high on the survival (full-suite reproduction); medium on the mechanism, since I reasoned through `_prefer_whitespace_boundary`'s logic rather than constructing a concrete failing document that demonstrates a visibly different segment split under this mutation — that construction (a hard-split range with a `max_tokens`-exact fit and an earlier whitespace character) is the falsifying test for a verifier.

---

**C-S3M-005** — `_MODEL_CONTEXT_WINDOWS`'s exact-match table entries for `"gpt-4"` and `"gpt-3.5-turbo"` are only reachable through the exact-match branch and are unpinned by any numeric assertion.

- Claim: no test resolves `resolve_context_window(provider="openai", model="gpt-4")` (or `"gpt-3.5-turbo"`) and asserts the specific token count from the table; corrupting `_MODEL_CONTEXT_WINDOWS["gpt-4"]` from `8_192` to `1` passed the entire 766-test suite.
- Citation: `summarizer/budget.py:20-24` (`_MODEL_CONTEXT_WINDOWS`) @301cc4d.
- Surviving mutation: row #7. (Row #8, the analogous mutation on the prefix table's `"gpt-4o"` entry, is a strictly weaker variant of the same gap — it survived too, but only because the one test that reads that value compares two model names against each other rather than against a literal number, so both sides moved together.)
- Reachability: real. Any caller (CLI or library) that names the bare model `"gpt-4"` or `"gpt-3.5-turbo"` (no dated suffix) resolves through this exact table and gets whatever number is written there, unquestioned.
- Proposed tier: **major** under the stated rule — an unpinned, wrong context-window constant used for real budget arithmetic feeds directly into `fits` and the direct/hierarchical decision; a wrong-too-large number lets an over-budget document silently through to a request that a provider can truncate rather than reject (the same mechanism as C-S3M-002, but rooted in a plain data-table typo rather than a computed value).
- Confidence: medium-high. The survival is unambiguous and reproduced against the full suite. I am not claiming today's values (`8_192`, `32_768`, `16_385`) are wrong — only that nothing would catch it if they were.

---

**C-S3M-006** — `resolve_context_window`'s and `resolve_token_counter`'s `provider.strip().lower()` normalization is exercised by no test and is unreachable from every real caller in this codebase today.

- Claim: dropping `.lower()` from `resolve_context_window`'s provider check passed 198 tests spanning every file that touches strategy selection, direct execution, pipeline, CLI parsing, and audit/reliability/finalization/publication.
- Citation: `summarizer/budget.py:85` (`if provider.strip().lower() == "openai":`) and the identical pattern at `summarizer/tokenization.py:220` (`if provider.strip().lower() != "openai":`) @301cc4d.
- Surviving mutation: row #11.
- Reachability check (this is the weak part, flagged per the instructions): `AppConfig.provider` is typed `Literal["openai", "ollama"]` and validated exactly in `__post_init__` (`summarizer/config.py:67-68`, `if self.provider not in ("openai", "ollama"): raise ValueError`). Every real caller in this repo (`summarizer/cli.py:253-259`, `summarizer/pipeline.py:174`, `summarizer/pipeline.py:155`) passes `app.provider` or `config.app.provider`, which is always already exactly lowercase `"openai"`/`"ollama"` by the time it reaches `resolve_context_window`/`resolve_token_counter`. The `.strip().lower()` normalization is therefore dead from every real caller *in this codebase*; it would only matter to an external library consumer calling `select_strategy`/`resolve_context_window`/`resolve_token_counter` directly with an unnormalized string, bypassing `AppConfig`.
- Proposed tier: **minor**, specifically *because* of the reachability finding above — per the instructions, a guard unreachable from every real caller is a weaker candidate, and this one goes further: the normalization it performs is not just unreached today, it is provably unreachable while `AppConfig` remains the only production entry point.
- Confidence: high on both the survival and the reachability analysis (both are structural facts read directly from `config.py`, `cli.py`, and `pipeline.py`, not inference).

---

**C-S3M-007** — `_decide`'s priority between the "assumed window" reason and the "direct cap" reason is unordered by any test.

- Claim: swapping the order of the `if assumed:` and `if capped:` branches in `_decide` changed no test outcome (strategy is `"hierarchical"` either way; only the `reason` string's content would differ, and no existing test constructs a document where both conditions hold simultaneously to observe that).
- Citation: `summarizer/budget.py:351-359` @301cc4d.
- Surviving mutation: row #13.
- Reachability: real but narrow — requires a document with an assumed context window *and* a `max_direct_tokens` cap that the document also exceeds, simultaneously. Both knobs are independently reachable through the CLI; a caller could set both.
- Proposed tier: **minor**. `strategy` (the field every caller branches on) is unaffected either way; only `report.reason`'s wording would change, which is diagnostic/log text, not a decision.
- Confidence: high on the survival; low on this mattering to anyone, since I could not identify a caller that branches on the *content* of `reason` rather than `strategy`.

## 4. Nothing-found notes

Guards that were probed and found to be properly pinned (surviving-candidate list above already implies everything not listed here was killed cleanly, but calling out the strongest ones explicitly):

- `usable_input_capacity`'s `capacity <= 0` boundary is pinned exactly at zero by `tests/test_budget.py::test_capacity_of_exactly_zero_is_refused` (row #1).
- The `fits = document_tokens <= capacity` boundary is pinned on both sides by `tests/test_strategy_selection.py::test_selects_direct_exactly_at_capacity` and `::test_selects_hierarchical_one_token_over_capacity` (row #2).
- `safety_margin`'s `max(...)` term-selection is pinned in both directions by `tests/test_budget.py::test_safety_margin_takes_the_larger_term` (row #4).
- The full capacity-subtraction arithmetic (`window - overhead - output - margin`) is pinned exactly by `tests/test_budget.py::test_capacity_subtracts_every_term` (row #5).
- The explicit-`direct`-over-capacity `BudgetError` path is pinned by `tests/test_strategy_selection.py::test_explicit_direct_over_capacity_fails_with_its_arithmetic` (row #6).
- The explicit-`direct`-with-assumed-window `BudgetError` path (a scenario I checked because it looked at first like a gap) is in fact pinned, by `tests/test_direct.py::test_explicit_direct_refuses_an_assumed_context_window`.
- Longest-prefix resolution among overlapping model families is pinned precisely by `tests/test_context_windows.py::test_prefers_the_longest_matching_prefix` (row #9), which is deliberately built so `"o1"` and `"o1-mini"` carry different windows.
- `_pack_units`'s post-snap recount-and-fallback loop — the mechanism that protects against a non-monotonic counter's dip — is pinned exactly by `tests/test_segmentation.py::test_packing_recounts_the_boundary_it_snaps_to`, using the documented `DippingCounter` fixture (row #14).
- The heading-triggered run-boundary in `_pack_units` (a run never packs across a `HEADING` unit) is pinned by multiple tests simultaneously, most directly `tests/test_segmentation.py::test_heading_starts_a_new_section_and_packs_forward` (row #16).
- `TiktokenCounter.fitting_prefix`'s binary-search fallback (below the early-return fast path discussed in C-S3M candidates) is exercised and pinned by `tests/test_tokenization.py::test_tiktoken_prefix_search_is_bounded_near_the_budget` and `::test_tiktoken_prefix_extends_past_unstable_full_text_boundary`.
- `summarizer/ingestion.py::normalize_source_text` — BOM stripping, CRLF/CR normalization, per-line trailing-space/tab stripping, and leading/trailing blank-line collapse are all exercised together by one comprehensive fixture, `tests/test_ingestion.py::test_normalization_preserves_structure_and_unicode` (`"﻿\r\n# H  \r\n\r\n  - café\t \rNext\t\r\n\r\n"` → `"# H\n\n  - café\nNext"`). I did not mutate this function (time/priority budget went to budget.py and segmentation.py per the stated priority order), but the fixture's density made it look well-covered on inspection; I flag it as inspected-not-mutated rather than claim it as killed.
- `summarizer/text.py::chunk_text_by_sentences` was inspected but not mutated. It is legacy code superseded by segmentation per the module's own comment ("Issue #4 replaces this legacy chunker with the generalized segmentation pipeline") and is reachable only through `summarizer/legacy_workflow.py`, which `summarizer/cli.py` never imports — i.e. it is very likely unreachable from the real CLI entry point today. Per the instructions' reachability guidance, I judged mutation effort here to be low-value and spent the remaining budget on segmentation.py/budget.py instead.

## Files written by this review

Exactly one file was written: `/Users/kylenakamura/documents-local/development-local/side-projects/recursive-text-summarizer/.review/evidence/s3-mutation-review.md` (this file).

No other file under the repo was created, modified, or deleted. All mutation work happened in a disposable copy at `/tmp/s3-mutation-review/repo`, which has been removed.
