# Automatic Strategy Selection and Direct Summarization Design

## Scope

Issue #6 adds safe context-budget arithmetic and completes the `direct` and `auto` execution paths. A document that genuinely fits is summarized in one call; one that does not is routed toward hierarchical execution instead of being sent as an invalid request.

The boundaries against neighbouring issues are narrow and worth stating, because three of them are easy to drift into:

- **Issue #4 owns segmentation** and **issue #5 owns leaf records.** Both are merged. This issue consumes `SourceDocument`, `SourceSegment`, `TokenCounter`, `build_leaf_request`, `parse_leaf_summary`, and `SummaryNode` as they stand, and changes them only where noted below.
- **Issue #7 owns the hierarchy.** Tree construction, branching factor, merge prompts, and forward-progress guarantees are its work. This issue ships the budget calculator #7 will consume and *routes* to hierarchical without implementing it.
- **Issue #9 owns the audit artifact.** The run metadata here is an in-process value returned to callers, not a serialized file.
- **Issue #11 hashes behaviour-relevant configuration into cache keys**, so every configuration value added here becomes a future cache-key input.
- **Issue #12 owns the end-to-end demonstration** and the README rewrite. This issue therefore adds strategy configuration to the CLI, but does not rewire `main()` onto the new pipeline — the command line keeps running the legacy path, exactly as #4 and #5 left it.

## What the numbers say

The design is driven by measurements against the pinned dependencies rather than estimates, because two of them are counter-intuitive.

**The schema dominates the fixed overhead.** `leaf_summary_schema()` is sent on every request. Serialized compactly and counted with `o200k_base` it is **521 tokens**, against **253** for the instructions and **27** for the fencing — so the schema is 65% of an 801-token fixed cost. Of those 521, **113 are pydantic docstrings** rendered into JSON Schema `description` keys and 84 are auto-generated `title` keys, so 38% of the dominant term is documentation shipped to the model on every call.

**Overhead is not one constant.** When a segment carries overlap, the prompt gains a second instruction block and two more fences: overhead rises to roughly **930 tokens**. A calculator that assumes 801 under-reserves by ~130 tokens whenever overlap is configured.

**The conservative counter makes small windows unusable.** `ConservativeUtf8TokenCounter` — what `resolve_token_counter` returns for every Ollama tag — costs **4.2×** on the schema and **4.8×** on the instructions relative to a real tokenizer, because it charges one token per UTF-8 byte. At a 4,096-token window with a 1,024-token output reserve, usable input capacity computes to **−674**. Non-positive capacity is reachable on default local configuration and must therefore be a named error, not an arithmetic underflow.

**Direct is the common case, not the exotic one.** At a 128,000-token window the usable input capacity is 119,775 tokens, which is roughly 514 KB of prose. The entire 30-file corpus in this repository is 84,948 tokens and fits in a single request; `input.txt` is 307 tokens. On the project's own defaults, a window-only `auto` selects `direct` for every input this repository has ever handled, and the hierarchy #7 builds would be unreachable without configuration. That is a product consequence, not a bug, and it is addressed under strategy selection below.

**The two providers fail differently, and one of them does not fail at all.** OpenAI's Responses API defaults `truncation` to `disabled`, so oversized input is a 400 that the adapter already maps to `ProviderRequestError`. Ollama silently truncates: a ~1,700-token prompt sent with `num_ctx=512` returned `prompt_eval_count=258` and `done_reason="length"` with **no error and plausible-looking output**. On the local path, budget arithmetic is the only defence against a confidently ungrounded summary, because the response status does not reveal the loss.

## Considered approaches

The central question is how `direct` produces a result, given that #5's machinery is built around a segment.

### A dedicated direct record and prompt

Introduce a `DirectSummary` record with its own schema, prompt, and version constants. Maximum control, and it leaves #5 untouched. Rejected: it gives issue #9 two unrelated shapes to synthesize from, duplicates the schema and prompt versioning, and abandons the leaf design's explicit choice to keep one central record for the whole hierarchy.

### Reuse segmentation with a budget large enough to yield one segment

Attractive, and it was the first approach attempted: if the whole document fits, `segment_document` should pack it into a single segment and the leaf pipeline could be reused unchanged.

**This does not work, and the reason is worth recording.** Headings force a hard break in packing, so a document with more than one heading never collapses to a single segment regardless of budget. Measured: a 26-token document with three headings yields three segments even at a budget of 52. Only heading-free documents collapse. Verified rather than assumed, after the design initially depended on it.

### A document-spanning segment, constructed directly

Build one `SourceSegment` covering the whole canonical document and pass it through the existing request builder, parser, and validator.

This is the selected approach. It reuses the request construction, the tolerant parser, the strict validator, and the provenance rules that were hardened in #18, and it produces the same `SummaryNode` the rest of the hierarchy consumes. Because the record's core range spans the document, quotations are checked against the whole text and the single legal identifier is the document's own — so `_validate_provenance` needs no change.

It requires two small, honest additions rather than workarounds:

- **`BoundaryKind.DOCUMENT`.** The existing kinds all describe a boundary *within* a document. A unit that is the entire document has no such boundary, and forcing `HARD` or `PARAGRAPH` would misreport provenance. Naming the case is cheaper than lying about it.
- **A prompt that does not claim to be a fragment.** The leaf instructions open "You extract structured information from one region of a longer document." For a direct run there is no longer document, and telling a model it is reading a fragment invites it to hedge about missing context — the opposite of the cohesive result the acceptance criteria ask for. The framing sentence is therefore selected by whether the unit is the whole document, and `LEAF_PROMPT_VERSION` is bumped because output can change for identical input.

## Budget arithmetic

```
usable_input = window − overhead − output_reserve − safety_margin
```

Every term is either measured or configured; none is guessed.

**`overhead` is measured, not hard-coded.** An acceptance criterion covers "overhead changes", so a constant would be wrong by construction the moment the prompt or the record changes. It is computed as the token count of the filled instruction template, plus the compactly-serialized schema, plus the fencing — with the overlap-carrying variant taken whenever overlap is configured, since that is the larger of the two. A test pins the measurement so that editing either string fails loudly rather than silently shrinking the budget.

The schema is counted in its compact serialization. That is an approximation of what the provider's own tokenizer sees, and the direction of the error is stated: compact is the closer proxy to the wire, while an indented form would overstate by 67%. Approximating in the direction of a smaller reservation is the wrong direction for safety, which is one reason the safety margin exists.

**`safety_margin` is `max(fixed, fraction × window)`.** A fixed margin alone is negligible at 262,144 tokens; a fractional margin alone discards 13,000 tokens there for no reason while under-protecting a 4,096-token window. Taking the maximum gives a floor for small windows and proportionality for large ones. Defaults are 256 tokens and 2%.

**Non-positive capacity is an error with the arithmetic in it.** `BudgetError` reports the window, each subtracted term, and the result, because the operator's next action depends on which term dominated — a bigger model, a smaller output reserve, or an explicit window override.

## Context windows

There is no offline source of truth for a model's context window, and for OpenAI there is no *online* one either. The installed `openai` 3.7.0 exposes no window anywhere; `Model` carries only `{id, created, object, owned_by, shutdown_date}`. `tiktoken` maps names to encodings, never to sizes. Ollama does expose an architectural length through `show()`, but the key is architecture-prefixed rather than tag-named — tag `qwen3.8` reports `qwen35.context_length` — and it is a network call that is unavailable before a pull and unusable in the offline suite.

So the window comes from a hand-maintained table with a two-tier lookup, exact name then prefix, mirroring the shape `tiktoken` already uses and which is what makes `gpt-4o-mini` resolve at all.

**An unknown model does not fail, and does not silently get a default that pretends to be knowledge.** It resolves to a conservative assumed window, and the resolution records that it was assumed. `auto` then routes an unknown model to hierarchical rather than gambling a direct request on a guess — the only option that keeps arbitrary local tags working, as the epic requires, while never sending a request that cannot fit. An operator who knows better states the window explicitly and gets the direct path back.

This mirrors `TokenCounter.exact`, which already exists so that an estimate can be visibly an estimate.

## Strategy selection

`direct` is chosen only when the measured document token count fits the usable input capacity. The measurement is `counter.count(document.text)`, which is exactly the text a direct request sends — not the sum of segment counts, which differs under BPE and diverges further under overlap.

`auto` resolves to `direct` when the document fits **and** the window was known rather than assumed **and** no configured direct cap is exceeded; otherwise `hierarchical`.

The optional cap exists because of the measurement above: without one, window-only selection sends 85,000-token documents in a single call and #7's hierarchy is unreachable on defaults. Whether a single enormous call produces a *better* summary than a hierarchy is a quality question this issue is not entitled to answer, so the cap defaults to unset — window-only behaviour, exactly as the acceptance criterion words it — and exists so the question can be answered later by configuration rather than by a code change.

Explicit `direct` that does not fit fails before any provider call, with the same numbers the decision used.

## Domain model

Two frozen dataclasses, not pydantic models: these are computed locally from configuration and measurement, and the leaf design reserved pydantic for records built from untrusted model output.

- **`StrategyConfig`** carries the strategy name, an optional explicit context window, the output reserve, the two safety-margin terms, and the optional direct cap. It validates its own fields, so a bad value becomes an argument-parser error rather than a traceback.
- **`BudgetReport`** records what the decision saw: the resolved window and whether it was assumed, the counter's identity and whether it is exact, each measured overhead term, the reserve, the margin, the computed capacity, the document's token count, the selected strategy, and a machine-readable reason. It is what satisfies the run-metadata criterion and what makes a rejection explicable.

`StrategyConfig` becomes a fourth field on `ParsedConfig` rather than more fields on `AppConfig`. `AppConfig` is constructed positionally in existing tests, and a commit already exists in this repository's history solely to repair that after a field moved; keeping budget settings in their own record avoids re-opening that hazard and mirrors how `RetryPolicy` and `LegacyWorkflowConfig` are already separate.

## Errors and invariants

- A configuration whose computed capacity is non-positive raises `BudgetError` naming every term.
- Explicit `direct` with a document that does not fit raises `BudgetError` before a provider call.
- An unknown model resolves to an assumed window and is reported as assumed; it never silently claims exactness.
- Strategy validation happens during argument parsing, so the command line reports it as a usage error. This matters because `main()` currently catches only `OSError` and `ProviderError`, so a `ValueError` escaping into it would print a traceback.
- The direct path calls the provider exactly once and never converts a provider failure into summary text.

## Determinism

The same document, model, counter, and configuration produce an identical `BudgetReport` and an identical `GenerationRequest`. The report is therefore safe to hash for #11's cache keys, and a strategy decision is reproducible from its inputs alone.

## Module layout

- `summarizer/budget.py` — the context table, overhead measurement, capacity arithmetic, `BudgetError`, `BudgetReport`, and strategy selection.
- `summarizer/direct.py` — the document-spanning segment and the direct summarization stage.
- `summarizer/config.py` — `StrategyConfig`.
- `summarizer/cli.py` — the new flags.

## Testing

All offline, following the established discipline: sockets blocked by the autouse fixture, providers faked at the `client_factory` seam or by a hand-written `generate`, and deterministic injected counters wherever a boundary matters. A real `tiktoken` encoding is constructed in exactly one place, guarded so that only vocabulary availability can skip it.

Coverage follows the acceptance criteria, with the boundary cases named explicitly: a document one token under capacity and one token over it; an overhead change shifting the decision; each safety-margin term dominating; a direct success; a forced-direct rejection carrying its arithmetic; automatic hierarchical selection; an unknown model routing to hierarchical; and a configuration whose capacity underflows.

## Delivery boundary

This issue delivers the budget calculator, the context table, strategy selection, the direct path, `StrategyConfig`, the CLI flags, and their tests. The legacy command-line workflow, the merge stage, and the audit artifact are untouched.

## Decisions taken

1. **Direct reuses the leaf machinery through a document-spanning segment**, rather than a parallel record. It costs one new `BoundaryKind` member and one prompt branch, and it keeps a single central record for the hierarchy.
2. **`LEAF_PROMPT_VERSION` is bumped** because the framing sentence now varies. Nothing caches on it yet, so this is free today and would not be later.
3. **Overhead is measured at runtime** and pinned by a test, rather than hard-coded.
4. **The safety margin is `max(fixed, fraction × window)`**, defaulting to 256 tokens and 2%.
5. **An unknown model routes `auto` to hierarchical** instead of failing or guessing that a default window is knowledge.
6. **The direct cap defaults to unset**, so behaviour matches the acceptance criterion while leaving the quality question configurable.
7. **Budget settings live in `StrategyConfig`**, not `AppConfig`, to avoid the positional-construction hazard.
8. **The output reserve is accounted for but not enforced.** Enforcing it means adding an output cap to `GenerationRequest` and mapping it in both adapters, which re-opens the provider contract for a criterion that only requires the reservation be *accounted* for.

## Open decisions

These change the shape of the code or reflect a product judgment this issue should not make alone:

1. **Whether to trim `description` and `title` from the leaf schema**, reclaiming 197 of 521 tokens — a 38% cut to the dominant overhead term. It changes `leaf_summary_schema()` and breaks the equality test that pins it against the OpenAI SDK's converter, so it is #5's surface rather than this issue's.
2. **Whether `GenerationRequest` should gain an output cap** so the reservation is enforced rather than notional. Without it, a model may exceed its reserve and OpenAI will return `incomplete`, which the adapter currently reports without saying why.
3. **The default direct cap.** Unset preserves the criterion as written, at the cost of the hierarchy being unreachable on defaults for every realistic input.
4. **Whether the conservative counter should measure our own instructions and schema.** It charges 4.2–4.8× on ASCII that this project authors, which is what drives capacity negative at small windows. Calibrating it for controlled text would recover that, at the cost of the estimator no longer being uniformly safe.
5. **Whether a runtime Ollama `show()` probe belongs behind an injectable seam**, giving local models a real window instead of an assumed one.
6. **Whether to backfill `help=` on the ten pre-existing CLI flags.** None has one today, and the criterion asks for clear CLI help.
