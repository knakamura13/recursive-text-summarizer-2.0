# Adaptive Multi-Level Hierarchical Merging Design

## Scope

Issue #7 builds the summary tree: ordered leaves from segment summaries, recursively grouped until one root remains, with group sizes derived from measurement rather than a fixed map-reduce shape.

Boundaries against neighbouring issues:

- **Issues #4, #5, and #6 are merged.** This issue consumes `SourceSegment`, `SummaryNode`, `summarize_segments`, and the budget calculator as they stand, and changes only what is named below.
- **Issue #8 owns source grounding.** Retrieving original passages into merge requests, separating authoritative source from generated child summaries in the prompt, correcting a misleading child against source, and *narrowing* provenance are all its work. This issue must set provenance up so #8 can narrow it, without narrowing it here.
- **Issue #9 owns the serialized audit artifact.** The tree report here is an in-process value, following the precedent `BudgetReport` set.
- **Issue #10 owns claim verification; #11 owns caching, resume, and concurrency.** The merge prompt and schema versions added here become cache-key inputs, and the tree must not make concurrency impossible.
- **Issue #12 owns the end-to-end demonstration.** The command line stays on the legacy path, as it has since #4.

## What the numbers say

**A serialized child costs 55 to 870 tokens.** Measured with `o200k_base` on realistic records: a sparse node (summary only) is 54 tokens, a typical one (four content units with evidence) is 290, a rich one (ten units, quotations, contradictions) is 863. Pretty-printing adds 40%. Per-child fencing is about 40 tokens.

**Provenance is the only term that grows without bound, and it grows at exactly 4.0 tokens per identifier.** Measured by holding a rich body fixed and widening provenance: 1 id → 863 tokens, 16 → 923, 64 → 1,115, 256 → 1,883.

**That growth kills the recursion, which is the finding this design is built around.** If a node's provenance is the union of its children's, a child carrying more than roughly **30,660 identifiers cannot fit a merge request at all**, whatever else it contains. Simulated against the real capacity: rich children reach fanout 2 at level 3 and fanout **1 at level 4** — no forward progress, with the provenance field alone responsible several levels before content is. A design that unions provenance into the merge payload therefore terminates in a failure that looks like a budget bug but is structural.

**Three merge levels are unreachable on hosted defaults.** At `gpt-4o-mini`'s 123,618-token capacity, measured packing gives a fanout of 136 for rich children. Forcing three levels needs more than 20,000 leaves, i.e. roughly 20 million source tokens. The largest file in this repository is 17,262 tokens and produces 22 leaves at a 1,000-token budget — one merge level. Multi-level behaviour is therefore exercised by injecting a small counter or window, not by large input.

**The failure case is the default local configuration, not an exotic one.** With the assumed 8,192-token window and the byte estimator, usable capacity is 3,436 bytes and a single rich child (3,896 bytes) **does not fit a merge request**. That is criterion 3's "even one child cannot fit", reachable with no adversarial input — and `select_strategy` routes assumed windows to hierarchical, so it is exactly the path a local run lands on.

## Considered approaches

The central question is how provenance travels upward, because it is what decides whether the recursion terminates.

### Union provenance into the merge payload

The obvious reading of "the root can be traced to the exact original segments": each node's provenance is the union of its children's, and children are serialized whole into the merge request.

Rejected on the measurement above. It is not merely expensive; it halts the recursion at level 3 or 4 for any realistic tree.

### Narrow provenance during merging

Keep only identifiers supporting retained claims. This bounds the growth, but narrowing is explicitly issue #8's criterion, and doing it here would pre-empt the policy that issue is meant to establish — while silently discarding traceability in the meantime.

### Compute provenance locally, and exclude it from the payload

Provenance is a fact about the tree, not an opinion the model should be asked for: a node covers exactly the segments its children cover. So it is computed by union locally and never read from the response, and the children serialized into a merge request omit their provenance entirely.

This is the selected approach. It removes the 4-tokens-per-identifier growth from every request while keeping the union intact in the record, so #8 can narrow it later and #9 can trace a root to source. It also strengthens the injection boundary rather than weakening it: the leaf design's rule is that the legal set never comes from the payload, and here the *stored* provenance does not come from the payload either.

The model still emits a `provenance` field, because the schema requires every property. It is validated as a subset of the legal set and then replaced by the computed union — a discarded model opinion, deliberately.

## Grouping and forward progress

Group size is measured, not fixed: serialize each child, add per-child fencing, and derive how many fit the merge capacity. Groups are then **balanced** rather than greedily packed, because the epic requires that earlier content not be compressed more than later content, and greedy packing leaves a ragged final group that does exactly that.

Two guarantees make the recursion terminate:

- **Every level strictly reduces the node count**, asserted after each level rather than assumed. This is the only property that makes termination provable.
- **A single child that cannot fit its own merge request fails** with `BudgetError` naming the capacity, the child's serialized size, the fencing, and which child. Reachable on default local configuration, so this is a real path, not a guard.

A trailing group of one is allowed and passes its node upward without a provider call, since the count still falls when other groups merge.

An optional ceiling on children per merge exists and defaults to unset, so measurement governs. The ceiling is how the multi-level path is exercised in tests, and how an operator can narrow merges without a code change. Whether wide merges lose more information than narrow ones is a quality question this issue is not entitled to answer — but note the functional concern recorded under open decisions: a 136-way merge must express itself within the same output reserve as a 2-way one.

## Domain model

`SummaryNode` is reused unchanged, as its docstring intends. The schema needs no change either: every property is generic, `level` is an unbounded integer, and `provenance` is an unbounded array — verified against the dumped schema.

What `SummaryNode` cannot carry is the tree. It has no identifier, no children, and no order. Two frozen dataclasses supply that:

- **`TreeNode`** — a stable identifier, its level, its order within the level, the `SummaryNode`, the identifiers of its children, and the segment identifiers it covers. Carrying order explicitly rather than relying on list position keeps ordering deterministic even if merges later run concurrently, which the epic requires.
- **`HierarchyReport`** — per-level node counts, the fanout chosen at each level and why, the leaf count, the level count, and the provider call count.

Parent-to-child edges are stored even though this issue's criterion asks only for shape, because #8's root-to-segment traceability needs them and reconstructing them later would be guesswork.

## Validation

`_validate_provenance` currently hard-codes a single legal identifier and a single attributable text. It is generalized rather than duplicated: duplicating a validator that needed a dedicated hardening pass invites the two copies to drift on exactly the injection defences that pass established.

Three changes, and one invariant that must survive all of them:

- the legal set becomes a caller-supplied mapping from identifier to attributable text;
- the "must record provenance for itself" rule becomes "must cite within the legal set and not vacuously", so a merged node cannot pass by citing nothing;
- a quotation is checked against the core of *the segment it cites*, not a concatenation of all of them — a concatenation would let a quote straddle two segments and reopen the cross-attribution hole the overlap work closed.

The invariant: the legal set never comes from the payload.

## The merge prompt

A new template with its own version constant, since it is a distinct cache-key input. It states the rules the criteria require, and two that the request's own shape creates:

- **Adjacency is not evidence.** Children arrive as an ordered list and a model will read order as causation, so the prompt says so explicitly and forbids inventing causal or temporal links.
- **Deduplication must not lose evidence.** Collapsing two children's identical claim has to keep both children's support, or provenance silently narrows — which is #8's decision to make, not a side effect here.
- **Contradictions are preserved, not reconciled.** No averaging, no choosing.

The level is interpolated, so instruction length varies slightly by level; overhead measurement takes a representative level and the safety margin absorbs the difference.

Children are model-written text, so the injection boundary applies to them exactly as it applies to source: per-child fences derived the way leaf fences are, and a precedence rule that does not depend on the fence being unguessable. A merged node's legal identifiers are **not** enumerated in the prompt — at level two a legal set can run to thousands of identifiers and cost a third of the request — so the prompt says to cite only what appears in the children shown, and the validator enforces the actual set.

## Errors and invariants

- A single child that cannot fit raises `BudgetError` with the arithmetic.
- A level that fails to reduce the node count raises rather than looping.
- A merged node citing outside the legal set, or citing nothing, fails validation.
- A quotation that does not occur in the core of the segment it cites fails validation.
- Provider failures propagate; a validation failure fails the run without partial output, as the leaf stage already does.

## Determinism

The same document, counter, and configuration produce identical groups, an identical sequence of requests, and an identical report. Ordering comes from explicit order fields, never from arrival sequence. Packing recounts every chosen boundary rather than trusting a search, because the token counters are documented as non-maximal under a byte-pair encoder.

## Module layout

- `summarizer/hierarchy.py` — `TreeNode`, `HierarchyReport`, grouping, the merge stage, and the recursion.
- `summarizer/merge.py` — the merge prompt, request builder, and response parsing.
- `summarizer/leaf.py` — the generalized validator.

## Testing

Offline throughout, following the established discipline. Multi-level behaviour is forced with an injected small counter and a configured ceiling, since real input cannot reach three levels at hosted capacity. Coverage: adaptive grouping at two child sizes, deterministic ordering with shuffled input, deduplication and contradiction preservation surviving a merge, a single oversized child failing with its arithmetic, a level that fails to shrink, provenance unioning to exactly the covered set, and the genre sweep the leaf prompt already gets.

## Delivery boundary

This issue delivers the tree, grouping, the merge stage and prompt, the generalized validator, and the report. Source reconsultation, provenance narrowing, citations, the audit artifact, and the command line are untouched.

## Decisions taken

1. **Provenance is computed locally and excluded from the merge payload.** It is the difference between a recursion that terminates and one that does not.
2. **The model's `provenance` field is validated then discarded** in favour of the computed union.
3. **Groups are balanced, not greedily packed**, because the epic requires even compression.
4. **`SummaryNode` and its schema are reused unchanged**; the tree lives in separate records.
5. **`_validate_provenance` is generalized in place** rather than duplicated.
6. **Every level must strictly reduce the node count**, asserted.
7. **The children-per-merge ceiling defaults to unset**, so measurement governs and the quality question stays open.
8. **Parent-to-child edges are stored** despite not being required here.

## Measured on completion

Verified end to end against an injected counter and a ceiling of three children per merge: **20 leaves reduce to one root through 3 levels in 11 provider calls**, with per-level counts 20 → 7 → 3 → 1. The root's provenance equals the ordered union of all twenty segment identifiers, and parent-to-child edges are present on all 31 stored nodes.

The `require_provenance` flag on the validator was not in the design. It emerged from implementing it: the anti-vacuity rule added for leaves demands non-empty provenance, but a merged node's provenance is derived, so requiring a model to restate a value that is then discarded is ceremony and its absence proves nothing either way. The flag defaults to True, so leaf behaviour is untouched.

## Open decisions

1. **Whether a wide merge should be capped by default.** A 136-way merge must express itself within the same output reserve as a 2-way one, which is a functional argument for a default ceiling and not only a quality one. Left unset for consistency with the direct cap, and because no measurement in this repository bears on the quality question.
2. **Whether provenance should be range-compressed.** Identifiers are zero-padded and segmentation guarantees contiguous cores, so a node's coverage is almost always a contiguous range that would collapse 136 identifiers to about 10 tokens. It stops being lossless once #8 narrows.
3. **Whether `GenerationRequest` should gain an output cap.** Merge outputs are larger than leaf outputs, and a truncated merge response surfaces as "no JSON object found" rather than as a budget failure. This is the first stage where the 1,024-token default is plausibly too small.
4. **Whether merging may run concurrently.** This issue stays sequential; the explicit order fields are what keep that decision open.
