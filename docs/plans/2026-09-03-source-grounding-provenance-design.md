# Source Grounding and Provenance Propagation Design

## Scope

Issue #8 makes recursive merges reconsult authoritative source text and retain
traceable, machine-readable provenance. It follows issue #7's hierarchy rather
than changing its tree shape, CLI integration, cache policy, or final output.

## Existing seam

`build_hierarchy` already receives `attributable`, an ordered mapping from a
source-segment ID to its citable core text. That mapping is controlled by the
application, not model output, and is therefore the only valid source for
grounding passages. The current merge request serializes generated children
only; its parser validates model references but replaces all provenance with
the complete child union. This preserves reachability but cannot make a merge
source-grounded or narrow provenance to retained claims.

## Considered approaches

1. Send every covered source segment with every merge. This is maximally
   direct but can consume the entire context window and breaks multi-level
   progress.
2. Perform a separate retrieval or embedding stage. It could improve semantic
   ranking, but adds a provider and persistence boundary that the issue does
   not require.
3. Select deterministic passages from existing structured evidence and child
   coverage, under the request's actual remaining token budget. This reuses
   the established model-neutral token boundary and keeps offline tests fully
   deterministic.

The third approach is selected.

## Data flow

For each merge group, the hierarchy reserves one quarter of the merge payload
capacity for source grounding before calculating fanout. Once a concrete group
is known, it gives the selector every remaining token after its measured child
payloads and fences. The selector considers IDs in source order:

1. contradiction evidence;
2. qualification and uncertain-content evidence;
3. quotation evidence;
4. other content-unit evidence; and
5. declared provenance as a deterministic fallback.

It packs complete core passages only. A request fails clearly if the reserve
cannot hold evidence required by a contradiction, qualification, or uncertain
claim; it may omit only low-priority fallback IDs. This is conservative: a
later issue may replace ranking, but it cannot permit generated text to supply
its own source or silently turn an ambiguous claim into an ungrounded one.

The merge request has separate generated-summary and authoritative-source
blocks, each individually fenced. Its instructions say source passages are
authoritative, child summaries are provisional, and an output claim or
reference must be supported by the supplied passages. Source text remains data
and cannot override those instructions.

## Provenance policy

Only selected passage IDs are legal in a merge response. The shared validator
checks every content-unit evidence item, grounded qualification,
grounded contradiction, and quotation against those passages. It requires every
content unit and grounded annotation to name at least one source. Merged
responses must record provenance. The parser then canonicalizes every declared
and direct-evidence reference to source order and stores that narrowed
sequence; it no longer replaces it with the whole child union. The tree's
`covered_segments` continues to preserve the full structural reachability
independently.

This makes the two notions explicit:

- `covered_segments`: every original segment structurally represented by a
  tree node;
- `SummaryNode.provenance`: selected original passages that support claims
  retained by that node.

`GroundedAnnotation` replaces the existing bare strings for qualifications and
contradictions. This small schema extension gives every retained standalone
qualification or conflict explicit evidence without adding a separate assertion
framework. A merge cannot silently resolve or restate them without being shown
the original supporting material.

## Error handling and determinism

All candidates, selection, source-block serialization, and stored provenance
use source order. The selector never uses model output to retrieve text. An
unknown reference, a quotation absent from its cited passage, empty merge
provenance, an empty grounding selection, or a passage that cannot fit is a
validation error rather than partial output.

## Tests

Offline tests will pin passage priority, whole-passage budget accounting,
separation/fencing, invalid references, contradictory or
ambiguous child grounding, and a misleading child summary corrected against
an authoritative passage. Existing hierarchy tests will prove full tree
coverage remains available while root `SummaryNode.provenance` narrows.
