# Final Editorial Synthesis, Citations, and Audit Design

## Scope

Issue #9 adds the library-level stage after either direct summarization or a
grounded hierarchy. It turns the root record into one reader-facing summary,
optionally renders stable segment citations, and can emit a deterministic audit
artifact. The CLI deliberately remains on the legacy workflow until #12.

## Chosen design

`summarizer.editorial` owns a single final-writing request. Its input is the
already grounded `SummaryNode`, not raw document text: this stage may improve
organization and remove redundancy, but must not introduce facts, resolve
uncertainty, or silently alter meaning. The request returns a minimal strict
JSON record containing only final prose. It is genre-neutral, fences the root
record as untrusted data, and accepts an explicit requested target length.

Citations are not generated afresh. They are deterministic views of the root's
validated, source-ordered provenance. A citation therefore cannot be invented
by a model and every emitted identifier is checked against recorded segment
metadata. Default rendering returns only prose; opt-in rendering appends a
short source-segment list.

`summarizer.audit` owns serializable records and writing. The artifact includes
sanitized configuration, source and segment metadata (never raw source text),
the root/tree shape, each node's structured content and evidence links,
warnings/failures, known provider usage metadata, and citation-to-segment
mappings. Its schema version is explicit. Construction validates all cross
references, recursively redacts authentication-shaped keys and values, and
serializes with sorted compact JSON; parsing the bytes back through the schema
is required before an atomic write.

## Alternatives rejected

1. **Use the legacy CLI as the integration point.** Rejected because #4–#8
   intentionally keep it compatible while reusable pipeline stages mature, and
   #12 owns the end-to-end command-line workflow.
2. **Ask the final writer to invent inline citations.** Rejected because a
   textual citation marker cannot be safely tied to a generated sentence.
   Root provenance already carries validated support, so deterministic rendered
   citations are both simpler and stricter.
3. **Put raw source text in the audit JSON.** Rejected because an audit is
   designed to be retained/shared and source contents can contain credentials.
   Stable IDs, offsets, token counts, structure, and evidence links provide the
   trace without copying source secrets.

## Invariants

- Final prose is a dedicated provider call and plain text by default.
- Citation IDs are unique, source ordered, and resolve to supplied segment
  metadata; unknown root provenance is an error, not a dangling footnote.
- Audit records contain no raw source text or provider-authentication fields.
  Known credential patterns are replaced with `[REDACTED]` in prose, metadata,
  warnings, failures, and configuration values.
- Audit bytes are deterministic for deterministic inputs and validate against
  the versioned schema before they are written.
- Direct and hierarchical callers share the same finalization API; hierarchy
  callers simply supply the ordered `TreeNode` records.

## Tests

Offline fakes will prove the direct and forced-multi-level hierarchy paths each
make a final editorial call; the prompt is genre-neutral and fenced; citations
are stable and reject unknown IDs; default rendering stays plain; audit records
resolve all source/tree/evidence links; artifacts redact credentials; bytes are
deterministic and validated; invalid artifacts never replace an existing file.
