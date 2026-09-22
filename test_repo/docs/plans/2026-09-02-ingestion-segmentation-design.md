# Generalized Ingestion and Token-Aware Segmentation Design

## Scope

Issue #4 replaces the legacy character-count chunker with reusable, offline ingestion and segmentation components. It does not switch the CLI to the new hierarchical pipeline. Issue #5 will consume the segments for structured leaf summarization.

The design optimizes for two properties that matter throughout the hierarchy:

1. Pack as much useful source text as possible into each model request without exceeding its budget.
2. Preserve exact, machine-resolvable evidence spans so later summaries can remain grounded in the source.

## Considered approaches

### One universal tokenizer

Use one tokenizer, such as `cl100k_base`, for every provider and model. This is deterministic and simple, but its estimates can be unsafe or wasteful for unrelated Ollama model families.

### Provider calls for token counts

Ask each provider to tokenize or evaluate every candidate segment. This would be model-accurate, but OpenAI and Ollama do not expose one common preflight tokenization contract. Ollama documents token counts on completed generation responses, not a public tokenize endpoint. Counting through inference would add latency, load models unnecessarily, require a live service during segmentation, and break offline tests.

### Injectable counters with conservative fallback

Use a small `TokenCounter` protocol. Resolve an exact local counter when the selected model has a supported tokenizer, and otherwise use a conservative UTF-8 estimator with an explicit safety margin. Tests inject deterministic counters.

This is the selected approach. It preserves a provider-neutral segmentation pipeline, supports arbitrary Ollama tags, and permits tighter model-specific packing without making ingestion depend on the network.

## Domain model

`SourceDocument` is the canonical representation of an ingested file:

- `text`: UTF-8 text with line endings normalized to `\n`.
- `source_id`: a stable digest of the canonical text.
- `path`: the originating file, or `None` for text ingested directly. The source encoding is an argument to `read_source` rather than a field, because the canonical text is already decoded and the digest is taken over its UTF-8 bytes.

Normalization is deliberately narrow. It removes a leading UTF-8 BOM, converts CRLF and CR line endings to LF, removes trailing spaces and tabs from lines, and trims only outer blank lines. It preserves headings, indentation, paragraph breaks, list markers, and internal blank lines. Empty or whitespace-only input raises an actionable `EmptySourceError`.

All character offsets resolve against `SourceDocument.text`. A segment must satisfy:

```python
document.text[segment.context_start:segment.context_end] == segment.text
```

The canonical document and its digest make those offsets unambiguous even when the original file used different line endings.

`SourceSegment` records:

- a deterministic identifier such as `S000001`;
- source order;
- a disjoint core character range;
- the possibly expanded context character range;
- exact text for the context range;
- core and context token counts;
- the structural boundary used to end the core range;
- explicit leading and trailing overlap sizes.

Core ranges partition all meaningful source content in order. Context ranges may overlap, but overlap never changes identifiers or core ownership. Later stages should attribute evidence to core ranges and treat overlap as context only.

## Token accounting

`TokenCounter` exposes `count(text) -> int` plus a stable `identity` used by future cache keys and audit artifacts. Segmentation receives a counter directly and never branches on provider names.

The initial resolvers are:

- `TiktokenCounter` for OpenAI models recognized by `tiktoken`, with an explicit encoding fallback for unknown OpenAI model names.
- `ConservativeUtf8TokenCounter` for arbitrary Ollama or otherwise unsupported models. It estimates no fewer than one token per UTF-8 byte and therefore intentionally leaves substantial headroom. The configured segment budget remains an independent hard ceiling.
- Any future exact Ollama-family adapter can implement the same protocol without changing segmentation.

The fallback is an estimate, so its identity and exact-versus-estimated status must be visible. It favors safety over packing density; users can inject or later select an exact model-family counter to recover that capacity. No network request is allowed during counting. A zero or negative budget is invalid.

## Structure detection and splitting

The segmenter scans canonical text into contiguous structural blocks without rewriting their contents. Boundaries have this preference order:

1. Markdown-style headings and the sections they introduce.
2. Paragraph or list-block boundaries separated by blank lines.
3. Sentence boundaries inside an oversized block.
4. A token-safe fallback split inside an oversized sentence or indivisible block.

Packing is greedy and stable. It adds the next complete unit while the counter remains within the budget. If a unit does not fit, the segmenter recursively tries the next weaker boundary. The final fallback searches character offsets, then backs up to a Unicode-safe whitespace boundary when possible. Every emitted segment is recounted and must be within budget.

### Non-monotonic counters

A byte-pair encoder is not monotonic in text length: appending a character can *reduce* the token count, because it can change how the surrounding run merges. With `cl100k_base`, a 12-token budget over a long run of `a` characters is exceeded at 93 characters and satisfied again at 96. The instability is not bounded by the longest single token either, so no window makes a coarse search exact.

Two rules follow, and the implementation depends on both:

1. **Verify, never infer.** Every boundary a search returns is recounted against the budget before use, and so is any boundary later snapped to a structural or unit edge. A slice that fits is not evidence that a shorter slice of it fits.
2. **Under-packing is acceptable; over-packing is not.** A search returns a boundary that provably fits, not necessarily the largest one that would have. Segments may come in slightly under budget.

Counters that *are* monotonic — including the conservative UTF-8 estimator — can be searched with a plain binary search. Counters that are not must supply their own boundary searches, `fitting_prefix` and `fitting_suffix`, since only the counter knows its tokenizer; segmentation refuses a non-monotonic counter that supplies neither rather than guessing. Deriving anything from a tokenizer's vocabulary must use its public interface and tolerate reserved identifiers that have no byte mapping, which real encodings contain.

The algorithm never drops source text. Separators remain attached to one adjacent core range, so concatenating core slices in order reconstructs the canonical document except for outer whitespace removed by ingestion.

## Overlap

Overlap is configured in tokens and defaults to zero. When enabled, each segment after the first expands its context start backward from its core start up to the overlap budget, and each segment before the last expands its context end forward the same way. Expansion prefers sentence and paragraph boundaries and falls back to a token-safe character boundary. Neither direction reaches past the immediately adjacent core, so overlap never pulls in text from two segments away.

Leading context is resolved first, and trailing context receives only the room left over, so look-back continuity wins when a core leaves too little for both.

The combined context must still fit the segment budget. If the core alone consumes the budget, overlap is reduced to zero. Metadata records the resulting overlap rather than only the requested value.

## Prompt-injection boundary

Ingestion and segmentation treat every byte of source text as inert data. They do not parse instructions, execute markup, or interpolate source text into system prompts. Later generation stages must continue to delimit source content and maintain this trust boundary.

## Errors and invariants

Public failures use focused application exceptions:

- unreadable or undecodable input identifies the path and encoding;
- empty input raises `EmptySourceError`;
- invalid budgets or overlap raise configuration errors before processing;
- a counter that returns a negative value raises a token-accounting error;
- failure to make forward progress raises an internal segmentation error rather than looping.

The implementation asserts ordered, nonempty, nonoverlapping core ranges, resolvable context slices, stable IDs, and budget compliance.

## Testing

All tests are offline and use deterministic counters where exact boundaries matter. Coverage includes line-ending and whitespace normalization, Unicode and BOM input, empty input, headings, paragraphs, lists, sentences, oversized indivisible content, stable identifiers, exact offsets, overlap metadata, reconstruction, budget compliance, provider-specific counter selection, unknown model fallback, and injection-like source text.

Property-style cases should verify that segmenting the same canonical document twice yields identical results and that no input causes an infinite loop or a segment above budget.

## Delivery boundary

Issue #4 delivers domain models, ingestion, token accounting, segmentation, and their tests. The legacy workflow remains unchanged to preserve the existing CLI until issue #5 introduces structured leaf summaries. This keeps the PR independently verifiable and avoids temporarily replacing working behavior with raw segment output.
