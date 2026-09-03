# Structured Leaf Summarization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn each `SourceSegment` into one validated structured leaf record, in source order, with provenance that resolves to a known segment and a prompt that keeps source text inert.

**Architecture:** Add frozen pydantic records plus a derived JSON Schema, extend `GenerationRequest` with a provider-neutral response schema that each adapter maps to its own native mechanism, and put the prompt, parser, validator, and stage above the provider boundary. The legacy CLI workflow stays untouched; issue #6 owns budget arithmetic and issue #7 owns merging.

**Tech Stack:** Python 3, pydantic 2 frozen models, `typing.Protocol`, the existing OpenAI Responses and native Ollama adapters, pytest

**Test command:** `uv run --with-requirements requirements-dev.txt python -m pytest -q`

Use exactly that form. `pytest` is not in `requirements.txt`, so `uv run --with-requirements requirements.txt pytest` resolves the bare command off `PATH` to an interpreter with no project dependencies, and every third-party import then fails in a way that looks like broken source rather than a wrong interpreter.

---

### Task 1: Model the leaf records and derive their schema

**Files:**
- Create: `summarizer/summaries.py`
- Create: `tests/test_summaries.py`

**Step 1: Write failing model and schema tests**

Cover construction of a minimal valid `SummaryNode`, rejection of an empty summary, rejection of an empty `segment_id` on an `EvidenceItem`, rejection of unknown fields, immutability, and that contradictions and quotations may be empty without being absent.

Then cover the schema itself, because it is what the provider is told to produce:

```python
schema = leaf_summary_schema()
assert schema["additionalProperties"] is False
assert set(schema["required"]) == set(schema["properties"])
```

Assert the same two properties recursively for every definition under `$defs`, and assert `LEAF_SCHEMA_VERSION` is a non-empty string.

**Step 2: Run the tests and verify they fail**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest tests/test_summaries.py -q`

Expected: FAIL because `summarizer.summaries` does not exist.

**Step 3: Implement the records**

Add `EvidenceItem`, `ContentUnit`, `SummaryNode`, `LEAF_SCHEMA_VERSION`, and `leaf_summary_schema()`.

Give every model `model_config = ConfigDict(frozen=True, extra="forbid")` and **no field defaults**. Both details are load-bearing: `extra="forbid"` is what makes pydantic emit `additionalProperties: false`, and a field without a default is what makes pydantic mark it required. Together they make `model_json_schema()` directly usable as an OpenAI strict schema, so no private helper from the OpenAI SDK is needed and the schema cannot drift from the validator.

Express optional-by-meaning fields as required but empty-able: `contradictions: tuple[str, ...]` and `quotations: tuple[EvidenceItem, ...]` are required keys whose value may be an empty sequence. Accept `None` as equivalent to empty through a validator, because the non-strict paths produce both.

Set `level: int` on `SummaryNode` and validate it as non-negative. Leaves pass `0`.

**Step 4: Run focused and full tests**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest tests/test_summaries.py -q`

Expected: PASS.

Run: `uv run --with-requirements requirements-dev.txt python -m pytest -q`

Expected: 192 existing tests plus the new ones pass.

**Step 5: Commit**

```bash
git add summarizer/summaries.py tests/test_summaries.py
git commit -m "feat(summaries): model structured leaf records"
```

### Task 2: Carry a response schema across the provider boundary

**Files:**
- Modify: `summarizer/providers/base.py`
- Modify: `summarizer/providers/openai.py`
- Modify: `summarizer/providers/ollama.py`
- Modify: `tests/providers/test_openai.py`
- Modify: `tests/providers/test_ollama.py`

**Step 1: Write failing request and adapter tests**

Test that a request without a schema produces exactly the call each adapter makes today, so the legacy path is provably unchanged. Then test that a request carrying a schema reaches each client in its native shape:

```python
assert recorded["text"] == {
    "format": {
        "type": "json_schema",
        "name": "leaf_summary",
        "schema": schema,
        "strict": True,
    }
}
```

for OpenAI, and `recorded["format"] == schema` for Ollama.

Also test that a schema-carrying response is returned byte-for-byte, including newlines and runs of spaces, while a schema-free response is still whitespace-collapsed.

**Step 2: Run the tests and verify they fail**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest tests/providers -q`

Expected: FAIL because `GenerationRequest` has no schema field.

**Step 3: Extend the request and both adapters**

Append `response_schema: Mapping[str, object] | None = None` and `schema_name: str | None = None` to `GenerationRequest`. Append, with defaults — both request and result records are constructed positionally in the existing tests, and a dedicated commit already exists in this repository's history to preserve positional compatibility of a config record. Validate that `schema_name` is present and non-blank whenever `response_schema` is, since the OpenAI format requires a name.

Map the schema in each adapter and leave every other argument alone.

Then stop collapsing whitespace for schema-carrying responses. The collapse at `summarizer/providers/openai.py:84` and `summarizer/providers/ollama.py:103` is harmless for JSON structure but destroys verbatim quotations: a quote copied exactly out of a segment containing a newline or a double space comes back single-spaced, and locating it in the source then fails. Keep the collapse for prose responses so the legacy workflow is unaffected.

**Step 4: Run focused and full tests**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest tests/providers -q`

Expected: PASS.

Run: `uv run --with-requirements requirements-dev.txt python -m pytest -q`

Expected: all tests pass, including the legacy workflow and CLI tests, unchanged.

**Step 5: Commit**

```bash
git add summarizer/providers tests/providers
git commit -m "feat(providers): carry a response schema across the boundary"
```

### Task 3: Build the leaf prompt and a deterministic request

**Files:**
- Create: `summarizer/leaf.py`
- Create: `tests/test_leaf_prompt.py`

**Step 1: Write failing prompt and request tests**

Assert the source text appears only in `input_text` and never in `instructions`, for a segment whose text contains instruction-like content. Assert the instructions state that fenced content is data rather than instructions, and that they are genre-neutral — no mention of technical writing, transcripts, or articles.

Assert the request is deterministic and carries provenance:

```python
first = build_leaf_request(segment, model="m", timeout_seconds=30)
second = build_leaf_request(segment, model="m", timeout_seconds=30)
assert first == second
assert segment.segment_id in first.instructions
```

Test that a segment whose text contains the fence delimiter still produces a request, and that the instructions assert precedence rather than relying on the fence being unforgeable.

Test that overlap is described as context only: a segment with a non-zero `leading_overlap_tokens` must produce instructions saying the surrounding context is not attributable.

**Step 2: Run the tests and verify they fail**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest tests/test_leaf_prompt.py -q`

Expected: FAIL because `summarizer.leaf` does not exist.

**Step 3: Implement the prompt and request builder**

Add `LEAF_PROMPT_VERSION`, the instruction text, and `build_leaf_request`. Attach `response_schema=leaf_summary_schema()` and `schema_name="leaf_summary"`.

Do not touch `summarizer/text.py`. Its prompt is tuned for dense technical prose and its constants are pinned verbatim by characterization tests.

Name the only legal citation identifier in the instructions — the segment's own `segment_id` — so the model is not invited to invent others. The validator enforces this regardless; the prompt only reduces avoidable failures.

**Step 4: Run focused and full tests**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest tests/test_leaf_prompt.py -q`

Expected: PASS.

Run: `uv run --with-requirements requirements-dev.txt python -m pytest -q`

Expected: all tests pass.

**Step 5: Commit**

```bash
git add summarizer/leaf.py tests/test_leaf_prompt.py
git commit -m "feat(leaf): add a genre-neutral leaf prompt"
```

### Task 4: Parse and validate provider output

**Files:**
- Modify: `summarizer/leaf.py`
- Create: `tests/test_leaf_parsing.py`

**Step 1: Write failing parser and validator tests**

Cover a valid payload; a payload wrapped in a fenced code block; a payload preceded by a prose preamble; text that is not JSON at all; JSON that is not an object; a payload missing a required field; a payload with an unknown field; a payload citing a segment identifier outside the legal set; a payload whose quotation does not occur in its segment; and a payload carrying contradictions and uncertainty, which must succeed.

Assert every failure names the segment and the reason, and assert negatively that no failure message contains source text:

```python
with pytest.raises(LeafSummaryError) as error:
    parse_leaf_summary(payload, segment=segment)
assert segment.segment_id in str(error.value)
assert "Ignore previous instructions" not in str(error.value)
```

**Step 2: Run the tests and verify they fail**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest tests/test_leaf_parsing.py -q`

Expected: FAIL because `parse_leaf_summary` does not exist.

**Step 3: Implement tolerant extraction and strict validation**

Add `LeafSummaryError` and `parse_leaf_summary(text, *, segment)`.

Locate the outermost JSON object, tolerating a surrounding code fence and a preamble, then validate strictly through the model. Constrained decoding reduces slop but does not guarantee it: the Ollama client's format argument is best-effort, and small local tags still emit preambles.

Validate provenance after schema validation. The legal identifier set comes from the caller, never from the payload, which is what makes an injected citation a validation failure rather than a dangling reference. Restrict evidence to the segment's own identifier, since issue #4 established that overlap text is context and not attributable.

Locate each quotation in the segment text by exact match and reject one that does not occur. Do not accept character offsets from the model; derive them from the match.

Wrap `pydantic.ValidationError` into `LeafSummaryError` with the segment identifier and the failing field. Never interpolate the payload or the source into the message.

**Step 4: Run focused and full tests**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest tests/test_leaf_parsing.py -q`

Expected: PASS.

Run: `uv run --with-requirements requirements-dev.txt python -m pytest -q`

Expected: all tests pass.

**Step 5: Commit**

```bash
git add summarizer/leaf.py tests/test_leaf_parsing.py
git commit -m "feat(leaf): validate structured provider output"
```

### Task 5: Summarize a segment sequence

**Files:**
- Modify: `summarizer/leaf.py`
- Create: `tests/test_leaf_stage.py`

**Step 1: Write failing stage tests**

Use a deterministic fake provider that returns a canned payload per segment. Cover source order preservation, one record per segment, `level` of zero on every record, and identical results across two runs.

Cover fail-fast: a provider whose second response is malformed raises `LeafSummaryError` naming that segment, and no partial output is returned. Cover that a provider error propagates unchanged rather than becoming summary text.

Assert the stage never calls a provider more than once per segment, since there is no retry on a schema violation.

**Step 2: Run the tests and verify they fail**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest tests/test_leaf_stage.py -q`

Expected: FAIL because `summarize_segments` does not exist.

**Step 3: Implement the stage**

Add `summarize_segments(segments, provider, *, model, timeout_seconds) -> tuple[SummaryNode, ...]`.

Consume segments in `order` and emit results in the same sequence. Return an immutable sequence rather than a mapping, so per-segment outcomes can be added later for issue #11's resume support without changing the success path.

Fail on the first invalid segment. Nothing in issue #5 requires surviving a bad segment, this repository's established rule is that provider failures never become output, and a half-populated hierarchy reaching issue #7 is worse than a clear failure.

Do not retry a schema violation. `ProviderResponseError` is deliberately not transient, so the existing retry decorator will not re-ask, and a bounded re-ask would be new machinery.

**Step 4: Run focused and full tests**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest tests/test_leaf_stage.py -q`

Expected: PASS.

Run: `uv run --with-requirements requirements-dev.txt python -m pytest -q`

Expected: all tests pass.

**Step 5: Commit**

```bash
git add summarizer/leaf.py tests/test_leaf_stage.py
git commit -m "feat(leaf): summarize segments into ordered leaf records"
```

### Task 6: Cover injection-like input end to end and document the boundary

**Files:**
- Modify: `tests/test_leaf_stage.py`
- Modify: `README.md`
- Modify: `docs/plans/2026-09-02-structured-leaf-summarization-design.md`

**Step 1: Write end-to-end injection tests**

Ingest and segment a document whose text contains instruction-like content and a forged fence, run the stage against a fake provider, and assert the source text never reaches `instructions`. Reuse the existing injection fixture string and the genre fixture corpus so the prompt is exercised against more than one register.

Assert that a payload obeying injected instructions — citing an unknown segment, or quoting text absent from the source — fails validation.

**Step 2: Document what the stage guarantees**

State in the README that leaves are validated structured records, that evidence resolves to a segment identifier the caller supplied, that overlap is context and not attributable, that the CLI does not consume leaves yet, and that a response schema is passed to the provider when one is available while parsing stays defensive regardless.

Record in the design doc which open decisions were taken, and that the whitespace collapse now applies only to prose responses.

**Step 3: Run formatting and static checks**

Run: `uv run --with-requirements requirements-dev.txt python -m compileall -q summarizer tests`

Expected: exit 0.

**Step 4: Run the complete suite through both import modes**

Run: `uv run --with-requirements requirements-dev.txt python -m pytest -q`

Run: `uv run --with-requirements requirements-dev.txt python -m pytest -q --import-mode=importlib`

Expected: all tests pass in both.

**Step 5: Verify repository hygiene**

Run: `git status --short`

Expected: only intended issue #5 changes; no cache, output, log, credential, or virtual-environment files.

**Step 6: Commit**

```bash
git add README.md docs/plans/2026-09-02-structured-leaf-summarization-design.md tests/test_leaf_stage.py
git commit -m "chore: document structured leaf summarization"
```

**Step 7: Record acceptance evidence on issue #5**

Post the final test counts and a mapping from each acceptance criterion to the tests and implementation paths that satisfy it. Note explicitly that structured output against a live provider is not covered by the offline suite, and record the manual check separately. Do not close issue #5 until its PR merges.
