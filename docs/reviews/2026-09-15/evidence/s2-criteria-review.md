# S2 Subsystem Design Review: Hierarchy, Merge, Grounding, Provenance

**Commit:** 301cc4d  
**Review Date:** 2026-09-16  
**Scope:** Static code and design document analysis (no tests run)

---

## File Responsibilities Summary

### `summarizer/hierarchy.py` (641 lines)
Orchestrates multi-level tree reduction via balanced grouping. Computes merged node IDs deterministically, measures fanout from child serialization, handles provenance as a computed union (excluded from merge payloads), validates grounding boundaries, and sequences concurrent or serial merge work through `BoundedScheduler`. Core: `build_hierarchy()`, `merge_fanout()`, `group_children()`, `TreeNode`, `MergeGrounding`.

### `summarizer/merge.py` (304 lines)
Builds and parses merge requests. Constructs deterministic delimiters via SHA256 fencing, serializes children without provenance, builds complete merge request payloads with separated child and source blocks, validates responses, and derives provenance from model output and preserved union. Core: `build_merge_request()`, `parse_merged_summary()`, `serialize_child()`, `measure_merge_overhead()`.

### `summarizer/grounding.py` (141 lines)
Selects authoritative source passages under token budget. Prioritizes evidence by type (contradictions > qualifications/uncertain > quotations > content-unit evidence > provenance), packs complete `SourcePassage` objects, and fails early on mandatory evidence overflow. Returns `GroundingSelection` with selected/omitted IDs. Core: `select_source_passages()`, `_candidates()`.

### `summarizer/leaf.py` (494 lines)
Validates leaf summaries and builds leaf requests. The generalized `validate_provenance()` now accepts caller-supplied legal ID mapping and quotation source subset. Checks references against legal set, quotations against cited segment's text alone (not concatenated), and enforces evidence on all content units and grounded annotations. Core: `validate_provenance()`, `derive_provenance()`, `parse_leaf_summary()`.

### `summarizer/summaries.py` (165 lines)
Defines immutable record types: `ContentKind`, `EvidenceItem`, `ContentUnit`, `GroundedAnnotation`, `SummaryNode`. Each carries evidence pointers and quotations. Schema enforces max 5 quotes per node, max 500 chars per quote. Core: `SummaryNode`, validation fields.

### `summarizer/audit.py` (1446 lines)
Builds versioned audit artifacts. Stores grounding metadata per merge (`AuditGroundingSelection` with selected_ids, omitted_ids, reserve_tokens, request_capacity_tokens). Distinguishes `AuditNodeV4` (with grounding) from `AuditNode` (without). Validates end-to-end tree reachability, citation order, segment references. Core: `build_audit_artifact()`, `_audit_node()`, `AuditArtifactV4`.

---

## Domain Models Correspondence

### Required Models (from issues #5–#29)
| Domain Concept | Implementation | Status |
|---|---|---|
| SourceSegment | `summarizer/segmentation.py:SourceSegment` (external) | ✓ Accepted |
| EvidenceItem | `summarizer/summaries.py:EvidenceItem` | ✓ Present |
| ContentUnit | `summarizer/summaries.py:ContentUnit` | ✓ Present |
| SummaryNode | `summarizer/summaries.py:SummaryNode` | ✓ Present |
| TreeNode | `summarizer/hierarchy.py:TreeNode` | ✓ Present |

All required domain models are present and frozen (immutable). `EvidenceItem` and `ContentUnit` now properly carry quotations. `TreeNode` adds tree structure (node_id, level, order, children, covered_segments, grounding).

---

## Leaf Summary Structure (Level 0)

### Data Carriers
- **summary** (str): Required, non-blank  
- **content_units** (tuple[ContentUnit, ...]): Each has evidence, qualification, uncertain flag  
- **entities** (tuple[str, ...]): Names and concepts  
- **qualifications** (tuple[GroundedAnnotation, ...]): Hedges with evidence pointers  
- **contradictions** (tuple[GroundedAnnotation, ...]): Conflicting claims with evidence  
- **quotations** (tuple[EvidenceItem, ...]): Salient pull-quotes (max 5, max 500 chars each)  
- **provenance** (tuple[str, ...]): Segment IDs supporting the summary  
- **level** (int): Must be 0 for leaf

### Validation (`validate_provenance`)
**Location:** `summarizer/leaf.py:307–394`

Checks:
- Every content unit has non-empty evidence (line 341–344)
- Every grounded qualification/contradiction has non-empty evidence (lines 346–351)
- All cited segment IDs resolve to legal set (line 354–359)
- Quotations occur in the segment they cite, checking only the cited segment's core text (lines 386–393)
- Optional: provenance is non-empty (line 361–364, controlled by `require_provenance` flag)

**Finding:** Validation correctly isolates quotations to individual segment text (line 389: `available_quotation_sources[segment_id]`), not concatenated text. This blocks cross-segment attribution.

---

## Hierarchy Building: Ordered Nodes and Deterministic Grouping

**Location:** `summarizer/hierarchy.py:230–483`

### Leaf Preparation (Level 0)
- Each input leaf is wrapped in `TreeNode` with node_id `L0N{order:04d}` (lines 294–305)
- Provenance is derived via `derive_provenance()` on validated leaf (line 287)
- Segment coverage tuples are extracted and validated to exist in attributable text (lines 273–281)

### Fanout Calculation
**Location:** `summarizer/hierarchy.py:154–204`

```python
def merge_fanout(children, counter, *, capacity, ceiling=None):
    # Measure serialized size of largest child
    costs = [measure_child_tokens(child, counter) for child in children]
    largest = max(costs)
    measured = capacity // largest if largest else len(children)
    
    # Fail if pair cannot fit
    if measured < 2:
        raise BudgetError(...)
    
    # Apply ceiling if present
    if ceiling is not None and ceiling < measured:
        return ceiling, reason
    return measured, reason
```

**Finding:** Fanout is computed from the largest child, not average. This ensures no group is assembled that only fits on average — a real failure mode. Minimum fanout is 2; a capacity holding only 1 child rejects with arithmetic.

### Grouping
**Location:** `summarizer/hierarchy.py:207–227`

```python
def group_children(count, fanout):
    groups = -(-count // fanout)  # ceiling division
    base, remainder = divmod(count, groups)
    # Distribute remainder to earlier groups
    for position in range(groups):
        size = base + (1 if position < remainder else 0)
```

**Finding:** Grouping is **balanced**, not greedy. A greedy pack would leave a small final group, compressing end of document more than beginning (violates epic requirement). Balanced groups preserve even compression.

### Merge Preparation and Execution
**Location:** `summarizer/hierarchy.py:486–640`

1. **Coverage union (line 506–512):** Covered segments deduplicated in document order (using `dict.fromkeys()`)
2. **Legal set (line 521):** Mapped from attributable text; validates all are present
3. **Grounding selection (line 549–555):** Passes to `select_source_passages()` with selection cost calculated via full merge request measurement (lines 523–547)
4. **Preserved provenance (line 557–564):** Union of child provenance minus grounded passages, kept as fallback for ungrounded claims
5. **Request building (line 565–572):** Full merge request with level, passages, source_id
6. **Descriptor creation (line 581–605):** For cache coordinator, records merge inputs and grounding policy

**Finding:** The flow correctly:
- Measures the complete merge request including grounding (lines 574, 636), not child size alone
- Retries with narrower fanout if adaptive grounding exhausts budget (lines 342–400)
- Handles passthrough nodes (single-child groups, lines 357–373) without provider calls
- Keeps both reserve_tokens and request_capacity_tokens to distinguish fixed vs. adaptive budget (line 616–622)

---

## Grounding: Merge Retrieves and Validates Original Passages

### Selection Algorithm
**Location:** `summarizer/grounding.py:86–140`

1. **Candidate extraction (line 104):** Iterates children in order, collecting evidence IDs with mandatory/optional flag
   - Contradictions (mandatory)
   - Qualifications + uncertain content (mandatory)
   - Quotations (optional)
   - Content-unit evidence (optional)
   - Provenance fallback (optional)
   - Deduplicates via ordered dict, first-occurrence-wins (lines 77–83)

2. **Packing (lines 105–127):**
   - Builds passage objects incrementally (line 111)
   - Measures cost of tentative set with optional `selection_cost` callable (lines 113–116)
   - Accepts if cost ≤ policy.max_tokens
   - Rejects optional passages; fails early on mandatory overflow

3. **Result (lines 135–140):** Returns `GroundingSelection` with passages, selected_ids, omitted_ids

**Finding:** Selection is deterministic (candidate order fixed, no randomization) and conservative (mandatory evidence fails hard, optional degrades gracefully). Omitted IDs are tracked for provenance narrowing later.

### Request Structure
**Location:** `summarizer/merge.py:191–254`

```
-----BEGIN ...-----
-----GENERATED-CHILD-SUMMARIES-BEGIN-----
[SUMMARY-BEGIN ... SUMMARY-END (no provenance field)]
...
-----GENERATED-CHILD-SUMMARIES-END-----
-----AUTHORITATIVE-ORIGINAL-SOURCE-PASSAGES-BEGIN-----
[SOURCE-PASSAGE-BEGIN ... SOURCE-PASSAGE-END]
...
-----AUTHORITATIVE-ORIGINAL-SOURCE-PASSAGES-END-----
-----END-----
```

- Children serialized without provenance field (line 112): `payload.pop("provenance", None)`
- Fences derived deterministically from `MERGE_PROMPT_VERSION:source_id:level:label:ordinal` (lines 85–98)
- Passages fenced separately and labeled by ordinal

**Finding:** Separation is explicit and deterministically fenced. Instructions (line 33–82) state:
- Passages are authoritative, children provisional
- Omitted references from children remain citable (line 44–48 in design)
- No invented identifiers or causal links
- Quotations copied character-for-character from children or authoritative source

### Reference Validation
**Location:** `summarizer/merge.py:257–303` (`parse_merged_summary`)

1. **Response parsing (lines 269–273):** Extract JSON object, sanitize errors
2. **Schema validation (lines 276–281):** `SummaryNode.model_validate(payload)`
3. **Level check (lines 283–285):** Must match expected level
4. **Provenance validation (lines 288–293):** Call `validate_provenance()` with:
   - `legal`: full union of covered segment IDs
   - `quotation_sources`: only grounded passages (optional; if None, full legal set is checked)
5. **Provenance replacement (lines 294–302):** Computed union of preserved + derived provenance in source order

**Finding:** The validator receives:
- `legal`: all covered segments (full authoritatively sourced set)
- `quotation_sources` (optional): only selected passages; quotation checks confined to this subset
- `preserved_provenance`: union of child provenance minus grounded IDs

This allows a child's ungrounded reference to be cited in merged output without being shown to the model (safe), while enforcing quotations only against passages actually supplied.

---

## Provenance: Union, Narrow, Deterministically Traceable

### Union Construction
**Location:** `summarizer/leaf.py:396–404` (`derive_provenance`)

```python
def derive_provenance(node, *, source_order):
    referenced = set(node.provenance)
    for unit in node.content_units:
        referenced.update(item.segment_id for item in unit.evidence)
    for annotation in (*node.qualifications, *node.contradictions):
        referenced.update(item.segment_id for item in annotation.evidence)
    referenced.update(item.segment_id for item in node.quotations)
    return tuple(identifier for identifier in source_order if identifier in referenced)
```

Collects all referenced IDs from provenance, content-unit evidence, grounded annotations, and quotations, then canonicalizes to source order.

### Preserved References
**Location:** `summarizer/hierarchy.py:557–564`

```python
preserved_provenance = tuple(
    dict.fromkeys(
        identifier
        for member in members
        for identifier in member.summary.provenance
        if identifier not in grounded
    )
)
```

Ungrounded child references are collected and retained as a separate tuple. If a child cites a segment that was not selected for grounding, that reference stays in the merge output (validated later against full legal set, not subject to quotation check).

### Narrow + Replace
**Location:** `summarizer/merge.py:294–302`

```python
canonical_order = source_order or tuple(legal)
retained = set(preserved_provenance)
retained.update(derive_provenance(node, source_order=canonical_order))
return node.model_copy(
    update={
        "provenance": tuple(
            identifier for identifier in canonical_order if identifier in retained
        )
    }
)
```

Final provenance is:
1. All preserved (ungrounded child) references
2. Plus all derived references from the response
3. Ordered by source order (preserves document flow)

**Finding:** Provenance is intentionally kept representable as a conflict. A child carrying a claim about segment X with evidence from X, when X is not grounded, produces an entry in preserved_provenance. The merge response may cite X or not; both choices are allowed. This preserves traceability while allowing narrowing by issue #8.

---

## Audit: Merge Node Lists Grounding Segment IDs

### Grounding Metadata
**Location:** `summarizer/audit.py:435–460` (`AuditGroundingSelection`, `AuditNodeV4`)

```python
class AuditGroundingSelection:
    selected_ids: tuple[str, ...]        # IDs with passages supplied
    omitted_ids: tuple[str, ...]          # IDs candidates but not selected
    reserve_tokens: int | None = None     # Fixed budget (mutually exclusive)
    request_capacity_tokens: int | None = None  # Adaptive budget
    omission_reason: Literal["budget"]
```

Stored per node only for executed merges (len(children) > 1). Leaf and passthrough nodes have `grounding=None`.

### Node Audit Structure
**Location:** `summarizer/audit.py:839–914` (`_audit_node`)

```python
values = {
    "node_id": node.node_id,
    "level": node.level,
    "order": node.order,
    "children": node.children,
    "covered_segments": node.covered_segments,
    "summary": audit_summary,
}
if include_grounding:  # AuditNodeV4
    grounding=(
        AuditGroundingSelection(
            selected_ids=node.grounding.selection.selected_ids,
            omitted_ids=node.grounding.selection.omitted_ids,
            reserve_tokens=node.grounding.reserve_tokens,
            request_capacity_tokens=node.grounding.request_capacity_tokens,
            omission_reason="budget",
        )
        if node.grounding is not None
        else None
    )
```

### Document Order Preservation
**Location:** `summarizer/audit.py:490–500`

```python
@model_validator(mode="after")
def _links_resolve(self) -> _AuditArtifactBase:
    # ... validation ...
    if tuple(segment.segment_id for segment in self.source_segments) != tuple(
        segment.segment_id
        for segment in sorted(self.source_segments, key=lambda item: item.order)
    ):
        raise ValueError("source segments must be in source order")
```

Asserts segments remain in source order in the audit artifact.

**Finding:** Audit correctly records selected vs. omitted IDs per merge and validates:
- Merge nodes have grounding; leaf/passthrough do not (lines 514–518)
- Selected and omitted IDs are disjoint and both resolve to covered segments (lines 520–531)
- Budget mode (fixed reserve vs. adaptive capacity) is recorded (line 532–535)

---

## Quotation Limits: Enforced Per Leaf and Merge

### Schema Validation
**Location:** `summarizer/summaries.py:154–159`

```python
@field_validator("quotations")
@classmethod
def _validate_quotation_count(cls, value: tuple[EvidenceItem, ...]):
    if len(value) > MAX_QUOTATIONS_PER_NODE:
        raise ValueError(f"must not exceed {MAX_QUOTATIONS_PER_NODE} quotations")
    return value
```

Per-node limit: max 5 quotations.

### Per-Quote Limit
**Location:** `summarizer/summaries.py:79–93`

```python
@field_validator("quote")
@classmethod
def _validate_quote(cls, value: str | None) -> str | None:
    if value is not None:
        if not value.strip():
            raise ValueError("quote must be null rather than blank")
        if len(value) > MAX_QUOTE_CHARS:
            raise ValueError(f"quote must not exceed {MAX_QUOTE_CHARS} characters")
    return value
```

Per-quote limit: max 500 characters.

### Enforcement in Prompts
- Leaf prompt (line 57–58 in `summarizer/leaf.py`): states max 500 chars, max 5 quotations
- Merge prompt (line 67–69 in `summarizer/merge.py`): "Keep at most {max_quotations} quotations, each at most {max_quote_chars} characters"

**Finding:** Limits are enforced at the `EvidenceItem` level (per quote) and `SummaryNode` level (per node). Merge prompt instructs narrowing if inherited quotations exceed limits, but validation enforces the hard limit on output.

---

## Candidate Findings

### Finding 1: Provenance Exclusion from Merge Payload
**Claim:** Provenance is deliberately excluded from serialized children sent to merge, computed locally instead.  
**Evidence:** `summarizer/merge.py:111–113` (`serialize_child`), design doc 2026-09-03-hierarchical-merging, line 43–45.  
**Criterion:** Provenance (deterministic union, representable conflicts)  
**Assessment:** By design. Provenance is excluded to prevent 4 tokens-per-ID growth that would halt recursion at level 3–4. The model emits a provenance field that is validated and discarded (lines 297–302 in merge.py). This is sound architectural decision but **depends critically on validator never trusting model provenance**. If a caller invoked without the replace step, traceability would vanish.

### Finding 2: Grounding Selection Cost Measurement
**Claim:** Grounding cost is measured inclusively (full merge request + passages) not just passages alone, allowing adaptive fanout retry.  
**Evidence:** `summarizer/hierarchy.py:523–547`, lines 524–536 compute full request cost before accepting.  
**Criterion:** Grounding (merge retrieves original passages, evidence references resolve)  
**Assessment:** Correct. The `selection_cost` callable receives tentative passage set and returns the cost of the full request with those passages, allowing `select_source_passages` to make bounded-capacity decisions that the downstream merge can consume.

### Finding 3: Quotation Checks Isolated to Cited Segment
**Claim:** A quotation is checked against only the core text of the segment it cites, not concatenated passages.  
**Evidence:** `summarizer/leaf.py:386–393`, line 389: `available_quotation_sources[segment_id]`.  
**Criterion:** Grounding (valid refs fail validation)  
**Assessment:** Correct. This blocks a quote from straddling two segments and being attributed to a neighbor. The check applies only to passages actually supplied to the model (`quotation_sources` parameter, defaulting to full legal set for leaf).

### Finding 4: Adaptive vs. Fixed Grounding Budget
**Claim:** Two budget modes exist: fixed reserve (explicit `GroundingPolicy`) and adaptive (computed per-request).  
**Evidence:** `summarizer/hierarchy.py:268–270`, 326–337, 386–391, 616–622. `MergeGrounding` has `reserve_tokens XOR request_capacity_tokens`.  
**Criterion:** Hierarchy (fanout deterministic, context budget respected)  
**Assessment:** Correct. Fixed policy allocates a pre-determined token pool. Adaptive mode measures merge overhead + candidates, sizes fanout to fit children, then measures the request with actual passages. If passages don't fit adaptive budget, fanout is narrowed (lines 342–400). This is well-designed but the distinction must be tracked in audit for reproducibility.

### Finding 5: Level Assertion After Merge
**Claim:** Node count is asserted to strictly fall at each level.  
**Evidence:** `summarizer/hierarchy.py:459–463`.  
**Criterion:** Hierarchy (fanout ≥ 2 guarantees progress)  
**Assessment:** Correct. Given fanout ≥ 2, the count must fall. The assertion is defensive but appropriate, and would catch bugs in grouping or passthrough logic.

### Finding 6: Deterministic Fencing and Ordering
**Claim:** Delimiters are deterministic via SHA256, and node ordering is explicit (not list position).  
**Evidence:** `summarizer/merge.py:85–98`, `summarizer/hierarchy.py:105–109` (TreeNode.order field).  
**Criterion:** Hierarchy (deterministic grouping)  
**Assessment:** Correct. Fences derived from hash of version+source_id+level+label+ordinal ensure bit-identical requests across runs. Explicit order fields prevent concurrency from reordering nodes invisibly. Design doc acknowledges this keeps the door open for concurrency.

### Finding 7: Missing Schema Version for Merge
**Claim:** `LEAF_SCHEMA_VERSION = "leaf/2"` is defined, but no distinct merge schema version constant exists.  
**Evidence:** `summarizer/merge.py:29` defines `MERGE_PROMPT_VERSION` but not a schema version.  
**Criterion:** Grounding (merge retrieves passages, provenance narrowing)  
**Assessment:** **Candidate concern.** Merge uses the same `leaf_summary_schema()` (line 252) as leaves, which is correct (SummaryNode is reused). But if the audit descriptor records schema version (line 586: `schema_version=LEAF_SCHEMA_VERSION`), both leaf and merge are tagged with `leaf/2`. If a future change modifies the schema, the descriptor won't distinguish which stage changed it. Not a bug in this commit, but a gap for issue #27 (audit).

### Finding 8: Preserved Provenance Tuple May Contain Duplicates Before Merge
**Claim:** Child provenance is unioned with deduplication via `dict.fromkeys()` at line 558 in hierarchy.py.  
**Evidence:** `summarizer/hierarchy.py:557–564`.  
**Criterion:** Provenance (deterministic traceability)  
**Assessment:** Correct. Deduplication is applied, preserving document order. The comment acknowledges this is for issue #8's narrowing step.

### Finding 9: Audit Grounding Validation
**Claim:** Audit enforces grounding is present only for executed merges and validates selected+omitted resolve to coverage.  
**Evidence:** `summarizer/audit.py:513–531`.  
**Criterion:** Audit (merge node lists grounded segment ids and omitted ids)  
**Assessment:** Correct. The validator checks:
- Merges (children > 1) have grounding; leaves/passthrough (≤1 child) do not (lines 515–518)
- Selected + omitted IDs are disjoint (lines 520–527)
- Both resolve to covered_segments (line 528–530)
- Budget mode (reserve_tokens XOR request_capacity_tokens) is complete (line 532–535)

### Finding 10: Legal Set Passed to Merge Grounding
**Claim:** The full legal set (all covered segments) is passed to `validate_provenance` for merge, not the grounded set.  
**Evidence:** `summarizer/merge.py:288–292`, `summarizer/hierarchy.py:521`.  
**Criterion:** Grounding (evidence references resolve to known segments)  
**Assessment:** Correct. The legal set is the union of all covered segments, and this is what allows a child's ungrounded reference to remain citable. The quotation check (optional `quotation_sources` parameter) is what restricts quote validation to supplied passages.

---

## Limitations of This Pass

1. **No runtime verification:** This review does not run tests, execute merges, or check that grounding actually corrects misleading summaries. The prompt instructions are read but not validated against actual model behavior.

2. **No budget arithmetic verification:** Token counting is assumed correct; not independently validated against the tokenizer or provider models.

3. **No concurrent execution:** The code paths for `BoundedScheduler` and cache coordination are present but not traced for thread safety or ordering guarantees.

4. **No end-to-end flow:** The handoff between hierarchy, grounding, audit, and external verification phases is assumed correct but not traced in detail.

5. **Audit descriptor input coverage:** Descriptor construction at lines 587–600 in hierarchy.py is not exhaustively checked against actual cache usage.

---

## Summary Assessment

**S2 Architecture:** **Yes, appears sound**

### Strengths

1. **Domain models are clean and immutable.** `EvidenceItem`, `ContentUnit`, `SummaryNode`, `TreeNode` are well-separated frozen dataclasses with clear responsibilities.

2. **Provenance handling is sophisticated and deliberate.** Computed locally to avoid payload bloat, preserved as a representable conflict set, narrowable downstream. The design doc articulates the constraint (provenance grows 4 tokens per ID) and the solution (exclude from payload).

3. **Grounding is deterministic and conservative.** Candidate selection follows priority order (contradictions > qualifications > quotations > content evidence > fallback), and packing fails hard on mandatory overflow but degrades gracefully on optional.

4. **Validation is injection-hardened.** The legal set never comes from model output. Quotation checks are isolated to individual segment cores. Merged responses have their provenance replaced by computed union.

5. **Fanout guarantees forward progress.** Minimum fanout is 2; a capacity holding one child fails with arithmetic. Grouping is balanced to avoid ragged final groups.

6. **Audit captures grounding decisions.** Per-merge selected/omitted IDs and budget mode (fixed vs. adaptive) are recorded and validated for reachability.

### Gaps (Not Defects)

1. **Merge schema versioning:** Merge uses `LEAF_SCHEMA_VERSION` in audit descriptors, so schema changes cannot be attributed to a specific stage. Acceptable for now; flag for issue #27.

2. **Model provenance handling not tested in this pass:** The response's provenance field is validated then discarded. This is correct by design, but relies on the caller invoking the replace step. A caller that omitted lines 297–302 would silently lose traceability.

3. **Adaptive grounding retry logic is complex:** Lines 342–400 in hierarchy.py retry with narrower fanout if passages overflow. This is correct but the condition (adaptive_grounding and fanout > 2) is tightly coupled; clarifying comments are sparse.

4. **Preserved provenance semantics not explicit in code:** The tuple is collected and passed but not named distinctly in function signatures or type hints until used in parse_merged_summary. A domain type alias would improve clarity.

### Correspondence to Acceptance Criteria

| Criterion | Evidence | Status |
|---|---|---|
| Domain models (SourceSegment, EvidenceItem, ContentUnit, SummaryNode) | summarizer/summaries.py, hierarchy.py | ✓ Present |
| Leaf structure (summary, content units, segment refs, entities, qualifications, contradictions, quotations) | summarizer/summaries.py:119–159, leaf.py validation | ✓ Present |
| Hierarchy (ordered nodes, deterministic grouping) | hierarchy.py:207–227, 154–204 | ✓ Present |
| Grounding (merge retrieves passages, evidence resolves) | grounding.py:86–140, merge.py:191–254, hierarchy.py:523–547 | ✓ Present |
| Provenance (union, narrow, representable conflicts, traceability) | leaf.py:396–404, merge.py:294–302, hierarchy.py:557–564 | ✓ Present |
| Audit (merge node lists grounding segment ids + omitted ids, order preserved) | audit.py:435–460, 513–531, 490–500 | ✓ Present |
| Quotation limits (length & count per leaf/merge) | summaries.py:79–93, 154–159, leaf.py:57–58, merge.py:67–69 | ✓ Present |

---

## Recommendation

The S2 subsystem architecture is **sound and ready for integration testing.** All required domain models, validation rules, and audit structures are in place. The central design decision (compute provenance locally, exclude from merge payload) is well-motivated and correctly implemented.

Proceed to:
1. Execute hierarchy integration tests (multi-level tree building)
2. Verify grounding actually corrects misleading child summaries (against live model)
3. Validate audit artifact round-trip and reachability checks
4. Benchmark provenance union growth and fanout determination on realistic documents
