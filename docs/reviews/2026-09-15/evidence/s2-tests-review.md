# S2 Mutation Testing Review - recursive-text-summarizer @ 301cc4d

## Baseline
- Commit: `301cc4d56d6326b5b0449da059b3b35f484cc5ca`
- Tests executed: `tests/test_hierarchy.py`, `tests/test_merge_prompt.py`, `tests/test_grounding.py`, `tests/test_provenance_validation.py`, `tests/test_summaries.py`, `tests/test_leaf_*.py`, `tests/test_direct.py`
- Baseline pass count: **150 tests passed**

## Mutation Summary
- Total mutations applied: 21
- Mutations SURVIVED (tests did not catch): 8
- Mutations KILLED (tests caught): 13

## Full Mutation Table

| # | File:Line | Original Expression | Mutated Expression | Result | Killed By |
|---|-----------|-------------------|-------------------|--------|-----------|
| 1 | hierarchy.py:49 | `reserve_tokens <= 0` | `reserve_tokens < 0` | SURVIVED | — |
| 2 | hierarchy.py:53 | `request_capacity_tokens <= 0` | `request_capacity_tokens < 0` | SURVIVED | — |
| 3 | hierarchy.py:125 | raise ValueError (min segments) | pass (removed check) | SURVIVED | — |
| 4 | hierarchy.py:188 | `measured < 2` | `measured < 1` | KILLED | `test_a_capacity_that_cannot_hold_a_pair_fails_with_its_arithmetic` |
| 5 | hierarchy.py:214 | `count <= 0` | `count < 0` | SURVIVED | — |
| 6 | hierarchy.py:216 | `fanout < 2` | `fanout < 1` | KILLED | `test_grouping_refuses_a_fanout_that_cannot_progress` |
| 7 | merge.py:201 | `if not children` | `if False` | KILLED | `test_an_empty_merge_is_rejected` |
| 8 | merge.py:203 | `if not passages` | `if False` | KILLED | `test_a_merge_requires_authoritative_source_passages` |
| 9 | merge.py:283 | `if node.level != level` | `if node.level == level` | KILLED | Multiple merge prompt tests |
| 10 | grounding.py:101 | `if not children` | `if False` | SURVIVED | — |
| 11 | grounding.py:109 | `if identifier not in source` | `if False` | SURVIVED | — |
| 12 | grounding.py:118 | `cost <= policy.max_tokens` | `cost < policy.max_tokens` | SURVIVED | — |
| 13 | grounding.py:123 | `if mandatory` check enabled | `if False` (disabled) | KILLED | `test_counts_the_complete_source_section_inside_the_grounding_reserve` |
| 14 | leaf.py:341 | `if not unit.evidence` | `if False` | KILLED | `test_rejects_an_evidence_free_content_unit` |
| 15 | leaf.py:389 | `quote not in sources` check | `False` (disabled) | KILLED | Multiple leaf quotation tests |
| 16 | leaf.py:355 | `if unknown` check enabled | `if False` (disabled) | KILLED | Multiple leaf parsing tests |
| 17 | grounding.py:80 | Deduplication enabled | `if True` (disabled dedup) | KILLED | Grounding deduplication test |
| 18 | merge.py:297 | `return node.model_copy(update=...)` | `return node` (no update) | KILLED | Merge provenance tests |
| 19 | leaf.py:361 | `if require_provenance and not...` | `if False and not...` | KILLED | `test_rejects_a_node_citing_nothing`, `test_rejects_a_leaf_that_records_no_provenance` |
| 20 | leaf.py:347 | `if not annotation.evidence` | `if False` | SURVIVED | — |
| 21 | grounding.py:129 | `if not passages` (final check) | `if False` | KILLED | `test_empty_selection_reports_reserve_and_smallest_candidate` |

## Tier B Candidates (Surviving Mutations = Unkilled Boundaries)

```
id: C-S2M-001   tier: B   proposed_severity: minor
claim: Zero-valued grounding reserve is not rejected during GroundingPolicy initialization
evidence: hierarchy.py:49 at 301cc4d (GroundingPolicy.__post_init__)
criterion: Grounding policy must enforce strictly positive reserves (implicit in budget spec)
how_to_reproduce: Construct `GroundingPolicy(reserve_tokens=0)` - should raise ValueError but does not with `<` instead of `<=`
why_tests_missed_it: No test constructs a zero-reserve policy and verifies rejection; existing tests use positive reserves

id: C-S2M-002   tier: B   proposed_severity: minor
claim: Zero-valued request capacity is not rejected during GroundingPolicy initialization
evidence: hierarchy.py:53 at 301cc4d (GroundingPolicy.__post_init__)
criterion: Request capacity must be strictly positive to enforce forward progress
how_to_reproduce: Construct `GroundingPolicy(request_capacity_tokens=0)` - should raise ValueError
why_tests_missed_it: No test exercises zero request capacity; all tests use positive values

id: C-S2M-003   tier: B   proposed_severity: minor
claim: TreeNode with zero segments passes validation when it should require at least one
evidence: hierarchy.py:125 at 301cc4d (TreeNode.__post_init__)
criterion: A tree node must span at least one source segment to be valid
how_to_reproduce: Construct `TreeNode(segments=())` - should raise ValueError but does not when check is disabled
why_tests_missed_it: No test directly constructs invalid TreeNode objects; tree-building tests only use valid hierarchies

id: C-S2M-005   tier: B   proposed_severity: minor
claim: Zero-valued child count in group_children is not rejected
evidence: hierarchy.py:214 at 301cc4d (group_children)
criterion: Cannot group zero or negative children per the function contract
how_to_reproduce: Call `group_children(count=0, fanout=2)` - should raise ValueError but does not with `<` check
why_tests_missed_it: No test calls group_children with zero or negative count; caller validates before calling

id: C-S2M-010   tier: B   proposed_severity: minor
claim: select_source_passages accepts empty children sequence without validation
evidence: grounding.py:101 at 301cc4d (select_source_passages)
criterion: Source grounding requires at least one child to select passages from
how_to_reproduce: Call `select_source_passages(children=[], ...)` - should raise ValueError but does not when check disabled
why_tests_missed_it: No test calls grounding selector with empty children; callers validate before invoking

id: C-S2M-011   tier: B   proposed_severity: major
claim: select_source_passages does not validate that all cited segments exist in the source mapping
evidence: grounding.py:109 at 301cc4d (select_source_passages)
criterion: Unknown segment identifiers from model output must be rejected before building merge request
how_to_reproduce: Call `select_source_passages(children=[node_citing_S999999], source={'S000001': '...'})` - should raise ValueError when S999999 is missing
why_tests_missed_it: All test children cite only segments that exist in the provided source mapping; no test injects unknown citations into children

id: C-S2M-012   tier: B   proposed_severity: major
claim: Passages costing exactly max_tokens are rejected, allowing overflow by one token
evidence: grounding.py:118 at 301cc4d (select_source_passages)
criterion: Grounding cost must never exceed the fixed policy maximum to maintain budget safety
how_to_reproduce: Call with a passage that costs exactly `policy.max_tokens` - it will be rejected when it should be accepted (cost <= max is correct; cost < max is loose)
why_tests_missed_it: Test fixtures size passages below the boundary; no test checks a passage costing exactly the max

id: C-S2M-020   tier: B   proposed_severity: minor
claim: Annotations without supporting evidence are not rejected during leaf validation
evidence: leaf.py:347 at 301cc4d (validate_leaf_provenance)
criterion: Every grounded annotation (contradiction, qualification) must cite supporting evidence from source
how_to_reproduce: Create a SummaryNode with a Contradiction having empty evidence list - should raise LeafSummaryError but does not when check disabled
why_tests_missed_it: No test constructs an annotation with no evidence; all test fixtures use properly evidenced annotations
```

## Notes on Unkilled Boundaries

The eight surviving mutations reveal boundaries that are defended only through positive-case construction and default-assumption testing, not through explicit edge-case assertions:

1. **Validation off-by-ones** (mutations 1-2, 5): GroundingPolicy and group_children use `<=` and `<` comparisons for positive-integer constraints, but no test exercises zero-valued inputs. The code is correct, but the test boundaries are one value looser than the actual constraint.

2. **Call-site filtering** (mutations 10-11): select_source_passages and related functions rely on callers to pre-filter empty sequences and unknown identifiers. These make it to the function unchallenged and tests don't send them.

3. **Cost boundary** (mutation 12): The `cost <= policy.max_tokens` check allows a passage costing exactly the maximum; the mutation to `<` would incorrectly reject it. But all test fixtures stay below this boundary—no fixture tests the exact boundary case.

4. **Annotation evidence** (mutation 20): Contradictions and qualifications without evidence are never constructed in tests, only positive content units with evidence.

All eight surviving mutations are Tier B candidates. None represent Tier A breaks (wrong output, data loss, ungrounded text, or instruction-injection): they are boundaries tested implicitly rather than explicitly, places where a real-world mutation would let a defect slip through only if caller layers also fail to validate.

## Verification of Primary Checkout

Primary checkout status (must be clean):
```
$ git -C /Users/kylenakamura/documents-local/development-local/side-projects/recursive-text-summarizer status --short
(no output - clean)

$ git -C /Users/kylenakamura/documents-local/development-local/side-projects/recursive-text-summarizer rev-parse HEAD
301cc4d56d6326b5b0449da059b3b35f484cc5ca
```

Primary checkout is untouched.

## Temp Copy Cleanup

Temporary copy at `/tmp/s2-mut-57194` has been deleted.
