"""Repro for the S2H investigation, claim 4 ("coverage by concatenation
rather than union") and its claim-5 test-hole angle.

This script does NOT prove claim 4 is present in main at 301cc4d - it isn't;
`_prepare_merge` in summarizer/hierarchy.py already computes coverage with
`dict.fromkeys(...)` (a document-order union), not a bare concatenation.

What it proves instead:

1. Main's real code, exercised through the real `build_hierarchy` /
   `_prepare_merge` path (no hand-built TreeNode/coverage tuples), correctly
   deduplicates overlapping `covered` input. This is evidence that claim 4 is
   ABSENT from main.

2. No fixture anywhere in tests/test_hierarchy.py or tests/test_merge_prompt.py
   ever passes overlapping `covered` identifiers across sibling leaves, so
   this property is exercised only by this script, never by the suite. The
   only real caller (summarizer/pipeline.py) always builds `covered` as
   `[(segment.segment_id,) for segment in segments]`, i.e. disjoint
   singletons, one per segment - so the union-vs-concatenation distinction
   currently has no live path from a real caller. It matters only as a
   defensive property of `build_hierarchy`'s own contract (the module's
   comment references "issue #8" narrowing coverage further, implying a
   future caller might not keep leaves disjoint).

3. A companion mutation test (run separately against a scratch copy of the
   tree, not this script) showed that reverting the `dict.fromkeys(...)`
   wrapper in `_prepare_merge` back to a bare concatenation leaves the FULL
   766-test suite green. That is the live test hole: nothing in the suite
   would catch a regression of this specific shape today.

Run with:
    cd <repo> && UV_OFFLINE=1 uv run --with-requirements requirements-dev.txt \
        python .review/wip/issue7/repro_claim4_coverage_union_hole.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from summarizer.grounding import GroundingPolicy
from summarizer.hierarchy import build_hierarchy
from summarizer.providers.base import GenerationRequest, GenerationResult


@dataclass(frozen=True)
class CharacterCounter:
    """Same fake counter tests/test_hierarchy.py uses: 1 token per character."""

    identity: str = "repro:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


def leaf_payload(index: int, provenance: str) -> dict:
    return {
        "summary": f"Summary of part {index}.",
        "content_units": [],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": [provenance],
        "level": 0,
    }


class MergingProvider:
    """Always answers a merge request with a valid level-1 payload."""

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        body = {
            "summary": "Merged.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            # require_provenance defaults True even on the merge path (the
            # branch's own design doc notes this flag was not planned but
            # emerged during implementation), so cite something legal here.
            "provenance": ["S000001"],
            "level": 1,
        }
        return GenerationResult(text=json.dumps(body), provider="fake", model=request.model)


def main() -> None:
    from summarizer.summaries import SummaryNode

    leaves = [
        SummaryNode.model_validate(leaf_payload(1, "S000001")),
        SummaryNode.model_validate(leaf_payload(2, "S000002")),
        SummaryNode.model_validate(leaf_payload(3, "S000003")),
    ]

    # The defect surface: two leaves are declared to cover the SAME segment
    # identifier (S000002). No real caller in this codebase does this today
    # (summarizer/pipeline.py always hands build_hierarchy disjoint
    # singletons), but build_hierarchy's own signature does not forbid it,
    # and nothing in the test suite exercises this input shape.
    covered = [
        ("S000001", "S000002"),  # leaf 1: overlaps leaf 2 on S000002
        ("S000002",),            # leaf 2
        ("S000003",),            # leaf 3
    ]
    attributable = {
        "S000001": "Part 1 said something.",
        "S000002": "Part 2 said something.",
        "S000003": "Part 3 said something.",
    }

    provider = MergingProvider()
    root, nodes, report = build_hierarchy(
        leaves,
        provider,
        CharacterCounter(),
        source_id="a" * 64,
        covered=covered,
        attributable=attributable,
        usable_tokens=100_000,
        model="m",
        timeout_seconds=30,
        max_merge_children=3,  # force all three leaves into one merge group
        grounding_policy=GroundingPolicy(max_tokens=1_000),
    )

    print(f"root.covered_segments = {root.covered_segments}")
    print(f"len(covered_segments)  = {len(root.covered_segments)}")
    print(f"len(set(covered_segments)) = {len(set(root.covered_segments))}")

    if len(root.covered_segments) == len(set(root.covered_segments)):
        print(
            "PASS (claim 4 ABSENT from main): duplicate S000002 was "
            "deduplicated by _prepare_merge's dict.fromkeys(...) union."
        )
    else:
        print(
            "FAIL (claim 4 PRESENT): coverage carries a duplicate identifier, "
            "exactly the concatenation defect the branch commit describes."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
