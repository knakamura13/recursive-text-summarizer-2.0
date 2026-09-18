#!/usr/bin/env python3
"""Verify F-006: TreeNode validation for zero segments."""

from summarizer.hierarchy import TreeNode
from summarizer.summaries import SummaryNode


def make_leaf(index: int) -> SummaryNode:
    """Construct a valid SummaryNode for testing."""
    body = {
        "summary": f"Summary of part {index}.",
        "content_units": [],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": [f"S{index:06d}"],
        "level": 0,
    }
    return SummaryNode.model_validate(body)


def test_treenode_empty_segments():
    """Test whether TreeNode constructor rejects empty segments tuple."""
    try:
        node = TreeNode(
            node_id="L0N0001",
            level=0,
            order=0,
            summary=make_leaf(1),
            children=(),
            covered_segments=(),  # Empty tuple: should this be rejected?
        )
        print("FAIL: TreeNode accepted empty covered_segments")
        print(f"  Created node: {node}")
        return False
    except ValueError as e:
        print(f"PASS: TreeNode rejected empty covered_segments")
        print(f"  Error: {e}")
        return True
    except Exception as e:
        print(f"ERROR: Unexpected exception: {type(e).__name__}: {e}")
        return False


def test_treenode_with_segments():
    """Test that TreeNode works with non-empty segments."""
    try:
        node = TreeNode(
            node_id="L0N0001",
            level=0,
            order=0,
            summary=make_leaf(1),
            children=(),
            covered_segments=("S000001",),
        )
        print("PASS: TreeNode accepted non-empty covered_segments")
        print(f"  Created node: {node}")
        return True
    except Exception as e:
        print(f"FAIL: TreeNode rejected valid segments: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: TreeNode with empty covered_segments")
    print("=" * 60)
    result1 = test_treenode_empty_segments()

    print("\n" + "=" * 60)
    print("Test 2: TreeNode with non-empty covered_segments")
    print("=" * 60)
    result2 = test_treenode_with_segments()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if result1 and result2:
        print("All tests passed: Guard is working correctly.")
        exit(0)
    else:
        print("Some tests failed.")
        exit(1)
