#!/usr/bin/env python3
"""
Verify F-005: Test whether MergeGrounding rejects zero and negative request_capacity_tokens.
"""

from summarizer.hierarchy import MergeGrounding
from summarizer.grounding import GroundingSelection, SourcePassage

def test_zero_request_capacity():
    """Test constructing MergeGrounding with request_capacity_tokens=0."""
    print("Testing request_capacity_tokens=0...")
    selection = GroundingSelection(
        passages=(),
        selected_ids=(),
        omitted_ids=()
    )
    
    try:
        mg = MergeGrounding(
            selection=selection,
            reserve_tokens=None,
            request_capacity_tokens=0
        )
        print("  ERROR: MergeGrounding accepted request_capacity_tokens=0 (no error raised)")
        return False
    except ValueError as e:
        print(f"  OK: ValueError raised: {e}")
        return True

def test_negative_request_capacity():
    """Test constructing MergeGrounding with request_capacity_tokens=-1."""
    print("Testing request_capacity_tokens=-1...")
    selection = GroundingSelection(
        passages=(),
        selected_ids=(),
        omitted_ids=()
    )
    
    try:
        mg = MergeGrounding(
            selection=selection,
            reserve_tokens=None,
            request_capacity_tokens=-1
        )
        print("  ERROR: MergeGrounding accepted request_capacity_tokens=-1 (no error raised)")
        return False
    except ValueError as e:
        print(f"  OK: ValueError raised: {e}")
        return True

def test_positive_request_capacity():
    """Test constructing MergeGrounding with request_capacity_tokens=100 (should work)."""
    print("Testing request_capacity_tokens=100...")
    selection = GroundingSelection(
        passages=(),
        selected_ids=(),
        omitted_ids=()
    )
    
    try:
        mg = MergeGrounding(
            selection=selection,
            reserve_tokens=None,
            request_capacity_tokens=100
        )
        print("  OK: MergeGrounding constructed successfully with positive value")
        return True
    except ValueError as e:
        print(f"  ERROR: ValueError raised unexpectedly: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("F-005: MergeGrounding request_capacity_tokens validation")
    print("=" * 60)
    
    results = []
    results.append(("zero value", test_zero_request_capacity()))
    results.append(("negative value", test_negative_request_capacity()))
    results.append(("positive value", test_positive_request_capacity()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = all(result for _, result in results)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    if all_passed:
        print("\nConclusion: Guard IS present and working correctly.")
    else:
        print("\nConclusion: Guard is NOT working as expected.")
    print("=" * 60)
