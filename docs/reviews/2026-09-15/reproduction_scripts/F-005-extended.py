#!/usr/bin/env python3
"""
Extended test: Check if existing tests would catch a zero request_capacity_tokens issue.
This simulates what happens if we allow zero values through the guard.
"""

from summarizer.hierarchy import MergeGrounding
from summarizer.grounding import GroundingSelection
from summarizer.budget import BudgetError
from dataclasses import dataclass, field, replace

def test_downstream_bude_error():
    """
    Check if allowing zero request_capacity_tokens through the guard
    would cause a downstream BudgetError in _prepare_merge.
    """
    print("Testing downstream impact of zero request_capacity_tokens...")
    print()
    
    # This is a hypothetical scenario - what if the guard allowed zero?
    # Would downstream code catch the problem?
    
    selection = GroundingSelection(
        passages=(),
        selected_ids=(),
        omitted_ids=()
    )
    
    # Create MergeGrounding with zero request_capacity_tokens
    # (We can't actually do this with the current guard, but let's document the risk)
    print("Current behavior: guard blocks zero values at construction time")
    print()
    
    try:
        mg = MergeGrounding(
            selection=selection,
            reserve_tokens=None,
            request_capacity_tokens=0
        )
        print("ERROR: Zero value was allowed! This shouldn't happen.")
    except ValueError as e:
        print(f"Guard caught it at construction: {e}")
        print()
        print("Risk assessment:")
        print("- If the guard were removed/inverted, zero would propagate to _prepare_merge")
        print("- At _prepare_merge line 575, we check: if request_tokens > usable_tokens")
        print("- With usable_tokens=0 and request_tokens>0, BudgetError would be raised")
        print("- BUT: That error would be at runtime during hierarchy building")
        print("- The guard catches it EARLY at construction, before any work is done")
        print()
        print("Verdict: Guard is protective but NOT TESTED")
        print("- No test constructs MergeGrounding with zero/negative values")
        print("- No test verifies the ValueError is raised")
        print("- This is a valid mutation testing gap: guard logic untested")

if __name__ == "__main__":
    print("=" * 70)
    print("F-005 Extended: Downstream impact analysis")
    print("=" * 70)
    print()
    
    test_downstream_bude_error()
    
    print()
    print("=" * 70)
