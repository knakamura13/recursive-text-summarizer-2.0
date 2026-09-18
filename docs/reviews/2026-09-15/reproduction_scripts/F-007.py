#!/usr/bin/env python3
"""Reproduction script for F-007: group_children with count=0."""

from summarizer.hierarchy import group_children

print("Test 1: Calling group_children(count=0, fanout=2)")
try:
    result = group_children(count=0, fanout=2)
    print(f"  Result: {result}")
    print("  [UNEXPECTED] No error raised - guard is NOT working!")
except ValueError as e:
    print(f"  Caught ValueError: {e}")
    print("  [EXPECTED] Guard validates count=0")
except Exception as e:
    print(f"  Caught unexpected exception: {type(e).__name__}: {e}")

print("\nTest 2: Calling group_children(count=-1, fanout=2)")
try:
    result = group_children(count=-1, fanout=2)
    print(f"  Result: {result}")
    print("  [UNEXPECTED] No error raised - guard is NOT working!")
except ValueError as e:
    print(f"  Caught ValueError: {e}")
    print("  [EXPECTED] Guard validates count=-1")
except Exception as e:
    print(f"  Caught unexpected exception: {type(e).__name__}: {e}")

print("\nTest 3: Calling group_children(count=1, fanout=2) - should work")
try:
    result = group_children(count=1, fanout=2)
    print(f"  Result: {result}")
    print("  [EXPECTED] Single child grouped successfully")
except Exception as e:
    print(f"  Caught exception: {type(e).__name__}: {e}")

print("\nTest 4: Calling group_children(count=2, fanout=2) - typical call from build_hierarchy")
try:
    result = group_children(count=2, fanout=2)
    print(f"  Result: {result}")
    print("  [EXPECTED] Two children grouped successfully")
except Exception as e:
    print(f"  Caught exception: {type(e).__name__}: {e}")
