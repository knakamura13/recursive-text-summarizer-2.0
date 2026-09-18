#!/usr/bin/env python3
"""Test what happens when group_children guard is disabled."""

from summarizer.hierarchy import group_children

print("Test: Calling group_children(count=0, fanout=2) with guard disabled")
try:
    result = group_children(count=0, fanout=2)
    print(f"  Result: {result}")
    print("  [UNEXPECTED] Returned successfully - no error raised!")
except ZeroDivisionError as e:
    print(f"  Caught ZeroDivisionError: {e}")
    print("  [EXPLAINED] With guard disabled, code hits divmod(0, 0) which raises ZeroDivisionError")
except ValueError as e:
    print(f"  Caught ValueError: {e}")
    print("  [GUARD STILL PRESENT] Guard is still raising ValueError")
except Exception as e:
    print(f"  Caught unexpected exception: {type(e).__name__}: {e}")
