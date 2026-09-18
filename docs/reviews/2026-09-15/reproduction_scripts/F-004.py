#!/usr/bin/env python3
"""Verify whether GroundingPolicy rejects zero/negative reserve_tokens."""

from summarizer.grounding import GroundingPolicy

print("Test 1: GroundingPolicy with max_tokens=0")
try:
    policy = GroundingPolicy(max_tokens=0)
    print("  FAIL: Constructor accepted max_tokens=0, no exception raised")
except ValueError as e:
    print(f"  PASS: ValueError raised: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

print("\nTest 2: GroundingPolicy with max_tokens=-1")
try:
    policy = GroundingPolicy(max_tokens=-1)
    print("  FAIL: Constructor accepted max_tokens=-1, no exception raised")
except ValueError as e:
    print(f"  PASS: ValueError raised: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

print("\nTest 3: GroundingPolicy with max_tokens=1 (positive, should work)")
try:
    policy = GroundingPolicy(max_tokens=1)
    print(f"  PASS: Constructor accepted max_tokens=1")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

print("\nTest 4: Checking if MergeGrounding has reserve_tokens field")
from summarizer.hierarchy import MergeGrounding
from summarizer.grounding import GroundingSelection, SourcePassage

print("  MergeGrounding fields:", MergeGrounding.__dataclass_fields__.keys())

print("\nTest 5: MergeGrounding with reserve_tokens=0")
try:
    grounding = MergeGrounding(
        selection=GroundingSelection(passages=(), selected_ids=(), omitted_ids=()),
        reserve_tokens=0,
        request_capacity_tokens=None,
    )
    print("  FAIL: MergeGrounding accepted reserve_tokens=0, no exception raised")
except ValueError as e:
    print(f"  PASS: ValueError raised: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

print("\nTest 6: MergeGrounding with reserve_tokens=-1")
try:
    grounding = MergeGrounding(
        selection=GroundingSelection(passages=(), selected_ids=(), omitted_ids=()),
        reserve_tokens=-1,
        request_capacity_tokens=None,
    )
    print("  FAIL: MergeGrounding accepted reserve_tokens=-1, no exception raised")
except ValueError as e:
    print(f"  PASS: ValueError raised: {e}")
except Exception as e:
    print(f"  UNEXPECTED: {type(e).__name__}: {e}")

print("\n\nTest 7: Integration test - constructing audit with MergeGrounding that has zero reserve_tokens should fail Pydantic validation")
from summarizer.audit import AuditGroundingSelection

try:
    audit_grounding = AuditGroundingSelection(
        selected_ids=("seg1",),
        omitted_ids=(),
        reserve_tokens=0,  # This should fail Pydantic's ge=1 validation
        request_capacity_tokens=None,
        omission_reason="budget",
    )
    print("  FAIL: AuditGroundingSelection accepted reserve_tokens=0")
except Exception as e:
    print(f"  PASS: {type(e).__name__}: {e}")

print("\nTest 8: AuditGroundingSelection accepts reserve_tokens=1")
try:
    audit_grounding = AuditGroundingSelection(
        selected_ids=("seg1",),
        omitted_ids=(),
        reserve_tokens=1,
        request_capacity_tokens=None,
        omission_reason="budget",
    )
    print(f"  PASS: AuditGroundingSelection accepted reserve_tokens=1")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
