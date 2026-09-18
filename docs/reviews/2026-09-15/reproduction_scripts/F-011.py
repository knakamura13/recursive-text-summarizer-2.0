#!/usr/bin/env python3
"""Reproduce validation of an annotation without supporting evidence."""

from __future__ import annotations

import json

import pytest

from summarizer.leaf import LeafSummaryError, validate_provenance
from summarizer.summaries import SummaryNode

LEGAL = {
    "S000001": "The archive moved in March.",
    "S000002": "The index was rebuilt afterwards.",
    "S000003": "Staffing was unchanged.",
}


def node(**overrides: object) -> SummaryNode:
    body: dict[str, object] = {
        "summary": "The archive moved and the index was rebuilt.",
        "content_units": [],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": ["S000001", "S000002"],
        "level": 1,
    }
    body.update(overrides)
    return SummaryNode.model_validate(json.loads(json.dumps(body)))


def test_annotation_without_evidence_is_rejected() -> None:
    """Test that a contradiction with empty evidence is rejected."""
    try:
        validate_provenance(
            node(
                contradictions=[
                    {
                        "text": "Another source disagrees.",
                        "evidence": [],  # Empty evidence list
                    }
                ]
            ),
            legal=LEGAL,
            subject="L1N001",
        )
        print("RESULT=ACCEPTED: Empty evidence on annotation was NOT rejected")
    except LeafSummaryError as e:
        print(f"RESULT=REJECTED: {e}")


def test_qualification_without_evidence_is_rejected() -> None:
    """Test that a qualification with empty evidence is rejected."""
    try:
        validate_provenance(
            node(
                qualifications=[
                    {
                        "text": "This needs more context.",
                        "evidence": [],  # Empty evidence list
                    }
                ]
            ),
            legal=LEGAL,
            subject="L1N001",
        )
        print("RESULT=ACCEPTED: Empty evidence on qualification was NOT rejected")
    except LeafSummaryError as e:
        print(f"RESULT=REJECTED: {e}")


if __name__ == "__main__":
    print("Test 1: Contradiction without evidence")
    test_annotation_without_evidence_is_rejected()
    print("\nTest 2: Qualification without evidence")
    test_qualification_without_evidence_is_rejected()
