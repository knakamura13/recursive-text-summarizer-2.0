#!/usr/bin/env python3
"""
Comprehensive verification that quote normalization fixes insufficient_support issues.

This script reproduces the scenario from the previous session where claims with
properly-quoted evidence were incorrectly marked as INSUFFICIENTLY_SUPPORTED
due to whitespace and quote-mark variations in the verifier output.
"""

import json
from pathlib import Path
from summarizer.verification import (
    Claim,
    ClaimAssessment,
    ClaimVerdict,
    DraftSpan,
    SourceLexicalIndex,
    VerificationConfig,
    VerificationRuntime,
    build_source_lexical_index,
    locate_quote,
    parse_claim_findings,
    reduce_batch_findings,
    split_draft_spans,
    verify_draft_once,
)
from summarizer.providers.base import Provider, GenerationResult
from summarizer.tokenization import ConservativeUtf8TokenCounter


# Test 1: Direct quote locating with whitespace and quote mark variations
def test_quote_location_with_variations():
    """Verify locate_quote handles all expected variations."""
    # Passage with typographic marks from PDF
    passage = 'The group\nhad "one rule" — it didn\'t\n\nbend.'

    # Quotes with variations that should match
    test_cases = [
        ('The group had "one rule" - it didn\'t bend.', True),  # straight quotes
        ('The group had "one rule" — it didn\'t bend.', True),   # em dash
        ('  The group\thad "one rule" - it didn\'t  bend.  ', True),  # spaces/tabs
        ('The group\nhad "one rule" — it didn\'t\n\nbend.', True),  # exact match
        ('The group had "one law" - it didn\'t bend.', False),    # content mismatch
        ('The group had "one rule" - it did not bend.', False),   # content mismatch
    ]

    results = []
    for quote, should_match in test_cases:
        located = locate_quote(quote, passage)
        matched = located is not None
        status = "✓" if matched == should_match else "✗"
        results.append({
            "quote": quote[:50] + "..." if len(quote) > 50 else quote,
            "expected": "match" if should_match else "no match",
            "actual": "match" if matched else "no match",
            "status": status,
            "located": located[:30] + "..." if located else None,
        })

    return results


# Test 2: Verification with PDF-style typographic variations
class MockProvider(Provider):
    """Mock provider that returns claims with typographic variations."""

    def __init__(self):
        self.calls = []

    def generate(self, request):
        """Generate mock responses for verification operations."""
        self.calls.append(request)

        if "decompose" in request.operation_id:
            # Return one decomposed claim
            return GenerationResult(
                json.dumps({
                    "spans": [{
                        "span_id": "V01S000001",
                        "anchors": [
                            "The group had one rule and it didn't bend."
                        ]
                    }]
                }),
                "test-model",
                request.model
            )

        # Classification response with typographic variations
        # The verifier found the quote but with spaces instead of line breaks
        # and straight quotes instead of curly ones
        return GenerationResult(
            json.dumps({
                "findings": [{
                    "claim_id": "V01C000001",
                    "verdict": "supported",
                    "evidence": [{
                        "segment_id": "S000001",
                        # Quote with variations that should match after normalization
                        "exact_quote": 'The group had "one rule" - it didn\'t bend.'
                    }]
                }]
            }),
            "test-model",
            request.model
        )


def test_verification_with_normalized_quotes():
    """Test that verification accepts quotes after normalization."""
    # Source passage with PDF-style typography
    source_text = 'Before.  The group\nhad "one rule" — it didn\'t\n\nbend. After.'
    source_id = "S000001"

    # Create a draft
    draft = "The group had one rule and it didn't bend."

    # Build the verification runtime
    counter = ConservativeUtf8TokenCounter()
    source_index = build_source_lexical_index(
        {source_id: source_text},
        counter=counter,
    )

    provider = MockProvider()
    runtime = VerificationRuntime(
        provider=provider,
        counter=counter,
    )

    config = VerificationConfig(enabled=True, max_repair_passes=0)

    # Run verification
    result = verify_draft_once(
        draft,
        source_id=source_id,
        source_index=source_index,
        runtime=runtime,
        config=config,
    )

    return {
        "passed": not result.failed,
        "exhausted": result.exhausted,
        "diagnostic_codes": result.diagnostic_codes,
        "verdict": result.assessments[0].verdict.value if result.assessments else "no assessment",
        "expected_verdict": "supported",
        "calls_made": len(provider.calls),
    }


# Test 3: Batch findings reduction logic
def test_batch_findings_reduction():
    """Test that reduce_batch_findings works correctly with normalized quotes."""
    from summarizer.verification import BatchFinding

    # Simulate findings after normalization
    findings = (
        BatchFinding(
            claim_id="V01C000001",
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=("S000001",),
            exact_quotes=('The group\nhad "one rule" — it didn\'t\n\nbend.',),  # Normalized from verifier
        ),
    )

    verdict, codes = reduce_batch_findings(
        "V01C000001",
        findings,
        retrieval_complete=True,
    )

    return {
        "verdict": verdict.value,
        "expected": "supported",
        "codes": list(codes),
        "passed": verdict is ClaimVerdict.SUPPORTED,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Quote Normalization Fix for insufficient_support Issues")
    print("=" * 80)
    print()

    # Test 1: Quote location
    print("TEST 1: Quote Location with Variations")
    print("-" * 80)
    results = test_quote_location_with_variations()
    for r in results:
        print(f"  {r['status']} {r['expected']:12} {r['actual']:12} {r['quote'][:40]}")
    print()

    all_passed = all(r['status'] == '✓' for r in results)
    print(f"  Result: {'PASS' if all_passed else 'FAIL'}")
    print()

    # Test 2: Full verification flow
    print("TEST 2: Verification with Normalized Quotes")
    print("-" * 80)
    result2 = test_verification_with_normalized_quotes()
    for key, value in result2.items():
        print(f"  {key:25}: {value}")
    print()
    print(f"  Result: {'PASS' if result2['passed'] and result2['verdict'] == 'supported' else 'FAIL'}")
    print()

    # Test 3: Batch findings reduction
    print("TEST 3: Batch Findings Reduction")
    print("-" * 80)
    result3 = test_batch_findings_reduction()
    for key, value in result3.items():
        print(f"  {key:25}: {value}")
    print()
    print(f"  Result: {'PASS' if result3['passed'] else 'FAIL'}")
    print()

    # Summary
    print("=" * 80)
    all_tests_passed = all_passed and result2['passed'] and result3['passed']
    print(f"Overall Result: {'PASS - Quote normalization fix is working!' if all_tests_passed else 'FAIL'}")
    print("=" * 80)
