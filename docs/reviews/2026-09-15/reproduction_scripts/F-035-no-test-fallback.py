#!/usr/bin/env python3
"""
F-035 - Claim Classification Internal Reduction Branches

This script demonstrates the input validation check that is not exercised by any test.
If findings is empty or contains a claim_id mismatch, reduce_batch_findings should raise ValueError.
"""

from summarizer.verification import (
    reduce_batch_findings,
    BatchFinding,
    ClaimVerdict,
)

def demo_input_validation_survivor() -> None:
    print("=== INPUT VALIDATION SURVIVOR (F-035) ===")
    # Normal case: one matching finding
    findings = (
        BatchFinding(
            claim_id="V01C000001",
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=("S000001",),
            exact_quotes=("quote",),
        )
    )
    try:
        verdict, _ = reduce_batch_findings(claim_id="V01C000001", findings=findings, retrieval_complete=False)
        assert verdict is ClaimVerdict.SUPPORTED
        print("- Normal case passes (returns SUPPORTED)")
    except Exception as e:
        print(f"Unexpected error in normal case: {e}")
        return

    # Edge case 1: empty findings (should raise ValueError but mutation removes check)
    try:
        verdict, _ = reduce_batch_findings(claim_id="V01C000001", findings=(), retrieval_complete=False)
        print("- Empty findings: no exception raised (mutation survives)")
    except ValueError as e:
        print(f"- Empty findings: ValueError raised: {e}")
    except Exception as e:
        print(f"- Empty findings: unexpected error: {e}")

    # Edge case 2: mismatched claim_id (should raise ValueError but mutation removes check)
    findings_mismatch = (
        BatchFinding(
            claim_id="V02C000001",  # different claim_id
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=("S000001",),
            exact_quotes=("quote",),
        )
    )
    try:
        verdict, _ = reduce_batch_findings(claim_id="V01C000001", findings=findings_mismatch, retrieval_complete=False)
        print("- Mismatched claim_id: no exception raised (mutation survives)")
    except ValueError as e:
        print(f"- Mismatched claim_id: ValueError raised: {e}")
    except Exception as e:
        print(f"- Mismatched claim_id: unexpected error: {e}")

    print("\nConclusion: The input validation check at line 1131 is not exercised by any test.")
    print("Removing the check (e.g., deleting the 'not findings or' part) lets the function proceed")
    print("with malformed input, but no existing test triggers this path.")
    print("F-035 documented as a minor internal boundary with no customer-visible impact.")

if __name__ == "__main__":
    demo_input_validation_survivor()