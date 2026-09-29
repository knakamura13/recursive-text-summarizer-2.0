#!/usr/bin/env python3
"""
Diagnose scenarios that produce INSUFFICIENTLY_SUPPORTED verdicts.

This script analyzes the verification logic to identify all possible paths
that lead to INSUFFICIENTLY_SUPPORTED verdicts, helping determine if there
are remaining issues after the quote normalization fix.
"""

from enum import Enum
from typing import NamedTuple

class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_MEANINGFULLY_VERIFIABLE = "not_meaningfully_verifiable"
    INSUFFICIENTLY_SUPPORTED = "insufficiently_supported"


class Scenario(NamedTuple):
    """A test scenario for verdict reduction."""
    name: str
    verdicts: set[Verdict]
    retrieval_complete: bool
    expected_result: Verdict
    reason: str


def reduce_batch_findings_logic(
    verdicts: set[Verdict],
    retrieval_complete: bool,
) -> tuple[Verdict, str]:
    """
    Replicate the reduce_batch_findings logic to identify output.

    Based on verification.py lines 1850-1867.
    """
    supported = Verdict.SUPPORTED in verdicts
    contradicted = Verdict.CONTRADICTED in verdicts
    nonverifiable = Verdict.NOT_MEANINGFULLY_VERIFIABLE in verdicts

    if supported and contradicted:
        return Verdict.INSUFFICIENTLY_SUPPORTED, "conflicting_evidence"
    if nonverifiable and len(verdicts) != 1:
        return Verdict.INSUFFICIENTLY_SUPPORTED, "inconsistent_meaningfulness"
    if verdicts == {Verdict.NOT_MEANINGFULLY_VERIFIABLE}:
        return Verdict.NOT_MEANINGFULLY_VERIFIABLE, "all_nonverifiable"
    if contradicted and retrieval_complete and not supported:
        return Verdict.CONTRADICTED, "contradicted_complete"
    if supported and not contradicted:
        return Verdict.SUPPORTED, "supported_clean"
    return Verdict.INSUFFICIENTLY_SUPPORTED, "default"


def analyze_scenarios():
    """Analyze all possible verdict reduction scenarios."""

    scenarios = [
        # Base cases - single verdict type
        Scenario(
            name="Only SUPPORTED",
            verdicts={Verdict.SUPPORTED},
            retrieval_complete=True,
            expected_result=Verdict.SUPPORTED,
            reason="Clear support → SUPPORTED",
        ),
        Scenario(
            name="Only SUPPORTED (incomplete retrieval)",
            verdicts={Verdict.SUPPORTED},
            retrieval_complete=False,
            expected_result=Verdict.SUPPORTED,
            reason="Support found (even if retrieval incomplete) → SUPPORTED",
        ),
        Scenario(
            name="Only CONTRADICTED (complete retrieval)",
            verdicts={Verdict.CONTRADICTED},
            retrieval_complete=True,
            expected_result=Verdict.CONTRADICTED,
            reason="Clear contradiction with exhaustive search → CONTRADICTED",
        ),
        Scenario(
            name="Only CONTRADICTED (incomplete retrieval)",
            verdicts={Verdict.CONTRADICTED},
            retrieval_complete=False,
            expected_result=Verdict.INSUFFICIENTLY_SUPPORTED,
            reason="Contradiction found but evidence not exhaustive → INSUFFICIENTLY_SUPPORTED",
        ),
        Scenario(
            name="Only NOT_MEANINGFULLY_VERIFIABLE",
            verdicts={Verdict.NOT_MEANINGFULLY_VERIFIABLE},
            retrieval_complete=True,
            expected_result=Verdict.NOT_MEANINGFULLY_VERIFIABLE,
            reason="Claim not meaningfully verifiable → NOT_MEANINGFULLY_VERIFIABLE",
        ),

        # Mixed verdicts
        Scenario(
            name="SUPPORTED + CONTRADICTED",
            verdicts={Verdict.SUPPORTED, Verdict.CONTRADICTED},
            retrieval_complete=True,
            expected_result=Verdict.INSUFFICIENTLY_SUPPORTED,
            reason="Evidence conflicts → conflicting_evidence → INSUFFICIENTLY_SUPPORTED",
        ),
        Scenario(
            name="SUPPORTED + NOT_MEANINGFULLY_VERIFIABLE",
            verdicts={Verdict.SUPPORTED, Verdict.NOT_MEANINGFULLY_VERIFIABLE},
            retrieval_complete=True,
            expected_result=Verdict.INSUFFICIENTLY_SUPPORTED,
            reason="Mixed meaningfulness → inconsistent_meaningfulness → INSUFFICIENTLY_SUPPORTED",
        ),
        Scenario(
            name="CONTRADICTED + NOT_MEANINGFULLY_VERIFIABLE",
            verdicts={Verdict.CONTRADICTED, Verdict.NOT_MEANINGFULLY_VERIFIABLE},
            retrieval_complete=True,
            expected_result=Verdict.INSUFFICIENTLY_SUPPORTED,
            reason="Mixed meaningfulness → inconsistent_meaningfulness → INSUFFICIENTLY_SUPPORTED",
        ),
        Scenario(
            name="All three verdict types",
            verdicts={Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.NOT_MEANINGFULLY_VERIFIABLE},
            retrieval_complete=True,
            expected_result=Verdict.INSUFFICIENTLY_SUPPORTED,
            reason="Mixed meaningfulness (and conflicting) → INSUFFICIENTLY_SUPPORTED",
        ),

        # Empty findings (would raise error in real code)
        Scenario(
            name="Empty findings list",
            verdicts=set(),
            retrieval_complete=True,
            expected_result=Verdict.INSUFFICIENTLY_SUPPORTED,
            reason="No findings (no verdict) → default → INSUFFICIENTLY_SUPPORTED",
        ),
    ]

    results = []
    for scenario in scenarios:
        actual_result, reason_code = reduce_batch_findings_logic(
            scenario.verdicts,
            scenario.retrieval_complete,
        )
        passed = actual_result == scenario.expected_result
        results.append({
            "scenario": scenario.name,
            "verdicts": scenario.verdicts,
            "retrieval": scenario.retrieval_complete,
            "expected": scenario.expected_result,
            "actual": actual_result,
            "passed": passed,
            "reason_code": reason_code,
            "explanation": scenario.reason,
        })

    return results


def categorize_insufficiently_supported():
    """Identify and categorize all paths leading to INSUFFICIENTLY_SUPPORTED."""

    insufficient_paths = []

    insufficient_paths.append({
        "path": "conflicting_evidence",
        "condition": "SUPPORTED and CONTRADICTED verdicts in same batch",
        "cause": "Quote matched in both supporting and contradicting evidence",
        "mitigation": "Model may have found conflicting passages or misunderstood context",
        "likelihood": "Medium - depends on source document complexity",
        "fixable_by_quote_norm": "Partially - quote norm fixes matching, not resolution",
    })

    insufficient_paths.append({
        "path": "inconsistent_meaningfulness",
        "condition": "NOT_MEANINGFULLY_VERIFIABLE mixed with other verdicts",
        "cause": "Model inconsistent about whether claim is verifiable",
        "mitigation": "Some evidence batches verifiable, others not",
        "likelihood": "Low - model should be consistent",
        "fixable_by_quote_norm": "No - quote norm doesn't affect meaningfulness",
    })

    insufficient_paths.append({
        "path": "incomplete_contradicted",
        "condition": "Only CONTRADICTED verdict with retrieval_complete=False",
        "cause": "Claims contradicted but evidence selection was cut short",
        "mitigation": "Evidence budget exhausted, supporting evidence may exist elsewhere",
        "likelihood": "Low-Medium - depends on evidence budget and document size",
        "fixable_by_quote_norm": "No - quote norm doesn't affect budget constraints",
    })

    insufficient_paths.append({
        "path": "empty_findings",
        "condition": "No findings returned for a claim",
        "cause": "Model returned empty or malformed findings",
        "mitigation": "Provider error or claim decomposition issue",
        "likelihood": "Very Low - normally caught as provider error",
        "fixable_by_quote_norm": "No - quote norm doesn't affect provider error handling",
    })

    return insufficient_paths


if __name__ == "__main__":
    print("=" * 100)
    print("INSUFFICIENTLY_SUPPORTED Verdict Analysis")
    print("=" * 100)
    print()

    # Section 1: Verdict reduction logic
    print("SECTION 1: Verdict Reduction Scenarios")
    print("-" * 100)
    results = analyze_scenarios()

    for r in results:
        status = "✓" if r["passed"] else "✗"
        verdicts_str = ", ".join([v.value for v in r["verdicts"]]) if r["verdicts"] else "(empty)"
        print(f"{status} {r['scenario']:40} | {verdicts_str:50}")
        print(f"   → {r['actual'].value} (reason: {r['reason_code']})")
        print(f"   → {r['explanation']}")
        print()

    # Count results
    passing = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"Results: {passing}/{total} scenarios behave as expected")
    print()

    # Section 2: All paths to INSUFFICIENTLY_SUPPORTED
    print("SECTION 2: Paths to INSUFFICIENTLY_SUPPORTED Verdict")
    print("-" * 100)

    insufficient_paths = categorize_insufficiently_supported()

    for i, path in enumerate(insufficient_paths, 1):
        print(f"{i}. {path['path'].upper()}")
        print(f"   Condition:      {path['condition']}")
        print(f"   Root Cause:     {path['cause']}")
        print(f"   Likelihood:     {path['likelihood']}")
        print(f"   Quote Norm Fix: {path['fixable_by_quote_norm']}")
        print()

    # Section 3: Summary
    print("SECTION 3: Summary & Implications")
    print("-" * 100)
    print()
    print("✓ Quote Normalization Fix Impact:")
    print("  - FIXES: Quote matching false negatives (exact match → normalized match)")
    print("  - PARTIALLY FIXES: conflicting_evidence (if due to quote matching failures)")
    print()
    print("✗ Paths NOT fixed by quote normalization:")
    print("  - inconsistent_meaningfulness: Model gives conflicting verdicts")
    print("  - incomplete_contradicted: Budget constraints limiting evidence search")
    print("  - empty_findings: Provider errors or decomposition issues")
    print()
    print("Next Investigation Steps:")
    print("  1. Measure frequency of each INSUFFICIENTLY_SUPPORTED path in real runs")
    print("  2. Determine if quote normalization actually reduced conflicting_evidence rate")
    print("  3. Investigate inconsistent_meaningfulness scenarios if they're common")
    print("  4. Consider evidence budget optimization if incomplete_contradicted is frequent")
    print()
    print("=" * 100)
