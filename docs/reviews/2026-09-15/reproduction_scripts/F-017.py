"""
F-017 Repair Mechanism Analysis

Goal: Trace what happens when the verification/editorial stage encounters
citations to non-existent segments or invalid quotes.
"""

import json
from pathlib import Path

print("=== ANALYZING LIVE-RUN EVIDENCE ===\n")

live_runs = [
    ".review/live-runs/s1-default-live",
    ".review/live-runs/s1-default-live-attempt-2",
    ".review/live-runs/s1-default-live-attempt-3",
]

for run in live_runs:
    stderr_path = Path(run) / "stderr.txt"
    if stderr_path.exists():
        error = stderr_path.read_text().strip()
        print(f"{run}:")
        print(f"  Error: {error}\n")

print("=== INTERPRETATION ===\n")
print("""
Error sources identified:

1. "response cited unknown segments" (s1-default-live)
   - This comes from leaf.py:357: validate_provenance()
   - Occurs during initial summary generation, BEFORE editorial/verification
   - The repair mechanism is NEVER invoked

2. "response failed validation (content_units.0.evidence.0.quote: value_error; ...)" (attempt-2)
   - This comes from either:
     a) leaf.py:289: parse_leaf_summary validation error, OR
     b) verification.py:1026: _validated_response error during verification
   - If from leaf level, repair not invoked
   - If from verification level, need to check if error triggers repair

3. "response failed validation (summary: value_error; quotations.0.quote: ...)" (attempt-3)
   - Same as attempt-2, similar origin uncertainty

KEY QUESTION: Are these errors from leaf-level or verification-level?
""")

print("\n=== CODE FLOW ANALYSIS ===\n")
print("""
LEAF-LEVEL (Early, before repair):
  summarize_direct (direct.py:56-97)
    -> parse_leaf_summary (leaf.py:271-304)
      -> validate_provenance (leaf.py:307-393)
        -> Raises LeafSummaryError if citations are invalid
           (not caught or handled by repair mechanism)

VERIFICATION-LEVEL (Where repair would be triggered):
  _finalize_summary (finalization.py:281-397)
    -> write_editorial (editorial.py)
    -> verify_and_repair (verification.py:1533)
      -> verify_draft_once (verification.py:1150)
        -> Claims classified based on evidence
        -> If contradicted, may trigger repair (line 1892)
        -> build_repair_request (verification.py:886)
        -> parse_repair_proposals (verification.py:940)
        -> apply_repairs (verification.py:969)
        -> Re-verify repaired draft

The issue: Citation/quote validation errors from leaf stage are NOT passed to repair.
They fail the pipeline immediately with a LeafSummaryError.
""")

print("\n=== CHECKING AUDIT FILES ===\n")
for run in live_runs:
    audit_path = Path(run) / "audit.json"
    if audit_path.exists():
        try:
            audit_data = json.loads(audit_path.read_text())
            print(f"{run}: audit file exists")
            if "verification" in audit_data:
                verif = audit_data["verification"]
                print(f"  enabled: {verif.get('enabled')}")
                print(f"  failed: {verif.get('failed')}")
            else:
                print(f"  No verification data in audit")
        except Exception as e:
            print(f"  Error reading audit: {e}")
    else:
        print(f"{run}: no audit file")

print("\n=== ANALYSIS ===\n")
print("""
The stderr messages show "Summarization failed:" which is printed by cli.py
when it catches LeafSummaryError (which extends ValueError).

This is strong evidence that the errors occur during the initial summary
generation (leaf-level), NOT during editorial verification.

If that's the case:
- The repair mechanism was never invoked
- This is not a defect in the repair mechanism
- It's expected behavior for generation-level errors

However, to be thorough, we need to check if ANY of the attempts reached
the verification stage. The audit files would show if verification was attempted.
""")

print("\n=== DEFINITIVE ANALYSIS ===\n")
print("""
ALL THREE ERRORS come from leaf.py:parse_leaf_summary (line 289):

Error format in leaf.py (line 289):
    f"{segment.segment_id}: response failed validation ({_describe(error)})"

This exactly matches the live-run errors:
1. "S000001: response cited unknown segments..." (line 357)
2. "S000001: response failed validation (content_units.0.evidence.0.quote: ...)" (line 289)
3. "S000002: response failed validation (summary: value_error; ...)" (line 289)

KEY FINDING: These are ALL LEAF-LEVEL errors, occurring during initial summary
generation BEFORE the editorial stage and verification/repair mechanism.

The error flow:
1. summarize_segments() or summarize_direct() calls parse_leaf_summary()
2. parse_leaf_summary() validates the model response against SummaryNode schema
3. validate_provenance() checks for citations to unknown segments
4. If validation fails, LeafSummaryError is raised
5. Exception propagates to cli.py and is printed as "Summarization failed: {error}"
6. Pipeline exits with code 1

The repair mechanism is NEVER invoked because the pipeline fails BEFORE
reaching _finalize_summary() which calls verify_and_repair().

This is NOT a defect in the repair mechanism. The repair mechanism is designed
to handle editorial verification errors, not initial summary generation errors.

VERDICT:
- The claim is REFUTED
- The repair mechanism was not invoked on these errors (expected)
- These are generation-level errors, not verification-level errors
- The claim author may have confused the two error stages
""")
