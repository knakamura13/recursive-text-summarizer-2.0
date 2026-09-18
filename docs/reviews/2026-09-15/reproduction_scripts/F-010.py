#!/usr/bin/env python
"""
Verify F-010: Test exact boundary condition where passage cost == policy.max_tokens

The claim is that passages costing exactly max_tokens are rejected.
We need to confirm whether the UNMUTATED code accepts or rejects them.

Current code at grounding.py:118 uses: if cost <= policy.max_tokens:
This should ACCEPT passages costing exactly max_tokens (inclusive comparison).
"""

from dataclasses import dataclass
from summarizer.grounding import GroundingPolicy, GroundingSelection, select_source_passages
from summarizer.summaries import SummaryNode


@dataclass(frozen=True)
class CharacterCounter:
    """Count characters to make token cost deterministic and controllable."""
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


def test_exact_boundary_passage_cost_equals_max_tokens() -> None:
    """
    Create a passage that costs EXACTLY policy.max_tokens.
    Verify whether it's ACCEPTED (correct) or REJECTED (bug).
    """
    max_tokens = 100
    policy = GroundingPolicy(max_tokens=max_tokens)
    
    # Create minimal passages with predictable character costs.
    # Each passage is serialized as JSON: {"segment_id":"...", "text":"..."}
    # We need to control the text length so the total serialized cost is exactly 100.
    
    # Let's build a single passage. The serialize_source_passage produces:
    # {"segment_id":"S0","text":"...content..."}
    # For a segment_id of "S0" and varying text length, we can hit the exact 100.
    
    # Through testing: text_len=71 gives exactly 100 characters when serialized
    source = {
        "S0": "x" * 71,  # When serialized with segment_id "S0", total = 100 chars
    }
    
    # Create a minimal summary node that requires this source
    children = (
        SummaryNode.model_validate(
            {
                "summary": "Test node.",
                "content_units": [
                    {
                        "text": "Claim.",
                        "kind": "claim",
                        "evidence": [{"segment_id": "S0", "quote": None}],
                        "qualification": None,
                        "uncertain": False,
                    }
                ],
                "entities": [],
                "qualifications": [],
                "contradictions": [],
                "quotations": [],
                "provenance": ["S0"],
                "level": 0,
            }
        ),
    )
    
    # Verify the serialization cost
    from summarizer.grounding import serialize_source_passage, SourcePassage
    test_passage = SourcePassage("S0", "x" * 71)
    serialized = serialize_source_passage(test_passage)
    counter = CharacterCounter()
    actual_cost = counter.count(serialized)
    
    print(f"Serialized passage: {serialized}")
    print(f"Serialized length: {actual_cost} characters")
    print(f"Policy max_tokens: {max_tokens}")
    print(f"Cost == max_tokens: {actual_cost == max_tokens}")
    print()
    
    # Now call select_source_passages with this exact boundary condition
    try:
        selection: GroundingSelection = select_source_passages(
            children,
            source=source,
            counter=counter,
            policy=policy,
        )
        print("✓ RESULT: Passage was ACCEPTED")
        print(f"  Selected passages: {selection.selected_ids}")
        print(f"  Omitted passages: {selection.omitted_ids}")
        return True
    except Exception as e:
        print(f"✗ RESULT: Passage was REJECTED")
        print(f"  Error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    accepted = test_exact_boundary_passage_cost_equals_max_tokens()
    
    if accepted:
        print("\n" + "="*70)
        print("ANALYSIS: The current unmutated code ACCEPTS passages costing")
        print("exactly max_tokens (using <= operator). This is CORRECT behavior.")
        print("The mutation finding reflects a TEST GAP, not a real bug.")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("ANALYSIS: The current unmutated code REJECTS passages costing")
        print("exactly max_tokens. This suggests the code uses < instead of <=,")
        print("indicating a REAL BUG in the boundary check.")
        print("="*70)
