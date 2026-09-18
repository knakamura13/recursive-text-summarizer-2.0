"""Test F-009: select_source_passages validation of unknown segment ids."""
import sys
from dataclasses import dataclass

from summarizer.grounding import GroundingPolicy, select_source_passages
from summarizer.summaries import SummaryNode


@dataclass(frozen=True)
class CharacterCounter:
    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


# Create a child node that references S999999 (unknown segment)
child_with_unknown = SummaryNode.model_validate(
    {
        "summary": "Child with unknown citation.",
        "content_units": [
            {
                "text": "A claim citing unknown segment.",
                "kind": "claim",
                "evidence": [{"segment_id": "S999999", "quote": None}],
                "qualification": None,
                "uncertain": False,
            },
        ],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": [],
        "level": 1,
    }
)

# Source mapping that doesn't include S999999
source = {
    "S000001": "Known source segment 1",
    "S000002": "Known source segment 2",
}

print("TEST: Calling select_source_passages with unknown segment S999999")
print(f"  Children: {[child_with_unknown]}")
print(f"  Source keys: {list(source.keys())}")

try:
    result = select_source_passages(
        [child_with_unknown],
        source=source,
        counter=CharacterCounter(),
        policy=GroundingPolicy(max_tokens=1000),
    )
    print(f"ERROR: Did not raise exception!")
    print(f"  Result: {result}")
    sys.exit(1)
except ValueError as e:
    print(f"SUCCESS: ValueError raised with message:")
    print(f"  {e}")
except KeyError as e:
    print(f"FAILURE: KeyError raised instead of ValueError:")
    print(f"  {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"FAILURE: Unexpected exception type {type(e).__name__}:")
    print(f"  {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
