"""#70 second branch — budget=2 nested exhaustion with contradiction still alive."""
# This pins the path where verify_and_repair runs budget 2 but contradiction
# remains after both passes. It complements the budget=1 case.
from summarizer.verification import verify_and_repair, VerificationConfig
from summarizer.indexing import build_source_lexical_index
from summarizer.tokenization import ConservativeUtf8TokenCounter
from summarizer.runtime import VerificationRuntime


class AlwaysContradictionProvider:
    """Every generation returns contradiction (so repair can never succeed)."""
    call_count = 0
    def generate(self, request):
        AlwaysContradictionProvider.call_count += 1
        # Return contradiction with evidence pointing at source segment
        payload = ("{\"findings\":[{\"claim_id\":\"C1\","
                   "\"verdict\":\"contradicted\","
                   "\"evidence\":[{\"segment_id\":\"S001\","
                   "\"exact_quote\":\"value is 41\"}]}]}".replace('\\', ''))
        # Minimal generation result wrapper that verify_and_repair accepts
        class FakeGen:
            pass
        g = FakeGen()
        g.text = payload
        g.provider = "scripted"
        g.model = "fake"
        return g


def test_budget2_exhaustion_contradiction_remains():
    draft = "The value is 42."
    provider = AlwaysContradictionProvider()
    result = verify_and_repair(
        draft,
        source_id="test-budget2",
        source_index=build_source_lexical_index(
            provenance_ids=("S001",), source={"S001": "The value is 41."}
        ),
        runtime=VerificationRuntime(
            provider, ConservativeUtf8TokenCounter(), "fake", 30, 10_000
        ),
        config=VerificationConfig(enabled=True, max_repair_passes=2),
    )
    # With contradiction alive and repair budget exhausted, result must fail.
    assert result.failed, "Expected failure after repair exhaustion"
    # The failure code should include the repair-reverification failure.
    codes = result.failure_codes or ()
    assert "repair_reverification_failed" in codes, (
        f"Expected repair_reverification_failed in codes, got {codes}"
    )
