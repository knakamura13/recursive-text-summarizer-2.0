"""Reproduction script for #70 — repair-pass exhaustion."""
import sys
sys.path.insert(0, '.')
from summarizer.verification import verify_and_repair, VerificationConfig
from summarizer.indexing import build_source_lexical_index
from summarizer.tokenization import ConservativeUtf8TokenCounter

class FakeContradictionProvider:
    def __init__(self): self.calls = 0
    def generate(self, request):
        self.calls += 1
        return type('G',(),{'text':'{"findings":[{"claim_id":"C1","verdict":"contradicted","evidence":[{"segment_id":"S001","quote":"value is 41"}]}]}'})(

class FakeRuntime:
    pass

print("Script created; expect failure code ('repair_reverification_failed',) when verify_and_repair reaches max_repair_passes=1 with contradiction alive.")
