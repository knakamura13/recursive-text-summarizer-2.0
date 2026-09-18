"""S4 desk-review repro: does verify_and_repair's pass-bound exhaustion path
actually get exercised by the existing suite, and what does it do when a
contradiction survives every configured repair pass?

Modeled on tests/test_verification_repair.py's ScriptedProvider pattern.
Read-only against the repo; does not modify any tracked file.

Run with:
  UV_OFFLINE=1 uv run --with-requirements requirements-dev.txt python \
    .review/wip/s4-criteria/repro_repair_exhaustion.py
from the repo root (a checkout at 301cc4d).
"""
from __future__ import annotations

import hashlib
import json
import sys

sys.path.insert(0, ".")

from summarizer.providers.base import GenerationResult
from summarizer.tokenization import ConservativeUtf8TokenCounter
from summarizer.verification import (
    VerificationConfig,
    VerificationRuntime,
    build_source_lexical_index,
    verify_and_repair,
)


def contradicted(pass_index: int, quote: str) -> tuple[str, str]:
    prefix = f"V{pass_index:02d}"
    return (
        json.dumps({"spans": [{"span_id": f"{prefix}S000001", "anchors": ["42"]}]}),
        json.dumps(
            {
                "findings": [
                    {
                        "claim_id": f"{prefix}C000001",
                        "verdict": "contradicted",
                        "evidence": [{"segment_id": "S000001", "exact_quote": quote}],
                    },
                    {
                        "claim_id": f"{prefix}C000002",
                        "verdict": "contradicted",
                        "evidence": [{"segment_id": "S000001", "exact_quote": quote}],
                    },
                ]
            }
        ),
    )


def repair(span_id: str, original_text: str, replacement: str) -> str:
    return json.dumps(
        {
            "repairs": [
                {
                    "span_id": span_id,
                    "original_hash": hashlib.sha256(original_text.encode()).hexdigest(),
                    "action": "replace",
                    "replacement": replacement,
                }
            ]
        }
    )


class ScriptedProvider:
    """Always finds the claim contradicted, no matter how many times we repair it."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return GenerationResult(next(self.responses), "scripted", "model")


def run(label: str, max_repair_passes: int, n_repair_rounds_scripted: int) -> None:
    draft = "The value is 42."
    responses: list[str] = []
    text = draft
    for round_index in range(n_repair_rounds_scripted):
        decompose_pass = 1 + 2 * round_index
        d, c = contradicted(decompose_pass, "value is 41")
        responses += [d, c]
        replacement = f"The value is 42 (attempt {round_index + 1})."
        responses.append(repair(f"V{decompose_pass:02d}S000001", text, replacement))
        text = replacement
        reverify_pass = decompose_pass + 1
        d2, c2 = contradicted(reverify_pass, "value is 41")
        responses += [d2, c2]

    provider = ScriptedProvider(responses)
    result = verify_and_repair(
        draft,
        source_id="a" * 64,
        source_index=build_source_lexical_index(
            provenance_ids=("S000001",), source={"S000001": "The value is 41."}
        ),
        runtime=VerificationRuntime(provider, ConservativeUtf8TokenCounter(), "model", 30, 10_000),
        config=VerificationConfig(enabled=True, max_repair_passes=max_repair_passes),
    )
    print(f"--- {label} (max_repair_passes={max_repair_passes}) ---")
    print("provider.calls           =", provider.calls)
    print("result.failed            =", result.failed)
    print("result.exhausted         =", result.exhausted)
    print("result.failure_codes     =", result.failure_codes)
    print("result.text              =", repr(result.text))
    print("result.repairs span_ids  =", [e.span_id for e in result.repairs])
    print("len(result.pass_results) =", len(result.pass_results))
    print()


if __name__ == "__main__":
    # Case A: default max_repair_passes=1. One repair round is scripted; its
    # reverification still shows a contradiction. With only 1 pass configured,
    # _verify_and_repair cannot recurse, so it must hit the direct `if failed:`
    # branch (verification.py ~1957-1986) and revert.
    run("default-config single exhaustion", max_repair_passes=1, n_repair_rounds_scripted=1)

    # Case B: max_repair_passes=2. Two repair rounds are scripted, both of
    # which still show a contradiction on reverification. This exercises the
    # recursive branch (config.max_repair_passes > 1 -> recurse) followed by
    # the nested call's own exhaustion (`continued.failed`), which is the
    # branch at verification.py ~1928-1943.
    run("nested exhaustion after 2 configured passes", max_repair_passes=2, n_repair_rounds_scripted=2)
