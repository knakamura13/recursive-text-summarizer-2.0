#!/usr/bin/env python3
"""S4 live angle: direct verify_and_repair probe against real Ollama models.

Tests the self-certification guard and repair exhaustion branches with a real
document that triggers verification failures and subsequent repair attempts.
"""
import asyncio
from pathlib import Path
import tempfile

from summarizer.providers.ollama import OllamaProvider
from summarizer.tokenization import ConservativeUtf8TokenCounter
from summarizer.verification import (
    VerificationConfig,
    VerificationRuntime,
    verify_and_repair,
    split_draft_spans,
    build_source_lexical_index,
)

# Test document with verifiable claims and some that will fail
TEST_DOC = """At sunrise on Monday, Lina unlocked the fictional Marrow Bay field station and found a thin layer of gray dust across the western windowsill. She photographed it, labeled a sample jar, and called Tomas, the station's equipment manager. Neither could identify the material, though Lina thought it might have blown from the dry lakebed overnight.

By noon, Tomas had checked the roof filters and found them intact. Lina carried the jar to Dr. Chen at the harbor laboratory, then returned to Marrow Bay before the afternoon tide survey. During the survey, she noticed the same gray color on three reeds near Marker Six, but rain began before she could collect another sample.

On Tuesday morning, Dr. Chen reported that the jar contained mostly clay and salt with a small amount of plant fiber. The result did not establish where the dust originated. Lina and Tomas inspected Marker Six again, found no fresh deposit, and installed a covered sampler beside the reeds. They agreed to check it daily for one week before drawing any conclusion.

The temperature on Monday was 14 degrees Celsius and the wind came from the northwest at 12 kilometers per hour. On Tuesday it dropped to 8 degrees and the wind shifted to the northeast at 8 kilometers per hour."""

# A draft summary containing both correct and incorrect claims
DRAFT_SUMMARY = """Lina found gray dust on the windowsill at Marrow Bay on Monday. The dust was analyzed by Dr. Chen and found to be clay, salt, and plant fiber. The temperature on Monday was 25 degrees Celsius. Tomas checked the roof filters and found them damaged. Lina installed a sampler on Wednesday."""

async def run_probe():
    # Setup provider and runtime
    provider = OllamaProvider()
    counter = ConservativeUtf8TokenCounter()
    
    runtime = VerificationRuntime(
        provider=provider,
        counter=counter,
        model="qwen3.5:9b",
        provider_identity="ollama",
        timeout_seconds=120,
        context_window_tokens=131072,
    )
    
    config = VerificationConfig(
        enabled=True,
        max_repair_passes=2,
    )
    
    source_id = "test-doc"
    source_cores = {"test-doc": TEST_DOC}
    source_index = build_source_lexical_index(
        provenance_ids=("test-doc",),
        source=source_cores,
    )
    
    print("Running verify_and_repair with live Ollama (qwen3.5:9b)...")
    print(f"Draft: {DRAFT_SUMMARY[:200]}...")
    print()
    
    result = verify_and_repair(
        draft=DRAFT_SUMMARY,
        source_id=source_id,
        source_index=source_index,
        runtime=runtime,
        config=config,
        coordinator=None,
    )
    
    print("=== RESULT ===")
    print(f"Failed: {result.failed}")
    print(f"Exhausted: {result.exhausted}")
    print(f"Failure codes: {result.failure_codes}")
    print(f"Limitation codes: {result.limitation_codes}")
    print(f"Pass count: {len(result.pass_results)}")
    for i, pass_result in enumerate(result.pass_results):
        print(f"  Pass {i+1}: failed={pass_result.failed}, spans={len(pass_result.spans)}")
        for span in pass_result.spans:
            print(f"    Span {span.span_id}: {span.text[:80]}...")
    print(f"Repairs: {len(result.repairs)}")
    print(f"Generations: {len(result.generations)}")
    for gen in result.phase_generations:
        g = gen.generation
        print(f"  {gen.phase.value} pass={gen.pass_index}: {g.text[:100] if g else 'None'}...")
    print()
    print(f"Final text: {result.text}")
    print(f"Diagnostic codes: {result.diagnostic_codes}")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(run_probe())