#!/usr/bin/env python3
"""F-014 repro: Empirical test of OllamaProvider context handling."""

import sys
import json

sys.path.insert(0, '/Users/kylenakamura/documents-local/development-local/side-projects/recursive-text-summarizer')

from summarizer.providers.ollama import OllamaProvider
from summarizer.providers.base import GenerationRequest
from summarizer.tokenization import TiktokenCounter
import time

def test_real_request():
    """Send a real 20K+ token request and measure prompt_eval_count."""
    
    provider = OllamaProvider(host="http://localhost:11434")
    counter = TiktokenCounter.for_encoding("cl100k_base")
    
    # Create content that will be ~25,000 tokens
    # Use real text from the project
    base_text = """NP-completeness is a fundamental concept in computational complexity theory. 
    The theory of NP-completeness provides a framework for understanding the hardness of computational problems.
    Many real-world problems fall into the NP-complete category, which means they are believed to be intractable.
    The P versus NP problem asks whether polynomial-time solvable problems are the same as polynomial-time verifiable problems.
    This is one of the most important open problems in computer science and mathematics.
    NP-complete problems include the satisfiability problem, the traveling salesman problem, and many others.
    The complexity classes P and NP are central to understanding computational limits.
    When a problem is NP-complete, it means that finding a solution is as hard as verifying it.
    """
    
    # Repeat to get to ~25,000 tokens
    repetitions = 85  # Should give us roughly 25K tokens
    content = base_text * repetitions
    
    content_tokens = counter.count(content)
    instruction = "Briefly summarize the key points about NP-completeness in 2-3 sentences."
    instruction_tokens = counter.count(instruction)
    
    print(f"Test 1: ~25K token request")
    print(f"  Content tokens: {content_tokens}")
    print(f"  Instruction tokens: {instruction_tokens}")
    print(f"  Expected total: {content_tokens + instruction_tokens}")
    
    request = GenerationRequest(
        model="qwen3.5:9b",
        instructions=instruction,
        input_text=content,
        timeout_seconds=60.0
    )
    
    print(f"\n  Sending request...")
    start = time.time()
    result = provider.generate(request)
    elapsed = time.time() - start
    
    print(f"  Response received in {elapsed:.1f}s")
    print(f"  Input tokens (prompt_eval_count): {result.input_tokens}")
    print(f"  Output tokens (eval_count): {result.output_tokens}")
    print(f"  Finish reason: {result.finish_status}")
    print(f"  Text preview: {result.text[:100]}...")
    
    if result.input_tokens is not None:
        expected = content_tokens + instruction_tokens
        actual = result.input_tokens
        print(f"\n  Token analysis:")
        print(f"    Expected: {expected}")
        print(f"    Actual: {actual}")
        print(f"    Match: {abs(expected - actual) < 50}")  # Within reasonable margin
    
    if result.finish_status == "length":
        print(f"\n  ⚠️  CONTEXT LIMIT HIT: finish_reason='length'")
        return False
    else:
        print(f"\n  ✓ Request completed normally (finish_reason='{result.finish_status}')")
        return True


def test_large_request():
    """Test if we can push beyond the real context limit."""
    
    provider = OllamaProvider(host="http://localhost:11434")
    counter = TiktokenCounter.for_encoding("cl100k_base")
    
    # Try to send a VERY large request - intentionally oversized
    # If num_ctx=4096, this will fail/truncate
    # If num_ctx=262144, this will fail (too large for model)
    # But we want to see the actual finish_reason
    
    base_text = "The quick brown fox jumps over the lazy dog. " * 100
    
    # Try 100K tokens
    repetitions = 1100
    content = base_text * repetitions
    
    content_tokens = counter.count(content)
    instruction = "Summarize this text."
    instruction_tokens = counter.count(instruction)
    
    total_expected = content_tokens + instruction_tokens
    
    print(f"\nTest 2: Very large request (~{content_tokens:,} tokens)")
    print(f"  Content tokens: {content_tokens:,}")
    print(f"  Instruction tokens: {instruction_tokens}")
    print(f"  Expected total: {total_expected:,}")
    
    request = GenerationRequest(
        model="qwen3.5:9b",
        instructions=instruction,
        input_text=content,
        timeout_seconds=60.0
    )
    
    print(f"\n  Sending large request...")
    start = time.time()
    try:
        result = provider.generate(request)
        elapsed = time.time() - start
        
        print(f"  Response received in {elapsed:.1f}s")
        print(f"  Input tokens (prompt_eval_count): {result.input_tokens}")
        print(f"  Output tokens (eval_count): {result.output_tokens}")
        print(f"  Finish reason: {result.finish_status}")
        
        if result.finish_status == "length":
            print(f"\n  ⚠️  CONTEXT LIMIT HIT: finish_reason='length'")
            print(f"    Server accepted {result.input_tokens:,} tokens before hitting limit")
            if result.input_tokens is not None and result.input_tokens < 5000:
                print(f"    ↳ This suggests num_ctx ≈ {result.input_tokens}")
            return True
        else:
            print(f"\n  ✓ Request completed with finish_reason='{result.finish_status}'")
            print(f"    Tokens processed: {result.input_tokens:,}")
            return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Exception after {elapsed:.1f}s: {e}")
        return None


if __name__ == "__main__":
    print("="*60)
    print("F-014: Empirical num_ctx Verification")
    print("="*60)
    
    success1 = test_real_request()
    success2 = test_large_request()
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  25K token request: {'PASS' if success1 else 'FAIL'}")
    print(f"  100K+ token request: {'HIT_LIMIT' if success2 is True else 'COMPLETED' if success2 is False else 'EXCEPTION'}")
    print("="*60)
