#!/usr/bin/env python3
"""H10: Adaptive grounding reserve with 150K+ tokens and real tiktoken.

This version reuses exact schema-valid helper patterns from the test suite
to construct fake provider responses, rather than hand-building JSON.
"""

import json, sys, re
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from summarizer.config import AppConfig, StrategyConfig
from summarizer.ingestion import ingest_text
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer.tokenization import TiktokenCounter


def leaf_payload(segment_id: str, **overrides) -> str:
    """Construct a valid leaf summary response using exact test-suite schema.
    
    Copied from tests/test_leaf_parsing.py::payload() to ensure schema validity.
    """
    body = {
        "summary": f"Summary of {segment_id}.",
        "content_units": [
            {
                "text": f"Key fact from {segment_id}.",
                "kind": "fact",
                "evidence": [{"segment_id": segment_id, "quote": None}],
                "qualification": None,
                "uncertain": False,
            }
        ],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": [segment_id],
        "level": 0,
    }
    body.update(overrides)
    return json.dumps(body)


def merge_payload(level: int, provenance_ids: list[str], **overrides) -> str:
    """Construct a valid merge summary response using exact test-suite schema.
    
    Based on tests/test_merge_prompt.py::merged_payload() to ensure validity.
    """
    body = {
        "summary": f"Merged summary at level {level}.",
        "content_units": [
            {
                "text": "Key fact from merge.",
                "kind": "fact",
                "evidence": [{"segment_id": pid, "quote": None} for pid in provenance_ids],
                "qualification": None,
                "uncertain": False,
            }
        ],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": provenance_ids if provenance_ids else ["S000001"],
        "level": level,
    }
    body.update(overrides)
    return json.dumps(body)


class SchemaValidProvider:
    """Fake provider using exact schema patterns from test suite."""

    def __init__(self, debug=False):
        self.calls = []
        self.debug = debug

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        op_id = request.operation_id or ""
        
        # For leaf requests: operation_id is the segment_id (e.g., "S000001")
        # For merge requests: operation_id is "merge-L{level}" (e.g., "merge-L1")
        # For editorial: operation_id is "editorial-final"
        if op_id == "editorial-final":
            # EDITORIAL REQUEST: Return only a text field
            payload_json = json.dumps({"text": "Final editorial summary of the document."})
        elif op_id.startswith("S"):
            # LEAF REQUEST: Use operation_id as the segment being summarized
            payload_json = leaf_payload(op_id)
        else:
            # MERGE REQUEST: Extract segments from request text and determine level
            segment_ids = re.findall(r'"segment_id":"([^"]+)"', request.input_text)
            
            # Determine level from operation_id (format: "merge-L{level}")
            level = 1  # Default
            if op_id.startswith("merge-L"):
                try:
                    level = int(op_id.split("merge-L")[1])
                except (ValueError, IndexError):
                    level = 1  # Fallback to default
            
            payload_json = merge_payload(level, segment_ids if segment_ids else ["S000001"])
        
        return GenerationResult(
            text=payload_json, provider="fake", model=request.model
        )

def create_synthetic_doc(min_tokens: int = 160000) -> str:
    """Create varied synthetic document >= min_tokens."""
    prose = [
        "Machine learning algorithms process information through layers of learned features.",
        "Natural language processing turns text into actionable insights and understanding.",
        "Distributed systems require consensus mechanisms for reliable coordination.",
        "Database indexing significantly improves query performance at scale.",
        "Cryptographic protocols establish trust across untrusted networks.",
        "Software testing uncovers bugs and ensures system reliability and correctness.",
        "API design balances flexibility, performance, and ease of use for developers.",
        "Cloud infrastructure abstracts hardware complexity into manageable services.",
        "Caching strategies reduce latency and improve throughput for interactive systems.",
        "Monitoring and observability provide visibility into system health and behavior.",
    ]

    doc = []
    est_tokens = 0
    idx = 0
    while est_tokens < min_tokens:
        para = prose[idx % len(prose)]
        doc.append(para + " " * (idx % 3))
        idx += 1
        est_tokens = len(" ".join(doc).split()) * 1.3

    return " ".join(doc)


def main():
    print("[START] H10: Adaptive Grounding Reserve Test (Schema-Valid Fake Provider)")

    print("[1/5] Creating synthetic document...")
    doc_text = create_synthetic_doc(160000)
    
    print("[2/5] Verifying with real tiktoken...")
    try:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        token_count = len(enc.encode_ordinary(doc_text))
        print(f"  Document: {token_count} tokens")
        if token_count < 150000:
            return {
                "error": f"Document too small: {token_count} < 150000",
                "completed": False
            }
    except Exception as e:
        return {"error": f"Token count failed: {e}", "completed": False}

    print("[3/5] Ingesting...")
    try:
        document = ingest_text(doc_text)
    except Exception as e:
        return {"error": f"Ingestion failed: {e}", "completed": False}

    print("[4/5] Running pipeline with fake provider...")
    provider = SchemaValidProvider(debug=False)
    result = None
    try:
        result = run_pipeline(
            document,
            provider,
            TiktokenCounter.for_model("gpt-4o-mini"),
            app=AppConfig(),
            strategy=StrategyConfig(),
            config=PipelineConfig(),
        )
        print(f"  Pipeline completed successfully")
    except Exception as e:
        exc_type = type(e).__name__
        exc_msg = str(e)
        
        return {
            "error": f"{exc_type}: {exc_msg}",
            "completed": False,
            "reached_merge": False,
            "provider_calls": len(provider.calls),
            "exception_type": exc_type,
        }

    print("[5/5] Analyzing result...")
    
    # Analyze hierarchy from result.root
    # The root.level tells us how many levels deep we went (0 = no merges, 1+ = merges happened)
    hierarchy_levels = result.root.level + 1
    reached_merge = result.root.level > 0
    
    result_data = {
        "completed": True,
        "reached_merge": reached_merge,
        "hierarchy_levels": hierarchy_levels,
        "provider_calls": len(provider.calls),
        "document_tokens": token_count,
        "strategy_selected": result.strategy.strategy,
        "root_level": result.root.level,
    }

    print(f"  Levels: {hierarchy_levels}, Reached merge: {reached_merge}")
    print(f"  Root level: {result.root.level}")
    print(f"  Strategy: {result.strategy.strategy}")

    return result_data


if __name__ == "__main__":
    result = main()
    print()
    print("=" * 60)
    print("RESULT:")
    print(json.dumps(result, indent=2))
    print("=" * 60)
    sys.exit(0 if result.get("completed") else 1)
