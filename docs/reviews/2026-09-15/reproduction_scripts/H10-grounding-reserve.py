#!/usr/bin/env python3
"""H10: Adaptive grounding reserve with 150K+ tokens and real tiktoken."""

import json, re, sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from summarizer.config import AppConfig, StrategyConfig
from summarizer.ingestion import ingest_text
from summarizer.pipeline import PipelineConfig, run_pipeline
from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer.tokenization import TiktokenCounter


class SchemaValidProvider:
    """Fake provider matching exact SummaryNode schema."""

    def __init__(self):
        self.calls = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        op_id = request.operation_id or ""

        if "editorial" in op_id.lower():
            payload = {"text": "Final summary."}
        elif op_id.startswith("S") or "leaf" in op_id.lower():
            # Leaf: level=0, return proper SummaryNode schema
            payload = {
                "summary": "Segment summary.",
                "content_units": [],  # Empty list of ContentUnit
                "entities": [],
                "qualifications": [],  # Empty list of GroundedAnnotation
                "contradictions": [],  # Empty list of GroundedAnnotation
                "quotations": [],  # Empty list of EvidenceItem
                "provenance": [op_id],
                "level": 0,
            }
        elif op_id.startswith("L") or "merge" in op_id.lower():
            # Merge: extract level, return proper SummaryNode schema
            level = int((op_id or "L1").rsplit("L", 1)[1])
            identifiers = re.findall(r'"segment_id":"([SD]\d+)"', request.input_text)
            payload = {
                "summary": f"Merged level {level}.",
                "content_units": [],
                "entities": [],
                "qualifications": [],
                "contradictions": [],
                "quotations": [],
                "provenance": identifiers[:1] if identifiers else [],
                "level": level,
            }
        else:
            payload = {"text": "Response."}

        return GenerationResult(
            text=json.dumps(payload), provider="fake", model=request.model
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
    print("[START] H10: Adaptive Grounding Reserve Test")

    print("[1/5] Creating synthetic document...")
    doc_text = create_synthetic_doc(160000)
    
    print("[2/5] Verifying with real tiktoken...")
    try:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        token_count = len(enc.encode_ordinary(doc_text))
        print(f"  Document: {token_count} tokens")
        if token_count < 150000:
            return {"error": f"Document too small: {token_count} < 150000", "completed": False}
    except Exception as e:
        return {"error": f"Token count failed: {e}", "completed": False}

    print("[3/5] Ingesting...")
    try:
        document = ingest_text(doc_text)
    except Exception as e:
        return {"error": f"Ingestion failed: {e}", "completed": False}

    print("[4/5] Running pipeline...")
    provider = SchemaValidProvider()
    try:
        result = run_pipeline(
            document,
            provider,
            TiktokenCounter.for_model("gpt-4o-mini"),
            app=AppConfig(),
            strategy=StrategyConfig(),
            config=PipelineConfig(),
        )
        print(f"  Pipeline completed")
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {str(e)[:100]}",
            "completed": False,
            "reached_merge": False,
            "provider_calls": len(provider.calls),
        }

    print("[5/5] Analyzing...")
    
    # Analyze hierarchy
    levels = 1
    if result.root and hasattr(result.root, 'children'):
        node = result.root
        while node.children:
            levels += 1
            node = node.children[0]
    
    merges = sum(1 for n in result.nodes if n.parent is not None)
    reached_merge = any(n.parent is not None for n in result.nodes)

    result_data = {
        "completed": True,
        "reached_merge": reached_merge,
        "hierarchy_levels": levels,
        "merge_count": merges,
        "provider_calls": len(provider.calls),
        "document_tokens": token_count,
        "strategy_selected": result.strategy.strategy,
    }

    print(f"  Levels: {levels}, Merges: {merges}, Reached merge: {reached_merge}")
    print(f"  Strategy: {result.strategy.strategy}")

    return result_data


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("completed") else 1)
