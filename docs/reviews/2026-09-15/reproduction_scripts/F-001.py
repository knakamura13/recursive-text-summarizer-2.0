#!/usr/bin/env python3
"""Reproduce acceptance of an Ollama length-terminated structured summary."""

from __future__ import annotations

import json
from types import SimpleNamespace

from nltk.tokenize import PunktSentenceTokenizer

from summarizer.direct import summarize_direct, whole_document_segment
from summarizer.ingestion import ingest_text
from summarizer.leaf import build_leaf_request
from summarizer.providers.ollama import OllamaProvider
from summarizer.tokenization import TiktokenCounter


class LengthTerminatedClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content),
            model="local-test-model",
            done=True,
            done_reason="length",
            prompt_eval_count=12,
            eval_count=64,
        )


def main() -> None:
    text = "The archive moved in March. The index was rebuilt twice."
    document = ingest_text(text)
    counter = TiktokenCounter.for_encoding("cl100k_base")
    sentences = PunktSentenceTokenizer().tokenize(text)
    payload = json.dumps(
        {
            "summary": "The archive moved and the index was rebuilt.",
            "content_units": [
                {
                    "text": "The archive moved in March.",
                    "kind": "fact",
                    "evidence": [{"segment_id": "D000001", "quote": None}],
                    "qualification": None,
                    "uncertain": False,
                }
            ],
            "entities": ["archive"],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": ["D000001"],
            "level": 0,
        }
    )
    client = LengthTerminatedClient(payload)
    provider = OllamaProvider(client_factory=lambda **_kwargs: client)

    raw = provider.generate(
        build_leaf_request(
            whole_document_segment(document, counter),
            model="local-test-model",
            timeout_seconds=30,
        )
    )
    node = summarize_direct(
        document,
        provider,
        counter,
        model="local-test-model",
        timeout_seconds=30,
    )

    print(f"PUNKT_SENTENCES={len(sentences)}")
    print(f"COUNTER={counter.identity};TOKENS={counter.count(text)}")
    print(f"ADAPTER_FINISH_STATUS={raw.finish_status}")
    print(f"ADAPTER_RAISED=False")
    print(f"DIRECT_NODE_SUMMARY={node.summary}")
    print(f"DIRECT_NODE_LEVEL={node.level}")
    print("RESULT=OBSERVED: terminal schema-valid response was accepted; finish status was preserved")


if __name__ == "__main__":
    main()
