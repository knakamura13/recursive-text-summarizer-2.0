"""Bounded live check for F-003's alleged installed-Ollama contract gap."""

from summarizer.direct import summarize_direct
from summarizer.ingestion import ingest_text
from summarizer.leaf import LeafSummaryError
from summarizer.providers.ollama import OllamaProvider
from summarizer.tokenization import TiktokenCounter


document = ingest_text(
    "The archive moved in March. The index was rebuilt afterwards. "
    "Staffing was unchanged."
)
try:
    node = summarize_direct(
        document,
        OllamaProvider(),
        TiktokenCounter.for_encoding("cl100k_base"),
        model="qwen3.5:9b",
        timeout_seconds=120,
    )
except LeafSummaryError as error:
    # A model producing schema-valid but semantically invalid placeholders is
    # a model response outcome. The real pipeline rejects it before output.
    print("live_structured_contract=failed_closed")
    print(f"error={error}")
else:
    assert node.level == 0
    assert node.provenance == ("D000001",)
    assert node.summary.strip()
    print("live_structured_contract=accepted")
    print(f"model=qwen3.5:9b level={node.level} provenance={node.provenance}")
