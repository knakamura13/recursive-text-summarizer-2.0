# Recursive Text Summarizer

This project is being rebuilt as a generalized, source-grounded hierarchical summarization pipeline for extremely long artifacts. The current application provides a provider-neutral foundation while preserving the legacy flat, sentence-chunked workflow.

Canonical ingestion and token-aware segmentation are now available as library components in `summarizer.ingestion`, `summarizer.tokenization`, and `summarizer.segmentation`. The command-line workflow still runs the legacy flat, sentence-chunked path and does not consume them yet.

Hierarchical reduction, source grounding, concurrency, and quality evaluation are not implemented yet.

## Requirements

- Python 3.10 or newer
- An OpenAI API key for hosted generation, or a running Ollama service for
  local generation

Create a virtual environment and install the dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the credential through the environment. The application never accepts credentials through command-line configuration:

```sh
export OPENAI_API_KEY="your-api-key"
```

For local generation, install [Ollama](https://ollama.com/download), start its
service, and pull a model. A local Ollama service does not require an API key:

```sh
ollama serve
```

The Ollama desktop application normally starts the service itself. Run
`ollama serve` in a separate terminal when operating Ollama without the desktop
application. After the service is available, pull the model you want to use:

```sh
ollama pull gemma3:4b
```

## Usage

The no-argument workflow reads `input.txt` and writes `output.txt` in the current directory:

```sh
python main.py
```

Paths and foundational settings can be overridden:

```sh
python main.py --input source.txt --output summary.txt
python main.py --model gpt-4o-mini --timeout 180 --max-retries 5
python main.py --chunk-size 1000 --max-chunks 10
```

Select Ollama and any model tag already installed in that service:

```sh
python main.py --provider ollama --model gemma3:4b
python main.py --provider ollama --model qwen3.8
```

The standard Ollama host is `http://localhost:11434`. Override it when the
service is reachable elsewhere:

```sh
python main.py --provider ollama --ollama-host http://ollama.internal:11434 --model gemma3:4b
```

The application uses Ollama's native non-streaming chat API. It does not pull or
manage models automatically, so a missing tag produces an actionable error.

Run the file workflow without constructing or calling a provider:

```sh
python main.py --dry-run
```

Dry-run mode still reads, chunks, normalizes, and writes the source. It is intended for configuration and file-workflow checks, not summary-quality evaluation.

Run `python main.py --help` for the complete option reference.

## Defaults and failures

The default provider is `openai`, and the default model is `gpt-4o-mini`. This intentionally replaces the legacy `gpt-4-1106-preview` mapping with a currently supported, inexpensive model compatible with the OpenAI Responses API. Model selection will be evaluated separately as the hierarchical pipeline develops.

Configuration errors are reported by the argument parser. Missing files, provider failures, and exhausted retries produce an actionable stderr message and a nonzero process exit. Provider errors are never written as summary text, and output is written only after all configured chunks succeed.

## Architecture

`main.py` is an import-safe entry point over `summarizer.cli`. The CLI constructs immutable configuration, the selected OpenAI or native Ollama adapter, a provider-independent retry decorator, and the transitional legacy workflow.

The provider contract returns structured generation metadata, including the resolved model, token usage, finish status, and request ID when available. This data will support later cost, throughput, and compression evaluation without coupling pipeline orchestration to provider response objects.

The current workflow still uses NLTK sentence tokenization, fixed character-size chunks, independent summaries, and newline concatenation. These are compatibility behaviors scheduled for replacement by later issues.

## Ingestion and segmentation

`summarizer.ingestion` reads a file into an immutable `SourceDocument` holding canonical text and a SHA-256 `source_id` derived from it. Canonicalization is narrow: it strips a leading byte order mark, converts CRLF and CR line endings to LF, removes trailing spaces and tabs from each line, and trims outer blank lines. Headings, indentation, list markers, and internal blank lines survive unchanged.

Every character offset produced by segmentation resolves against that canonical text, not against the original bytes, so `document.text[segment.context_start:segment.context_end] == segment.text` always holds. Concatenating the core ranges of all segments in order reproduces the canonical document.

`summarizer.segmentation` splits a document by structure first — headings, then paragraphs and lists, then sentences, then a token-safe character fallback — and packs consecutive units up to the configured token budget. Each segment owns a disjoint *core* range. Overlap is configured in tokens, defaults to zero, and only ever widens the *context* range around a core; it never moves a core boundary, changes a segment identifier, or transfers evidence ownership. When a core already fills the budget, overlap is reduced, to zero if necessary.

Token accounting is injected through the `TokenCounter` protocol, and segmentation never contacts a provider:

- OpenAI models recognized by `tiktoken` get exact counts for the selected encoding. Constructing that counter can download an uncached vocabulary; counting afterwards is local.
- Arbitrary Ollama tags and otherwise unsupported models fall back to a conservative estimator that treats every UTF-8 byte as a token. It deliberately under-packs, and reports `exact` as false. Injecting an exact counter for such a model recovers that capacity.

Boundary searches are verified rather than assumed: a returned boundary is always recounted against the budget. Because byte-pair encoders are not monotonic in text length — adding a character can *lower* a token count — a search is not guaranteed to find the largest boundary that would have fit. Segments may therefore be slightly smaller than the budget allows, which is safe; no segment ever exceeds it.

Historical scripts under `omscs-ml-lectures/` remain available but are not part of the modern application entry point.

## Tests

Install development dependencies and run the offline suite:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pytest -q --import-mode=importlib
```

The suite blocks outbound socket connections and uses deterministic provider
fakes. It does not require an OpenAI credential, a running Ollama service, or a
locally installed model.

## Contributing

For major changes, open an issue before submitting a pull request.
