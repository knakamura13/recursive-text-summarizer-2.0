# Provider Boundary and Thin CLI Design

## Context

The legacy executable combines configuration, sentence chunking, prompt construction, OpenAI client setup, retry behavior, logging, and file orchestration in `main.py`. Importing the module downloads NLTK data, constructs a credential-dependent client, and attempts to configure file logging.

Issue #3 establishes reusable application seams without implementing the later generalized hierarchy. The project north star remains efficient summarization of extremely long artifacts with minimal source-signal loss. This issue therefore records model-call metadata needed for later empirical optimization, while deferring segmentation, hierarchy, grounding, concurrency, and evaluation policy.

## Goals

- Keep `python main.py` as a no-argument input-to-output workflow.
- Add validated CLI overrides for foundational and transitional settings.
- Make `main.py` a thin entry point over import-safe application modules.
- Make model generation injectable and independent of OpenAI types.
- Use the current OpenAI Responses API through a narrow adapter.
- Represent failures as actionable application exceptions.
- Preserve useful legacy behavior through the characterization suite.
- Capture provider metadata needed to measure future pipeline cost and efficiency.

## Non-goals

- Token-aware or structure-aware segmentation.
- Hierarchical reduction or synthesis strategies.
- Source spans, provenance graphs, grounding, or citations.
- Async generation, concurrency, backpressure, or batch APIs.
- Caching, checkpoints, resumability, or deduplication.
- Adaptive model routing or global cost and latency budgets.
- Prompt versioning, structured outputs, or semantic quality evaluation.

## Alternatives considered

### Keep workflow code in `main.py`

Extracting only configuration and the OpenAI adapter would minimize the diff, but orchestration would remain coupled to an executable module and require another structural rewrite for later pipeline issues.

### Introduce an async pipeline now

An async provider could improve future leaf-stage throughput, but concurrency cannot be designed safely before rate limits, ordering, retry budgets, provenance, and checkpoint semantics are defined. A synchronous protocol is sufficient for issue #3 and does not prevent a later async companion interface.

### Pipeline-ready synchronous foundation

The selected approach uses narrow immutable configurations, structured generation values, a provider decorator for retries, and an explicitly transitional legacy workflow. It provides stable measurement and failure seams without guessing at the future hierarchy.

## Package structure

```text
summarizer/
  __init__.py
  config.py
  text.py
  legacy_workflow.py
  cli.py
  providers/
    __init__.py
    base.py
    openai.py
    retrying.py
main.py
```

`main.py` calls `summarizer.cli.main()` and exits with its return code. Importing any application module must not download data, read credentials, construct a provider client, configure logging, call the network, or write files.

## Configuration

Configuration is divided by responsibility instead of accumulated in one mutable object:

- `AppConfig` contains input path, output path, model, and request timeout.
- `RetryPolicy` contains maximum attempts, initial delay, and backoff multiplier.
- `LegacyWorkflowConfig` contains character chunk size, maximum chunks, and dry-run mode.

All configuration objects are frozen dataclasses and validate themselves at construction. Empty paths or model names are invalid. Input and output paths must differ. Timeout, attempts, chunk size, and backoff values must be positive. Maximum chunks must be `-1` for unlimited processing or a positive integer.

The CLI exposes `--input`, `--output`, `--model`, `--timeout`, `--max-retries`, `--chunk-size`, `--max-chunks`, and `--dry-run`. No arguments retain `input.txt` and `output.txt` in the current working directory.

Chunk size and maximum chunks are intentionally stored in `LegacyWorkflowConfig`. They preserve the existing workflow but are not declared universal properties of the future pipeline.

Dry-run has one precise meaning: execute the file workflow without provider calls, use normalized source chunks as generation results, and write the assembled output normally.

## Provider boundary

The application depends on a synchronous protocol:

```python
class ModelProvider(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
```

`GenerationRequest` contains the model, system instructions, user input, timeout, and an optional operation identifier. `GenerationResult` contains normalized provider output plus provider name, resolved model, optional input and output token counts, finish status, and request identifier.

Structured results make future cost, throughput, and compression experiments possible without exposing OpenAI response objects to orchestration. Hierarchy-specific fields such as source spans, depth, or reduction stage are deferred.

Provider-independent exceptions distinguish transient timeout, rate-limit, connection, and server failures from fatal authentication, invalid-request, and invalid-response failures. Exceptions preserve a causal chain and sanitized diagnostic context. Source text and credentials must not appear in exception messages.

## OpenAI adapter

`OpenAIProvider` constructs `OpenAI()` lazily on the first generation call. The official client reads `OPENAI_API_KEY` from the environment. Application configuration does not contain credentials and does not load a project-specific `API_KEY` value.

The adapter uses `client.responses.create(...)`, maps the system prompt to `instructions`, maps the delimited user prompt to `input`, validates `response.output_text`, and captures available usage, model, status, and request ID metadata.

SDK retries are explicitly disabled because retry policy is owned by the provider decorator. SDK exceptions are translated once at the adapter boundary. Empty or malformed successful responses raise an invalid-response exception.

The refactor replaces the deprecated legacy `gpt-4-1106-preview` default with `gpt-4o-mini`, an inexpensive general text model supported by the Responses API. A focused compatibility test and operator documentation will make this intentional change explicit. Selecting an empirically optimal model remains deferred to evaluation issue #12.

## Retry decorator

`RetryingProvider` wraps any `ModelProvider`. It retries only typed transient failures, immediately propagates fatal failures, applies the immutable `RetryPolicy`, and preserves the final causal exception when attempts are exhausted. An injected sleeper makes backoff tests deterministic.

Keeping retries outside both the workflow and OpenAI adapter avoids transport policy in hierarchy traversal, prevents SDK and application retry multiplication, and permits later global retry budgets.

## Transitional workflow

`LegacyWorkflow` preserves the current flat behavior behind reusable methods:

1. Read the UTF-8 input file.
2. Split it with the characterized sentence-based chunker.
3. Apply positive prefix truncation from `max_chunks`.
4. Build the characterized prompt for each chunk in source order.
5. Generate through the injected provider, or normalize the chunk in dry-run mode.
6. Normalize each returned summary.
7. Join summaries with one newline.
8. Write the output only after every chunk succeeds.

The workflow does not define a general pipeline or hierarchy abstraction. Later issues can replace it while reusing configuration, provider, retry, and CLI composition seams.

Provider failures are never converted into summary text. A failed run does not write a new output. General atomic replacement, checkpoints, and recovery are deferred to reliability issue #11.

## CLI and errors

`summarizer.cli` owns argument parsing, configuration construction, logging setup, provider composition, workflow execution, user-facing error reporting, and exit codes. Validation errors use standard `argparse` behavior. Application failures produce concise stderr diagnostics and a nonzero result.

Logging is configured only during CLI execution. Reusable modules obtain named loggers but do not install handlers.

## Testing strategy

Implementation follows red-green-refactor cycles. Offline tests will cover:

- configuration defaults and every validation boundary;
- exact CLI defaults and flag overrides;
- the no-argument input-to-output workflow with an injected fake provider;
- import-time absence of client construction, downloads, logging configuration, network access, and file writes;
- request mapping and structured response metadata extraction;
- lazy OpenAI client construction and environment-owned credentials;
- empty or malformed Responses output;
- SDK exception translation with sanitized messages;
- transient retry, fatal non-retry, deterministic backoff, and causal exhaustion;
- proof that SDK and decorator retries do not multiply;
- dry-run behavior with zero provider calls;
- source ordering and output preservation when a generation fails;
- Unicode and very large input plumbing without specifying future segmentation behavior;
- retained legacy utility behavior outside intentional issue #3 changes.

Characterization tests will remain unchanged when they describe preserved behavior. Tests that assert import-time side effects or provider failures becoming summary text will be replaced by tests for the approved behavior.

## Future compatibility

The synchronous provider protocol is deliberately small. Later work may add an async companion, bounded scheduling, caching, provenance, and hierarchical node metadata around it. The structured request and result objects provide measurement and tracing hooks without making issue #3 responsible for those policies.
