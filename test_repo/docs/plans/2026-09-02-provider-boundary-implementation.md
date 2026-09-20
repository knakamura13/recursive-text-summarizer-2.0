# Provider Boundary and Thin CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the legacy executable into an import-safe package with validated CLI configuration, a structured provider protocol, an OpenAI Responses adapter, and provider-independent retry behavior.

**Architecture:** A thin `main.py` delegates to CLI composition. Frozen configuration objects separate application, retry, and transitional legacy workflow settings. `LegacyWorkflow` consumes a synchronous `ModelProvider`; an OpenAI adapter translates SDK requests and responses, while a decorator owns deterministic retry policy.

**Tech Stack:** Python 3.10+, standard-library dataclasses, argparse, pathlib, logging and typing; NLTK sentence tokenization; OpenAI Python SDK Responses API; pytest.

---

Implementation must follow `@superpowers:test-driven-development` for every behavior change. Run tests through `.venv/bin/python -m pytest`. Keep all tests offline under the shared socket guard. Do not modify historical scripts under `omscs-ml-lectures/`.

### Task 1: Add immutable validated configuration

**Files:**
- Create: `summarizer/__init__.py`
- Create: `summarizer/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing default and immutability tests**

Create tests asserting:

```python
def test_configuration_defaults_are_legacy_compatible() -> None:
    app = AppConfig()
    retry = RetryPolicy()
    workflow = LegacyWorkflowConfig()

    assert app.input_path == Path("input.txt")
    assert app.output_path == Path("output.txt")
    assert app.model == "gpt-4o-mini"
    assert app.timeout_seconds == 180
    assert retry.max_attempts == 5
    assert retry.initial_delay_seconds == 1
    assert retry.backoff_multiplier == 2
    assert workflow.chunk_size == 1000
    assert workflow.max_chunks == -1
    assert workflow.dry_run is False
```

Also assert assigning to any field raises `dataclasses.FrozenInstanceError`.

The model default intentionally replaces the deprecated legacy `gpt-4-1106-preview` alias with `gpt-4o-mini`. This is a focused compatibility change: the no-argument workflow still uses an inexpensive general text model supported by the Responses API. Model-quality selection remains deferred to evaluation issue #12.

**Step 2: Run the tests and verify RED**

Run:

```sh
.venv/bin/python -m pytest tests/test_config.py -v
```

Expected: collection fails because `summarizer.config` does not exist.

**Step 3: Implement the minimal frozen dataclasses**

Add `AppConfig`, `RetryPolicy`, and `LegacyWorkflowConfig` with the defaults above. Use `Path` fields and `__post_init__` validation.

**Step 4: Run the tests and verify GREEN**

Run the targeted test file, then the full suite. Expected: all tests pass.

**Step 5: Add failing validation parameterization**

Cover:

- empty model;
- empty input or output path;
- identical input and output paths after `Path.resolve(strict=False)`;
- zero or negative timeout;
- zero or negative attempts, initial delay, or multiplier;
- zero or values below `-1` for maximum chunks;
- zero or negative chunk size.

Assert each raises `ValueError` with the offending field named.

**Step 6: Verify RED, implement validation, and verify GREEN**

Run the new cases before and after implementation. Then run the full suite.

**Step 7: Commit**

```sh
git add summarizer/__init__.py summarizer/config.py tests/test_config.py
git commit -m "feat: add validated summarizer configuration"
```

### Task 2: Define structured provider values and failure taxonomy

**Files:**
- Create: `summarizer/providers/__init__.py`
- Create: `summarizer/providers/base.py`
- Create: `tests/providers/test_base.py`

**Step 1: Write failing value-object tests**

Specify frozen `GenerationRequest` and `GenerationResult` values. Test construction, immutability, and rejection of blank model, instructions, input, provider, result model, result text, and nonpositive timeout. Token counts must accept `None` or nonnegative integers and reject negative integers.

Use this intended shape:

```python
request = GenerationRequest(
    model="gpt-4o-mini",
    instructions="Summarize accurately.",
    input_text='\n"""\nSource\n"""\n',
    timeout_seconds=30,
    operation_id="leaf-17",
)
result = GenerationResult(
    text="Summary",
    provider="openai",
    model="gpt-4o-mini-2024-07-18",
    input_tokens=20,
    output_tokens=4,
    finish_status="completed",
    request_id="req_123",
)
```

**Step 2: Run and verify RED**

Expected: the provider module is missing.

**Step 3: Implement values, protocol, and exceptions**

Define `ModelProvider` as a runtime-checkable protocol with `generate`. Define:

- `ProviderError`;
- `TransientProviderError`;
- `ProviderTimeoutError`;
- `ProviderRateLimitError`;
- `ProviderConnectionError`;
- `ProviderServerError`;
- `ProviderAuthenticationError`;
- `ProviderRequestError`;
- `ProviderResponseError`;
- `ProviderRetriesExhaustedError`.

Transient categories subclass `TransientProviderError`; fatal categories subclass only `ProviderError`. `ProviderRetriesExhaustedError` records the attempt count without copying source input into its message.

**Step 4: Verify GREEN and commit**

```sh
.venv/bin/python -m pytest tests/providers/test_base.py -v
.venv/bin/python -m pytest -q
git add summarizer/providers tests/providers/test_base.py
git commit -m "feat: define structured model provider contract"
```

### Task 3: Add the retrying provider decorator

**Files:**
- Create: `summarizer/providers/retrying.py`
- Create: `tests/providers/test_retrying.py`

**Step 1: Write one failing transient-retry test**

Use a small scripted provider that raises two `ProviderConnectionError` instances, then returns a real `GenerationResult`. Assert three calls, returned object identity, and delays `[1, 2]`.

**Step 2: Verify RED**

Expected: `RetryingProvider` is missing.

**Step 3: Implement the minimal decorator**

Constructor dependencies:

```python
RetryingProvider(
    provider: ModelProvider,
    policy: RetryPolicy,
    sleeper: Callable[[float], None] = time.sleep,
)
```

Retry only `TransientProviderError`. Compute the next delay from the immutable policy.

**Step 4: Verify GREEN**

Run the targeted test and full suite.

**Step 5: Add failing edge-case tests**

Cover:

- fatal failures are attempted once and re-raised unchanged;
- exhaustion raises `ProviderRetriesExhaustedError` from the last transient error;
- the exhaustion error exposes the configured attempt count;
- no sleep occurs after the final attempt;
- one configured attempt works without sleeping.

**Step 6: Verify RED, implement, and verify GREEN**

Inspect `exc_info.value.__cause__` in the causal-chain test.

**Step 7: Commit**

```sh
git add summarizer/providers/retrying.py tests/providers/test_retrying.py
git commit -m "feat: add provider retry policy decorator"
```

### Task 4: Adapt the OpenAI Responses API

**Files:**
- Create: `summarizer/providers/openai.py`
- Create: `tests/providers/test_openai.py`

**Step 1: Write a failing request and response adaptation test**

Inject a client factory instead of patching network internals. Its fake client must expose `responses.create`. Assert the adapter:

- does not call the factory at construction;
- calls the factory once on first generation and reuses the client;
- creates the SDK client with `max_retries=0` and no explicit API key;
- calls `responses.create` with `model`, `instructions`, `input`, and `timeout` from `GenerationRequest`;
- returns text, actual response model, token counts, status, and `_request_id` in `GenerationResult`.

Representative fake response:

```python
SimpleNamespace(
    output_text="  concise\nsummary  ",
    model="gpt-4o-mini-2024-07-18",
    status="completed",
    usage=SimpleNamespace(input_tokens=100, output_tokens=12),
    _request_id="req_123",
)
```

**Step 2: Verify RED**

Expected: `OpenAIProvider` is missing.

**Step 3: Implement minimal lazy adaptation**

Default the factory through a function that imports and constructs `OpenAI(max_retries=0)` only when invoked. Do not import or construct the client at module import time. Normalize output whitespace at the adapter boundary.

**Step 4: Verify GREEN**

Run the targeted test and full suite.

**Step 5: Add failing malformed-response tests**

Cover missing, `None`, blank, and non-string `output_text`; absent usage; and absent optional response metadata. Missing optional metadata must yield `None`, while invalid text raises `ProviderResponseError`.

**Step 6: Add failing exception-translation tests**

Use real OpenAI exception classes with minimal fake requests/responses where practical. Verify translation of:

- `APITimeoutError`;
- `RateLimitError`;
- `APIConnectionError`;
- `AuthenticationError`;
- other 4xx `APIStatusError`;
- 5xx `APIStatusError`.

Assert translated messages contain the failure category and status or request ID when available, but do not contain the generation input or a sentinel credential.

**Step 7: Verify RED, implement translation, and verify GREEN**

Order exception handlers from most specific to general. Chain translated errors with `raise ... from error`.

**Step 8: Add an import-isolation test**

Start a subprocess that removes `OPENAI_API_KEY`, imports `summarizer.providers.openai`, and asserts no client construction or network attempt occurs. Keep the global offline guard active.

**Step 9: Commit**

```sh
git add summarizer/providers/openai.py tests/providers/test_openai.py
git commit -m "feat: adapt OpenAI Responses provider"
```

### Task 5: Extract text utilities and prompt construction

**Files:**
- Create: `summarizer/text.py`
- Create: `tests/test_text.py`
- Modify: `tests/test_legacy_utilities.py`

**Step 1: Write failing tests against the package API**

Port the existing whitespace and sentence-chunk characterization cases to `summarizer.text`. Add a prompt test asserting the exact legacy system instruction and delimited user text.

Functions:

```python
normalize_whitespace(text: str) -> str
chunk_text_by_sentences(
    text: str,
    max_chunk_size: int,
    sentence_tokenizer: Callable[[str], list[str]] = sent_tokenize,
) -> list[str]
build_generation_request(
    chunk: str,
    *,
    model: str,
    timeout_seconds: float,
    operation_id: str | None = None,
) -> GenerationRequest
```

The injectable tokenizer prevents tests from downloading data and leaves segmentation replacement to issue #4.

**Step 2: Verify RED**

Expected: `summarizer.text` is missing.

**Step 3: Move the characterized behavior without improving it**

Preserve current separator accounting, long-sentence behavior, leading-space behavior, prompt text, and delimiter exactly. Importing NLTK tokenization is allowed; downloading resources is not.

**Step 4: Verify GREEN**

Run new text tests, legacy utility tests, and full suite.

**Step 5: Commit**

```sh
git add summarizer/text.py tests/test_text.py tests/test_legacy_utilities.py
git commit -m "refactor: extract characterized text utilities"
```

### Task 6: Implement the injected legacy workflow

**Files:**
- Create: `summarizer/legacy_workflow.py`
- Create: `tests/test_legacy_workflow.py`

**Step 1: Write a failing in-memory workflow test**

Construct `LegacyWorkflow` with an injected provider and sentence tokenizer. Run against an input string containing two characterized chunks. Assert:

- source-order `GenerationRequest` values with stable operation IDs;
- output equals normalized summaries joined by one newline;
- both structured `GenerationResult` values are retained in a returned workflow result for future metrics.

The workflow result should contain final text and an immutable tuple of provider results. It must not contain OpenAI types.

**Step 2: Verify RED**

Expected: `LegacyWorkflow` is missing.

**Step 3: Implement minimal string-to-result orchestration**

Keep file I/O separate from core transformation:

```python
workflow.summarize(text: str) -> WorkflowResult
run_file_workflow(
    app_config: AppConfig,
    workflow: LegacyWorkflow,
) -> WorkflowResult
```

`run_file_workflow` reads UTF-8, calls `summarize`, then writes only after successful completion.

**Step 4: Verify GREEN**

Run targeted and full tests.

**Step 5: Add failing behavior tests**

Cover:

- positive `max_chunks` keeps only the prefix;
- `-1` processes all chunks;
- dry-run makes zero provider calls and returns normalized chunks;
- provider failure leaves a preexisting output unchanged;
- provider failure does not create a missing output;
- Unicode input and output round-trip;
- a large synthetic input crosses the application boundary without truncation beyond configured `max_chunks`;
- provider error text is never treated as a successful summary.

**Step 6: Verify RED, implement, and verify GREEN**

Do not add checkpointing, atomic replacement, concurrency, or hierarchy.

**Step 7: Commit**

```sh
git add summarizer/legacy_workflow.py tests/test_legacy_workflow.py
git commit -m "feat: add injectable legacy summarization workflow"
```

### Task 7: Build the CLI composition root

**Files:**
- Create: `summarizer/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write failing parser tests**

Test `parse_args([])` exact defaults and one case overriding every supported flag:

```text
--input --output --model --timeout --max-retries
--chunk-size --max-chunks --dry-run
```

Assert parser errors for invalid numeric and path combinations without invoking a provider.

**Step 2: Verify RED, implement parsing, and verify GREEN**

Parse arguments into the three validated configuration objects, not an unvalidated global namespace.

**Step 3: Write a failing CLI workflow test**

Inject a provider factory and run `cli.main([])` from a temporary directory containing `input.txt`. Assert:

- `output.txt` contains the fake summary;
- provider requests use the default model and timeout;
- return code is zero;
- no real OpenAI client or network is used.

**Step 4: Verify RED, implement composition, and verify GREEN**

Allow tests to inject the provider factory while production defaults to `OpenAIProvider`. Wrap production and fake providers with `RetryingProvider` at the composition root.

**Step 5: Add failing error and logging tests**

Cover:

- provider failure prints an actionable message to stderr and returns nonzero;
- missing input returns nonzero without creating output;
- logging handlers are configured only inside `main` execution;
- dry-run succeeds without constructing the OpenAI client;
- SDK retry disabling and decorator attempts result in exactly the configured total provider calls.

**Step 6: Verify RED, implement, and verify GREEN**

Use named loggers in reusable modules. Do not call `logging.basicConfig` at import time.

**Step 7: Commit**

```sh
git add summarizer/cli.py tests/test_cli.py
git commit -m "feat: add validated summarizer CLI"
```

### Task 8: Replace `main.py` with an import-safe entry point

**Files:**
- Modify: `main.py`
- Modify: `tests/test_legacy_cli.py`
- Modify: `tests/test_legacy_import.py`
- Modify: `tests/test_legacy_provider.py`
- Modify: `tests/support/legacy_loader.py`
- Create: `tests/test_entrypoint.py`

**Step 1: Write failing entry-point tests**

Use a subprocess or `runpy` with injected CLI composition. Assert:

- importing `main` does not initialize OpenAI, download NLTK data, configure root logging, call the network, or write files;
- executing `python main.py` delegates once to `summarizer.cli.main`;
- its process status reflects the CLI return code;
- no arguments still use `input.txt` and `output.txt`.

**Step 2: Verify RED**

Expected: legacy import side effects violate the new assertions.

**Step 3: Replace the executable body**

Target shape:

```python
from summarizer.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 4: Verify GREEN**

Run entry-point, CLI, and import tests.

**Step 5: Reconcile characterization tests**

Remove the legacy loader and tests only where they assert intentionally replaced seams:

- import-time NLTK download;
- import-time OpenAI construction;
- import-time file logging;
- Chat Completions request shape;
- provider errors returned as summary text;
- the generic-exception `UnboundLocalError` defect.

Keep or port tests for unchanged utility, prompt, chunk ordering, retry-count, output-joining, and no-argument behavior. Delete `tests/support/legacy_loader.py` only when no retained test imports it.

**Step 6: Run the full suite**

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q --import-mode=importlib
```

Expected: all tests pass in both modes with no network access.

**Step 7: Commit**

```sh
git add main.py tests
git commit -m "refactor: make main a thin import-safe entry point"
```

### Task 9: Update dependencies and operator documentation

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`
- Modify: `docs/legacy-baseline.md`
- Create: `tests/test_documented_cli.py`

**Step 1: Write failing documentation smoke tests**

Test that `python main.py --help` exits zero and lists every documented flag. Add a test that parses the README command examples without calling the provider.

**Step 2: Verify RED**

Expected: the new flags and examples are not documented yet.

**Step 3: Update dependencies**

- Require a tested OpenAI SDK version that supports Responses and `max_retries=0`.
- Remove `pathlib`, because it is part of the standard library.
- Remove `requests` and `python-dotenv` if no production code imports them after the refactor.
- Retain NLTK and tqdm only if production code still uses them.

Regenerate no lockfile unless the project adopts one in a separate decision.

**Step 4: Update documentation**

Document:

- no-argument usage;
- every CLI override;
- `OPENAI_API_KEY` credential setup;
- dry-run behavior;
- current default model and the explicit legacy-model change;
- nonzero failure behavior;
- provider and retry architecture;
- which legacy limitations issue #3 fixes and which remain for later issues.

Do not claim hierarchical, source-grounded, concurrent, or token-aware behavior yet.

**Step 5: Verify GREEN and dependency integrity**

```sh
.venv/bin/python -m pytest tests/test_documented_cli.py -v
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q --import-mode=importlib
```

**Step 6: Commit**

```sh
git add requirements.txt README.md docs/legacy-baseline.md tests/test_documented_cli.py
git commit -m "chore: document provider-neutral CLI workflow"
```

### Task 10: Verify issue #3 acceptance and scope

**Files:**
- Modify if needed: files changed in Tasks 1 through 9

**Step 1: Run focused acceptance tests**

```sh
.venv/bin/python -m pytest \
  tests/test_config.py \
  tests/providers \
  tests/test_text.py \
  tests/test_legacy_workflow.py \
  tests/test_cli.py \
  tests/test_entrypoint.py \
  tests/test_documented_cli.py -v
```

Expected: all focused tests pass without network access.

**Step 2: Run full verification**

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q --import-mode=importlib
.venv/bin/python -m pip check
git diff --check main...HEAD
```

Expected: all commands succeed.

**Step 3: Verify import behavior directly**

From an empty temporary directory with `OPENAI_API_KEY` unset, import `main`, `summarizer.cli`, and every provider module. Assert the directory remains empty and no network guard fires.

**Step 4: Verify repository scope**

Confirm:

- `omscs-ml-lectures/` is unchanged;
- no credentials, logs, generated summaries, caches, or virtual-environment files are tracked;
- the executable has no mutable configuration globals;
- only the OpenAI adapter imports OpenAI;
- only the CLI composition root constructs the production provider;
- no provider exception is converted into summary text;
- no async, concurrency, hierarchy, provenance, caching, or evaluation behavior was introduced.

Use codebase-memory `detect_changes` for the branch blast radius, then call `check_index_coverage` for every relied-on changed code path. Read any reported missed ranges directly.

**Step 5: Request independent review**

Invoke `@superpowers:requesting-code-review`. Ask for both issue #3 acceptance compliance and architecture quality, with special attention to retry multiplication, sanitized errors, import side effects, compatibility, and future long-artifact efficiency.

**Step 6: Address findings with TDD**

For each accepted behavior change, write or strengthen a failing test before changing production code. Re-run focused and full verification.

**Step 7: Commit final corrections if required**

Use only `feat`, `fix`, `chore`, or `refactor` commit types.

**Step 8: Stop at the delivery gate**

Report the verified branch state. Do not push, create a pull request, merge, close issue #3, or remove the worktree without separate user direction.
