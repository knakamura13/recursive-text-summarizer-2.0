# Native Ollama Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a native Ollama provider and CLI selection so the existing summarizer can run against local models without changing workflow orchestration.

**Architecture:** Add a lazy, injectable `OllamaProvider` beside the OpenAI adapter and map native chat responses into the existing provider-neutral values. Extend validated configuration and CLI composition to choose a provider, then retain the existing retry decorator and `LegacyWorkflow` unchanged.

**Tech Stack:** Python 3.12, official `ollama` Python client, `argparse`, frozen dataclasses, pytest, deterministic fakes

---

### Task 1: Add validated provider configuration

**Files:**
- Modify: `summarizer/config.py`
- Modify: `summarizer/cli.py`
- Test: `tests/test_config.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing configuration tests**

Add tests asserting:

```python
assert AppConfig().provider == "openai"
assert AppConfig().ollama_host == "http://localhost:11434"

with pytest.raises(ValueError, match="provider"):
    AppConfig(provider="other")

with pytest.raises(ValueError, match="ollama_host"):
    AppConfig(ollama_host="")
```

Extend CLI parsing tests to assert `--provider ollama --ollama-host
http://ollama.internal:11434 --model qwen3.8` reaches `AppConfig` unchanged.

**Step 2: Run tests to verify they fail**

Run: `uv run pytest -q tests/test_config.py tests/test_cli.py`

Expected: FAIL because `AppConfig` and the parser do not expose provider or host.

**Step 3: Implement minimal validated configuration**

Add immutable fields:

```python
provider: Literal["openai", "ollama"] = "openai"
ollama_host: str = "http://localhost:11434"
```

Reject unsupported providers and blank hosts. Add parser choices for `--provider`
and a string `--ollama-host`, then pass both values into `AppConfig`.

**Step 4: Run focused tests**

Run: `uv run pytest -q tests/test_config.py tests/test_cli.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add summarizer/config.py summarizer/cli.py tests/test_config.py tests/test_cli.py
git commit -m "feat: configure model provider selection"
```

### Task 2: Implement the native Ollama adapter

**Files:**
- Create: `summarizer/providers/ollama.py`
- Create: `tests/providers/test_ollama.py`
- Modify: `requirements.txt`

**Step 1: Add the official client dependency**

Add a bounded compatible `ollama` dependency to `requirements.txt`, install the
updated environment, and inspect the installed exception and response types
before finalizing adapter catches.

Run: `uv pip install -r requirements.txt`

Expected: dependency installation succeeds.

**Step 2: Write failing request and response mapping tests**

Use an injected fake client factory and assert the provider:

```python
result = provider.generate(REQUEST)

assert calls == [{
    "model": "gemma3:4b",
    "messages": [
        {"role": "system", "content": REQUEST.instructions},
        {"role": "user", "content": REQUEST.input_text},
    ],
    "stream": False,
    "think": False,
}]
assert result.provider == "ollama"
assert result.text == "local summary"
assert result.input_tokens == 42
assert result.output_tokens == 11
assert result.finish_status == "stop"
```

Also assert client construction is lazy, receives the configured host and
timeout, and happens once.

**Step 3: Run mapping tests to verify they fail**

Run: `uv run pytest -q tests/providers/test_ollama.py`

Expected: FAIL because `summarizer.providers.ollama` does not exist.

**Step 4: Implement minimal native mapping**

Create `OllamaProvider` with an injectable factory. Lazily construct the
official client, call native `chat` with non-streaming messages, validate
terminal assistant content, normalize whitespace, and return
`GenerationResult` with model, token counts, and completion reason.

**Step 5: Write failing malformed-response and error tests**

Cover:

- blank or absent `message.content`;
- `done` explicitly false;
- connection and timeout errors;
- 404 missing-model responses;
- 429 rate limits;
- 500 and 502 server errors;
- other rejected requests;
- prompt and source text absent from translated messages.

**Step 6: Run failure tests to verify they fail**

Run: `uv run pytest -q tests/providers/test_ollama.py`

Expected: FAIL in the new validation and translation cases.

**Step 7: Implement error translation**

Translate installed-client exception types and status codes into the existing
provider exception hierarchy. Preserve causal exceptions, provide an actionable
missing-model message, and sanitize all messages.

**Step 8: Run adapter tests**

Run: `uv run pytest -q tests/providers/test_ollama.py`

Expected: PASS.

**Step 9: Commit**

```bash
git add requirements.txt summarizer/providers/ollama.py tests/providers/test_ollama.py
git commit -m "feat: add native Ollama provider"
```

### Task 3: Compose the selected provider in the CLI

**Files:**
- Modify: `summarizer/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing provider-selection tests**

Refactor injection around a provider builder and assert:

- the default selects `OpenAIProvider`;
- `--provider ollama` selects `OllamaProvider`;
- the configured host is passed only to Ollama;
- the same `RetryingProvider` wraps either adapter once;
- dry-run constructs neither adapter;
- an unavailable Ollama service returns exit code 1 with concise stderr and no
  final output.

**Step 2: Run tests to verify they fail**

Run: `uv run pytest -q tests/test_cli.py`

Expected: FAIL because `main` still accepts only the OpenAI default factory.

**Step 3: Implement provider composition**

Add a small provider-building function that receives `AppConfig`, selects the
adapter, and remains replaceable in tests. Keep selection out of
`LegacyWorkflow`. Preserve dry-run provider isolation and apply
`RetryingProvider` only after selection.

**Step 4: Run focused and provider tests**

Run: `uv run pytest -q tests/test_cli.py tests/providers`

Expected: PASS.

**Step 5: Commit**

```bash
git add summarizer/cli.py tests/test_cli.py
git commit -m "feat: select OpenAI or Ollama from CLI"
```

### Task 4: Preserve import safety and document local usage

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-09-02-provider-boundary-design.md`
- Modify: `tests/test_import_safety.py`
- Modify: `tests/test_documented_cli.py`

**Step 1: Write failing documentation and import tests**

Assert the application-module import matrix includes
`summarizer.providers.ollama`, no client is constructed during import, and the
README documents `--provider`, `--ollama-host`, `qwen3.8`, and `gemma3:4b`.

**Step 2: Run tests to verify they fail**

Run: `uv run pytest -q tests/test_import_safety.py tests/test_documented_cli.py`

Expected: FAIL because the new flags and provider are not documented in the
existing operator guide.

**Step 3: Update documentation**

Document:

```bash
ollama pull gemma3:4b
python main.py --provider ollama --model gemma3:4b
python main.py --provider ollama --ollama-host http://localhost:11434 --model qwen3.8
```

Explain that Ollama is an external prerequisite, local use needs no API key,
the default remains OpenAI, model tags must already exist locally, and automated
tests do not contact Ollama. Amend the earlier provider-boundary design so its
OpenAI-only wording reflects the approved extension.

**Step 4: Run documentation and import tests**

Run: `uv run pytest -q tests/test_import_safety.py tests/test_documented_cli.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add README.md docs/plans/2026-09-02-provider-boundary-design.md tests/test_import_safety.py tests/test_documented_cli.py
git commit -m "chore: document local Ollama workflow"
```

### Task 5: Run final acceptance and update PR #13

**Files:**
- Modify if needed: files implicated by verification failures
- External update: GitHub PR #13

**Step 1: Run the full offline suite**

Run: `uv run pytest -q`

Expected: PASS with no network access.

**Step 2: Run importlib mode and dependency checks**

Run: `uv run pytest -q --import-mode=importlib`

Run: `uv run pip check`

Expected: both PASS.

**Step 3: Verify import-time behavior from an empty directory**

Import every application module in subprocesses with `OPENAI_API_KEY` removed
and assert the temporary directory remains empty.

Expected: all imports succeed with no client construction, network calls, or
files written.

**Step 4: Run diff and repository checks**

Run: `git diff --check origin/main...HEAD`

Run: `git status --short --branch`

Expected: no whitespace errors and no uncommitted files.

**Step 5: Request independent review**

Use `superpowers:requesting-code-review` to review issue #3 compliance, native
Ollama request semantics, failure classification, import safety, and provider
neutrality. Fix all material findings with new failing tests before production
changes.

**Step 6: Push and update PR #13**

Push the reviewed commits, update the PR summary and verification counts, and
retain the required AI-agent disclaimer. Keep the PR in Draft unless the user
explicitly authorizes marking it ready.
