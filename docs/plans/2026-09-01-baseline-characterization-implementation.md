# Baseline Characterization Harness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an offline, deterministic characterization harness for the unchanged legacy summarizer and document its accepted compatibility constraints and known defects.

**Architecture:** Pytest loads `main.py` through a test-only module loader that installs scripted doubles for OpenAI, NLTK, dotenv, tqdm, and requests before import. Unit tests exercise the actual legacy functions, while a runpy-based executable test drives the default `input.txt` to `output.txt` workflow inside a temporary directory.

**Tech Stack:** Python 3, pytest, standard-library unittest.mock, importlib, runpy, dataclasses

---

### Task 1: Establish offline pytest infrastructure

**Files:**
- Modify: `.gitignore`
- Create: `requirements-dev.txt`
- Create: `tests/conftest.py`
- Create: `tests/test_offline_guard.py`

**Step 1: Write the failing network-guard test**

```python
# tests/test_offline_guard.py
import socket

import pytest


def test_network_access_is_disabled() -> None:
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        socket.create_connection(("example.com", 443))
```

**Step 2: Add the development dependency file and install it**

```text
# requirements-dev.txt
-r requirements.txt
pytest>=8.3
```

Run: `venv/bin/python -m pip install -r requirements-dev.txt`

Expected: pytest installs successfully in the worktree virtual environment.

**Step 3: Run the test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_offline_guard.py -v`

Expected: FAIL because the attempted connection raises a DNS or connection error rather than the required guard error.

**Step 4: Add the automatic network guard**

```python
# tests/conftest.py
import socket
from collections.abc import Iterator

import pytest
from pytest import MonkeyPatch


@pytest.fixture(autouse=True)
def deny_external_network(monkeypatch: MonkeyPatch) -> Iterator[None]:
    def blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Network access is disabled during tests")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    yield
```

Append these generated-path rules to `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
```

**Step 5: Run the test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_offline_guard.py -v`

Expected: PASS, 1 test.

**Step 6: Commit**

```bash
git add .gitignore requirements-dev.txt tests/conftest.py tests/test_offline_guard.py
git commit -m "chore: establish offline pytest harness"
```

### Task 2: Add a deterministic legacy-module loader

**Files:**
- Create: `tests/support/__init__.py`
- Create: `tests/support/legacy_loader.py`
- Create: `tests/test_legacy_import.py`

**Step 1: Write the failing import test**

```python
# tests/test_legacy_import.py
from tests.support.legacy_loader import FakeHarness, load_legacy_main


def test_import_uses_test_doubles_without_credentials_or_downloads() -> None:
    harness = FakeHarness()

    module = load_legacy_main(harness)

    assert module.client is harness.client
    assert harness.api_keys == [None]
    assert harness.downloads == ["punkt"]
    assert harness.dotenv_load_count == 1
```

Create an empty `tests/support/__init__.py` so the support package imports explicitly.

**Step 2: Run the test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_legacy_import.py -v`

Expected: FAIL with `ModuleNotFoundError: tests.support.legacy_loader`.

**Step 3: Implement the scripted doubles and loader**

```python
# tests/support/legacy_loader.py
from __future__ import annotations

import importlib.util
import runpy
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Iterator
from unittest.mock import patch
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeRequestException(Exception):
    pass


@dataclass
class FakeHarness:
    sentences: list[str] = field(default_factory=list)
    scripted_results: deque[object] = field(default_factory=deque)
    calls: list[dict[str, object]] = field(default_factory=list)
    api_keys: list[str | None] = field(default_factory=list)
    downloads: list[str] = field(default_factory=list)
    dotenv_load_count: int = 0

    def __post_init__(self) -> None:
        completions = _FakeCompletions(self)
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

    def queue(self, *results: object) -> None:
        self.scripted_results.extend(results)


class _FakeCompletions:
    def __init__(self, harness: FakeHarness) -> None:
        self._harness = harness

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        timeout: int,
    ) -> object:
        self._harness.calls.append(
            {"model": model, "messages": messages, "timeout": timeout}
        )
        if not self._harness.scripted_results:
            raise AssertionError("Fake model was called without a scripted result")
        result = self._harness.scripted_results.popleft()
        if isinstance(result, BaseException):
            raise result
        message = SimpleNamespace(content=result)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )


def _fake_modules(harness: FakeHarness) -> dict[str, ModuleType]:
    openai = ModuleType("openai")

    def openai_client(*, api_key: str | None = None) -> object:
        harness.api_keys.append(api_key)
        return harness.client

    openai.OpenAI = openai_client

    nltk = ModuleType("nltk")

    def download(resource: str) -> None:
        harness.downloads.append(resource)

    nltk.download = download

    nltk_tokenize = ModuleType("nltk.tokenize")
    nltk_tokenize.sent_tokenize = lambda text: list(harness.sentences)
    nltk.tokenize = nltk_tokenize

    dotenv = ModuleType("dotenv")

    def load_dotenv() -> None:
        harness.dotenv_load_count += 1

    dotenv.load_dotenv = load_dotenv

    tqdm = ModuleType("tqdm")
    tqdm.tqdm = lambda iterable, **kwargs: iterable

    requests = ModuleType("requests")
    requests.exceptions = SimpleNamespace(
        RequestException=FakeRequestException
    )

    return {
        "dotenv": dotenv,
        "nltk": nltk,
        "nltk.tokenize": nltk_tokenize,
        "openai": openai,
        "requests": requests,
        "tqdm": tqdm,
    }


@contextmanager
def installed_legacy_modules(
    harness: FakeHarness,
) -> Iterator[None]:
    with patch.dict(sys.modules, _fake_modules(harness)):
        yield


def load_legacy_main(harness: FakeHarness) -> ModuleType:
    module_name = f"_legacy_main_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        module_name, PROJECT_ROOT / "main.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not construct a loader for main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with installed_legacy_modules(harness):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def run_legacy_main(harness: FakeHarness) -> dict[str, object]:
    with installed_legacy_modules(harness):
        return runpy.run_path(
            str(PROJECT_ROOT / "main.py"),
            run_name="__main__",
        )
```

If mypy-style assignment warnings make the helper noisy, keep the runtime code as above; static typing is not configured in this repository and is not part of #2.

**Step 4: Run the import test**

Run: `venv/bin/python -m pytest tests/test_legacy_import.py -v`

Expected: PASS, proving import uses only deterministic doubles.

**Step 5: Commit**

```bash
git add tests/support tests/test_legacy_import.py
git commit -m "test: add deterministic legacy module loader"
```

### Task 3: Characterize file, whitespace, and chunk behavior

**Files:**
- Create: `tests/test_legacy_utilities.py`

**Step 1: Add utility characterization tests**

```python
# tests/test_legacy_utilities.py
from pathlib import Path

import pytest

from tests.support.legacy_loader import FakeHarness, load_legacy_main


@pytest.fixture
def legacy_module():
    return load_legacy_main(FakeHarness())


def test_open_file_reads_utf8(legacy_module, tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Résumé — 東京", encoding="utf-8")

    assert legacy_module.open_file(str(source)) == "Résumé — 東京"


def test_save_file_creates_parent_directories(
    legacy_module, tmp_path: Path
) -> None:
    output = tmp_path / "nested" / "output.txt"

    legacy_module.save_file("summary", str(output))

    assert output.read_text(encoding="utf-8") == "summary"


def test_save_file_rejects_path_without_parent_directory(
    legacy_module, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        legacy_module.save_file("summary", "output.txt")


def test_remove_extra_whitespace_collapses_all_runs(
    legacy_module,
) -> None:
    assert (
        legacy_module.remove_extra_whitespace("  Alpha\n\tBeta   Gamma  ")
        == "Alpha Beta Gamma"
    )


def test_chunking_packs_scripted_sentences_by_approximate_characters(
    legacy_module, monkeypatch
) -> None:
    monkeypatch.setattr(
        legacy_module,
        "sent_tokenize",
        lambda text: ["Alpha.", "Beta.", "Gamma."],
    )

    assert legacy_module.chunk_text_by_sentences("ignored", 12) == [
        " Alpha. Beta.",
        "Gamma.",
    ]


def test_oversized_first_sentence_is_not_split(
    legacy_module, monkeypatch
) -> None:
    monkeypatch.setattr(
        legacy_module, "sent_tokenize", lambda text: ["Oversized sentence."]
    )

    assert legacy_module.chunk_text_by_sentences("ignored", 4) == [
        " Oversized sentence."
    ]
```

**Step 2: Run the tests and inspect any mismatch**

Run: `venv/bin/python -m pytest tests/test_legacy_utilities.py -v`

Expected: PASS. If `save_file` raises a platform-specific `OSError` subtype, narrow the assertion to the observed subtype and record it in the baseline document.

**Step 3: Commit**

```bash
git add tests/test_legacy_utilities.py
git commit -m "test: characterize legacy utility behavior"
```

### Task 4: Characterize model requests and successful summaries

**Files:**
- Create: `tests/test_legacy_provider.py`

**Step 1: Add the successful-provider and dry-run tests**

```python
# tests/test_legacy_provider.py
from pathlib import Path

from tests.support.legacy_loader import FakeHarness, load_legacy_main


def test_successful_summary_records_prompt_model_timeout_and_log(
    monkeypatch, tmp_path: Path
) -> None:
    harness = FakeHarness()
    harness.queue("  concise\n summary  ")
    module = load_legacy_main(harness)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "time", lambda: 123.0)

    result = module.summarize_with_gpt("Source text", _model="gpt-4")

    assert result == "concise summary"
    assert len(harness.calls) == 1
    call = harness.calls[0]
    assert call["model"] == "gpt-4-1106-preview"
    assert call["timeout"] == 180
    assert call["messages"][0]["role"] == "system"
    assert '"""\nSource text\n"""' in call["messages"][1]["content"]
    assert (tmp_path / "gpt_logs" / "123.0_gpt.txt").read_text(
        encoding="utf-8"
    ).endswith("RESPONSE:\n\nconcise summary")


def test_default_model_alias_uses_gpt_35_turbo(tmp_path, monkeypatch) -> None:
    harness = FakeHarness()
    harness.queue("summary")
    module = load_legacy_main(harness)
    monkeypatch.chdir(tmp_path)

    module.summarize_with_gpt("Source")

    assert harness.calls[0]["model"] == "gpt-3.5-turbo"


def test_dry_run_collapses_source_without_calling_provider() -> None:
    harness = FakeHarness()
    module = load_legacy_main(harness)
    module.DRY_RUN = True

    assert module.summarize_with_gpt("  Alpha\n Beta  ") == "Alpha Beta"
    assert harness.calls == []
```

**Step 2: Run the tests**

Run: `venv/bin/python -m pytest tests/test_legacy_provider.py -v`

Expected: PASS, 3 tests.

**Step 3: Commit**

```bash
git add tests/test_legacy_provider.py
git commit -m "test: characterize legacy model requests"
```

### Task 5: Characterize retry and terminal failure behavior

**Files:**
- Modify: `tests/test_legacy_provider.py`

**Step 1: Add retry tests**

```python
from tests.support.legacy_loader import FakeRequestException


def test_request_failures_retry_five_times_with_exponential_delays(
    monkeypatch,
) -> None:
    harness = FakeHarness()
    harness.queue(*(FakeRequestException("offline") for _ in range(5)))
    module = load_legacy_main(harness)
    delays: list[int] = []
    monkeypatch.setattr(module, "sleep", delays.append)

    result = module.summarize_with_gpt("Source")

    assert result == "GPT error: Unknown error"
    assert len(harness.calls) == 5
    assert delays == [1, 2, 4, 8]


def test_generic_terminal_failure_exposes_legacy_unbound_error(
    monkeypatch,
) -> None:
    harness = FakeHarness()
    harness.queue(*(ValueError("bad response") for _ in range(5)))
    module = load_legacy_main(harness)
    monkeypatch.setattr(module, "sleep", lambda delay: None)

    with pytest.raises(UnboundLocalError):
        module.summarize_with_gpt("Source")
```

Also add `import pytest` at the top of the file.

**Step 2: Run the targeted tests**

Run: `venv/bin/python -m pytest tests/test_legacy_provider.py -v`

Expected: PASS, including five calls and four recorded delays. The generic-failure test intentionally records a legacy defect; do not fix production code in #2.

**Step 3: Commit**

```bash
git add tests/test_legacy_provider.py
git commit -m "test: characterize legacy retry failures"
```

### Task 6: Characterize the default executable workflow

**Files:**
- Create: `tests/test_legacy_cli.py`

**Step 1: Add the multi-chunk executable test**

```python
# tests/test_legacy_cli.py
from pathlib import Path

from tests.support.legacy_loader import FakeHarness, run_legacy_main


def test_default_workflow_summarizes_chunks_and_joins_responses(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("source", encoding="utf-8")
    harness = FakeHarness(
        sentences=["A" * 600 + ".", "B" * 600 + "."]
    )
    harness.queue("first summary", "second summary")

    namespace = run_legacy_main(harness)

    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == (
        "first summary\nsecond summary"
    )
    assert [call["model"] for call in harness.calls] == [
        "gpt-4-1106-preview",
        "gpt-4-1106-preview",
    ]
    assert namespace["MAX_CHUNKS"] == -1


def test_missing_input_is_logged_without_partial_output(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    monkeypatch.chdir(tmp_path)
    harness = FakeHarness()

    run_legacy_main(harness)

    assert not (tmp_path / "output.txt").exists()
    assert "Fatal error in main application" in caplog.text
```

**Step 2: Run the executable tests**

Run: `venv/bin/python -m pytest tests/test_legacy_cli.py -v`

Expected: PASS, 2 tests. If pytest's installed logging handler prevents the second assertion from observing the message, patch `logging.critical` with a recorder in that test; do not change `main.py`.

**Step 3: Commit**

```bash
git add tests/test_legacy_cli.py
git commit -m "test: characterize default summarizer workflow"
```

### Task 7: Add representative long-form fixtures

**Files:**
- Create: `tests/fixtures/article.txt`
- Create: `tests/fixtures/report.txt`
- Create: `tests/fixtures/transcript.txt`
- Create: `tests/fixtures/structured.md`
- Create: `tests/fixtures/narrative.txt`
- Create: `tests/test_fixture_corpus.py`

**Step 1: Write the failing fixture-corpus test**

```python
# tests/test_fixture_corpus.py
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "name",
    [
        "article.txt",
        "report.txt",
        "transcript.txt",
        "structured.md",
        "narrative.txt",
    ],
)
def test_representative_fixture_is_nonempty_utf8(name: str) -> None:
    content = (FIXTURES / name).read_text(encoding="utf-8")

    assert len(content.split()) >= 80
```

**Step 2: Run the test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_fixture_corpus.py -v`

Expected: FAIL because the five fixtures do not exist.

**Step 3: Create compact genre-distinct fixtures**

Write 100–200 words per fixture:

- `article.txt`: a reported municipal tree-canopy initiative with attribution, dates, and a qualification.
- `report.txt`: a quarterly transit reliability report with headings expressed in plain text, metrics, causes, and recommendations.
- `transcript.txt`: labeled speakers who correct one another and distinguish a proposal from a decision.
- `structured.md`: Markdown headings, paragraphs, nested lists, and a warning block about a software migration.
- `narrative.txt`: chronological prose with recurring entities and an uncertain observation.

Keep all facts fictional, avoid lecture or textbook framing, and include at least one qualification or uncertainty in every fixture.

**Step 4: Run the fixture test**

Run: `venv/bin/python -m pytest tests/test_fixture_corpus.py -v`

Expected: PASS, 5 parameterized cases.

**Step 5: Commit**

```bash
git add tests/fixtures tests/test_fixture_corpus.py
git commit -m "test: add representative source fixtures"
```

### Task 8: Document the observed baseline and verify #2

**Files:**
- Create: `docs/legacy-baseline.md`
- Verify unchanged: `main.py`
- Verify unchanged: `omscs-ml-lectures/`

**Step 1: Write the baseline record**

Use this structure:

```markdown
# Legacy Summarizer Baseline

## Running the offline suite

1. Create and activate a Python virtual environment.
2. Install `requirements-dev.txt`.
3. Run `python -m pytest -v`.

The suite blocks socket connections and supplies deterministic test doubles.

## Compatibility constraints

- `python main.py` reads UTF-8 `input.txt` from the working directory.
- Successful execution writes UTF-8 `output.txt`.
- Each source chunk produces one provider call in source order.
- Chunk responses are joined with a single newline.

## Known legacy limitations

- Chunk budgets count characters and can exceed the configured limit.
- Oversized sentences are not split.
- Import downloads NLTK data and constructs the OpenAI client.
- Prompts and model aliases are fixed in source.
- Intermediate summaries are concatenated without recursive synthesis.
- Provider errors may become output text; generic terminal failures can raise an
  unbound-local error.
- The executable logs fatal errors without returning a nonzero status.
- Configuration requires editing module constants.

These limitations are characterization findings, not target behavior.

## Fixture corpus

Describe the five compact genres and state that historical lecture data remains
unchanged.
```

**Step 2: Run the complete offline suite**

Run: `venv/bin/python -m pytest -v`

Expected: all tests pass with no external network access.

**Step 3: Verify production and historical sources were not changed**

Run: `git diff origin/main -- main.py omscs-ml-lectures/`

Expected: no output.

Run: `git diff --check`

Expected: no output.

Run: `git status --short`

Expected: only `docs/legacy-baseline.md` is uncommitted.

**Step 4: Commit**

```bash
git add docs/legacy-baseline.md
git commit -m "docs: record legacy summarizer baseline"
```

**Step 5: Re-run acceptance verification**

Run: `venv/bin/python -m pytest -q`

Expected: all tests pass.

Run: `git diff origin/main --name-only -- main.py omscs-ml-lectures/`

Expected: no output.

Run: `git status --short --branch`

Expected: clean `knakamura/issue-2-baseline-harness` worktree.
