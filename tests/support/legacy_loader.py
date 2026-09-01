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
