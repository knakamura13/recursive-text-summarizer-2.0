from __future__ import annotations

import importlib.util
import logging
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import nltk
import ollama
import openai
import pytest

import summarizer.cli
import summarizer.providers.ollama


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def test_importing_main_has_no_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(nltk, "download", lambda *_args, **_kwargs: calls.append("download"))
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda *_args, **_kwargs: calls.append("client") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        logging,
        "basicConfig",
        lambda *_args, **_kwargs: calls.append("logging"),
    )
    module_name = f"_entrypoint_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_executing_main_delegates_exit_code_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fake_main() -> int:
        calls.append(object())
        return 7

    monkeypatch.setattr(summarizer.cli, "main", fake_main)
    monkeypatch.setattr(nltk, "download", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MAIN_PATH), run_name="__main__")

    assert exc_info.value.code == 7
    assert len(calls) == 1


def test_importing_ollama_adapter_does_not_construct_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        ollama,
        "Client",
        lambda *_args, **_kwargs: calls.append(object()),
    )

    importlib.reload(summarizer.providers.ollama)

    assert calls == []
