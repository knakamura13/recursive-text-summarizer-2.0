"""Independent verification for the OpenAI half of C-S5-001.

Does NOT make any network call to OpenAI (or anywhere else) -- per the hard
constraint against ever calling OpenAI live. This only inspects:

  1. `summarizer.providers.openai._create_client` / `OpenAIProvider` never
     passes `base_url` to `openai.OpenAI(...)`, confirmed by intercepting the
     constructor with a fake that records its kwargs and raises before any
     HTTP client is built.
  2. The installed `openai` SDK's own `OpenAI.__init__` resolves an omitted
     `base_url` from the `OPENAI_BASE_URL` environment variable, confirmed by
     reading its source directly (`inspect.getsource`), not by calling it.

Together these establish whether the mechanism the author flagged is a
*currently reachable* path in this codebase's pinned dependency, or only a
theoretical one -- without ever placing a real request.

Run offline with:
    UV_OFFLINE=1 uv run --with-requirements requirements-dev.txt \
        python -m pytest -q -v \
        .review/wip/v-s5-001/test_openai_base_url_reachability.py
"""

from __future__ import annotations

import inspect

import openai
import pytest

from summarizer.providers.openai import OpenAIProvider, _create_client


def test_create_client_never_passes_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_create_client` (used by `OpenAIProvider._get_client`) never supplies
    `base_url`, `AppConfig` has no OpenAI base-url field anywhere in this
    codebase, so the only way `base_url` is ever set is the SDK's own
    environment fallback -- entirely outside this application's control.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    captured_kwargs: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)
            raise RuntimeError("stop before any HTTP client is constructed")

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    with pytest.raises(RuntimeError, match="stop before any HTTP client"):
        _create_client(max_retries=0)

    assert "base_url" not in captured_kwargs
    assert captured_kwargs == {"max_retries": 0}

    # Same result reached through the real call chain OpenAIProvider uses.
    provider = OpenAIProvider()
    captured_kwargs.clear()
    with pytest.raises(RuntimeError, match="stop before any HTTP client"):
        provider._get_client()
    assert "base_url" not in captured_kwargs


def test_installed_openai_sdk_falls_back_to_openai_base_url_env_var() -> None:
    """Read-only confirmation that the *installed* SDK version (pinned by
    requirements.txt: `openai>=3.7.0,<4`; resolved here to 3.7.0) resolves an
    omitted `base_url` from `OPENAI_BASE_URL`. Source inspection only -- the
    assertion never constructs a real client or opens a socket.
    """
    source = inspect.getsource(openai.OpenAI.__init__)
    assert 'os.environ.get("OPENAI_BASE_URL")' in source
    # And there is no config surface in this app that ever sets that env var
    # or threads an OpenAI base URL into AppConfig -- the mechanism, if it
    # fires at all, fires purely from whatever the OS process environment
    # already contains when the CLI is launched.
    import summarizer.config as config_module

    config_source = inspect.getsource(config_module)
    assert "OPENAI_BASE_URL" not in config_source
    assert "base_url" not in config_source
