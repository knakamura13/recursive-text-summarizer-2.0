"""S5 supporting check for C-S5-001: does the installed `openai` SDK resolve
its base_url from the OPENAI_BASE_URL environment variable when the
summarizer's OpenAIProvider does not pass `base_url` explicitly?

summarizer/providers/openai.py's `_create_client` calls
`openai.OpenAI(**kwargs)` with only `max_retries=0` -- no `base_url`. If the
SDK itself defaults to os.environ["OPENAI_BASE_URL"], then the same
cache-key blindness documented for Ollama's `ollama_host` in
test_repro_ollama_host_collision.py also applies to the OpenAI path: two
different OPENAI_BASE_URL values (e.g. pointed at an OpenAI-compatible proxy
or a differently configured deployment) with the same `model` string would
collide on an identical CacheDescriptor, since `AppConfig` carries no
openai-base-url field at all and the client is constructed with no explicit
override.

This is inspection-only evidence (reads the installed SDK's own default
resolution), not a claim that this repo calls it -- summarizer/config.py
has no OpenAI base-url field, so this documents the SDK-level mechanism the
gap would ride on if such a field is ever added, and explains why the
Ollama case (which DOES have a first-class `ollama_host` field, deliberately
excluded) is the concrete, currently-reachable half of C-S5-001.
"""

from __future__ import annotations

import inspect

import openai


def test_openai_client_constructor_defaults_base_url_from_environment() -> None:
    signature = inspect.signature(openai.OpenAI.__init__)
    assert "base_url" in signature.parameters
    base_url_param = signature.parameters["base_url"]
    # The SDK's own default (not one this repo supplies) is its NOT_GIVEN
    # sentinel, not a fixed literal -- meaning "resolve from environment /
    # library default" rather than a hardcoded value baked into the client.
    # A caller that never passes base_url (as
    # summarizer/providers/openai.py's _create_client does -- it only passes
    # max_retries=0) inherits whatever OPENAI_BASE_URL is set in the process
    # environment, per the SDK's own __init__ source.
    assert type(base_url_param.default).__name__ == "NotGiven"

    source = inspect.getsource(openai.OpenAI.__init__)
    assert 'os.environ.get("OPENAI_BASE_URL")' in source
