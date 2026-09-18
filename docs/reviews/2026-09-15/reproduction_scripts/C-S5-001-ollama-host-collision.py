"""S5 repro for C-S5-001, expressed as a pytest test (the sandbox here only
permits `python -m pytest` invocations, not arbitrary `python <script>.py`
execution, so this evidence is captured as a test node rather than a __main__
script).

Demonstrates that `AppConfig.ollama_host` never reaches `CacheDescriptor`, so
two Ollama endpoints serving different models/weights under the identical
`model` string collide on the SAME cache object key and SAME cache object.

Uses the real production classes (`CacheCoordinator` from
`summarizer.segmentation`, `CacheStore`/`CacheDescriptor` from
`summarizer.cache`) and mirrors the exact `CacheCoordinator` construction the
real pipeline performs in `summarizer/pipeline.py::run_pipeline` (lines
~254-281), rather than hand-building a descriptor, to keep this reachable
from a real caller rather than a synthetic worst case.

Run offline with:
    UV_OFFLINE=1 uv run --with-requirements requirements-dev.txt \
        python -m pytest -q -v \
        .review/wip/s5-criteria/test_repro_ollama_host_collision.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from summarizer.cache import CacheStore
from summarizer.config import AppConfig
from summarizer.segmentation import CacheCoordinator
from summarizer.tokenization import resolve_token_counter


def _source_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != {"summary"}:
        raise ValueError("expected a summary payload")
    return {"summary": payload["summary"]}  # type: ignore[index]


def _build_coordinator(app: AppConfig, store: CacheStore) -> CacheCoordinator:
    """Mirror summarizer/pipeline.py::run_pipeline's CacheCoordinator construction.

    Only the fields pipeline.py actually threads through are reproduced here
    (source_id/provider/model/counter identity/context window/behavior). The
    point under test is that `app.ollama_host` never appears anywhere in this
    construction -- exactly as in the real pipeline.
    """
    counter = resolve_token_counter(provider=app.provider, model=app.model)
    return CacheCoordinator(
        store=store,
        source_id=_source_id("The quick brown fox jumps over the lazy dog."),
        provider=app.provider,
        model=app.model,
        counter_identity=counter.identity,
        counter_exact=counter.exact,
        context_window_tokens=8192,
        behavior={
            "strategy_config": {
                "strategy": "direct",
                "context_window": 8192,
                "max_output_tokens": 1024,
                "safety_margin_tokens": 256,
            },
            "budget": {
                "context_window": 8192,
                "max_output_tokens": 1024,
                "safety_margin_tokens": 256,
                "safety_margin_fraction": 0.02,
            },
        },
        session=None,
        allow_unreferenced_cache=True,
        max_in_flight=1,
        reliability_tracker=None,
    )


def test_different_ollama_hosts_collide_on_the_same_cache_key(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")

    # Two AppConfigs a real operator could produce: same provider label
    # ("ollama"), same model *name* ("llama3.2:3b"), but pointed at two
    # different Ollama hosts -- e.g. a laptop dev server and a team's shared
    # GPU box that happens to have a differently-tuned/fine-tuned model
    # registered under the same tag. This is exactly the scenario
    # docs/plans/2026-09-06-reliability-cache-resume-design.md calls out as
    # excluded "to avoid leaking a credential" -- but the host here carries
    # no credential, only endpoint identity.
    app_a = AppConfig(
        provider="ollama", model="llama3.2:3b",
        ollama_host="http://gpu-box-a.internal:11434",
    )
    app_b = AppConfig(
        provider="ollama", model="llama3.2:3b",
        ollama_host="http://gpu-box-b.internal:11434",
    )

    coordinator_a = _build_coordinator(app_a, store)
    coordinator_b = _build_coordinator(app_b, store)

    descriptor_a = coordinator_a.descriptor_for(
        stage="direct", work_id="D000001",
        prompt_version="direct-prompt/1", schema_version="direct-summary/1",
        input_value={"instructions": "summarize", "input_text": "same document"},
        behavior={},
    )
    descriptor_b = coordinator_b.descriptor_for(
        stage="direct", work_id="D000001",
        prompt_version="direct-prompt/1", schema_version="direct-summary/1",
        input_value={"instructions": "summarize", "input_text": "same document"},
        behavior={},
    )

    assert app_a.ollama_host != app_b.ollama_host
    # The falsifiable claim: descriptor construction is blind to ollama_host.
    assert descriptor_a.canonical_value() == descriptor_b.canonical_value()
    assert descriptor_a.key == descriptor_b.key

    # Second, independent proof at the store layer: server A's answer is
    # served back as a validated HIT when server B's descriptor is queried,
    # with no signal that the two came from different hosts.
    store.store_winner(
        descriptor_a,
        {"summary": "Answer produced by gpu-box-a's model"},
        _validate,
    )
    lookup_b = store.load(descriptor_b, _validate)

    assert lookup_b.hit
    assert lookup_b.payload == {"summary": "Answer produced by gpu-box-a's model"}
