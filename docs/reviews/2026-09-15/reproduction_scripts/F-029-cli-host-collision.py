"""Independent verification repro for C-S5-001, at the CLI boundary.

Unlike the author's repro (`.review/repro/C-S5-001-ollama-host-collision.py`),
which builds a `CacheCoordinator`/`CacheDescriptor` directly, this drives
`summarizer.cli.main()` twice with real argv, using the same offline
`provider_factory=`/`counter_factory=` seam `tests/test_cli.py` uses. The
question this answers: does an ordinary two-command CLI session -- no
`--resume`, two different `--run-id` values, same `--cache-dir` -- actually
serve run A's provider output back to run B when only `--ollama-host` differs?

Run offline with:
    UV_OFFLINE=1 uv run --with-requirements requirements-dev.txt \
        python -m pytest -q -v \
        .review/wip/v-s5-001/test_cli_host_collision.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from summarizer.cli import main
from summarizer.config import AppConfig
from summarizer.providers.base import GenerationRequest, GenerationResult


class CharacterCounter:
    """Mirrors tests/test_cli.py::CharacterCounter -- offline, exact, deterministic."""

    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


def counter_factory(_config: AppConfig) -> CharacterCounter:
    return CharacterCounter()


class RecordingProvider:
    """Mirrors tests/test_cli.py::RecordingProvider: a fake ModelProvider that
    never touches the network, records every call it receives, and answers
    with a caller-supplied sentinel string so two "servers" are trivially
    distinguishable in the final output.
    """

    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if request.operation_id == "editorial-final":
            text = json.dumps({"text": self.outcome})
        elif request.operation_id == "D000001":
            text = json.dumps(
                {
                    "summary": self.outcome,
                    "content_units": [],
                    "entities": [],
                    "qualifications": [],
                    "contradictions": [],
                    "quotations": [],
                    "provenance": ["D000001"],
                    "level": 0,
                }
            )
        else:
            identifiers = re.findall(r'"segment_id":"([SD]\d+)"', request.input_text)
            text = json.dumps(
                {
                    "summary": self.outcome,
                    "content_units": [],
                    "entities": [],
                    "qualifications": [],
                    "contradictions": [],
                    "quotations": [],
                    "provenance": identifiers[-1:] or ["S000001"],
                    "level": 0,
                }
            )
        return GenerationResult(text, "fake", request.model)


def test_two_new_runs_sharing_a_cache_dir_collide_across_ollama_hosts(
    tmp_path: Path,
) -> None:
    """Two ordinary `summarizer` CLI invocations, no --resume, distinct
    --run-id each time, same --cache-dir, differing ONLY in --ollama-host.

    If C-S5-001 is real at the CLI boundary, run B's output file will contain
    run A's provider answer, and run B's fake provider will never be called.
    """
    input_path = tmp_path / "input.txt"
    input_path.write_text("Source.", encoding="utf-8")

    cache_dir = tmp_path / "cache"
    output_a = tmp_path / "output_a.txt"
    output_b = tmp_path / "output_b.txt"
    audit_a = tmp_path / "audit_a.json"
    audit_b = tmp_path / "audit_b.json"

    provider_a = RecordingProvider("ANSWER-FROM-GPU-BOX-A")
    provider_b = RecordingProvider("ANSWER-FROM-GPU-BOX-B")

    common = [
        "--input", str(input_path),
        "--provider", "ollama",
        "--model", "llama3.2:3b",
        # llama3.2:3b has no entry in the local context-window table, so
        # --strategy auto would otherwise refuse "direct" (window "assumed")
        # and fall back to a one-leaf hierarchical path (work id S000001
        # instead of D000001). Supplying --context-window keeps this on the
        # exact same direct-strategy shape as tests/test_cli.py's baseline,
        # per the CLI's own --context-window help text.
        "--context-window", "8192",
        "--cache-dir", str(cache_dir),
    ]

    exit_code_a = main(
        [
            *common,
            "--output", str(output_a),
            "--ollama-host", "http://gpu-box-a.internal:11434",
            "--run-id", "run-a",
            "--audit", str(audit_a),
        ],
        provider_factory=lambda _config: provider_a,
        counter_factory=counter_factory,
    )
    assert exit_code_a == 0
    assert output_a.read_text(encoding="utf-8") == "ANSWER-FROM-GPU-BOX-A"
    # Sanity: run A actually called its own fake provider (no pre-existing cache).
    assert [call.operation_id for call in provider_a.calls] == [
        "D000001",
        "editorial-final",
    ]

    # Run B: brand-new run_id, run_mode="new" (no --resume anywhere), only
    # --ollama-host and --run-id/--output/--audit differ from run A.
    exit_code_b = main(
        [
            *common,
            "--output", str(output_b),
            "--ollama-host", "http://gpu-box-b.internal:11434",
            "--run-id", "run-b",
            "--audit", str(audit_b),
        ],
        provider_factory=lambda _config: provider_b,
        counter_factory=counter_factory,
    )
    assert exit_code_b == 0

    # THE CLAIM: run B's output is server A's stale, cross-host answer, and
    # run B's own provider was never invoked at all -- a silent, validated
    # cache HIT purely from sharing --cache-dir with a different --ollama-host.
    assert output_b.read_text(encoding="utf-8") == "ANSWER-FROM-GPU-BOX-A"
    assert provider_b.calls == []

    # Confirm the audit trail for run B does not surface anything host-related
    # that would let an operator notice: no "host" or endpoint text anywhere,
    # and no descriptor-invalidation reason recorded (a genuine key collision
    # produces a clean HIT, not a typed miss/invalidation).
    audit_b_body = json.loads(audit_b.read_text(encoding="utf-8"))
    audit_b_text = json.dumps(audit_b_body)
    assert "gpu-box" not in audit_b_text
    assert "11434" not in audit_b_text
    reliability = audit_b_body.get("reliability") or {}
    cache_block = reliability.get("cache") or {}
    assert cache_block.get("invalidation_reasons", []) == []
    # Both run-B work items (D000001 direct stage + editorial-final) show as
    # plain cache hits, not misses/recomputes -- the reliability snapshot
    # records a bare "hit" code per work item, not the work id itself, so a
    # host-only collision is indistinguishable from an ordinary same-host hit.
    assert cache_block.get("cache_hits", []) == ["hit", "hit"]
    assert cache_block.get("cache_misses", []) == []
