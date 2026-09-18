#!/usr/bin/env python3
"""F-019 repro: a real hierarchical run, with --audit-equivalent (audit_path set),
over a document ending in a fenced code block, crashes audit-artifact construction.

This does NOT hand-construct AuditSegment. It calls the real, public entry point
(summarizer.pipeline.run_pipeline) with a fake in-process provider (mirroring the
PipelineProvider test double in tests/test_pipeline.py, per the repo's own offline
test convention -- no network, no Ollama). The goal is to prove or disprove that a
CODE_FENCE-boundary SourceSegment, produced by the real segmentation code, survives
packing and reaches summarizer.audit._audit_segment via the real call path
run_pipeline -> _run_pipeline -> _finalize_summary -> _build_audit ->
build_audit_artifact -> _audit_segment.

Run with:
    uv run --with-requirements requirements-dev.txt python .review/repro/F-019.py
"""

import json
import re
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from summarizer.config import AppConfig, StrategyConfig  # noqa: E402
from summarizer.ingestion import ingest_text  # noqa: E402
from summarizer.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from summarizer.providers.base import GenerationRequest, GenerationResult  # noqa: E402
from summarizer.segmentation import BoundaryKind  # noqa: E402


class CharacterCounter:
    """Same fake counter test doubles use in tests/test_pipeline.py / test_segmentation.py."""

    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


class PipelineProvider:
    """Copied verbatim in shape from tests/test_pipeline.py's PipelineProvider fake.
    No network calls, no Ollama -- pure in-process stand-in, per the review's
    offline-only constraint."""

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if request.operation_id == "editorial-final":
            payload = {"text": "A concise, coherent final summary."}
        elif request.operation_id == "D000001":
            payload = self._node(0, "D000001")
        elif (request.operation_id or "").startswith("S"):
            payload = self._node(0, request.operation_id or "S000001")
        else:
            level = int((request.operation_id or "merge-L1").rsplit("L", 1)[1])
            identifiers = re.findall(r'"segment_id":"([SD]\d+)"', request.input_text)
            payload = self._node(level, identifiers[-1] if identifiers else "S000001")
        return GenerationResult(json.dumps(payload), "fake", request.model, 1, 1, "stop")

    @staticmethod
    def _node(level: int, identifier: str) -> dict:
        return {
            "summary": f"Grounded level {level}.",
            "content_units": [],
            "entities": [],
            "qualifications": [],
            "contradictions": [],
            "quotations": [],
            "provenance": [identifier],
            "level": level,
        }


DOCUMENT_WITH_TRAILING_CODE_FENCE = (
    "This report walks through the deployment script used to roll out the "
    "nightly batch job, and the snippet below is exactly what operators run "
    "by hand today when the scheduler is down.\n\n"
    "```python\n"
    "def deploy():\n"
    "    print('rolling out nightly batch job')\n"
    "```\n"
)


def main() -> int:
    document = ingest_text(DOCUMENT_WITH_TRAILING_CODE_FENCE)

    with tempfile.TemporaryDirectory() as tmp:
        audit_path = Path(tmp) / "audit.json"
        provider = PipelineProvider()

        # Force the hierarchical path explicitly -- this is the same mechanism
        # tests/test_pipeline.py uses (StrategyConfig(strategy="hierarchical", ...)),
        # not a hand-rolled shortcut. It corresponds to a live CLI run whenever
        # --strategy hierarchical is passed, OR whenever "auto" strategy selection
        # decides the document doesn't fit a single context window (see
        # summarizer/budget.py:338-361). Forcing it here isolates the reachability
        # question (can a CODE_FENCE segment reach audit construction at all) from
        # the separate, uninteresting question of exactly which auto-selection
        # thresholds apply to this toy document.
        strategy_config = StrategyConfig(
            strategy="hierarchical",
            context_window=100_000,
            max_output_tokens=1,
            safety_margin_tokens=0,
            safety_margin_fraction=0,
        )
        pipeline_config = PipelineConfig(
            target_words=40,
            # audit_path set == the CLI's --audit <path> flag (summarizer/cli.py:86,200-218).
            audit_path=audit_path,
        )

        print("=== Running real pipeline (run_pipeline) with audit_path set ===")
        try:
            result = run_pipeline(
                document,
                provider,
                CharacterCounter(),
                app=AppConfig(model="gpt-4o-mini", timeout_seconds=30),
                strategy=strategy_config,
                config=pipeline_config,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\nrun_pipeline itself raised before returning: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 1

        print(f"strategy selected: {result.strategy.strategy}")
        print(f"number of segments produced by the REAL segmentation code: {len(result.final.audit.source_segments) if result.final.audit else 'n/a (audit is None!)'}")

        # If we got here, audit construction succeeded end-to-end through the real
        # call path. Show what boundary_kind values the real segmenter actually
        # produced, and whether a CODE_FENCE segment made it into the artifact.
        if result.final.audit is not None:
            kinds = [seg.boundary_kind for seg in result.final.audit.source_segments]
            print(f"boundary_kind values in the resulting AuditArtifact.source_segments: {kinds}")
            if "code_fence" in kinds:
                print(
                    "\nUNEXPECTED (per the claim): a code_fence-boundary AuditSegment "
                    "was constructed successfully. The claim in the source review "
                    "appears REFUTED for this input."
                )
            else:
                print(
                    "\nNo code_fence segment reached the audit artifact for this input "
                    "-- the CODE_FENCE unit must have been merged/absorbed/reclassified "
                    "upstream. Need to check whether that's guaranteed or just this input."
                )
            print("\nRESULT: run_pipeline succeeded; no ValidationError raised for this input.")
            return 0
        else:
            print("\naudit_path was set but result.final.audit is None -- unexpected, investigate.")
            return 2


if __name__ == "__main__":
    sys.exit(main())
