"""Observer events, re-asks, Stop, and Resume through the real pipeline."""

import json
import multiprocessing
import re
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic, sleep

import pytest

from summarizer.config import AppConfig, CacheConfig, ReliabilityConfig, StrategyConfig
from summarizer.ingestion import ingest_text
from summarizer.leaf import LeafSummaryError
from summarizer.pipeline import PipelineConfig, PipelineResult, run_pipeline
from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderResponseError,
)
from summarizer.runtime.observers import (
    ItemEvent,
    ItemFailedError,
    PipelineStopped,
    RuntimeObserver,
    SegmentInfo,
    StageEvent,
    StageName,
)
from summarizer.segmentation import SegmentationConfig
from tests.support.compression_provider import compression_generation_payload

TEXT = "one two three four five six seven eight nine ten " * 12
RUN_ID = "runtime-events"
THREE_SEGMENTS = 200
FOUR_SEGMENTS = 150


class Counter:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


def _summary(level: int, identifier: str) -> dict[str, object]:
    return {
        "summary": f"Grounded level {level} from {identifier}.",
        "content_units": [],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": [identifier],
        "level": level,
    }


class ScriptedProvider:
    """Answer leaves, merges, and the editorial validly, with scripted faults.

    `invalid` maps a work id to how many unparseable answers it gets before a
    valid one. Calls for a work id in `block` wait until `release` is set.
    """

    def __init__(
        self,
        *,
        invalid: dict[str, int] | None = None,
        block: frozenset[str] = frozenset(),
    ) -> None:
        self.invalid = dict(invalid or {})
        self.block = block
        self.started = {work_id: Event() for work_id in block}
        self.release = Event()
        self.requests: list[GenerationRequest] = []
        self._lock = Lock()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        work_id = request.audit_work_id or request.operation_id or ""
        with self._lock:
            self.requests.append(request)
            spoiled = self.invalid.get(work_id, 0)
            if spoiled:
                self.invalid[work_id] = spoiled - 1
        if work_id in self.block:
            self.started[work_id].set()
            self.release.wait(timeout=30)
        if spoiled:
            return GenerationResult("I would rather not.", "fake", request.model)
        if work_id == "editorial-final":
            payload: dict[str, object] = {"text": "A concise final summary."}
        elif work_id.startswith("compression:"):
            payload = compression_generation_payload(request)
        elif work_id.startswith(("S", "D")):
            payload = _summary(0, work_id)
        else:
            identifiers = re.findall(r'"segment_id":"([SD]\d+)"', request.input_text)
            payload = _summary(int(work_id[1 : work_id.index("N")]), identifiers[-1])
        return GenerationResult(json.dumps(payload), "fake", request.model)

    def leaf_calls(self) -> list[str]:
        with self._lock:
            return [
                request.operation_id or ""
                for request in self.requests
                if (request.operation_id or "").startswith("S")
            ]


TREE_KINDS = frozenset({"leaf", "merge", "passthrough"})
TREE_STAGES = frozenset(
    {
        StageName.PREPARING,
        StageName.SEGMENTING,
        StageName.SUMMARIZING,
        StageName.MERGING,
    }
)


class Recorder:
    """Collect the tree stages, segments, and tree items, in arrival order.

    Writing, verification, and publication events come from finalization and
    are not what these tests are about, so they are left out.
    """

    def __init__(self) -> None:
        self.timeline: list[StageEvent | ItemEvent | tuple[SegmentInfo, ...]] = []
        self.stop = Event()
        self._lock = Lock()

    def observer(self) -> RuntimeObserver:
        return RuntimeObserver(
            on_stage=self._record,
            on_segments=self._record,
            on_item=self._record,
            should_stop=self.stop.is_set,
        )

    def _record(self, event: StageEvent | ItemEvent | tuple[SegmentInfo, ...]) -> None:
        if isinstance(event, ItemEvent) and event.kind not in TREE_KINDS:
            return
        if isinstance(event, StageEvent) and event.stage not in TREE_STAGES:
            return
        with self._lock:
            self.timeline.append(event)

    @property
    def items(self) -> list[ItemEvent]:
        with self._lock:
            return [event for event in self.timeline if isinstance(event, ItemEvent)]

    @property
    def stages(self) -> list[StageEvent]:
        with self._lock:
            return [event for event in self.timeline if isinstance(event, StageEvent)]

    def states(self, work_id: str) -> list[str]:
        return [event.state for event in self.items if event.work_id == work_id]


def summarize(
    provider: ScriptedProvider,
    recorder: Recorder | None = None,
    *,
    cache_root: Path | None = None,
    run_id: str = RUN_ID,
    mode: str = "new",
    max_tokens: int = THREE_SEGMENTS,
    max_in_flight: int = 1,
    strategy: str = "hierarchical",
    audit_path: Path | None = None,
) -> PipelineResult:
    return run_pipeline(
        ingest_text(TEXT),
        provider,
        Counter(),
        app=AppConfig(
            model="gpt-4o-mini",
            timeout_seconds=30,
            output_path=(
                audit_path.with_name("summary.txt")
                if audit_path is not None
                else Path("output.txt")
            ),
        ),
        strategy=StrategyConfig(
            strategy=strategy,
            context_window=100_000,
            max_output_tokens=1,
            safety_margin_tokens=0,
            safety_margin_fraction=0,
        ),
        config=PipelineConfig(
            target_words=40,
            segmentation=SegmentationConfig(max_tokens=max_tokens),
            max_merge_children=2,
            audit_path=audit_path,
            cache=CacheConfig(
                enabled=cache_root is not None,
                root=cache_root or Path(".summarizer-cache"),
            ),
            reliability=ReliabilityConfig(
                run_id=run_id if cache_root is not None else None,
                run_mode=mode,
                max_in_flight=max_in_flight,
            ),
        ),
        observer=recorder.observer() if recorder is not None else None,
    )


def test_reasks_count_as_attempts_of_their_own_item_without_a_retry_reason(
    tmp_path: Path,
) -> None:
    """audit/3 only accepts transient retry reasons, so a re-ask adds none."""

    class RejectedTwiceProvider(ScriptedProvider):
        def __init__(self) -> None:
            super().__init__()
            self.rejections: list[Exception | str] = [
                ProviderResponseError(
                    "Ollama stopped at the model's output limit; the response is incomplete"
                ),
                "I would rather not.",
            ]

        def generate(self, request: GenerationRequest) -> GenerationResult:
            if request.operation_id == "D000001" and self.rejections:
                rejection = self.rejections.pop(0)
                if isinstance(rejection, Exception):
                    raise rejection
                return GenerationResult(rejection, "fake", request.model)
            return super().generate(request)

    audit_path = tmp_path / "audit.json"
    recorder = Recorder()

    summarize(
        RejectedTwiceProvider(),
        recorder,
        cache_root=tmp_path / "cache",
        strategy="direct",
        audit_path=audit_path,
    )

    assert recorder.states("L0N0001") == [
        "planned",
        "active",
        "retrying",
        "retrying",
        "completed",
    ]
    assert [
        (event.attempt, event.message)
        for event in recorder.items
        if event.state == "retrying"
    ] == [
        (1, "Ollama stopped at the model's output limit; the response is incomplete"),
        (2, "D000001: response was not a single JSON object (no JSON object found)"),
    ]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["reliability"]["attempts"] == [
        {"work_id": "D000001", "attempt_count": 3, "failure_reasons": []},
        {"work_id": "editorial-final", "attempt_count": 1, "failure_reasons": []},
    ]


def test_hierarchical_run_reports_every_leaf_and_merge_under_its_node_id() -> None:
    recorder = Recorder()

    result = summarize(ScriptedProvider(), recorder)

    tree_and_stages = [
        (event.stage.value, event.state)
        if isinstance(event, StageEvent)
        else (event.work_id, event.state)
        if isinstance(event, ItemEvent)
        else ("segments", tuple(info.segment_id for info in event))
        for event in recorder.timeline
    ]
    assert tree_and_stages == [
        ("preparing", "active"),
        ("preparing", "completed"),
        ("segmenting", "active"),
        ("segments", ("S000001", "S000002", "S000003")),
        ("segmenting", "completed"),
        ("summarizing", "active"),
        ("L0N0001", "planned"),
        ("L0N0002", "planned"),
        ("L0N0003", "planned"),
        ("L0N0001", "active"),
        ("L0N0001", "completed"),
        ("L0N0002", "active"),
        ("L0N0002", "completed"),
        ("L0N0003", "active"),
        ("L0N0003", "completed"),
        ("summarizing", "completed"),
        ("merging", "active"),
        ("L1N0001", "planned"),
        ("L1N0002", "planned"),
        ("L1N0002", "completed"),
        ("L1N0001", "active"),
        ("L1N0001", "completed"),
        ("L2N0001", "planned"),
        ("L2N0001", "active"),
        ("L2N0001", "completed"),
        ("merging", "completed"),
    ]
    assert recorder.stages[1].detail == "hierarchical"

    planned = {
        event.work_id: (
            event.kind,
            event.level,
            event.order,
            event.total,
            event.child_ids,
            event.covered_segment_ids,
        )
        for event in recorder.items
        if event.state == "planned"
    }
    assert planned == {
        "L0N0001": ("leaf", 0, 0, 3, (), ("S000001",)),
        "L0N0002": ("leaf", 0, 1, 3, (), ("S000002",)),
        "L0N0003": ("leaf", 0, 2, 3, (), ("S000003",)),
        "L1N0001": ("merge", 1, 0, 2, ("L0N0001", "L0N0002"), ("S000001", "S000002")),
        "L1N0002": ("passthrough", 1, 1, 2, ("L0N0003",), ("S000003",)),
        "L2N0001": (
            "merge",
            2,
            0,
            1,
            ("L1N0001", "L1N0002"),
            ("S000001", "S000002", "S000003"),
        ),
    }
    # Completed events carry exactly the summaries the final tree holds.
    assert {
        event.work_id: event.summary
        for event in recorder.items
        if event.state == "completed"
    } == {node.node_id: node.summary.model_dump(mode="json") for node in result.nodes}

    document = ingest_text(TEXT)
    (segments,) = [
        event for event in recorder.timeline if isinstance(event, tuple)
    ]
    assert segments[0].core_start == 0
    assert segments[-1].core_end == len(document.text)
    assert all(
        previous.core_end == following.core_start
        for previous, following in zip(segments, segments[1:])
    )


def test_direct_run_reports_the_whole_document_segment_and_one_leaf() -> None:
    recorder = Recorder()

    summarize(ScriptedProvider(), recorder, strategy="direct")

    document = ingest_text(TEXT)
    whole = len(document.text)
    assert [event for event in recorder.timeline if isinstance(event, tuple)] == [
        (SegmentInfo("D000001", 0, 0, whole, 0, whole, whole),)
    ]
    stages = {(event.stage, event.state): event for event in recorder.stages}
    assert stages[(StageName.PREPARING, "completed")].detail == "direct"
    assert (StageName.SEGMENTING, "skipped") in stages
    assert (StageName.MERGING, "skipped") in stages
    assert [
        (event.work_id, event.state, event.covered_segment_ids)
        for event in recorder.items
    ] == [
        ("L0N0001", "planned", ("D000001",)),
        ("L0N0001", "active", ("D000001",)),
        ("L0N0001", "completed", ("D000001",)),
    ]


def test_a_reasked_leaf_succeeds_and_is_cached_under_its_original_request(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    provider = ScriptedProvider(invalid={"S000002": 1})
    recorder = Recorder()

    summarize(provider, recorder, cache_root=cache_root, max_in_flight=2)

    assert recorder.states("L0N0002") == ["planned", "active", "retrying", "completed"]
    (retrying,) = [event for event in recorder.items if event.state == "retrying"]
    assert retrying.attempt == 1
    assert retrying.message == (
        "S000002: response was not a single JSON object (no JSON object found)"
    )
    assert provider.leaf_calls().count("S000002") == 2

    # A new run finds the re-asked answer by the original request's descriptor.
    fresh = ScriptedProvider()
    fresh_recorder = Recorder()
    summarize(fresh, fresh_recorder, cache_root=cache_root, run_id="runtime-again")

    assert fresh.leaf_calls() == []
    assert fresh_recorder.states("L0N0002") == ["planned", "reused"]


def test_a_leaf_that_stays_invalid_fails_naming_it_and_resume_redoes_only_it(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    recorder = Recorder()

    with pytest.raises(ItemFailedError) as failure:
        summarize(
            ScriptedProvider(invalid={"S000002": 3}),
            recorder,
            cache_root=cache_root,
            max_tokens=FOUR_SEGMENTS,
            max_in_flight=4,
        )

    error = failure.value
    assert (error.stage, error.kind, error.work_id, error.covered_segment_ids) == (
        StageName.SUMMARIZING,
        "leaf",
        "L0N0002",
        ("S000002",),
    )
    assert str(error) == (
        "S000002: response was not a single JSON object (no JSON object found)"
    )
    assert isinstance(error.__cause__, LeafSummaryError)
    assert recorder.states("L0N0002") == [
        "planned",
        "active",
        "retrying",
        "retrying",
        "failed",
    ]
    (failed,) = [event for event in recorder.items if event.state == "failed"]
    assert failed.message == str(error)

    healthy = ScriptedProvider()
    resumed = Recorder()
    summarize(
        healthy,
        resumed,
        cache_root=cache_root,
        mode="resume",
        max_tokens=FOUR_SEGMENTS,
        max_in_flight=4,
    )

    assert healthy.leaf_calls() == ["S000002"]
    assert resumed.states("L0N0002") == ["planned", "active", "completed"]
    for finished in ("L0N0001", "L0N0003", "L0N0004"):
        assert resumed.states(finished) == ["planned", "reused"]
    reused = {
        event.work_id: event.summary
        for event in resumed.items
        if event.state == "reused"
    }
    assert reused["L0N0001"] == _summary(0, "S000001")


def test_stop_ends_the_run_while_calls_are_blocked_and_resume_redoes_them(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    provider = ScriptedProvider(block=frozenset({"S000002", "S000004"}))
    recorder = Recorder()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            summarize(
                provider,
                recorder,
                cache_root=cache_root,
                max_tokens=FOUR_SEGMENTS,
                max_in_flight=2,
            )
        except BaseException as error:  # noqa: BLE001 - surface the outcome
            errors.append(error)

    thread = Thread(target=run, daemon=True)
    thread.start()
    try:
        assert provider.started["S000002"].wait(timeout=10)
        assert provider.started["S000004"].wait(timeout=10)
        stopped_at = monotonic()
        recorder.stop.set()
        thread.join(timeout=3)
        # Both blocked calls are still waiting for `release`.
        assert not thread.is_alive()
        assert monotonic() - stopped_at < 3
    finally:
        provider.release.set()

    assert len(errors) == 1
    assert isinstance(errors[0], PipelineStopped)
    assert recorder.states("L0N0001")[-1] == "completed"
    assert recorder.states("L0N0003")[-1] == "completed"
    assert recorder.states("L0N0002") == ["planned", "active"]
    assert recorder.states("L0N0004") == ["planned", "active"]

    healthy = ScriptedProvider()
    resumed = Recorder()
    summarize(
        healthy,
        resumed,
        cache_root=cache_root,
        mode="resume",
        max_tokens=FOUR_SEGMENTS,
        max_in_flight=2,
    )

    assert sorted(healthy.leaf_calls()) == ["S000002", "S000004"]
    assert resumed.states("L0N0001") == ["planned", "reused"]
    assert resumed.states("L0N0003") == ["planned", "reused"]


def _summarize_until_killed(cache_root: str, markers: str) -> None:
    """Child body: leave S000002 and S000004 in flight until the parent kills it."""
    marker_dir = Path(markers)

    class MarkingProvider(ScriptedProvider):
        def generate(self, request: GenerationRequest) -> GenerationResult:
            if request.operation_id in self.block:
                (marker_dir / request.operation_id).touch()
            return super().generate(request)

    summarize(
        MarkingProvider(block=frozenset({"S000002", "S000004"})),
        cache_root=Path(cache_root),
        max_tokens=FOUR_SEGMENTS,
        max_in_flight=2,
    )


def test_resume_after_a_kill_mid_batch_recomputes_only_unfinished_leaves(
    tmp_path: Path,
) -> None:
    """SIGKILL while two leaves are in flight; the two that finished survive."""
    cache_root = tmp_path / "cache"
    markers = tmp_path / "markers"
    markers.mkdir()
    child = multiprocessing.get_context("spawn").Process(
        target=_summarize_until_killed,
        args=(str(cache_root), str(markers)),
        daemon=True,
    )
    child.start()
    try:
        deadline = monotonic() + 60
        while not all((markers / work_id).exists() for work_id in ("S000002", "S000004")):
            assert child.is_alive(), "the run ended before it could be killed"
            assert monotonic() < deadline, "the blocked leaves never started"
            sleep(0.05)
    finally:
        child.kill()
        child.join(timeout=10)

    manifest = json.loads((cache_root / "runs" / f"{RUN_ID}.json").read_text())
    assert [reference["work_id"] for reference in manifest["completed"]] == [
        "segmentation",
        "S000001",
        "S000003",
    ]

    healthy = ScriptedProvider()
    resumed = Recorder()
    summarize(
        healthy,
        resumed,
        cache_root=cache_root,
        mode="resume",
        max_tokens=FOUR_SEGMENTS,
        max_in_flight=2,
    )

    assert sorted(healthy.leaf_calls()) == ["S000002", "S000004"]
    assert resumed.states("L0N0001") == ["planned", "reused"]
    assert resumed.states("L0N0003") == ["planned", "reused"]
    assert resumed.states("L0N0002") == ["planned", "active", "completed"]
    assert resumed.states("L0N0004") == ["planned", "active", "completed"]
