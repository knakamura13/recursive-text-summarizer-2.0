"""GET /runs/{id}/tree, /nodes/{node_id}, and /segments against live and legacy rows."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.support.web_views import (
    SeededDocument,
    audit_segment,
    paged_text,
    seed_document,
    seed_event,
    seed_node,
    seed_run,
    seed_segments,
    web_client,
)

PAGES = (
    "The bridge deck spans 200 feet across the river. Engineers inspected it in 1998.",
    "Cracks appeared in the north pier. Repairs began in 2001 and finished in 2003.",
)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with web_client(tmp_path, monkeypatch) as test_client:
        yield test_client


@pytest.fixture()
def document(client: TestClient) -> SeededDocument:
    text, page_map = paged_text(PAGES)
    return seed_document(text=text, page_map=page_map)


def _segment_bounds(document: SeededDocument) -> tuple[tuple[str, int, int], ...]:
    split = document.page_map[1]["start"] - len("\n\n--- Page 2 ---\n")
    return ("S000001", 0, split), ("S000002", split, len(document.text))


def _seed_live_run(document: SeededDocument) -> int:
    seed_run(document)
    seed_segments("run-1", _segment_bounds(document), document.page_map)
    first = seed_event("run-1")
    seed_node(
        "run-1",
        "L0N0001",
        level=0,
        order=0,
        label="Segment 1 · p. 1",
        parent_id="L1N0001",
        kind="leaf",
        state="completed",
        covered=("S000001",),
        child_ids=(),
        started_at="2026-09-23T12:00:00+00:00",
        completed_at="2026-09-23T12:00:12.500000+00:00",
        updated_event_id=first,
        summary={
            "summary": "The 200-foot deck was inspected in 1998.",
            "content_units": [
                {
                    "text": "The deck spans 200 feet.",
                    "kind": "fact",
                    "evidence": [
                        {"segment_id": "S000001", "quote": "spans 200 feet across the river"}
                    ],
                    "qualification": None,
                    "uncertain": False,
                },
                {
                    "text": "The pier cracked.",
                    "kind": "claim",
                    # Quoted from page 2, which this leaf's segment does not cover.
                    "evidence": [{"segment_id": "S000001", "quote": "north pier"}],
                    "qualification": "Reported once.",
                    "uncertain": True,
                },
            ],
            "entities": ["bridge deck", "river"],
            "qualifications": [
                {"text": "Only one inspection is described.", "evidence": [{"segment_id": "S000001", "quote": None}]}
            ],
            "contradictions": [
                {"text": "Dates conflict.", "evidence": [{"segment_id": "S000009", "quote": "1997"}]}
            ],
            "quotations": [{"segment_id": "S000001", "quote": "Engineers inspected it in 1998."}],
            "provenance": ["S000001"],
            "level": 0,
        },
    )
    seed_node(
        "run-1",
        "L0N0002",
        level=0,
        order=1,
        label="Segment 2 · p. 2",
        parent_id="L1N0001",
        kind="leaf",
        state="failed",
        covered=("S000002",),
        child_ids=(),
        started_at="2026-09-23T12:00:13+00:00",
        error="Segment 2 · p. 2 produced invalid output after 3 tries: missing evidence",
        updated_event_id=seed_event("run-1"),
    )
    last = seed_event("run-1")
    seed_node(
        "run-1",
        "L1N0001",
        level=1,
        order=0,
        label="Level 1 · Group 1 · pp. 1–2",
        kind="merge",
        state="pending",
        covered=("S000001", "S000002"),
        child_ids=("L0N0001", "L0N0002"),
        updated_event_id=last,
    )
    return last


def test_tree_lists_nodes_by_level_then_order_with_the_event_cursor(
    client: TestClient, document: SeededDocument
) -> None:
    last_event = _seed_live_run(document)

    response = client.get("/api/v1/runs/run-1/tree")

    assert response.status_code == 200
    body = response.json()
    assert body["cursor"] == last_event
    assert [(node["node_id"], node["state"]) for node in body["nodes"]] == [
        ("L0N0001", "completed"),
        ("L0N0002", "failed"),
        ("L1N0001", "pending"),
    ]
    root = body["nodes"][2]
    assert (root["kind"], root["child_count"], root["page_start"], root["page_end"]) == (
        "merge",
        2,
        1,
        2,
    )
    assert body["nodes"][0]["duration_seconds"] == pytest.approx(12.5)


def test_node_detail_locates_quotes_inside_the_cited_segment_only(
    client: TestClient, document: SeededDocument
) -> None:
    _seed_live_run(document)

    response = client.get("/api/v1/runs/run-1/nodes/L0N0001")

    assert response.status_code == 200
    node = response.json()
    first, second = node["content_units"]
    found = first["evidence"][0]
    start, end = document.offset("spans 200 feet across the river")
    assert (found["quote_found"], found["start"], found["end"]) == (True, start, end)
    assert (found["page_start"], found["page_end"]) == (1, 1)
    # "north pier" exists on page 2, but S000001's core does not contain it.
    missed = second["evidence"][0]
    core_end = _segment_bounds(document)[0][2]
    assert (missed["quote_found"], missed["start"], missed["end"]) == (False, 0, core_end)
    assert (missed["page_start"], missed["page_end"]) == (1, 1)
    assert (second["kind"], second["uncertain"], second["qualification"]) == (
        "claim",
        True,
        "Reported once.",
    )
    assert node["summary_text"] == "The 200-foot deck was inspected in 1998."
    assert node["entities"] == ["bridge deck", "river"]
    assert [(item["kind"], item["text"]) for item in node["annotations"]] == [
        ("qualification", "Only one inspection is described."),
        ("contradiction", "Dates conflict."),
    ]
    unknown = node["annotations"][1]["evidence"][0]
    assert unknown == {
        "segment_id": "S000009",
        "quote": "1997",
        "quote_found": False,
        "start": None,
        "end": None,
        "page_start": None,
        "page_end": None,
    }
    quotation = node["quotations"][0]
    assert quotation["quote_found"] is True
    assert document.text[quotation["start"] : quotation["end"]] == "Engineers inspected it in 1998."
    assert [(segment["segment_id"], segment["page_start"]) for segment in node["covered_segments"]] == [
        ("S000001", 1)
    ]
    assert node["duration_seconds"] == pytest.approx(12.5)
    assert (node["kind"], node["state"], node["parent_id"]) == ("leaf", "completed", "L1N0001")


def test_node_detail_reports_failure_and_pending_merges(
    client: TestClient, document: SeededDocument
) -> None:
    _seed_live_run(document)

    failed = client.get("/api/v1/runs/run-1/nodes/L0N0002").json()
    pending = client.get("/api/v1/runs/run-1/nodes/L1N0001").json()

    assert failed["state"] == "failed"
    assert failed["error"].endswith("missing evidence")
    assert failed["summary_text"] is None and failed["content_units"] == []
    assert failed["duration_seconds"] is None
    assert pending["child_ids"] == ["L0N0001", "L0N0002"]
    assert [segment["segment_id"] for segment in pending["covered_segments"]] == [
        "S000001",
        "S000002",
    ]
    assert pending["covered_segments"][1]["page_start"] == 2


def test_node_and_run_lookups_fail_with_codes(client: TestClient, document: SeededDocument) -> None:
    _seed_live_run(document)

    missing_node = client.get("/api/v1/runs/run-1/nodes/L9N0001")
    missing_run = client.get("/api/v1/runs/nope/nodes/L0N0001")

    assert missing_node.status_code == 404
    assert missing_node.json()["code"] == "node_not_found"
    assert missing_run.status_code == 404
    assert missing_run.json()["code"] == "run_not_found"
    for path in ("tree", "segments"):
        response = client.get(f"/api/v1/runs/nope/{path}")
        assert (response.status_code, response.json()["code"]) == (404, "run_not_found")


def test_segments_carry_offsets_and_pages(client: TestClient, document: SeededDocument) -> None:
    _seed_live_run(document)

    response = client.get("/api/v1/runs/run-1/segments")

    assert response.status_code == 200
    (first_id, first_start, first_end), (second_id, second_start, second_end) = _segment_bounds(document)
    assert response.json()["segments"] == [
        {
            "segment_id": first_id,
            "order": 0,
            "start": first_start,
            "end": first_end,
            "core_start": first_start,
            "core_end": first_end,
            "page_start": 1,
            "page_end": 1,
        },
        {
            "segment_id": second_id,
            "order": 1,
            "start": second_start,
            "end": second_end,
            "core_start": second_start,
            "core_end": second_end,
            "page_start": 2,
            "page_end": 2,
        },
    ]


def _seed_legacy_run(document: SeededDocument) -> None:
    """A completed Run from before live projections: no kind/state history,
    content units, child ids, or run_segments; segments live in audit.json."""
    (first_id, first_start, first_end), (second_id, second_start, second_end) = _segment_bounds(document)
    sha = document.source_sha256
    seed_run(
        document,
        run_id="run-legacy",
        summary="Deck and pier summary.",
        audit={
            "schema_version": "audit/4",
            "source_segments": [
                audit_segment(first_id, 0, first_start, first_end, sha),
                audit_segment(second_id, 1, second_start, second_end, sha),
                # A verification sub-passage: cited by evidence, never a leaf.
                audit_segment("S000003", 2, second_start, second_start + 20, sha),
            ],
            "tree_nodes": [
                {"node_id": "L0N0001", "covered_segments": [first_id]},
                {"node_id": "L0N0002", "covered_segments": [second_id]},
                {"node_id": "L1N0001", "covered_segments": [first_id, second_id]},
            ],
        },
    )
    for order, (node_id, segment_id) in enumerate((("L0N0001", first_id), ("L0N0002", second_id))):
        seed_node(
            "run-legacy",
            node_id,
            level=0,
            order=order,
            label=f"Segment {order + 1}",
            parent_id="L1N0001",
            summary={"summary": f"Leaf {order + 1}."},
            covered=(segment_id,),
            legacy_evidence=[
                {"segment_id": segment_id, "quote": "Cracks appeared in the north pier."},
                {"segment_id": segment_id, "quote": "Cracks appeared in the north pier."},
                {"segment_id": segment_id, "quote": None},
            ],
        )
    seed_node(
        "run-legacy",
        "L1N0001",
        level=1,
        order=0,
        label="Merge group 1",
        kind="merge",
        summary={"summary": "Deck and pier summary."},
        covered=(first_id, second_id),
        legacy_evidence=[],
    )


def test_legacy_runs_degrade_to_parent_links_and_audit_segments(
    client: TestClient, document: SeededDocument
) -> None:
    _seed_legacy_run(document)

    tree = client.get("/api/v1/runs/run-legacy/tree").json()
    root = client.get("/api/v1/runs/run-legacy/nodes/L1N0001").json()
    leaf = client.get("/api/v1/runs/run-legacy/nodes/L0N0002").json()
    segments = client.get("/api/v1/runs/run-legacy/segments").json()["segments"]

    assert tree["cursor"] == 0
    assert [(node["node_id"], node["state"], node["child_count"]) for node in tree["nodes"]] == [
        ("L0N0001", "completed", 0),
        ("L0N0002", "completed", 0),
        ("L1N0001", "completed", 2),
    ]
    assert root["child_ids"] == ["L0N0001", "L0N0002"]
    assert root["kind"] == "merge"
    assert [segment["segment_id"] for segment in segments] == ["S000001", "S000002"]
    assert [(segment["page_start"], segment["page_end"]) for segment in segments] == [(1, 1), (2, 2)]
    assert leaf["content_units"] == [] and leaf["annotations"] == []
    assert leaf["summary_text"] == "Leaf 2."
    # Legacy evidence survives as deduplicated quotations resolved in the text.
    found, bare = leaf["quotations"]
    assert found["quote_found"] is True and found["page_start"] == 2
    assert document.text[found["start"] : found["end"]] == "Cracks appeared in the north pier."
    assert (bare["quote"], bare["quote_found"], bare["start"]) == (
        None,
        False,
        _segment_bounds(document)[1][1],
    )
    assert [segment["segment_id"] for segment in leaf["covered_segments"]] == ["S000002"]
