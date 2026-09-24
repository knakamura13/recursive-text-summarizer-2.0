"""GET /runs/{id}/summary: sentences, evidence, removed sentences, notices, and legacy Runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from summarizer.verification import split_draft_spans
from summarizer_web.models.api import RunConfig
from tests.support.web_views import (
    SeededDocument,
    audit_segment,
    paged_text,
    seed_document,
    seed_run,
    seed_segments,
    web_client,
)

PAGES = (
    "The bridge deck spans 200 feet across the river. Engineers inspected it in 1998.",
    "Cracks appeared in the north pier. Repairs began in 2001 and finished in 2003.",
)
PUBLISHED = "The deck spans 200 feet. Engineers inspected it in 1998.\n\nRepairs finished in 2003."


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with web_client(tmp_path, monkeypatch) as test_client:
        yield test_client


@pytest.fixture()
def document(client: TestClient) -> SeededDocument:
    text, page_map = paged_text(PAGES)
    return seed_document(text=text, page_map=page_map)


def _page_body(document: SeededDocument, page: int) -> tuple[int, int]:
    entry = document.page_map[page - 1]
    return entry["start"], entry["end"]


def _direct_audit(document: SeededDocument, **fields: object) -> dict:
    """A direct Run's audit: D000001 plus verification sub-passages S000001/S000002."""
    sha = document.source_sha256
    audit = {
        "schema_version": "audit/2",
        "source_id": sha,
        "strategy": "direct",
        "source_segments": [
            audit_segment("D000001", 0, 0, len(document.text), sha),
            audit_segment("S000001", 1, *_page_body(document, 1), sha),
            audit_segment("S000002", 2, *_page_body(document, 2), sha),
        ],
        "tree_nodes": [{"node_id": "L0N0001", "covered_segments": ["D000001"]}],
        "root_node_id": "L0N0001",
        "citations": [
            {"segment_id": "S000001", "source_id": sha, "order": 1},
            {"segment_id": "S000002", "source_id": sha, "order": 2},
        ],
        "warnings": [],
        "failures": [],
        "verification": {
            "enabled": True,
            "pass_count": 1,
            "exhausted": False,
            "failed": False,
            "passes": [],
            "repairs": [],
            "usage": [],
            "warning_codes": [],
            "limitation_codes": [],
            "failure_codes": [],
        },
    }
    audit.update(fields)
    return audit


def _subset_publication(document: SeededDocument) -> dict:
    quote = "spans 200 feet across the river"
    start, end = document.offset(quote)
    return {
        "kind": "verified_subset",
        "sentences": [
            {
                "index": 0,
                "paragraph": 0,
                "start": 0,
                "end": 24,
                "text": "The deck spans 200 feet.",
                "verdict": "supported",
                "evidence": [{"segment_id": "S000001", "quote": quote, "start": start, "end": end}],
            },
            {
                "index": 1,
                "paragraph": 0,
                "start": 25,
                "end": 56,
                "text": "Engineers inspected it in 1998.",
                "verdict": "not_meaningfully_verifiable",
                "evidence": [],
            },
            {
                "index": 2,
                "paragraph": 1,
                "start": 58,
                "end": 83,
                "text": "Repairs finished in 2003.",
                "verdict": "supported",
                # A redacted quote the audit could not place: resolves to the passage.
                "evidence": [
                    {"segment_id": "S000002", "quote": "finished in [REDACTED]", "start": None, "end": None}
                ],
            },
        ],
        "removed_sentences": [
            {
                "text": "The deck is 300 feet long.",
                "verdict": "contradicted",
                "reason": 'The source contradicts "300 feet".',
            }
        ],
    }


def test_verified_subset_summary_maps_sentences_to_evidence_and_explains_removals(
    client: TestClient, document: SeededDocument
) -> None:
    audit = _direct_audit(
        document,
        publication=_subset_publication(document),
        warnings=["verified_sentence_subset"],
    )
    audit["verification"]["warning_codes"] = ["invalid_anchors_omitted", "future_diagnostic"]
    audit["verification"]["limitation_codes"] = ["repair_not_eligible"]
    seed_run(document, summary=PUBLISHED + "\n\nSources: S000001, S000002", audit=audit)
    seed_segments("run-1", (("D000001", 0, len(document.text)),), document.page_map)

    response = client.get("/api/v1/runs/run-1/summary")

    assert response.status_code == 200
    summary = response.json()
    assert summary["available"] is True
    assert summary["text"] == PUBLISHED
    assert summary["publication"] == "verified_subset"
    assert summary["verification_state"] == "completed"
    assert [(item["index"], item["paragraph"], item["verdict"]) for item in summary["sentences"]] == [
        (0, 0, "supported"),
        (1, 0, "not_meaningfully_verifiable"),
        (2, 1, "supported"),
    ]
    exact = summary["sentences"][0]["evidence"][0]
    assert exact["quote_found"] is True
    assert document.text[exact["start"] : exact["end"]] == "spans 200 feet across the river"
    assert (exact["page_start"], exact["page_end"]) == (1, 1)
    # S000002 is a verification sub-passage: it exists only in audit.json.
    passage = summary["sentences"][2]["evidence"][0]
    assert (passage["quote_found"], passage["start"], passage["end"]) == (
        False,
        *_page_body(document, 2),
    )
    assert passage["page_start"] == 2
    assert summary["removed_sentences"] == [
        {
            "text": "The deck is 300 feet long.",
            "verdict": "contradicted",
            "reason": 'The source contradicts "300 feet".',
        }
    ]
    assert [(item["citation_id"], item["segment_id"], item["page_start"]) for item in summary["citations"]] == [
        ("1", "S000001", 1),
        ("2", "S000002", 2),
    ]
    assert summary["word_count"] == 14
    assert (summary["target_words"], summary["short_of_target"]) == (40, True)
    codes = [notice["code"] for notice in summary["notices"]]
    assert codes == [
        "verified_sentence_subset",
        "short_of_target",
        "invalid_anchors_omitted",
        "future_diagnostic",
        "repair_not_eligible",
    ]
    notices = {notice["code"]: notice for notice in summary["notices"]}
    assert "14" in notices["short_of_target"]["message"] and "40" in notices["short_of_target"]["message"]
    assert notices["short_of_target"]["severity"] == "warning"
    assert notices["verified_sentence_subset"]["severity"] == "warning"
    for notice in summary["notices"]:
        assert notice["message"].strip() and notice["message"] != notice["code"]


def test_content_unit_fallback_is_named_with_its_shortfall(
    client: TestClient, document: SeededDocument
) -> None:
    publication = _subset_publication(document) | {"kind": "content_unit_fallback"}
    audit = _direct_audit(
        document, publication=publication, warnings=["verified_content_unit_fallback"]
    )
    seed_run(
        document,
        config=RunConfig(model="llama3.2:3b", target_words=25),
        summary=PUBLISHED,
        audit=audit,
    )

    summary = client.get("/api/v1/runs/run-1/summary").json()

    assert summary["publication"] == "content_unit_fallback"
    assert [notice["code"] for notice in summary["notices"]] == [
        "verified_content_unit_fallback",
        "short_of_target",
    ]
    assert [notice["severity"] for notice in summary["notices"]] == ["warning", "warning"]


def test_editorial_summary_at_its_target_has_no_notices(
    client: TestClient, document: SeededDocument
) -> None:
    audit = _direct_audit(document, publication=_subset_publication(document) | {"kind": "editorial"})
    seed_run(
        document,
        config=RunConfig(model="llama3.2:3b", target_words=25),
        summary=PUBLISHED + " " + "word " * 11,
        audit=audit,
    )

    summary = client.get("/api/v1/runs/run-1/summary").json()

    assert summary["publication"] == "editorial"
    assert (summary["word_count"], summary["short_of_target"]) == (25, False)
    assert summary["notices"] == []


def test_verification_off_leaves_sentences_unchecked(
    client: TestClient, document: SeededDocument
) -> None:
    publication = {
        "kind": "editorial",
        "sentences": [
            {"index": 0, "paragraph": 0, "start": 0, "end": 24, "text": "The deck spans 200 feet.", "verdict": "unchecked", "evidence": []}
        ],
        "removed_sentences": [],
    }
    audit = _direct_audit(document, publication=publication, citations=[])
    audit["verification"] = {
        "enabled": False,
        "pass_count": 0,
        "exhausted": False,
        "failed": False,
        "passes": [],
        "repairs": [],
        "usage": [],
        "warning_codes": [],
        "limitation_codes": [],
        "failure_codes": [],
    }
    seed_run(
        document,
        config=RunConfig(model="llama3.2:3b", target_words=25, verify=False),
        summary="The deck spans 200 feet.",
        audit=audit,
    )

    summary = client.get("/api/v1/runs/run-1/summary").json()

    assert summary["verification_state"] == "not_run"
    assert [sentence["verdict"] for sentence in summary["sentences"]] == ["unchecked"]
    assert [notice["code"] for notice in summary["notices"]] == ["short_of_target", "verification_off"]
    assert summary["notices"][0]["severity"] == "info"


def _legacy_pass(text: str) -> dict:
    """The audit/2 record of one verification pass over `text`, as the verifier wrote it."""
    spans = split_draft_spans(text, pass_index=1)
    verdicts = ("supported", "not_meaningfully_verifiable", "supported")
    evidence = (("S000001",), (), ("S000002", "S000001"))
    return {
        "pass_index": 1,
        "complete": True,
        "spans": [
            {
                "span_id": span.span_id,
                "ordinal": span.ordinal,
                "start": span.start,
                "end": span.end,
                "content_hash": hashlib.sha256(span.text.encode("utf-8")).hexdigest(),
            }
            for span in spans
        ],
        "claims": [
            {"claim_id": f"V01C{index:06d}", "span_id": span.span_id, "ordinal": 1, "is_fallback": False}
            for index, span in enumerate(spans, start=1)
        ],
        "selections": [],
        "assessments": [
            {
                "claim_id": f"V01C{index:06d}",
                "verdict": verdict,
                "verifier_provider": "ollama",
                "verifier_model": "llama3.2:3b",
                "prompt_version": "verification-classification/6",
                "findings": [
                    {"claim_id": f"V01C{index:06d}", "verdict": verdict, "evidence_ids": list(ids)}
                ],
            }
            for index, (verdict, ids) in enumerate(zip(verdicts, evidence), start=1)
        ],
    }


def test_legacy_audit_rebuilds_sentences_from_the_last_verification_pass(
    client: TestClient, document: SeededDocument
) -> None:
    audit = _direct_audit(document, warnings=["verified_content_unit_fallback"])
    audit["verification"]["passes"] = [_legacy_pass(PUBLISHED)]
    seed_run(document, summary=PUBLISHED + "\n\nSources: S000001, S000002", audit=audit)

    summary = client.get("/api/v1/runs/run-1/summary").json()

    assert summary["publication"] == "content_unit_fallback"
    assert summary["removed_sentences"] == []
    assert [
        (item["text"], item["paragraph"], item["verdict"]) for item in summary["sentences"]
    ] == [
        ("The deck spans 200 feet.", 0, "supported"),
        ("Engineers inspected it in 1998.", 0, "not_meaningfully_verifiable"),
        ("Repairs finished in 2003.", 1, "supported"),
    ]
    last = summary["sentences"][2]["evidence"]
    assert [(item["segment_id"], item["quote_found"], item["page_start"]) for item in last] == [
        ("S000002", False, 2),
        ("S000001", False, 1),
    ]
    assert (last[0]["start"], last[0]["end"]) == _page_body(document, 2)


def test_runs_without_an_audit_still_show_their_text_and_sources(
    client: TestClient, document: SeededDocument
) -> None:
    run_dir = seed_run(document, summary=PUBLISHED + "\n\nSources: S000001")
    (run_dir / "verification.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")

    summary = client.get("/api/v1/runs/run-1/summary").json()

    assert summary["available"] is True
    assert summary["text"] == PUBLISHED
    assert summary["publication"] == "editorial"
    assert summary["verification_state"] == "completed"
    assert [sentence["verdict"] for sentence in summary["sentences"]] == ["unchecked"] * 3
    assert [sentence["paragraph"] for sentence in summary["sentences"]] == [0, 0, 1]
    assert [(item["segment_id"], item["start"]) for item in summary["citations"]] == [("S000001", None)]


def test_unfinished_runs_report_no_summary_and_why(client: TestClient, document: SeededDocument) -> None:
    seed_run(document, run_id="run-active", state="running")
    seed_run(document, run_id="run-stopped", state="stopped")
    failure = _direct_audit(document, warnings=["verified_content_unit_fallback_failed"], citations=[])
    failure["verification"].update(failed=True, failure_codes=["material_contradiction"])
    seed_run(document, run_id="run-failed", state="failed", audit=failure)

    active = client.get("/api/v1/runs/run-active/summary").json()
    stopped = client.get("/api/v1/runs/run-stopped/summary").json()
    failed = client.get("/api/v1/runs/run-failed/summary").json()

    assert (active["available"], active["verification_state"]) == (False, "in_progress")
    assert (stopped["available"], stopped["verification_state"], stopped["notices"]) == (
        False,
        "not_run",
        [],
    )
    assert (failed["available"], failed["verification_state"]) == (False, "failed")
    assert [(notice["code"], notice["severity"]) for notice in failed["notices"]] == [
        ("verified_content_unit_fallback_failed", "error"),
        ("material_contradiction", "error"),
    ]


def test_unknown_run_is_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/runs/missing/summary")

    assert response.status_code == 404
    assert response.json()["code"] == "run_not_found"
