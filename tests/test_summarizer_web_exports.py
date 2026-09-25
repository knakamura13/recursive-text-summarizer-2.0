"""GET /runs/{id}/export/{txt|md|json}: attachments, footnoted evidence, and availability."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from tests.support.web_views import (
    SeededDocument,
    audit_segment,
    paged_text,
    seed_document,
    seed_run,
    web_client,
)

PAGES = (
    "The bridge deck spans 200 feet across the river. Engineers inspected it in 1998.",
    "Cracks appeared in the north pier. Repairs began in 2001 and finished in 2003.",
)
PUBLISHED = "The deck spans 200 feet. It was inspected in 1998.\n\nThe pier cracked."
SUMMARY_FILE = PUBLISHED + "\n\nSources: S000001, S000002"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with web_client(tmp_path, monkeypatch) as test_client:
        yield test_client


def _document(title: str = "Bridge Report") -> SeededDocument:
    text, page_map = paged_text(PAGES)
    return seed_document(title=title, text=text, page_map=page_map)


def _audit(document: SeededDocument) -> dict:
    sha = document.source_sha256
    first_end = document.page_map[0]["end"] if document.page_map else document.offset("Cracks appeared")[0]
    deck = document.offset("spans 200 feet across the river")
    pier = document.offset("Cracks appeared in the north pier.")
    deck_evidence = {"segment_id": "S000001", "quote": "spans 200 feet across the river", "start": deck[0], "end": deck[1]}
    return {
        "schema_version": "audit/2",
        "source_segments": [
            audit_segment("S000001", 0, 0, first_end, sha),
            audit_segment("S000002", 1, first_end, len(document.text), sha),
        ],
        "citations": [
            {"segment_id": "S000001", "source_id": sha, "order": 0},
            {"segment_id": "S000002", "source_id": sha, "order": 1},
        ],
        "warnings": [],
        "verification": {"enabled": True, "failed": False, "passes": []},
        "publication": {
            "kind": "editorial",
            "sentences": [
                {"index": 0, "paragraph": 0, "text": "The deck spans 200 feet.", "verdict": "supported", "evidence": [deck_evidence]},
                {
                    "index": 1,
                    "paragraph": 0,
                    "text": "It was inspected in 1998.",
                    "verdict": "supported",
                    # The same quote again, and one without a quote.
                    "evidence": [deck_evidence, {"segment_id": "S000001", "quote": None, "start": None, "end": None}],
                },
                {
                    "index": 2,
                    "paragraph": 1,
                    "text": "The pier cracked.",
                    "verdict": "supported",
                    "evidence": [{"segment_id": "S000002", "quote": "Cracks appeared in the north pier.", "start": pier[0], "end": pier[1]}],
                },
            ],
            "removed_sentences": [],
        },
    }


def _filename(response) -> str:
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    return unquote(disposition.split("filename*=UTF-8''", 1)[1])


def test_txt_export_is_the_published_file(client: TestClient) -> None:
    document = _document()
    seed_run(document, summary=SUMMARY_FILE, audit=_audit(document))

    response = client.get("/api/v1/runs/run-1/export/txt")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == SUMMARY_FILE
    assert _filename(response) == "Bridge Report-summary.txt"


def test_md_export_footnotes_each_distinct_evidence_with_pages(client: TestClient) -> None:
    document = _document()
    seed_run(document, summary=SUMMARY_FILE, audit=_audit(document))

    response = client.get("/api/v1/runs/run-1/export/md")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert _filename(response) == "Bridge Report-summary.md"
    lines = response.text.splitlines()
    assert lines[0] == "# Bridge Report"
    paragraphs = [line for line in lines[1:] if line and not line.startswith("[^")]
    assert paragraphs == [
        "The deck spans 200 feet.[^1] It was inspected in 1998.[^1][^2]",
        "The pier cracked.[^3]",
    ]
    footnotes = [line for line in lines if line.startswith("[^")]
    assert footnotes == [
        "[^1]: p. 1: “spans 200 feet across the river”",
        "[^2]: p. 1",
        "[^3]: p. 2: “Cracks appeared in the north pier.”",
    ]


def test_md_export_without_sentence_evidence_lists_the_sources(client: TestClient) -> None:
    document = _document()
    audit = _audit(document)
    for sentence in audit["publication"]["sentences"]:
        sentence["evidence"] = []
    seed_run(document, summary=SUMMARY_FILE, audit=audit)

    text = client.get("/api/v1/runs/run-1/export/md").text

    assert "[^" not in text
    assert text.rstrip().endswith("Sources:\n\n- p. 1\n- p. 2")


def test_json_export_is_the_audit(client: TestClient) -> None:
    document = _document()
    audit = _audit(document)
    seed_run(document, summary=SUMMARY_FILE, audit=audit)

    response = client.get("/api/v1/runs/run-1/export/json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == audit
    assert _filename(response) == "Bridge Report-summary.json"


def test_filenames_keep_unicode_titles_and_drop_path_characters(client: TestClient) -> None:
    document = _document(title='Brücke: "Nord/Süd" Überblick')
    seed_run(document, summary=SUMMARY_FILE)

    response = client.get("/api/v1/runs/run-1/export/txt")

    assert _filename(response) == "Brücke Nord Süd Überblick-summary.txt"
    fallback = response.headers["content-disposition"].split('filename="', 1)[1].split('"', 1)[0]
    assert fallback.isascii() and "/" not in fallback


def test_exports_are_unavailable_until_the_run_publishes(client: TestClient) -> None:
    document = _document()
    seed_run(document, run_id="run-active", state="running")
    seed_run(document, run_id="run-failed", state="failed")

    for run_id in ("run-active", "run-failed"):
        for export_format in ("txt", "md", "json"):
            response = client.get(f"/api/v1/runs/{run_id}/export/{export_format}")
            assert (response.status_code, response.json()["code"]) == (404, "export_unavailable")

    missing = client.get("/api/v1/runs/missing/export/txt")
    assert (missing.status_code, missing.json()["code"]) == (404, "run_not_found")
    unsupported = client.get("/api/v1/runs/run-active/export/pdf")
    assert (unsupported.status_code, unsupported.json()["code"]) == (422, "invalid_request")


def test_failed_run_audit_can_still_be_exported(client: TestClient) -> None:
    document = _document()
    failure = {"schema_version": "audit/2", "warnings": ["verified_content_unit_fallback_failed"]}
    seed_run(document, state="failed", audit=failure)

    response = client.get("/api/v1/runs/run-1/export/json")
    summary = client.get("/api/v1/runs/run-1/export/txt")

    assert response.status_code == 200
    assert json.loads(response.content) == failure
    assert summary.status_code == 404



def test_md_export_names_passages_readably_when_the_document_has_no_pages(client: TestClient) -> None:
    text = " ".join(PAGES)
    document = seed_document(title="Bridge Report", text=text, page_map=None)
    seed_run(document, summary=SUMMARY_FILE, audit=_audit(document))

    response = client.get("/api/v1/runs/run-1/export/md")

    footnotes = [line for line in response.text.splitlines() if line.startswith("[^")]
    assert footnotes[0] == "[^1]: Passage 1: “spans 200 feet across the river”"
    assert "S000001" not in response.text
