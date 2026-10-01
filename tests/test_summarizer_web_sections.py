"""Section mode through the web worker: toggle, section nodes, audit records, notices (#170)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import summarizer_web.worker.runner as runner
from summarizer_web.config import load_paths
from summarizer_web.models.api import RunConfig
from summarizer_web.services import runs_service
from summarizer_web.services.events_service import start_attempt
from tests.support.web_runs import seed_document, seed_run
from tests.support.web_views import web_client
from tests.test_section_prose import DRAFTS, TEXT, CharacterCounter, SectionProvider, pages_of

NO_HEADINGS = "This document has no headings, so it was summarized without preserving sections."
CONFIG = {
    "strategy": "hierarchical",
    "chunk_tokens": 300,
    "max_merge_children": 2,
    "context_window": 100_000,
    "max_output_tokens": 128,
    "safety_margin_tokens": 0,
    "safety_margin_fraction": 0.0,
    "target_words": 300,
    "citations": False,
    "verify": True,
}


class WebSectionProvider(SectionProvider):
    def configure_context_window(self, model: str, requested: int | None, *, timeout_seconds: float) -> int:
        return requested or 100_000


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with web_client(tmp_path, monkeypatch) as test_client:
        monkeypatch.setattr(runner, "build_provider", lambda app: WebSectionProvider(DRAFTS))
        monkeypatch.setattr(runner, "build_counter", lambda app: CharacterCounter())
        yield test_client


def outline_of(text: str) -> list[dict[str, object]]:
    return [
        {
            "title": match.group(2),
            "level": len(match.group(1)),
            "start": match.start(2),
            "end": len(text),
            "page_start": None,
            "page_end": None,
        }
        for match in re.finditer(r"^(#+) (.+)$", text, re.M)
    ]


def complete(run_id: str) -> None:
    assert start_attempt(run_id) is not None
    assert runner.run_job(run_id) == "completed"


def seed(*, outline: list[dict[str, object]] | None, **config: object) -> str:
    pages = [{"page": p.page, "start": p.start, "end": p.end} for p in pages_of(TEXT)]
    document_id, revision_id = seed_document(TEXT, pages=pages, outline=outline)
    return seed_run(document_id, revision_id, {**CONFIG, **config})


def audit_of(run_id: str) -> dict:
    return json.loads((load_paths().runs / run_id / "audit.json").read_text(encoding="utf-8"))


def test_a_document_with_headings_yields_section_nodes_publications_and_audit_records(
    client: TestClient,
) -> None:
    run_id = seed(outline=outline_of(TEXT), preserve_sections=True)
    complete(run_id)

    nodes = client.get(f"/api/v1/runs/{run_id}/tree").json()["nodes"]
    by_id = {node["node_id"]: node for node in nodes}
    sections = {
        node["section"]["heading"]: node["section"] for node in nodes if node["section"]
    }
    assert set(sections) == {"Alpha", "Alpha one", "Beta", "Gamma"}
    alpha, child = sections["Alpha"], sections["Alpha one"]
    assert (alpha["level"], alpha["parent_section_id"]) == (1, None)
    assert (child["level"], child["parent_section_id"]) == (2, alpha["section_id"])
    assert (child["page_start"], child["page_end"]) == (1, 2)
    # Only the nodes that merge top-level sections belong to no section.
    assert all(node["section"] is None for node in nodes if node["kind"] == "merge" and node["level"] == 2)
    assert sum(1 for node in nodes if node["section"] and node["section"]["is_root"]) == 4
    detail = client.get(f"/api/v1/runs/{run_id}/nodes/L0N0002").json()
    assert detail["section"] == by_id["L0N0002"]["section"]

    audit = audit_of(run_id)
    records = {record["heading"]: record for record in audit["sections"]}
    assert records["Alpha one"]["parent_id"] == records["Alpha"]["section_id"]
    assert records["Alpha"]["child_ids"] == [records["Alpha one"]["section_id"]]
    assert records["Alpha"]["node_id"] in by_id
    assert records["Beta"]["publication"]["status"] == "verified"
    assert records["Gamma"]["publication"]["status"] == "empty"

    response = client.get(f"/api/v1/runs/{run_id}/sections").json()
    assert response["available"] is True and response["notices"] == []
    published = {item["heading"]: item for item in response["sections"]}
    beta = published["Beta"]
    assert (beta["status"], beta["text"]) == ("verified", "The heron did something.")
    assert [(item["text"], item["verdict"]) for item in beta["removed_sentences"]] == [
        ("The zebra did something else.", "insufficiently_supported")
    ]
    assert beta["sentences"][0]["evidence"][0]["segment_id"] in {c["segment_id"] for c in beta["citations"]}
    assert (beta["publication_page_start"], beta["publication_page_end"]) == (3, 4)
    assert published["Gamma"]["status"] == "empty" and published["Gamma"]["reason"]
    assert published["Alpha"]["child_section_ids"] == [published["Alpha one"]["section_id"]]
    # The run's final summary is unchanged by the mode.
    assert client.get(f"/api/v1/runs/{run_id}/summary").json()["available"] is True


def test_the_audit_holds_headings_and_prose_it_wrote_but_never_source_prose(
    client: TestClient,
) -> None:
    run_id = seed(outline=outline_of(TEXT), preserve_sections=True)
    complete(run_id)

    raw = (load_paths().runs / run_id / "audit.json").read_text(encoding="utf-8")
    assert '"heading":"Alpha one"' in raw
    for sentence in re.findall(r"The \w+ number \d+ did something\.", TEXT):
        assert sentence not in raw


@pytest.mark.parametrize(
    ("outline", "code", "reimport"),
    [([], "sections_no_outline", False), (None, "sections_outline_not_extracted", True)],
)
def test_a_document_without_an_outline_runs_without_sections_and_says_so(
    client: TestClient, outline: list[dict[str, object]] | None, code: str, reimport: bool
) -> None:
    run_id = seed(outline=outline, preserve_sections=True)
    complete(run_id)

    audit = audit_of(run_id)
    assert "sections" not in audit and code in audit["warnings"]
    nodes = client.get(f"/api/v1/runs/{run_id}/tree").json()["nodes"]
    assert all(node["section"] is None for node in nodes)
    summary = client.get(f"/api/v1/runs/{run_id}/summary").json()
    [notice] = [item for item in summary["notices"] if item["code"] == code]
    assert notice["message"].startswith(NO_HEADINGS)
    assert ("importing it again" in notice["message"]) is reimport
    sections = client.get(f"/api/v1/runs/{run_id}/sections").json()
    assert sections["available"] is False and sections["sections"] == []
    assert [item["code"] for item in sections["notices"]] == [code]


def test_mode_off_leaves_the_audit_and_notices_as_before(client: TestClient) -> None:
    run_id = seed(outline=outline_of(TEXT))
    complete(run_id)

    audit = audit_of(run_id)
    assert "sections" not in audit
    assert not any(code.startswith("sections_") for code in audit["warnings"])
    nodes = client.get(f"/api/v1/runs/{run_id}/tree").json()["nodes"]
    assert all(node["section"] is None for node in nodes)
    assert client.get(f"/api/v1/runs/{run_id}/sections").json()["available"] is False
    assert RunConfig().preserve_sections is False


def test_the_toggle_is_accepted_on_a_run_and_kept_in_settings(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class QueueOnly:
        def enqueue(self, _run_id: str) -> None:
            pass

    monkeypatch.setattr(runs_service, "get_supervisor", QueueOnly)
    document_id, _ = seed_document(TEXT, outline=outline_of(TEXT))
    headers = {"X-CSRF-Token": client.get("/api/v1/health").headers["x-csrf-token"]}

    created = client.post(
        "/api/v1/runs",
        json={"document_id": document_id, "config": {"model": "m", "preserve_sections": True}},
        headers=headers,
    )
    assert created.status_code == 201
    assert created.json()["config"]["preserve_sections"] is True
    assert client.get("/api/v1/settings").json()["defaults"]["preserve_sections"] is False
    patched = client.patch(
        "/api/v1/settings", json={"defaults": {"preserve_sections": True}}, headers=headers
    )
    assert patched.json()["defaults"]["preserve_sections"] is True
