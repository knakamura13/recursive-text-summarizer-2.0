"""Section records in the audit: structure, redaction and validation (#170)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from summarizer.audit import AuditArtifact, AuditError, with_sections
from summarizer.sections import SectionOutline
from summarizer.verification import VerificationConfig
from tests.test_section_prose import pages_of, run


@dataclass(frozen=True)
class Heading:
    title: str
    level: int
    start: int


def audit_of(tmp_path: Path, **options: object) -> dict:
    path = tmp_path / "audit.json"
    run(audit_path=path, **options)
    return json.loads(path.read_text(encoding="utf-8"))


def test_section_mode_audit_records_each_section_with_its_publication(tmp_path: Path) -> None:
    audit = audit_of(tmp_path)
    records = {record["heading"]: record for record in audit["sections"]}

    assert [record["section_id"] for record in audit["sections"]] == ["s1", "s2", "s3", "s4"]
    alpha, child = records["Alpha"], records["Alpha one"]
    assert (alpha["level"], alpha["parent_id"], alpha["child_ids"]) == (1, None, ["s2"])
    assert (child["level"], child["parent_id"], child["page_start"], child["page_end"]) == (2, "s1", 1, 2)
    node_ids = {node["node_id"] for node in audit["tree_nodes"]}
    assert {record["node_id"] for record in audit["sections"]} <= node_ids
    # A parent's own segments exclude its subsection's; its prose covers both.
    assert set(child["segment_ids"]).isdisjoint(alpha["segment_ids"])
    assert set(alpha["publication"]["segment_ids"]) == set(alpha["segment_ids"]) | set(child["segment_ids"])
    beta = records["Beta"]["publication"]
    assert beta["status"] == "verified" and beta["words"] == 4
    assert [(item["text"], item["verdict"]) for item in beta["removed_sentences"]] == [
        ("The zebra did something else.", "insufficiently_supported")
    ]
    assert all(item["evidence"] for item in beta["sentences"])
    assert {item["segment_id"] for item in beta["citations"]} <= set(beta["segment_ids"])
    empty = records["Gamma"]["publication"]
    assert empty["status"] == "empty" and empty["reason"] and empty["sentences"] == []
    # The run's own publication stays the root editorial.
    assert audit["publication"]["sentences"]


def test_section_prose_without_verification_is_recorded_unverified(tmp_path: Path) -> None:
    audit = audit_of(tmp_path, verification=VerificationConfig(enabled=False))

    statuses = {record["publication"]["status"] for record in audit["sections"]}
    assert statuses == {"unverified"}
    assert all(
        sentence["verdict"] == "unchecked"
        for record in audit["sections"]
        for sentence in record["publication"]["sentences"]
    )


def test_a_run_without_section_mode_writes_no_section_key(tmp_path: Path) -> None:
    audit = audit_of(tmp_path, sections=False)

    assert "sections" not in audit


def test_headings_are_kept_as_stored_but_credentials_in_them_are_redacted(tmp_path: Path) -> None:
    text = "Release notes\n\n" + "The zebra grazed. " * 12 + "\n\nAPI token: hunter2secret\n\n" + "The otter swam. " * 12
    headings = (Heading("Release notes", 1, 0), Heading("API token: hunter2secret", 1, text.index("API")))
    audit = audit_of(tmp_path, text=text, outline=SectionOutline(headings, pages_of(text)))

    stored = [record["heading"] for record in audit["sections"]]
    assert stored == ["Release notes", "API [REDACTED]"]
    assert "hunter2secret" not in json.dumps(audit)


def test_section_records_must_resolve_to_the_audit(tmp_path: Path) -> None:
    audit = audit_of(tmp_path)
    artifact = AuditArtifact.model_validate(audit)
    records = list(artifact.sections or ())
    assert records

    broken = records[0].model_copy(update={"node_id": "L9N9999"})
    with pytest.raises(AuditError):
        with_sections(artifact, [broken, *records[1:]])
    orphan = records[1].model_copy(update={"parent_id": "s4"})
    with pytest.raises(AuditError):
        with_sections(artifact, [records[0], orphan, *records[2:]])
