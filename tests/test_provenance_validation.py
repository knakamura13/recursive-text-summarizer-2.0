import json

import pytest

from summarizer.leaf import LeafSummaryError, validate_provenance
from summarizer.summaries import SummaryNode

LEGAL = {
    "S000001": "The archive moved in March.",
    "S000002": "The index was rebuilt afterwards.",
    "S000003": "Staffing was unchanged.",
}


def node(**overrides: object) -> SummaryNode:
    body: dict[str, object] = {
        "summary": "The archive moved and the index was rebuilt.",
        "content_units": [],
        "entities": [],
        "qualifications": [],
        "contradictions": [],
        "quotations": [],
        "provenance": ["S000001", "S000002"],
        "level": 1,
    }
    body.update(overrides)
    return SummaryNode.model_validate(json.loads(json.dumps(body)))


def test_accepts_a_node_citing_a_subset_of_the_legal_set() -> None:
    validate_provenance(node(), legal=LEGAL, subject="L1N001")


def test_rejects_an_identifier_outside_the_legal_set() -> None:
    with pytest.raises(LeafSummaryError, match="S999999"):
        validate_provenance(
            node(provenance=["S000001", "S999999"]), legal=LEGAL, subject="L1N001"
        )


def test_rejects_a_node_citing_nothing() -> None:
    """Otherwise a merged node satisfies resolvable evidence vacuously."""
    with pytest.raises(LeafSummaryError, match="provenance"):
        validate_provenance(node(provenance=[]), legal=LEGAL, subject="L1N001")


def test_accepts_a_quotation_from_the_segment_it_cites() -> None:
    validate_provenance(
        node(
            quotations=[
                {"segment_id": "S000003", "quote": "Staffing was unchanged."}
            ],
            provenance=["S000003"],
        ),
        legal=LEGAL,
        subject="L1N001",
    )


def test_rejects_a_quotation_belonging_to_a_different_legal_segment() -> None:
    """A concatenated check would wrongly accept this.

    The quote does occur in the legal material, but not in the segment the
    evidence cites, which is the cross-attribution hole the overlap work
    closed for leaves.
    """
    with pytest.raises(LeafSummaryError, match="quotation"):
        validate_provenance(
            node(
                quotations=[
                    {"segment_id": "S000001", "quote": "Staffing was unchanged."}
                ]
            ),
            legal=LEGAL,
            subject="L1N001",
        )


def test_rejects_a_quotation_straddling_two_segments() -> None:
    straddling = "The archive moved in March.The index was rebuilt afterwards."

    with pytest.raises(LeafSummaryError, match="quotation"):
        validate_provenance(
            node(quotations=[{"segment_id": "S000001", "quote": straddling}]),
            legal=LEGAL,
            subject="L1N001",
        )


def test_checks_evidence_on_content_units_too() -> None:
    with pytest.raises(LeafSummaryError, match="S999999"):
        validate_provenance(
            node(
                content_units=[
                    {
                        "text": "The archive moved.",
                        "kind": "fact",
                        "evidence": [{"segment_id": "S999999", "quote": None}],
                        "qualification": None,
                        "uncertain": False,
                    }
                ]
            ),
            legal=LEGAL,
            subject="L1N001",
        )


def test_rejects_an_evidence_free_content_unit() -> None:
    with pytest.raises(LeafSummaryError, match="content unit"):
        validate_provenance(
            node(
                content_units=[
                    {
                        "text": "The archive moved.",
                        "kind": "fact",
                        "evidence": [],
                        "qualification": None,
                        "uncertain": False,
                    }
                ]
            ),
            legal=LEGAL,
            subject="L1N001",
        )


def test_checks_evidence_on_grounded_annotations_too() -> None:
    with pytest.raises(LeafSummaryError, match="S999999"):
        validate_provenance(
            node(
                contradictions=[
                    {
                        "text": "Another source disagrees.",
                        "evidence": [{"segment_id": "S999999", "quote": None}],
                    }
                ]
            ),
            legal=LEGAL,
            subject="L1N001",
        )


@pytest.mark.parametrize("field", ("qualifications", "contradictions"))
def test_rejects_an_annotation_quote_absent_from_its_cited_source(field: str) -> None:
    with pytest.raises(LeafSummaryError, match="quotation"):
        validate_provenance(
            node(
                **{
                    field: [
                        {
                            "text": "The record is qualified.",
                            "evidence": [
                                {"segment_id": "S000001", "quote": "invented"}
                            ],
                        }
                    ]
                }
            ),
            legal=LEGAL,
            subject="L1N001",
        )


def test_failures_name_the_subject_and_bound_payload_text() -> None:
    hostile = "S9\nINJECTED: " + "x" * 400

    with pytest.raises(LeafSummaryError) as error:
        validate_provenance(
            node(provenance=[hostile]), legal=LEGAL, subject="L1N001"
        )

    message = str(error.value)
    assert message.startswith("L1N001:")
    assert "\n" not in message
    assert len(message) < 120
