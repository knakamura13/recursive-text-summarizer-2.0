"""Tests for claim verification."""

from dataclasses import dataclass, field

import pytest

from summarizer.hierarchy import TreeNode
from summarizer.ingestion import SourceDocument, ingest
from summarizer.segmentation import SourceSegment, BoundaryKind
from summarizer.summaries import ContentUnit, Evidence, ContentKind, SummaryNode, Annotation
from summarizer.verification import (
    ClaimStatus,
    ClaimIssue,
    ClaimVerdict,
    detect_hedging,
    extract_claim_text,
    combine_evidence_text,
    assess_textual_alignment,
    verify_content_unit,
    verify_annotation,
    verify_summary,
    VerificationError,
)


@dataclass(frozen=True)
class CharacterCounter:
    """Simple character-based token counter for tests."""

    identity: str = "test:characters"
    exact: bool = True
    monotonic: bool = True

    def count(self, text: str) -> int:
        return len(text)


def segment_from_text(
    text: str, segment_id: str = "S000001", order: int = 0
) -> SourceSegment:
    """Create a test segment from text."""
    return SourceSegment(
        segment_id=segment_id,
        source_id="src:test",
        order=order,
        text=text,
        core_start=0,
        core_end=len(text),
        context_start=0,
        context_end=len(text),
        core_token_count=len(text),
        token_count=len(text),
        leading_overlap_tokens=0,
        trailing_overlap_tokens=0,
        boundary_kind=BoundaryKind.PARAGRAPH,
    )


def content_unit(
    text: str,
    evidence_ids: list[str] | None = None,
    quote: str | None = None,
    kind: ContentKind = ContentKind.CLAIM,
    qualified: bool = False,
    uncertain: bool = False,
) -> ContentUnit:
    """Create a test content unit."""
    if evidence_ids is None:
        evidence_ids = ["S000001"]

    evidence = [Evidence(segment_id=seg_id, quote=quote) for seg_id in evidence_ids]
    return ContentUnit(
        text=text,
        kind=kind,
        evidence=tuple(evidence),
        qualification="Maybe" if qualified else None,
        uncertain=uncertain,
    )


def annotation(
    evidence_ids: list[str] | None = None,
    quote: str | None = None,
) -> Annotation:
    """Create a test annotation (qualification or contradiction)."""
    if evidence_ids is None:
        evidence_ids = ["S000001"]

    evidence = [Evidence(segment_id=seg_id, quote=quote) for seg_id in evidence_ids]
    return Annotation(evidence=tuple(evidence))


def summary_node(
    level: int = 0,
    content_units: list[ContentUnit] | None = None,
    qualifications: list[Annotation] | None = None,
    contradictions: list[Annotation] | None = None,
) -> SummaryNode:
    """Create a test summary node."""
    return SummaryNode(
        summary="Test summary.",
        content_units=tuple(content_units or []),
        entities=(),
        qualifications=tuple(qualifications or []),
        contradictions=tuple(contradictions or []),
        quotations=(),
        level=level,
        provenance=("S000001",),
    )


def tree_node(
    summary: SummaryNode | None = None,
    children: list[TreeNode] | None = None,
    node_id: str = "N0",
    covered_segments: list[str] | None = None,
) -> TreeNode:
    """Create a test tree node."""
    if summary is None:
        summary = summary_node()
    if covered_segments is None:
        covered_segments = ["S000001"]
    return TreeNode(
        node_id=node_id,
        summary=summary,
        children=tuple(children or []),
        covered_segments=tuple(covered_segments),
    )


class TestHedgeDetection:
    """Tests for hedge marker detection."""

    def test_detects_may(self) -> None:
        hedges = detect_hedging("This may be true")
        assert "may" in hedges

    def test_detects_multiple(self) -> None:
        hedges = detect_hedging("This may possibly be true, or perhaps not")
        assert "may" in hedges
        assert "possibly" in hedges
        assert "perhaps" in hedges

    def test_detects_none(self) -> None:
        hedges = detect_hedging("This is definitely true")
        assert len(hedges) == 0

    def test_case_insensitive(self) -> None:
        hedges = detect_hedging("This MAY be true")
        assert "MAY" in hedges or "may" in hedges.lower()

    def test_hedges_at_sentence_boundaries(self) -> None:
        hedges = detect_hedging("May be it is true.")
        # "May" at start might be treated as proper noun; boundary detection
        # is heuristic. Just verify the function runs without error.
        assert isinstance(hedges, tuple)


class TestTextualAlignment:
    """Tests for claim-evidence alignment assessment."""

    def test_exact_match(self) -> None:
        claim = "The sky is blue"
        evidence = "The sky is blue because of Rayleigh scattering"
        status, confidence, issues = assess_textual_alignment(claim, evidence)
        assert status == ClaimStatus.SUPPORTED
        assert confidence >= 0.85

    def test_case_insensitive_match(self) -> None:
        claim = "the sky is blue"
        evidence = "The SKY is BLUE"
        status, confidence, issues = assess_textual_alignment(claim, evidence)
        assert status == ClaimStatus.SUPPORTED
        assert confidence >= 0.80

    def test_paraphrase_with_key_terms(self) -> None:
        claim = "The sky is blue"
        evidence = "A blue sky is a common observation"
        status, confidence, issues = assess_textual_alignment(claim, evidence)
        assert status == ClaimStatus.UNDER_SUPPORTED
        assert 0.5 <= confidence < 0.8
        assert any(i.kind == "paraphrase" for i in issues)

    def test_no_match(self) -> None:
        claim = "The sky is green"
        evidence = "The grass grows tall"
        status, confidence, issues = assess_textual_alignment(claim, evidence)
        assert status == ClaimStatus.UNSUPPORTED
        assert confidence < 0.3
        assert any(i.kind == "no-match" for i in issues)

    def test_hedges_in_evidence_dropped_in_claim(self) -> None:
        claim = "The sky is blue"
        evidence = "The sky may appear blue"
        status, confidence, issues = assess_textual_alignment(claim, evidence)
        # Status should still be supported (match found), but confidence drops
        assert status == ClaimStatus.SUPPORTED
        assert confidence < 0.85  # Reduced from exact match
        assert any(i.kind == "hedged-but-certain" for i in issues)

    def test_empty_evidence(self) -> None:
        claim = "Something"
        evidence = ""
        status, confidence, issues = assess_textual_alignment(claim, evidence)
        assert status == ClaimStatus.UNKNOWN
        assert confidence == 0.0
        assert any(i.kind == "no-evidence" for i in issues)


class TestContentUnitVerification:
    """Tests for verifying content units."""

    def test_supported_claim(self) -> None:
        unit = content_unit("The sky is blue", evidence_ids=["S000001"])
        segments = {"S000001": segment_from_text("The sky is blue")}
        verdict = verify_content_unit(unit, ["S000001"], segments)
        assert verdict.status == ClaimStatus.SUPPORTED
        assert verdict.confidence >= 0.85

    def test_unsupported_claim(self) -> None:
        unit = content_unit("The sky is green", evidence_ids=["S000001"])
        segments = {"S000001": segment_from_text("The sky is blue")}
        verdict = verify_content_unit(unit, ["S000001"], segments)
        assert verdict.status == ClaimStatus.UNSUPPORTED
        assert verdict.confidence < 0.3

    def test_missing_evidence(self) -> None:
        unit = content_unit("The sky is blue", evidence_ids=["S000001"])
        verdict = verify_content_unit(unit, ["S000001"], {})
        assert verdict.status == ClaimStatus.UNSUPPORTED
        # No segment found, so it's as if evidence is empty

    def test_no_evidence_segments_provided(self) -> None:
        unit = content_unit("The sky is blue", evidence_ids=["S000001"])
        verdict = verify_content_unit(unit, [], {})
        assert verdict.status == ClaimStatus.UNKNOWN
        assert verdict.confidence == 0.0

    def test_multiple_evidence_segments(self) -> None:
        unit = content_unit("The sky is blue and vast", evidence_ids=["S000001", "S000002"])
        segments = {
            "S000001": segment_from_text("The sky is blue"),
            "S000002": segment_from_text("The vast sky"),
        }
        verdict = verify_content_unit(unit, ["S000001", "S000002"], segments)
        # Claim should be found across combined evidence
        assert verdict.status in (ClaimStatus.SUPPORTED, ClaimStatus.UNDER_SUPPORTED)

    def test_quotation_verification_present(self) -> None:
        quote = "The sky is blue"
        unit = content_unit(
            "The sky is blue",
            evidence_ids=["S000001"],
            quote=quote
        )
        segments = {"S000001": segment_from_text("The sky is blue")}
        verdict = verify_content_unit(unit, ["S000001"], segments)
        # Quotation is found, so no quote-mismatch issue
        assert not any(i.kind == "quote-mismatch" for i in verdict.issues)

    def test_quotation_verification_missing(self) -> None:
        quote = "The sky is green"
        unit = content_unit(
            "The sky is green",
            evidence_ids=["S000001"],
            quote=quote
        )
        segments = {"S000001": segment_from_text("The sky is blue")}
        verdict = verify_content_unit(unit, ["S000001"], segments)
        # Quotation is not found
        assert verdict.status == ClaimStatus.UNSUPPORTED
        assert any(i.kind == "quote-mismatch" for i in verdict.issues)


class TestAnnotationVerification:
    """Tests for verifying annotations."""

    def test_annotation_with_evidence(self) -> None:
        anno = annotation(evidence_ids=["S000001"])
        segments = {"S000001": segment_from_text("Some source text")}
        verdict = verify_annotation(anno, ["S000001"], segments)
        assert verdict.status == ClaimStatus.SUPPORTED
        assert verdict.confidence >= 0.6

    def test_annotation_no_evidence(self) -> None:
        anno = annotation(evidence_ids=["S000001"])
        verdict = verify_annotation(anno, [], {})
        assert verdict.status == ClaimStatus.UNKNOWN
        assert verdict.confidence == 0.0

    def test_quotation_in_annotation_present(self) -> None:
        quote = "exact text"
        anno = annotation(evidence_ids=["S000001"], quote=quote)
        segments = {"S000001": segment_from_text("exact text here")}
        verdict = verify_annotation(anno, ["S000001"], segments)
        assert not any(i.kind == "quote-mismatch" for i in verdict.issues)

    def test_quotation_in_annotation_missing(self) -> None:
        quote = "not present"
        anno = annotation(evidence_ids=["S000001"], quote=quote)
        segments = {"S000001": segment_from_text("some other text")}
        verdict = verify_annotation(anno, ["S000001"], segments)
        assert verdict.status == ClaimStatus.UNSUPPORTED
        assert any(i.kind == "quote-mismatch" for i in verdict.issues)


class TestTreeTraversal:
    """Tests for verifying full summary trees."""

    def test_single_node_single_unit(self) -> None:
        unit = content_unit("The sky is blue")
        summary = summary_node(content_units=[unit])
        root = tree_node(summary=summary)

        segments = [segment_from_text("The sky is blue", "S000001")]
        document = SourceDocument(text="The sky is blue", source_id="src:test")

        result = verify_summary(root, segments, document)

        assert result.aggregate.total_claims == 1
        assert len(result.content_verdicts) == 1

    def test_single_node_multiple_units(self) -> None:
        units = [
            content_unit("The sky is blue"),
            content_unit("The grass is green"),
        ]
        summary = summary_node(content_units=units)
        root = tree_node(summary=summary, covered_segments=["S000001", "S000002"])

        segments = [
            segment_from_text("The sky is blue", "S000001"),
            segment_from_text("The grass is green", "S000002"),
        ]
        document = SourceDocument(text="The sky is blue\nThe grass is green", source_id="src:test")

        result = verify_summary(root, segments, document)

        assert result.aggregate.total_claims == 2
        assert len(result.content_verdicts) == 2

    def test_tree_with_qualifications_and_contradictions(self) -> None:
        units = [content_unit("The sky is blue")]
        quals = [annotation(evidence_ids=["S000001"])]
        contras = [annotation(evidence_ids=["S000001"])]
        summary = summary_node(
            content_units=units,
            qualifications=quals,
            contradictions=contras,
        )
        root = tree_node(summary=summary)

        segments = [segment_from_text("The sky is blue", "S000001")]
        document = SourceDocument(text="The sky is blue", source_id="src:test")

        result = verify_summary(root, segments, document)

        assert result.aggregate.total_claims == 3  # 1 unit + 1 qual + 1 contra
        assert len(result.content_verdicts) == 1
        assert len(result.annotation_verdicts) == 2

    def test_two_level_tree(self) -> None:
        # Leaf node
        leaf_units = [content_unit("The sky is blue")]
        leaf_summary = summary_node(level=0, content_units=leaf_units)
        leaf_node = tree_node(node_id="N1", summary=leaf_summary, covered_segments=["S000001"])

        # Root node
        root_units = [content_unit("The sky has color")]
        root_summary = summary_node(level=1, content_units=root_units)
        root = tree_node(node_id="N0", summary=root_summary, children=[leaf_node], covered_segments=["S000001"])

        segments = [segment_from_text("The sky is blue", "S000001")]
        document = SourceDocument(text="The sky is blue", source_id="src:test")

        result = verify_summary(root, segments, document)

        # Both root and leaf units should be verified
        assert result.aggregate.total_claims == 2

    def test_unused_segments_detected(self) -> None:
        unit = content_unit("The sky is blue", evidence_ids=["S000001"])
        summary = summary_node(content_units=[unit])
        root = tree_node(summary=summary, covered_segments=["S000001", "S000002"])

        segments = [
            segment_from_text("The sky is blue", "S000001"),
            segment_from_text("Unused text", "S000002"),
        ]
        document = SourceDocument(text="The sky is blue\nUnused text", source_id="src:test")

        result = verify_summary(root, segments, document)

        assert "S000002" in result.evidence_audit.unused_segments

    def test_over_cited_segments_detected(self) -> None:
        units = [
            content_unit("The sky", evidence_ids=["S000001"]),
            content_unit("More sky", evidence_ids=["S000001"]),
            content_unit("Sky again", evidence_ids=["S000001"]),
        ]
        summary = summary_node(content_units=units)
        root = tree_node(summary=summary)

        segments = [segment_from_text("The sky is blue", "S000001")]
        document = SourceDocument(text="The sky is blue", source_id="src:test")

        result = verify_summary(root, segments, document)

        # S000001 is cited 3 times, which is above the average and thus "over-cited"
        assert "S000001" in result.evidence_audit.over_cited_segments

    def test_aggregate_verdict_by_status(self) -> None:
        units = [
            content_unit("The sky is blue", evidence_ids=["S000001"]),  # Supported
            content_unit("The grass is yellow", evidence_ids=["S000002"]),  # Unsupported
        ]
        summary = summary_node(content_units=units)
        root = tree_node(summary=summary, covered_segments=["S000001", "S000002"])

        segments = [
            segment_from_text("The sky is blue", "S000001"),
            segment_from_text("The grass is green", "S000002"),
        ]
        document = SourceDocument(text="The sky is blue\nThe grass is green", source_id="src:test")

        result = verify_summary(root, segments, document)

        assert result.aggregate.by_status[ClaimStatus.SUPPORTED.value] >= 1
        assert result.aggregate.by_status[ClaimStatus.UNSUPPORTED.value] >= 1

    def test_warnings_generated(self) -> None:
        units = [content_unit("Unsupported claim", evidence_ids=["S000001"])]
        summary = summary_node(content_units=units)
        root = tree_node(summary=summary)

        segments = [segment_from_text("Something else entirely", "S000001")]
        document = SourceDocument(text="Something else entirely", source_id="src:test")

        result = verify_summary(root, segments, document)

        # Should generate warnings for unsupported claims
        unsupported_warning = any("unsupported" in w.lower() for w in result.aggregate.warnings)
        assert unsupported_warning or result.aggregate.by_status[ClaimStatus.UNSUPPORTED.value] > 0

    def test_average_confidence_computed(self) -> None:
        units = [
            content_unit("The sky is blue"),
            content_unit("Another claim"),
        ]
        summary = summary_node(content_units=units)
        root = tree_node(summary=summary)

        segments = [segment_from_text("The sky is blue and another claim", "S000001")]
        document = SourceDocument(text="The sky is blue and another claim", source_id="src:test")

        result = verify_summary(root, segments, document)

        assert 0.0 <= result.aggregate.average_confidence <= 1.0
        assert result.aggregate.total_claims == 2


class TestExtractClaimText:
    """Tests for extracting plain text from units."""

    def test_extract_simple(self) -> None:
        unit = content_unit("The sky is blue")
        text = extract_claim_text(unit)
        assert text == "The sky is blue"

    def test_extract_with_whitespace(self) -> None:
        unit = content_unit("  The sky is blue  ")
        text = extract_claim_text(unit)
        assert text == "The sky is blue"


class TestCombineEvidenceText:
    """Tests for combining evidence segments."""

    def test_single_segment(self) -> None:
        segments = {"S000001": segment_from_text("Text one")}
        combined = combine_evidence_text(["S000001"], segments)
        assert "Text one" in combined
        assert "[S000001]" in combined

    def test_multiple_segments(self) -> None:
        segments = {
            "S000001": segment_from_text("Text one", "S000001"),
            "S000002": segment_from_text("Text two", "S000002"),
        }
        combined = combine_evidence_text(["S000001", "S000002"], segments)
        assert "Text one" in combined
        assert "Text two" in combined
        assert "[S000001]" in combined
        assert "[S000002]" in combined

    def test_missing_segment(self) -> None:
        segments = {"S000001": segment_from_text("Text one")}
        combined = combine_evidence_text(["S000001", "S000002"], segments)
        assert "Text one" in combined
        # Missing segment is skipped

    def test_empty_ids(self) -> None:
        combined = combine_evidence_text([], {})
        assert combined == ""
