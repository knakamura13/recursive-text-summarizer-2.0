"""The final summary of a section mode run: section prose under source headings (#178)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from summarizer.audit import AuditArtifact
from summarizer.compression import word_count
from summarizer.finalization import (
    HeadingRefusedError,
    assemble_sections,
    heading_line,
)
from summarizer.providers.base import GenerationRequest, GenerationResult
from summarizer.verification import VerificationConfig
from tests.test_section_prose import DRAFTS, TEXT, SectionProvider, body, by_heading, run

ALPHA = "The zebra did something."
EXPECTED = (
    f"# Alpha\n\n{ALPHA}\n\n"
    "## Alpha one\n\nThe otter did something.\n\n"
    "# Beta\n\nThe heron did something.\n\n"
    "# Gamma"
)


@dataclass(frozen=True)
class Heading:
    title: str
    level: int
    start: int


def audit_of(tmp_path: Path, **options: object) -> dict:
    path = tmp_path / "audit.json"
    run(audit_path=path, **options)
    return json.loads(path.read_text(encoding="utf-8"))


def test_sections_come_in_source_order_under_headings_of_their_level() -> None:
    result, _ = run()

    # Gamma's prose is empty (no sentence was supported), so its heading stands alone.
    assert result.final.text == EXPECTED


def test_the_untitled_opening_prose_comes_first_without_a_heading() -> None:
    text = body("zebra", 8) + TEXT
    result, _ = run(text, drafts={**DRAFTS, "": "The zebra opened."})

    assert result.final.text.startswith("The zebra opened.\n\n# Alpha\n\n")
    assert result.final.text.count("The zebra opened.") == 1


def test_a_heading_only_section_emits_its_heading_and_its_subsections_follow() -> None:
    text = "# Parent\n\n## Child\n\n" + body("zebra", 8) + "# Other\n\n" + body("heron", 8)
    result, provider = run(text, drafts={"Child": ALPHA, "Other": "The heron did something."})

    assert result.final.text == (
        f"# Parent\n\n## Child\n\n{ALPHA}\n\n# Other\n\nThe heron did something."
    )
    parent = result.section_publications[by_heading(result)["Parent"]]
    assert parent.status == "heading_only"
    assert not [r for r in provider.requests if f"Q{parent.section_id}-" in (r.operation_id or "")]


def test_an_empty_section_emits_its_heading_and_keeps_its_reason_in_the_audit(
    tmp_path: Path,
) -> None:
    audit = audit_of(tmp_path)
    publication = audit["publication"]
    gamma = next(item for item in audit["sections"] if item["heading"] == "Gamma")

    assert [item["text"] for item in publication["headings"]][-1] == "Gamma"
    assert gamma["publication"]["status"] == "empty" and gamma["publication"]["reason"]
    assert publication["headings"][-1]["end"] == len(EXPECTED)


def test_a_level_beyond_six_is_capped_at_six() -> None:
    assert heading_line(9, "Deep", {"Deep"}) == "###### Deep"
    assert heading_line(1, "Top", {"Top"}) == "# Top"


def test_a_heading_that_is_not_an_outline_heading_is_refused() -> None:
    result, _ = run()
    tree = result.sections
    assert tree is not None
    outline = [Heading(node.heading, node.level, node.start) for node in tree.nodes]
    publications = result.section_publications

    assert assemble_sections(tree, publications, outline).text == EXPECTED
    with pytest.raises(HeadingRefusedError, match="Beta"):
        assemble_sections(tree, publications, [item for item in outline if item.title != "Beta"])
    with pytest.raises(HeadingRefusedError, match="Not in the outline"):
        heading_line(1, "Not in the outline", {"Alpha"})


def test_heading_words_count_toward_no_sections_words() -> None:
    result, _ = run()
    publications = result.section_publications
    text = result.final.text
    prose_words = sum(item.words for item in publications.values())
    heading_words = sum(
        word_count(line.lstrip("#")) for line in text.splitlines() if line.startswith("#")
    )

    assert prose_words == 4 + 4 + 4 + 0 and heading_words == 1 + 2 + 1 + 1
    assert word_count(text) == prose_words + heading_words + 4  # the four heading marks
    for item in publications.values():
        assert item.words == word_count(item.text)


def test_sentence_verification_under_each_heading_is_unchanged() -> None:
    result, _ = run()
    ids = by_heading(result)

    beta = result.section_publications[ids["Beta"]]
    assert beta.text == "The heron did something."
    assert [(r.text, r.verdict) for r in beta.removed_sentences] == [
        ("The zebra did something else.", "insufficiently_supported")
    ]
    assert "The zebra did something else." not in result.final.text
    assert "The otter did something. The heron" not in result.final.text


def test_the_audit_places_headings_and_sentences_in_the_assembled_text(tmp_path: Path) -> None:
    audit = audit_of(tmp_path)
    publication = audit["publication"]
    text = EXPECTED

    assert [(h["level"], h["text"]) for h in publication["headings"]] == [
        (1, "Alpha"),
        (2, "Alpha one"),
        (1, "Beta"),
        (1, "Gamma"),
    ]
    for heading in publication["headings"]:
        assert text[heading["start"] : heading["end"]] == f"{'#' * heading['level']} {heading['text']}"
    assert [s["text"] for s in publication["sentences"]] == [
        ALPHA,
        "The otter did something.",
        "The heron did something.",
    ]
    assert [s["index"] for s in publication["sentences"]] == [0, 1, 2]
    for sentence in publication["sentences"]:
        assert text[sentence["start"] : sentence["end"]] == sentence["text"]
        assert sentence["paragraph"] == len(re.findall(r"\n\n", text[: sentence["start"]]))
        assert sentence["verdict"] == "supported" and sentence["evidence"]
    assert publication["kind"] == "verified_subset"
    AuditArtifact.model_validate(audit)


def test_the_audit_records_each_sections_requested_and_published_words(tmp_path: Path) -> None:
    audit = audit_of(tmp_path)
    words = {item["section_id"]: item for item in audit["publication"]["section_words"]}
    records = {item["section_id"]: item["publication"] for item in audit["sections"]}

    assert set(words) == set(records) == {"s1", "s2", "s3", "s4"}
    for section_id, item in words.items():
        assert item["requested"] == records[section_id]["target_words"] >= 1
        assert item["published"] == records[section_id]["words"]
    assert words["s4"]["published"] == 0 and words["s1"]["published"] == 4


def test_a_publication_whose_section_words_disagree_with_the_records_is_invalid(
    tmp_path: Path,
) -> None:
    audit = audit_of(tmp_path)
    audit["publication"]["section_words"][0]["published"] += 1

    with pytest.raises(ValueError):
        AuditArtifact.model_validate(audit)


def test_the_root_editorial_and_its_verification_are_neither_planned_nor_requested(
    tmp_path: Path,
) -> None:
    from summarizer.config import CacheConfig, ReliabilityConfig

    cache = CacheConfig(enabled=True, root=tmp_path / "cache")
    _, provider = run(cache=cache, reliability=ReliabilityConfig(run_id="assembled", run_mode="new"))
    work_ids = json.loads((tmp_path / "cache" / "runs" / "assembled.json").read_text())["work_ids"]

    assert not [r for r in provider.requests if r.operation_id == "editorial-final"]
    assert not [r for r in provider.requests if "The zebra met the otter" in r.input_text]
    assert "editorial-final" not in work_ids and "V01" not in work_ids


def test_citations_and_the_output_carry_the_assembled_text() -> None:
    result, _ = run(include_citations=True)

    assert result.final.text.startswith(EXPECTED)
    assert result.final.text.endswith(
        "Sources: " + ", ".join(c.segment_id for c in result.final.citations)
    )
    cited = {c.segment_id for c in result.final.citations}
    assert cited and cited <= {
        segment for item in result.section_publications.values() for segment in item.segment_ids
    }


def test_without_verification_the_unverified_prose_is_assembled() -> None:
    result, _ = run(verification=VerificationConfig())

    assert result.final.text == (
        "# Alpha\n\n" + DRAFTS["Alpha"] + "\n\n## Alpha one\n\n" + DRAFTS["Alpha one"]
        + "\n\n# Beta\n\n" + DRAFTS["Beta"] + "\n\n# Gamma\n\n" + DRAFTS["Gamma"]
    )


def test_a_run_with_no_section_prose_at_all_fails_with_a_failure_audit(tmp_path: Path) -> None:
    from summarizer.finalization import FinalizationVerificationError

    path = tmp_path / "audit.json"
    with pytest.raises(FinalizationVerificationError):
        run(drafts={heading: "The platypus did something." for heading in DRAFTS}, audit_path=path)
    assert "section_prose_empty" in json.loads(path.read_text())["failures"]


# -- the length rule -------------------------------------------------------------

LONG = " ".join(
    f"The unicorn {adverb} did something." for adverb in
    ("calmly", "boldly", "slowly", "gladly", "madly", "sadly", "badly", "oddly", "idly", "daily")
    * 2
)


class TrimProvider(SectionProvider):
    """Shortens a trim request by dropping its last sentence, unless told it cannot.

    `plan` says, for the first trim requests in turn, whether each shortens its
    text; later requests follow `shortens`.
    """

    def __init__(
        self,
        drafts: dict[str, str],
        *,
        shortens: bool = True,
        suffix: str = "",
        plan: tuple[bool, ...] = (),
    ) -> None:
        super().__init__(drafts)
        self.shortens = shortens
        self.suffix = suffix
        self.plan = list(plan)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        operation = request.operation_id or ""
        if "-TC" not in operation:
            return super().generate(request)
        self.requests.append(request)
        shortens = self.plan.pop(0) if self.plan else self.shortens
        chunk = request.input_text.splitlines()[1]
        sentences = re.split(r"(?<=\.) ", chunk)
        kept = sentences[: max(1, int(len(sentences) * 0.8))] if shortens else sentences
        text = " ".join(kept) + self.suffix
        return GenerationResult(json.dumps({"text": text}), "fake", request.model, 1, 1, "stop")


def trim_requests(provider: SectionProvider) -> list[str]:
    return [r.operation_id for r in provider.requests if "-TC" in (r.operation_id or "")]


def test_prose_above_its_target_is_trimmed_until_within_the_band_and_verified_again() -> None:
    provider = TrimProvider({**DRAFTS, "Gamma": LONG})
    result, _ = run(provider=provider)
    gamma = result.section_publications[by_heading(result)["Gamma"]]
    target = gamma.target_words

    assert word_count(LONG) > target * 1.1
    assert trim_requests(provider) and all(
        item.startswith("compression:Qs4-TC") for item in trim_requests(provider)
    )
    assert gamma.status == "verified" and gamma.words <= target * 1.1
    assert gamma.words == word_count(gamma.text) and LONG.startswith(gamma.text)
    assert gamma.text in result.final.text
    # The shorter prose was checked again: its first sentence went through
    # verification once with the draft and once more after the trim.
    checks = [
        r
        for r in provider.requests
        if (r.operation_id or "").startswith("verification-decompose")
        and "The unicorn calmly did something." in r.input_text
    ]
    assert len(checks) == 2


def test_prose_that_cannot_be_shortened_is_published_longer_after_three_rejected_passes() -> None:
    provider = TrimProvider({**DRAFTS, "Gamma": LONG}, shortens=False)
    result, _ = run(provider=provider)
    gamma = result.section_publications[by_heading(result)["Gamma"]]

    assert gamma.text == LONG
    assert gamma.words > gamma.target_words * 1.1
    # Three discarded passes in a row stop the loop, each under its own pass
    # index, and the longer verified prose stands.
    assert [item.split("-TC")[1] for item in trim_requests(provider)] == [
        "01K000001",
        "02K000001",
        "03K000001",
    ]
    assert gamma.removed_sentences == ()


def test_a_rejected_trim_pass_is_retried_and_the_shorter_prose_is_verified_again() -> None:
    provider = TrimProvider({**DRAFTS, "Gamma": LONG}, plan=(False, True))
    result, _ = run(provider=provider)
    gamma = result.section_publications[by_heading(result)["Gamma"]]

    assert trim_requests(provider)[:2] == [
        "compression:Qs4-TC01K000001",
        "compression:Qs4-TC02K000001",
    ]
    assert gamma.status == "verified" and gamma.words <= gamma.target_words * 1.1
    assert gamma.words < word_count(LONG) and LONG.startswith(gamma.text)
    checked = [
        r
        for r in provider.requests
        if (r.operation_id or "").startswith("verification-decompose")
        and "The unicorn calmly did something." in r.input_text
    ]
    assert len(checked) == 2  # the written draft, then the trimmed prose


def test_a_trim_that_loses_a_sentence_to_verification_leaves_the_verified_prose_unchanged() -> None:
    provider = TrimProvider({**DRAFTS, "Gamma": LONG}, suffix=" The zebra did something odd.")
    result, _ = run(provider=provider)
    gamma = result.section_publications[by_heading(result)["Gamma"]]

    assert trim_requests(provider)
    assert gamma.text == LONG and "zebra" not in result.final.text.split("# Gamma")[1]
    assert gamma.status == "verified" and gamma.kind == "editorial"
    assert gamma.removed_sentences == ()


def test_the_rejected_trim_stays_out_of_the_removed_sentences_the_first_check_recorded() -> None:
    draft = "The zebra did something. " + LONG
    baseline = TrimProvider({**DRAFTS, "Gamma": draft}, shortens=False)
    kept, _ = run(provider=baseline)
    provider = TrimProvider({**DRAFTS, "Gamma": draft}, suffix=" The zebra did something odd.")
    result, _ = run(provider=provider)
    gamma = result.section_publications[by_heading(result)["Gamma"]]
    first = kept.section_publications[by_heading(kept)["Gamma"]]

    assert [item.text for item in gamma.removed_sentences] == ["The zebra did something."]
    assert gamma.removed_sentences == first.removed_sentences
    assert gamma.text == first.text == LONG and gamma.kind == first.kind == "verified_subset"


def test_unverified_prose_above_its_target_is_trimmed_too() -> None:
    provider = TrimProvider({**DRAFTS, "Gamma": LONG})
    result, _ = run(provider=provider, verification=VerificationConfig())
    gamma = result.section_publications[by_heading(result)["Gamma"]]

    assert gamma.status == "unverified" and gamma.words <= gamma.target_words * 1.1
    assert not [r for r in provider.requests if (r.operation_id or "").startswith("verification")]


def test_prose_within_its_band_is_not_trimmed() -> None:
    provider = TrimProvider(DRAFTS)
    run(provider=provider)

    assert trim_requests(provider) == []

