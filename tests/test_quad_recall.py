"""Offline tests for the SPOC quad recall spike (issue #150). No network or model."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.quad_recall.degrade import drop_phrase, drop_sentences_with
from experiments.quad_recall.quads import extract_quads, parse_quads, quads_from_json
from experiments.quad_recall.score import FULL, MISSED, PARTIAL, LexicalJudge, LLMJudge, degradation_check, score_summary
from summarizer.providers.base import GenerationRequest, GenerationResult

REFERENCE = Path(__file__).resolve().parents[1] / "experiments" / "quad_recall" / "reference"
SEGMENT = "The council approved $1.8 million on February 27. A grant will cover half, but the award is unsigned."


class ScriptedProvider:
    def __init__(self, *texts: str) -> None:
        self.texts, self.requests = list(texts), []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(text=self.texts.pop(0), provider="fake", model=request.model)


def _row(**overrides: str) -> dict[str, str]:
    row = {"subject": "council", "predicate": "approved", "object": "$1.8 million", "context": "on February 27",
           "evidence": "The council approved $1.8 million on February 27."}
    return {**row, **overrides}


def test_parse_drops_malformed_rows_and_flags_unanchored_evidence() -> None:
    text = json.dumps({"quads": [_row(), _row(evidence="made up"), _row(subject=""), {"subject": 3}, "x"]})
    quads = parse_quads(text, "S1", SEGMENT)
    assert [q.anchored for q in quads] == [True, False]
    assert parse_quads("not json", "S1", SEGMENT) == []
    assert parse_quads(json.dumps({"quads": "nope"}), "S1", SEGMENT) == []


def test_extract_sends_one_schema_request_per_segment() -> None:
    provider = ScriptedProvider(json.dumps({"quads": [_row()]}), json.dumps({"quads": [_row()]}))
    quads, requests = extract_quads(provider, "m", [("S1", SEGMENT), ("S2", SEGMENT)])
    assert requests == 2 and [q.quad_id for q in quads] == ["S1-Q001", "S2-Q002"]
    assert all(r.response_schema is not None and "untrusted" in r.instructions for r in provider.requests)


def test_llm_judge_counts_an_unanswered_fact_as_missed() -> None:
    quads = parse_quads(json.dumps({"quads": [_row(), _row(subject="grant", predicate="covers", object="half", context="")]}), "S1", SEGMENT)
    judge = LLMJudge(ScriptedProvider(json.dumps({"verdicts": [{"id": "S1-Q001", "status": FULL}]})), "m")
    score = score_summary(judge, "x", quads)
    assert score.verdicts == {"S1-Q001": FULL, "S1-Q002": MISSED}
    assert score.recall_full == 0.5


def test_reference_article_lexical_signal_lands_on_the_hit_quad_only() -> None:
    quads = quads_from_json((REFERENCE / "article.quads.json").read_text())
    summary = (REFERENCE / "article.summary.txt").read_text().strip()
    judge = LexicalJudge()
    base = score_summary(judge, summary, quads)
    assert base.counts[FULL] == len(quads)

    targets = [q.quad_id for q in quads if "not yet been signed" in q.text()]
    qualifier = degradation_check(base, score_summary(judge, drop_phrase(summary, "but the award is not yet signed"), quads), targets)
    assert qualifier["all_targets_caught"] and not qualifier["collateral"]

    ruiz = [q.quad_id for q in quads if "Ruiz" in q.text()]
    entity = score_summary(judge, drop_sentences_with(summary, "Ruiz"), quads)
    assert {i for i in ruiz if entity.verdicts[i] == MISSED} == set(ruiz)


def test_partial_is_reported_separately_from_missed() -> None:
    quads = quads_from_json((REFERENCE / "article.quads.json").read_text())
    summary = (REFERENCE / "article.summary.txt").read_text().strip()
    degraded = drop_phrase(summary, "but the award is not yet signed")
    score = score_summary(LexicalJudge(), degraded, quads)
    assert score.counts[PARTIAL] == 1 and score.recall_any == 1.0 and score.recall_full < 1.0


def test_llm_judge_ignores_ids_from_other_batches_and_non_list_payloads() -> None:
    rows = [_row(object=f"item {i}") for i in range(3)]
    quads = parse_quads(json.dumps({"quads": rows}), "S1", SEGMENT)
    stray = {"verdicts": [{"id": "S1-Q001", "status": MISSED}, {"id": "S1-Q002", "status": FULL}]}
    judge = LLMJudge(ScriptedProvider(json.dumps({"verdicts": [{"id": "S1-Q001", "status": FULL}]}), json.dumps(stray)), "m", batch_size=1)
    verdicts = judge.judge("x", quads[:2])
    # Batch 2 answered for Q001 (not its own id) and for Q002, so only Q002 is taken.
    assert verdicts == {"S1-Q001": FULL, "S1-Q002": FULL}
    for payload in ({"verdicts": None}, {"verdicts": 3}, {"verdicts": "x"}):
        assert LLMJudge(ScriptedProvider(json.dumps(payload)), "m").judge("x", quads[:1]) == {"S1-Q001": MISSED}


def test_an_empty_target_set_is_not_a_caught_degradation() -> None:
    quads = quads_from_json((REFERENCE / "article.quads.json").read_text())
    summary = (REFERENCE / "article.summary.txt").read_text().strip()
    base = score_summary(LexicalJudge(), summary, quads)
    assert degradation_check(base, base, [])["all_targets_caught"] is False


def test_drop_sentences_does_not_split_after_a_title() -> None:
    summary = "Lina found dust. Testing by Dr. Chen showed clay. It rained."
    assert drop_sentences_with(summary, "Dr. Chen") == "Lina found dust. It rained."
