"""Command line for the SPOC quad recall spike (issue #150).

  extract  source text -> quads JSON (needs a live provider)
  score    quads JSON + summary -> counts and recall (LLM judge, or --judge lexical offline)
  check    score a summary, then a degraded copy, and report whether the drop lands on the hit quads
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.quad_recall.degrade import drop_phrase, drop_sentences_with  # noqa: E402
from experiments.quad_recall.quads import extract_quads, quads_from_json, quads_to_json  # noqa: E402
from experiments.quad_recall.score import LexicalJudge, LLMJudge, degradation_check, score_summary  # noqa: E402
from summarizer.ingestion import ingest_text  # noqa: E402
from summarizer.segmentation import SegmentationConfig, segment_document  # noqa: E402


class _CharCounter:
    identity, exact, monotonic = "quad-recall:characters", True, True

    def count(self, text: str) -> int:
        return len(text)


def _provider(args: argparse.Namespace):
    if args.provider == "ollama":
        from summarizer.providers.ollama import OllamaProvider

        return OllamaProvider(args.ollama_host)
    from summarizer.providers.openai import OpenAIProvider

    return OpenAIProvider()


def _add_provider_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=("openai", "ollama"), default="openai")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--ollama-host", default="http://localhost:11434")


def _load_judge(args: argparse.Namespace):
    return LexicalJudge() if args.judge == "lexical" else LLMJudge(_provider(args), args.model)


def _score_dict(score) -> dict[str, object]:
    return {k: v for k, v in score.__dict__.items() if k != "verdicts"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--source", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--segment-chars", type=int, default=6000)
    _add_provider_args(e)

    s = sub.add_parser("score")
    s.add_argument("--quads", type=Path, required=True)
    s.add_argument("--summary", type=Path, required=True)
    s.add_argument("--judge", choices=("llm", "lexical"), default="llm")
    _add_provider_args(s)

    c = sub.add_parser("check")
    c.add_argument("--quads", type=Path, required=True)
    c.add_argument("--summary", type=Path, required=True)
    c.add_argument("--drop-phrase", help="qualifier phrase to remove from the summary")
    c.add_argument("--drop-sentences-with", help="entity whose sentences to remove from the summary")
    c.add_argument("--target-contains", required=True, help="text marking the quads the degradation hits")
    c.add_argument("--judge", choices=("llm", "lexical"), default="llm")
    _add_provider_args(c)

    args = parser.parse_args(argv)

    if args.command == "extract":
        document = ingest_text(args.source.read_text(encoding="utf-8"), path=args.source)
        segments = segment_document(document, _CharCounter(), SegmentationConfig(max_tokens=args.segment_chars))
        started = time.monotonic()
        quads, requests = extract_quads(_provider(args), args.model, [(s.segment_id, s.text) for s in segments])
        args.out.write_text(quads_to_json(quads), encoding="utf-8")
        print(json.dumps({"segments": len(segments), "requests": requests, "quads": len(quads),
                          "anchored": sum(q.anchored for q in quads), "seconds": round(time.monotonic() - started, 1)}))
        return 0

    quads = quads_from_json(args.quads.read_text(encoding="utf-8"))
    summary = args.summary.read_text(encoding="utf-8").strip()
    judge = _load_judge(args)

    if args.command == "score":
        print(json.dumps(_score_dict(score_summary(judge, summary, quads)), indent=2))
        return 0

    if bool(args.drop_phrase) == bool(args.drop_sentences_with):
        parser.error("give exactly one of --drop-phrase or --drop-sentences-with")
    degraded = drop_phrase(summary, args.drop_phrase) if args.drop_phrase else drop_sentences_with(summary, args.drop_sentences_with)
    base, worse = score_summary(judge, summary, quads), score_summary(judge, degraded, quads)
    targets = [q.quad_id for q in quads if args.target_contains.lower() in q.text().lower()]
    print(json.dumps({"base": _score_dict(base), "degraded": _score_dict(worse),
                      "check": degradation_check(base, worse, targets)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
