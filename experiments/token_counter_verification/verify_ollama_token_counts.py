"""Check a tokenizer.json counter against Ollama's own prompt token counts (#190).

Not run by the test suite against a real server, and never run by the
summarizer. It sends a small fixed set of short chat requests to an Ollama
server on this machine, each limited to one generated token, and compares what
the counter says each prompt costs with the ``prompt_eval_count`` Ollama
returns, the method of #120: counted system and user tokens plus a fixed chat
template. This is real model inference, so run it deliberately. Requests have
the shape of the summarizer's own, including ``think: false``, because the
thinking marker changes the template and so the overhead.

The counter must never count fewer tokens than Ollama processes. Probes come in
three kinds:

- ``clean``: no whitespace at either edge of either message. They set the
  template overhead, which is observed, not assumed: every clean probe must
  share one value, and the run fails if they disagree. They include internal
  runs of spaces, tabs, newlines and indentation, and two probes that carry a
  format schema, which also tests that the schema is not part of the prompt
  count.
- ``boundary``: one run of one kind of whitespace at one edge of one message,
  each alone. A server may trim it, so Ollama may count fewer than the counter,
  never more.
- ``marker``: special-token markers in text. The counter splits a marker into
  pieces, which can only count more, so Ollama may count fewer, never more.

``--set long`` swaps the short probes for a few long requests of a few thousand
tokens, the size of the summarizer's own leaf segments (a quarter of the usable
input capacity, about 5,000 to 6,000 tokens at a 32,768-token window). Three are
the app's real leaf requests, built with ``build_leaf_request`` from passages of
committed public-domain novels and sent with the Ollama wire schema, so the large
schema is tested against the prompt count too. The same three kinds apply.

A pass prints the counting fingerprint to add to
``_VERIFIED_TOKENIZER_JSON_FINGERPRINTS`` in ``summarizer/tokenization.py``. The
fixed set is a smoke check on a few dozen tokens each, so say in the commit that
adds a fingerprint what else was compared.

Counts are read from the model's own tokenizer files on this machine, so the
host must be this machine. Output carries no document text, only the fixed
probes' names, counts and the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from summarizer.leaf import build_leaf_request  # noqa: E402
from summarizer.providers.ollama import _local_response_schema  # noqa: E402
from summarizer.segmentation import BoundaryKind, SourceSegment  # noqa: E402
from summarizer.tokenization import (  # noqa: E402
    TokenAccountingError,
    TokenCounter,
    TokenizerJsonCounter,
)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_DOCUMENTS = _REPO_ROOT / "sample-documents"
# The chat template overhead #120 measured for gemma4 with think false. It is a
# reference to compare the observed value with, not an input to the verdict.
REFERENCE_OVERHEAD = 14
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_MINIMUM_CLEAN_PROBES = 3
EXIT_PASSED, EXIT_FAILED, EXIT_INCOMPLETE = 0, 1, 2

_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "keywords": {"type": "array", "items": {"type": "string"}}},
    "required": ["summary", "keywords"],
}


@dataclass(frozen=True)
class Probe:
    """One fixed chat request. `kind` is "clean", "boundary" or "marker"."""

    name: str
    kind: str
    system: str
    user: str
    schema: dict[str, Any] | None = None


def _long_ascii() -> str:
    return " ".join(
        f"Section {index} reports that the harbor moved {index * 7} tonnes of cargo in {1990 + index}."
        for index in range(1, 21)
    )


def _long_non_ascii() -> str:
    return " ".join(
        f"Café número {index}: «naïve» façade, 東京タワー ({index}), 한국어 문장, Ελληνικά — 🙂."
        for index in range(1, 13)
    )


# The text every boundary probe wraps in one run of whitespace.
BASE_USER = "Count these words."


def probes() -> list[Probe]:
    """The fixed set. Each system message opens with a different word, so
    consecutive prompts share only the chat template header and a reused cache
    prefix cannot differ from request to request."""
    return [
        Probe("ascii-short", "clean", "Alpha: you summarize text.", "The harbor moved 24 tonnes of cargo in 1998."),
        Probe("ascii-tiny", "clean", "Bravo: answer briefly.", "Hi."),
        Probe("digits-punctuation", "clean", "Charlie: you check numbers.", "3.14159; 2,718,281; 1e-9; 0xFF; 2026-10-19; $1,234.56 (50%)!"),
        Probe("unicode", "clean", "Delta: you read many scripts.", "Café naïve résumé — “quotes” … 東京タワー 한국어 العربية Ελληνικά 🙂🚀"),
        Probe("whitespace-internal", "clean", "Echo: you keep layout.", "a  b   c\t\td\n\n\ne        f \n g"),
        Probe("code-internal", "clean", "Foxtrot: you read code.", "def f(x):\n    return {'a': [1, 2, 3], \"b\": x**2}  # comment"),
        Probe("markup", "clean", "Golf: you read markup.", "<table><tr><td>cell</td></tr></table> <b>bold</b> &amp; [link](http://example.test/a?b=c)"),
        Probe("ascii-long", "clean", "Hotel: you summarize reports.", _long_ascii()),
        Probe("non-ascii-long", "clean", "India: you summarize reports.", _long_non_ascii()),
        Probe("schema-ascii", "clean", "Juliet: reply as JSON.", "The harbor moved 24 tonnes of cargo in 1998.", _SCHEMA),
        Probe("schema-unicode", "clean", "Kilo: reply as JSON.", "Café naïve résumé — 東京タワー 🙂", _SCHEMA),
        Probe("trailing-newline", "boundary", "Lima: answer briefly.", BASE_USER + "\n"),
        Probe("trailing-space", "boundary", "Mike: answer briefly.", BASE_USER + " "),
        Probe("trailing-tab", "boundary", "November: answer briefly.", BASE_USER + "\t"),
        Probe("leading-newline", "boundary", "Oscar: answer briefly.", "\n" + BASE_USER),
        # Two spaces: a single leading space merges into the first word ("▁Count" is one token),
        # so trimming it would change no count and the probe could not tell a trim from none.
        Probe("leading-space", "boundary", "Papa: answer briefly.", "  " + BASE_USER),
        Probe("leading-tab", "boundary", "Quebec: answer briefly.", "\t" + BASE_USER),
        Probe("system-trailing-newline", "boundary", "Romeo: answer briefly.\n", BASE_USER),
        Probe(
            "special-markers",
            "marker",
            "Sierra: you quote text.",
            "Quote this literally: <bos> <eos> <pad> <start_of_turn>user <end_of_turn> <|turn> <|video|>",
        ),
    ]


class MissingSampleDocument(Exception):
    """A committed sample document the long set needs could not be read."""


# Public-domain novels committed under sample-documents/, and where in each to start.
_NOVELS = {
    "alice": ("novel_Alice's Adventures in Wonderland.txt", 0.15),
    "frankenstein": ("novel_Frankenstein - or, the Modern Prometheus.txt", 0.30),
    "moby": ("novel_Moby Dick - Or, The Whale.txt", 0.40),
}
_LONG_NON_ASCII_SENTENCE = "Café número {i}: «naïve» façade, 東京タワー ({i}), 한국어 문장, Ελληνικά — 🙂."
_BLANK_LINES = "\n\n"


def _passage(
    documents: Path, novel: str, counter: TokenCounter, target_tokens: int, *, fraction: float | None = None
) -> str:
    """A whole-sentence passage of at most `target_tokens`, starting at a paragraph."""
    filename, default_fraction = _NOVELS[novel]
    try:
        text = (documents / filename).read_text(encoding="utf-8")
    except OSError as error:
        raise MissingSampleDocument(f"sample document {filename!r} cannot be read in {documents}") from error
    text = text.replace("\r\n", "\n")
    start = text.find("\n\n", int(len(text) * (default_fraction if fraction is None else fraction)))
    window = text[max(start, 0) + 2 :][: target_tokens * 12]
    piece = window[: counter.fitting_prefix(window, 0, len(window), target_tokens)]
    sentences = list(re.finditer(r"[.!?][\"”’)]*(?=\s)", piece))
    return (piece[: sentences[-1].end()] if sentences else piece).strip()


def _non_ascii_passage(counter: TokenCounter, target_tokens: int) -> str:
    sentences: list[str] = []
    while True:
        sentences.append(_LONG_NON_ASCII_SENTENCE.format(i=len(sentences) + 1))
        if counter.count(" ".join(sentences)) > target_tokens:
            sentences.pop()
            return " ".join(sentences)


def _leaf_probe(name: str, segment_id: str, passage: str, counter: TokenCounter, model: str) -> Probe:
    """The request the summarizer sends for one leaf segment, with the schema Ollama is sent."""
    segment = SourceSegment(
        segment_id=segment_id,
        source_id=hashlib.sha256(passage.encode("utf-8")).hexdigest(),
        order=0,
        text=passage,
        core_start=0,
        core_end=len(passage),
        context_start=0,
        context_end=len(passage),
        core_token_count=counter.count(passage),
        token_count=counter.count(passage),
        leading_overlap_tokens=0,
        trailing_overlap_tokens=0,
        boundary_kind=BoundaryKind.PARAGRAPH,
    )
    request = build_leaf_request(segment, model=model, timeout_seconds=1)
    return Probe(name, "clean", request.instructions, request.input_text, _local_response_schema(request))


def long_probes(counter: TokenCounter, model: str, documents: Path = DEFAULT_DOCUMENTS) -> list[Probe]:
    """The long set. No two leaf probes are adjacent, so consecutive requests never
    share the leaf instructions as a cached prefix."""
    marker_passage = _passage(documents, "alice", counter, 2_000, fraction=0.6)
    paragraphs = marker_passage.split("\n\n")
    for position, marker in zip((1, 3, 5, 7), ("<bos>", "<|turn>", "<eos>", "<|video|>"), strict=False):
        if position < len(paragraphs):
            paragraphs.insert(position, marker)
    edge_passage = _passage(documents, "alice", counter, 2_000, fraction=0.5)
    return [
        _leaf_probe("long-leaf-2k", "S000001", _passage(documents, "alice", counter, 2_000), counter, model),
        Probe("long-plain-4k", "clean", "Tango: you summarize passages.", _passage(documents, "moby", counter, 4_000, fraction=0.7)),
        _leaf_probe("long-leaf-4k", "S000002", _passage(documents, "frankenstein", counter, 4_000), counter, model),
        Probe("long-non-ascii-3k", "clean", "Uniform: you read many scripts.", _non_ascii_passage(counter, 3_000)),
        _leaf_probe("long-leaf-6k", "S000003", _passage(documents, "moby", counter, 6_000), counter, model),
        Probe("long-trailing-blank-lines", "boundary", "Victor: answer briefly.", edge_passage + _BLANK_LINES),
        Probe("long-leading-blank-lines", "boundary", "Whiskey: answer briefly.", _BLANK_LINES + edge_passage),
        Probe("long-markers", "marker", "Xray: you quote text.", "\n\n".join(paragraphs)),
    ]


WARM_UP = Probe("warm-up", "clean", "Zulu: warm up.", "Ready.")


@dataclass(frozen=True)
class Measurement:
    name: str
    kind: str
    schema: bool
    counted: int
    reported: int | None

    @property
    def overhead(self) -> int | None:
        return None if self.reported is None else self.reported - self.counted


@dataclass(frozen=True)
class Verdict:
    passed: bool
    overhead: int | None
    problems: list[str]
    notes: list[str]


def counted_tokens(counter: TokenCounter, probe: Probe) -> int:
    """What the counter says the prompt costs, before the chat template."""
    return counter.count(probe.system) + counter.count(probe.user)


def request_body(model: str, probe: Probe) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        # OllamaProvider sends think false on every request; leaving it out changes the template.
        "think": False,
        "messages": [
            {"role": "system", "content": probe.system},
            {"role": "user", "content": probe.user},
        ],
        "options": {"num_predict": 1, "temperature": 0, "seed": 0},
    }
    if probe.schema is not None:
        body["format"] = probe.schema
    return body


def ask(client: httpx.Client, model: str, probe: Probe) -> int | None:
    """Send one request; return Ollama's prompt_eval_count, or None when it reports none."""
    response = client.post("/api/chat", json=request_body(model, probe))
    response.raise_for_status()
    count = response.json().get("prompt_eval_count")
    return count if isinstance(count, int) and not isinstance(count, bool) else None


def judge(measurements: Sequence[Measurement], *, reference_overhead: int = REFERENCE_OVERHEAD) -> Verdict:
    """Decide whether the counter never counts fewer tokens than Ollama processed.

    Clean probes set the overhead and must agree exactly. Boundary and marker
    probes only need ``reported <= counted + overhead``.
    """
    problems: list[str] = []
    notes: list[str] = []
    missing = [m.name for m in measurements if m.reported is None]
    if missing:
        problems.append(f"Ollama reported no prompt_eval_count for: {', '.join(missing)}")
    clean = [m for m in measurements if m.kind == "clean" and m.reported is not None]
    if len(clean) < _MINIMUM_CLEAN_PROBES:
        problems.append(f"only {len(clean)} clean requests were measured; at least {_MINIMUM_CLEAN_PROBES} are needed")
    by_overhead: dict[int, list[str]] = {}
    for measurement in clean:
        assert measurement.reported is not None
        by_overhead.setdefault(measurement.reported - measurement.counted, []).append(measurement.name)
    overhead: int | None = None
    if len(by_overhead) == 1:
        [overhead] = by_overhead
        if overhead < 0:
            problems.append(f"the counter counts {-overhead} more than Ollama on every clean request, so it is not exact")
            overhead = None
    elif by_overhead:
        detail = "; ".join(f"{value:+d}: {', '.join(names)}" for value, names in sorted(by_overhead.items()))
        problems.append(
            "the template overhead is not constant across clean requests "
            f"(reported minus counted: {detail}). The counter does not match Ollama's "
            "tokenization on the requests that differ, or a reused prompt cache changed the count"
        )
    if overhead is not None:
        if overhead != reference_overhead:
            notes.append(
                f"observed template overhead is {overhead}, not the {reference_overhead} #120 measured for gemma4 "
                "on Ollama 0.34.4; the counter's counts are unaffected. No summarizer budget charges the template "
                "by name, so it matters only if it exceeds the slack in the schema term and the safety margin"
            )
        for measurement in measurements:
            if measurement.kind == "clean" or measurement.reported is None:
                continue
            gap = measurement.reported - measurement.counted - overhead
            if gap > 0:
                problems.append(f"{measurement.name}: Ollama counted {gap} more than the counter plus overhead")
            elif gap < 0:
                notes.append(
                    f"{measurement.name}: Ollama counted {-gap} fewer, so it reads this input as fewer tokens than "
                    "the counter; the counter overcounts, which is safe"
                )
            else:
                notes.append(f"{measurement.name}: counted the same as the counter plus overhead")
    return Verdict(passed=not problems, overhead=overhead, problems=problems, notes=notes)


def measure(
    client: httpx.Client | None, counter: TokenCounter, model: str, selected: Sequence[Probe] | None = None
) -> list[Measurement]:
    """Count every probe; when a client is given, also ask Ollama (after one throwaway warm-up)."""
    if client is not None:
        ask(client, model, WARM_UP)
    return [
        Measurement(
            name=probe.name,
            kind=probe.kind,
            schema=probe.schema is not None,
            counted=counted_tokens(counter, probe),
            reported=None if client is None else ask(client, model, probe),
        )
        for probe in (probes() if selected is None else selected)
    ]


def _is_this_machine(host: str) -> bool:
    return urlsplit(host if "://" in host else f"http://{host}").hostname in _LOOPBACK_HOSTS


def _print_table(measurements: Sequence[Measurement]) -> None:
    print(f"{'probe':26} {'kind':8} {'schema':6} {'counted':>8} {'reported':>9} {'reported-counted':>17}")
    for m in measurements:
        reported = "-" if m.reported is None else str(m.reported)
        overhead = "-" if m.overhead is None else f"{m.overhead:+d}"
        print(f"{m.name:26} {m.kind:8} {'yes' if m.schema else 'no':6} {m.counted:8d} {reported:>9} {overhead:>17}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="installed Ollama model, for example gemma4:31b")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server on this machine")
    parser.add_argument("--models-dir", type=Path, default=None, help="Ollama model store (default: OLLAMA_MODELS or ~/.ollama/models)")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per request; a cold 31B load is slow")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    parser.add_argument(
        "--set",
        choices=("short", "long"),
        default="short",
        help="short probes (default), or long requests of a few thousand tokens built from committed novels",
    )
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS, help="committed sample documents for --set long")
    parser.add_argument("--dry-run", action="store_true", help="count the probes only; send nothing to Ollama")
    return parser


def main(argv: Sequence[str] | None = None, *, client: httpx.Client | None = None) -> int:
    args = _parser().parse_args(argv)
    if not _is_this_machine(args.host):
        print(
            f"Refusing {args.host}: the tokenizer is read from this machine's model files, "
            "so the Ollama host must be this machine.",
            file=sys.stderr,
        )
        return EXIT_INCOMPLETE
    try:
        counter = TokenizerJsonCounter.for_ollama_model(
            args.model, models_dir=args.models_dir, allow_unverified=True
        )
    except TokenAccountingError as error:
        print(f"No tokenizer.json counter for {args.model}: {error}", file=sys.stderr)
        return EXIT_INCOMPLETE

    try:
        selected = long_probes(counter, args.model, args.documents_dir) if args.set == "long" else probes()
    except MissingSampleDocument as error:
        print(f"Cannot build the long set: {error}", file=sys.stderr)
        return EXIT_INCOMPLETE

    print(f"model {args.model}, {args.set} set, counter {counter.identity}, counting fingerprint {counter.fingerprint}")
    if args.dry_run:
        measurements = measure(None, counter, args.model, selected)
        _print_table(measurements)
        print("Dry run: nothing was sent to Ollama.")
        return EXIT_PASSED

    owned = client is None
    http = client or httpx.Client(base_url=args.host, timeout=args.timeout)
    try:
        version = http.get("/api/version").json().get("version")
        print(f"Ollama {version}")
        measurements = measure(http, counter, args.model, selected)
    except (httpx.HTTPError, ValueError) as error:
        print(f"Ollama request failed: {error}", file=sys.stderr)
        return EXIT_INCOMPLETE
    finally:
        if owned:
            http.close()

    verdict = judge(measurements)
    _print_table(measurements)
    for note in verdict.notes:
        print(f"note: {note}")
    for problem in verdict.problems:
        print(f"PROBLEM: {problem}")
    if verdict.passed:
        print(
            f"PASSED: counted tokens plus {verdict.overhead} template tokens equal Ollama's count on every "
            "clean request, and no boundary or marker request was undercounted.\n"
            f"Add to _VERIFIED_TOKENIZER_JSON_FINGERPRINTS: {counter.fingerprint}"
        )
    else:
        print("FAILED: do not add this fingerprint.")
    if args.output is not None:
        report = {
            "model": args.model,
            "set": args.set,
            "ollama_version": version,
            "counter": counter.identity,
            "fingerprint": counter.fingerprint,
            "passed": verdict.passed,
            "overhead": verdict.overhead,
            "problems": verdict.problems,
            "notes": verdict.notes,
            "requests": [asdict(m) | {"overhead": m.overhead} for m in measurements],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return EXIT_PASSED if verdict.passed else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
