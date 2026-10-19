"""Check a tokenizer.json counter against Ollama's own prompt token counts (#190).

Not run by the test suite against a real server, and never run by the
summarizer. It sends a small fixed set of short chat requests to an Ollama
server on this machine, each limited to one generated token, and compares what
the counter says each prompt costs with the ``prompt_eval_count`` Ollama
returns, the method of #120: counted system and user tokens plus a fixed chat
template. This is real model inference, so run it deliberately.

The counter must never count fewer tokens than Ollama processes, so the verdict
fails on any request that Ollama counted higher than the counter plus the
template overhead. The overhead is not assumed: it is the one value that every
plain request must share, and the run fails if the plain requests disagree.
Requests that carry a format schema are checked the same way, which also tests
that the schema is not part of the prompt count. Requests that carry special-token
markers may count lower than Ollama (the counter splits a marker into pieces,
which can only count more), never higher.

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
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from summarizer.tokenization import (  # noqa: E402
    TokenAccountingError,
    TokenCounter,
    TokenizerJsonCounter,
)

DEFAULT_HOST = "http://localhost:11434"
# The chat template overhead #120 measured for gemma4 on its GGUF path. It is a
# reference to compare the observed value with, not an input to the verdict.
REFERENCE_OVERHEAD = 14
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_MINIMUM_PLAIN_PROBES = 3
EXIT_PASSED, EXIT_FAILED, EXIT_INCOMPLETE = 0, 1, 2

_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "keywords": {"type": "array", "items": {"type": "string"}}},
    "required": ["summary", "keywords"],
}


@dataclass(frozen=True)
class Probe:
    """One fixed chat request. `kind` is "plain" or "marker"."""

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


def probes() -> list[Probe]:
    """The fixed set. Each system message opens with a different word, so
    consecutive prompts share only the chat template header and a reused cache
    prefix cannot differ from request to request."""
    return [
        Probe("ascii-short", "plain", "Alpha: you summarize text.", "The harbor moved 24 tonnes of cargo in 1998."),
        Probe("ascii-tiny", "plain", "Bravo: answer briefly.", "Hi."),
        Probe("digits-punctuation", "plain", "Charlie: you check numbers.", "3.14159; 2,718,281; 1e-9; 0xFF; 2026-10-19; $1,234.56 (50%)!"),
        Probe("unicode", "plain", "Delta: you read many scripts.", "Café naïve résumé — “quotes” … 東京タワー 한국어 العربية Ελληνικά 🙂🚀"),
        Probe("whitespace", "plain", "Echo: you keep layout.", "a  b   c\t\td\n\n\ne        f \n g\n"),
        Probe("code", "plain", "Foxtrot: you read code.", "def f(x):\n    return {'a': [1, 2, 3], \"b\": x**2}  # comment\n"),
        Probe("markup", "plain", "Golf: you read markup.", "<table><tr><td>cell</td></tr></table> <b>bold</b> &amp; [link](http://example.test/a?b=c)"),
        Probe("ascii-long", "plain", "Hotel: you summarize reports.", _long_ascii()),
        Probe("non-ascii-long", "plain", "India: you summarize reports.", _long_non_ascii()),
        Probe("schema-ascii", "plain", "Juliet: reply as JSON.", "The harbor moved 24 tonnes of cargo in 1998.", _SCHEMA),
        Probe("schema-unicode", "plain", "Kilo: reply as JSON.", "Café naïve résumé — 東京タワー 🙂", _SCHEMA),
        Probe(
            "special-markers",
            "marker",
            "Lima: you quote text.",
            "Quote this literally: <bos> <eos> <pad> <start_of_turn>user <end_of_turn> <|turn> <|video|>",
        ),
    ]


WARM_UP = Probe("warm-up", "plain", "Zulu: warm up.", "Ready.")


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
    """Decide whether the counter never counts fewer tokens than Ollama processed."""
    problems: list[str] = []
    notes: list[str] = []
    missing = [m.name for m in measurements if m.reported is None]
    if missing:
        problems.append(f"Ollama reported no prompt_eval_count for: {', '.join(missing)}")
    plain = [m for m in measurements if m.kind == "plain" and m.reported is not None]
    if len(plain) < _MINIMUM_PLAIN_PROBES:
        problems.append(f"only {len(plain)} plain requests were measured; at least {_MINIMUM_PLAIN_PROBES} are needed")
    by_overhead: dict[int, list[str]] = {}
    for measurement in plain:
        assert measurement.reported is not None
        by_overhead.setdefault(measurement.reported - measurement.counted, []).append(measurement.name)
    overhead: int | None = None
    if len(by_overhead) == 1:
        [overhead] = by_overhead
        if overhead < 0:
            problems.append(f"the counter counts {-overhead} more than Ollama on every plain request, so it is not exact")
            overhead = None
    elif by_overhead:
        detail = "; ".join(f"{value:+d}: {', '.join(names)}" for value, names in sorted(by_overhead.items()))
        problems.append(
            "the template overhead is not constant across plain requests "
            f"(reported minus counted: {detail}). The counter does not match Ollama's "
            "tokenization on the requests that differ, or a reused prompt cache changed the count"
        )
    if overhead is not None:
        if overhead != reference_overhead:
            notes.append(
                f"observed template overhead is {overhead}, not the {reference_overhead} #120 measured for gemma4; "
                "the counter's counts are unaffected, but anything that assumes 14 should be checked"
            )
        for measurement in measurements:
            if measurement.kind != "marker" or measurement.reported is None:
                continue
            gap = measurement.reported - measurement.counted - overhead
            if gap > 0:
                problems.append(f"{measurement.name}: Ollama counted {gap} more than the counter plus overhead")
            elif gap < 0:
                notes.append(
                    f"{measurement.name}: Ollama counted {-gap} fewer, so it reads a marker as fewer tokens than "
                    "its pieces; the counter overcounts, which is safe"
                )
            else:
                notes.append(f"{measurement.name}: counted the same as the counter, as pieces")
    return Verdict(passed=not problems, overhead=overhead, problems=problems, notes=notes)


def measure(client: httpx.Client | None, counter: TokenCounter, model: str) -> list[Measurement]:
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
        for probe in probes()
    ]


def _is_this_machine(host: str) -> bool:
    return urlsplit(host if "://" in host else f"http://{host}").hostname in _LOOPBACK_HOSTS


def _print_table(measurements: Sequence[Measurement]) -> None:
    print(f"{'probe':22} {'kind':7} {'schema':6} {'counted':>8} {'reported':>9} {'reported-counted':>17}")
    for m in measurements:
        reported = "-" if m.reported is None else str(m.reported)
        overhead = "-" if m.overhead is None else f"{m.overhead:+d}"
        print(f"{m.name:22} {m.kind:7} {'yes' if m.schema else 'no':6} {m.counted:8d} {reported:>9} {overhead:>17}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="installed Ollama model, for example gemma4:31b")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server on this machine")
    parser.add_argument("--models-dir", type=Path, default=None, help="Ollama model store (default: OLLAMA_MODELS or ~/.ollama/models)")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per request; a cold 31B load is slow")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
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

    print(f"model {args.model}, counter {counter.identity}, counting fingerprint {counter.fingerprint}")
    if args.dry_run:
        measurements = measure(None, counter, args.model)
        _print_table(measurements)
        print("Dry run: nothing was sent to Ollama.")
        return EXIT_PASSED

    owned = client is None
    http = client or httpx.Client(base_url=args.host, timeout=args.timeout)
    try:
        version = http.get("/api/version").json().get("version")
        print(f"Ollama {version}")
        measurements = measure(http, counter, args.model)
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
            "plain request and never undercount.\n"
            f"Add to _VERIFIED_TOKENIZER_JSON_FINGERPRINTS: {counter.fingerprint}"
        )
    else:
        print("FAILED: do not add this fingerprint.")
    if args.output is not None:
        report = {
            "model": args.model,
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
