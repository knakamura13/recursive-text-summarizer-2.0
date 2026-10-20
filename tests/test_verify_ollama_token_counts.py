"""The #190 verification script, against a scripted Ollama and a synthetic tokenizer.

These tests prove the script's arithmetic and its refusals. They say nothing
about a real server. The recorded runs against one are in the experiment's README.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from summarizer.compression import compression_draft_schema
from summarizer.config import StrategyConfig
from summarizer.editorial import final_draft_schema
from summarizer.summaries import leaf_summary_schema
from summarizer.tokenization import TokenizerJsonCounter
from tests.test_tokenizer_json_counter import _tensor_model

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "token_counter_verification"
    / "verify_ollama_token_counts.py"
)
TEMPLATE = 14


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_ollama_token_counts", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up by name while the class body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify = _load_script()


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    _tensor_model(tmp_path)
    return tmp_path


class ScriptedOllama:
    """Answers /api/version and /api/chat; prompt_eval_count follows the counter plus `template`.

    With `trim`, it counts each message as the real server may: with leading
    and trailing whitespace removed.
    """

    def __init__(
        self,
        counter: TokenizerJsonCounter,
        *,
        template: int = TEMPLATE,
        extra: dict[str, int | None] | None = None,
        trim: bool = False,
        known: list | None = None,
    ) -> None:
        self.known = [verify.WARM_UP, *(known if known is not None else verify.probes())]
        self.counter = counter
        self.template = template
        self.extra = extra or {}
        self.trim = trim
        self.requests: list[tuple[str, dict | None]] = []

    def client(self) -> httpx.Client:
        return httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(self))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.url.path, body))
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.6"})
        assert request.url.path == "/api/chat" and body is not None
        system, user = (message["content"] for message in body["messages"])
        [probe] = [p for p in self.known if (p.system, p.user) == (system, user)]
        extra = self.extra.get(probe.name, 0)
        if extra is None:
            return httpx.Response(200, json={"message": {"content": "x"}})
        if self.trim:
            system, user = system.strip(), user.strip()
        count = self.counter.count(system) + self.counter.count(user) + self.template + extra
        return httpx.Response(200, json={"prompt_eval_count": count, "message": {"content": "x"}})

    def chats(self) -> list[dict]:
        return [body for path, body in self.requests if path == "/api/chat" and body is not None]


def _counter(models_dir: Path) -> TokenizerJsonCounter:
    return TokenizerJsonCounter.for_ollama_model("big:31b", models_dir=models_dir, allow_unverified=True)


def _run(
    models_dir: Path,
    *,
    extra: dict[str, int | None] | None = None,
    template: int = TEMPLATE,
    trim: bool = False,
    output: Path | None = None,
    long: bool = False,
) -> tuple[int, ScriptedOllama]:
    counter = _counter(models_dir)
    known = verify.long_probes(counter, "big:31b") if long else None
    ollama = ScriptedOllama(counter, template=template, extra=extra, trim=trim, known=known)
    argv = ["--model", "big:31b", "--models-dir", str(models_dir)]
    if long:
        argv += ["--set", "long"]
    if output is not None:
        argv += ["--output", str(output)]
    return verify.main(argv, client=ollama.client()), ollama


# --- the verdict -----------------------------------------------------------


def test_a_counter_that_matches_ollama_passes_and_prints_the_fingerprint_to_add(
    models_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "out" / "report.json"
    code, _ = _run(models_dir, output=report)

    out = capsys.readouterr().out
    fingerprint = TokenizerJsonCounter.for_ollama_model(
        "big:31b", models_dir=models_dir, allow_unverified=True
    ).fingerprint
    assert code == verify.EXIT_PASSED
    assert "PASSED" in out and fingerprint in out and "_VERIFIED_TOKENIZER_JSON_FINGERPRINTS" in out
    saved = json.loads(report.read_text())
    assert saved["passed"] is True and saved["overhead"] == TEMPLATE
    assert saved["fingerprint"] == fingerprint and saved["ollama_version"] == "0.40.6"
    assert {request["name"] for request in saved["requests"]} == {p.name for p in verify.probes()}


def test_whitespace_at_a_message_edge_that_ollama_trims_passes_and_is_reported(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, trim=True)

    out = capsys.readouterr().out
    counter = TokenizerJsonCounter.for_ollama_model(
        "big:31b", models_dir=models_dir, allow_unverified=True
    )
    assert code == verify.EXIT_PASSED
    # Every boundary probe loses its whitespace tokens, and the notes name the gap.
    for probe in verify.probes():
        if probe.kind == "boundary":
            lost = (counter.count(probe.system) + counter.count(probe.user)) - (
                counter.count(probe.system.strip()) + counter.count(probe.user.strip())
            )
            assert lost >= 1
            assert f"{probe.name}: Ollama counted {lost} fewer" in out


def test_whitespace_at_a_message_edge_that_ollama_keeps_reports_no_difference(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, trim=False)

    out = capsys.readouterr().out
    assert code == verify.EXIT_PASSED
    assert "fewer" not in out and "counted the same as the counter" in out


def test_a_boundary_probe_counted_higher_by_ollama_fails_and_names_it(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, extra={"trailing-space": 2})

    assert code == verify.EXIT_FAILED
    out = capsys.readouterr().out
    assert "trailing-space: Ollama counted 2 more" in out and "PASSED" not in out


def test_one_clean_probe_off_by_a_token_fails_the_run(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, extra={"unicode": 3})

    out = capsys.readouterr().out
    assert code == verify.EXIT_FAILED
    assert "not constant" in out and "unicode" in out and "FAILED" in out


def test_internal_whitespace_probes_must_agree_exactly_with_the_rest(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Only edge whitespace may differ. A server that collapses runs inside a message is a mismatch.
    code, _ = _run(models_dir, extra={"whitespace-internal": -2})

    assert code == verify.EXIT_FAILED
    assert "whitespace-internal" in capsys.readouterr().out


def test_a_counter_that_overcounts_every_clean_probe_is_not_exact(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, template=-5)

    assert code == verify.EXIT_FAILED
    assert "more than Ollama on every clean request" in capsys.readouterr().out


def test_a_schema_that_counts_as_prompt_tokens_is_an_undercount(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, extra={"schema-ascii": 40, "schema-unicode": 40})

    assert code == verify.EXIT_FAILED
    out = capsys.readouterr().out
    assert "schema-ascii" in out and "schema-unicode" in out


def test_a_marker_read_as_fewer_tokens_than_its_pieces_is_safe(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, extra={"special-markers": -6})

    assert code == verify.EXIT_PASSED
    assert "special-markers: Ollama counted 6 fewer" in capsys.readouterr().out


def test_a_marker_counted_higher_by_ollama_fails(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, extra={"special-markers": 2})

    assert code == verify.EXIT_FAILED
    assert "special-markers: Ollama counted 2 more" in capsys.readouterr().out


def test_a_missing_prompt_eval_count_is_not_a_pass(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, extra={"leading-tab": None})

    assert code == verify.EXIT_FAILED
    assert "no prompt_eval_count for: leading-tab" in capsys.readouterr().out


def test_a_different_template_overhead_is_reported_but_does_not_fail(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, template=16)

    assert code == verify.EXIT_PASSED
    assert "observed template overhead is 16" in capsys.readouterr().out


# --- the requests ----------------------------------------------------------


def test_it_sends_one_warm_up_then_each_probe_once_in_the_summarizers_request_shape(
    models_dir: Path,
) -> None:
    _, ollama = _run(models_dir)

    chats = ollama.chats()
    assert len(chats) == 1 + len(verify.probes())
    assert chats[0]["messages"][1]["content"] == verify.WARM_UP.user
    for body in chats:
        assert body["stream"] is False
        # The summarizer's OllamaProvider sends think false; omitting it changes the template.
        assert body["think"] is False
        assert body["options"] == {"num_predict": 1, "temperature": 0, "seed": 0}
        assert body["model"] == "big:31b"
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert [body.get("format") is not None for body in chats[1:]] == [
        p.schema is not None for p in verify.probes()
    ]
    assert {path for path, _ in ollama.requests} == {"/api/version", "/api/chat"}


def test_a_dry_run_counts_the_probes_and_sends_nothing(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"dry run sent {request.url}")

    client = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(refuse))
    code = verify.main(
        ["--model", "big:31b", "--models-dir", str(models_dir), "--dry-run"], client=client
    )

    out = capsys.readouterr().out
    assert code == verify.EXIT_PASSED
    assert "Dry run: nothing was sent to Ollama." in out
    assert all(probe.name in out for probe in verify.probes())


@pytest.mark.parametrize("host", ["http://10.0.0.5:11434", "http://gpu-box:11434", "https://example.test"])
def test_a_host_that_is_not_this_machine_is_refused_before_anything_is_sent(
    models_dir: Path, capsys: pytest.CaptureFixture[str], host: str
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"sent {request.url}")

    client = httpx.Client(base_url=host, transport=httpx.MockTransport(refuse))
    code = verify.main(
        ["--model", "big:31b", "--models-dir", str(models_dir), "--host", host], client=client
    )

    assert code == verify.EXIT_INCOMPLETE
    assert "must be this machine" in capsys.readouterr().err


def test_a_model_without_a_tokenizer_layer_is_incomplete_not_a_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"sent {request.url}")

    client = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(refuse))
    code = verify.main(["--model", "absent:1b", "--models-dir", str(tmp_path)], client=client)

    assert code == verify.EXIT_INCOMPLETE
    assert "No tokenizer.json counter for absent:1b" in capsys.readouterr().err


def test_an_unreachable_server_is_incomplete(models_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(down))
    code = verify.main(["--model", "big:31b", "--models-dir", str(models_dir)], client=client)

    assert code == verify.EXIT_INCOMPLETE
    assert "Ollama request failed" in capsys.readouterr().err


# --- the fixed set ---------------------------------------------------------


def _edge_whitespace(text: str) -> list[tuple[str, str]]:
    """The (position, character) of each run of whitespace at the edges of a message."""
    found = []
    if text != text.lstrip():
        found.append(("leading", text[: len(text) - len(text.lstrip())]))
    if text != text.rstrip():
        found.append(("trailing", text[len(text.rstrip()) :]))
    return found


def test_the_fixed_set_is_small_varied_and_distinguishable() -> None:
    fixed = verify.probes()
    kinds = [probe.kind for probe in fixed]

    assert kinds.count("clean") >= 8 and kinds.count("boundary") == 7 and kinds.count("marker") == 1
    assert set(kinds) == {"clean", "boundary", "marker"}
    assert sum(probe.schema is not None for probe in fixed) == 2
    # Different opening words keep a reused cache prefix to the chat template header.
    openings = [probe.system.split(":")[0] for probe in [verify.WARM_UP, *fixed]]
    assert len(set(openings)) == len(openings)
    assert len({probe.name for probe in fixed}) == len(fixed)
    # Short requests: one generated token each, a few hundred prompt tokens at most.
    assert all(len(probe.system) + len(probe.user) < 2_500 for probe in fixed)
    # The set exercises what a byte count gets wrong: non-ASCII, internal whitespace, code, markup.
    names = {probe.name for probe in fixed}
    assert {"unicode", "non-ascii-long", "whitespace-internal", "code-internal", "markup"} <= names
    assert any(ord(character) > 0x2FFF for probe in fixed for character in probe.user)


def test_clean_and_marker_probes_have_no_whitespace_at_a_message_edge() -> None:
    for probe in verify.probes():
        if probe.kind in ("clean", "marker"):
            assert _edge_whitespace(probe.system) == [] and _edge_whitespace(probe.user) == [], probe.name
    # The internal-whitespace probes keep their runs, tabs, newlines and indentation inside.
    internal = {p.name: p for p in verify.probes()}
    assert "\t\t" in internal["whitespace-internal"].user and "\n\n\n" in internal["whitespace-internal"].user
    assert "  " in internal["whitespace-internal"].user and "\n    " in internal["code-internal"].user


def test_each_boundary_probe_adds_exactly_one_run_of_one_kind_of_whitespace() -> None:
    triples = set()
    for probe in verify.probes():
        if probe.kind != "boundary":
            continue
        found = _edge_whitespace(probe.system) + _edge_whitespace(probe.user)
        assert len(found) == 1, probe.name
        side = "system" if _edge_whitespace(probe.system) else "user"
        [(position, run)] = found
        assert len(set(run)) == 1, probe.name
        # Stripped, the user text is the shared base, so the whitespace is the only change.
        assert probe.user.strip() == verify.BASE_USER, probe.name
        triples.add((side, position, run[0]))
    assert triples == {
        ("user", "trailing", "\n"),
        ("user", "trailing", " "),
        ("user", "trailing", "\t"),
        ("user", "leading", "\n"),
        ("user", "leading", " "),
        ("user", "leading", "\t"),
        ("system", "trailing", "\n"),
    }


# --- judge -----------------------------------------------------------------


def _m(name: str, kind: str, counted: int, reported: int | None) -> object:
    return verify.Measurement(name, kind, False, counted, reported)


def test_judge_needs_enough_clean_measurements() -> None:
    few = [_m("a", "clean", 5, 19), _m("b", "clean", 6, 20)]

    verdict = verify.judge(few)

    assert verdict.passed is False
    assert "at least 3 are needed" in verdict.problems[0]


def test_judge_passes_an_exact_constant_overhead() -> None:
    measurements = [_m(name, "clean", 5 + i, 19 + i) for i, name in enumerate("abc")]

    verdict = verify.judge(measurements)

    assert verdict.passed is True and verdict.overhead == 14 and verdict.notes == []


def test_judge_sets_the_overhead_from_clean_probes_only() -> None:
    measurements = [_m(name, "clean", 5 + i, 19 + i) for i, name in enumerate("abc")]
    measurements.append(_m("edge", "boundary", 9, 22))  # one fewer than counter + 14

    verdict = verify.judge(measurements)

    assert verdict.passed is True and verdict.overhead == 14
    assert verdict.notes == ["edge: Ollama counted 1 fewer, so it reads this input as fewer tokens than "
                             "the counter; the counter overcounts, which is safe"]


def test_judge_fails_a_boundary_probe_above_counter_plus_overhead() -> None:
    measurements = [_m(name, "clean", 5 + i, 19 + i) for i, name in enumerate("abc")]
    measurements.append(_m("edge", "boundary", 9, 24))

    verdict = verify.judge(measurements)

    assert verdict.passed is False
    assert verdict.problems == ["edge: Ollama counted 1 more than the counter plus overhead"]


# --- the long set ----------------------------------------------------------

LONG_TARGETS = {"leaf-2k": 2_000, "leaf-4k": 4_000, "leaf-6k": 6_000}


def test_the_long_set_has_a_clean_core_boundary_and_marker_probes(models_dir: Path) -> None:
    long = verify.long_probes(_counter(models_dir), "big:31b")
    kinds = [probe.kind for probe in long]

    assert kinds.count("clean") >= 5 and kinds.count("boundary") == 2 and kinds.count("marker") == 1
    assert len({probe.name for probe in long}) == len(long)
    assert [probe.name for probe in long if probe.name.startswith("long-leaf")] == [
        "long-leaf-2k",
        "long-leaf-4k",
        "long-leaf-6k",
    ]
    assert any(ord(character) > 0x2FFF for probe in long for character in probe.user)


def test_long_probe_sizes_are_close_to_the_summarizers_segment_sizes(models_dir: Path) -> None:
    counter = _counter(models_dir)
    long = {probe.name: probe for probe in verify.long_probes(counter, "big:31b")}

    for name, target in LONG_TARGETS.items():
        counted = counter.count(long[f"long-{name}"].user)
        # The user message is the app's fenced segment, so a little over the passage itself.
        assert 0.8 * target <= counted <= 1.05 * target, (name, counted)
    # Every user message is a few thousand tokens, up to the size of a real leaf segment
    # (a quarter of the usable input capacity, about 5,000 to 6,000 at a 32,768 window).
    users = [counter.count(p.user) for p in long.values()]
    assert 1_500 <= min(users) and max(users) <= 6_300


def test_the_long_set_is_deterministic(models_dir: Path) -> None:
    counter = _counter(models_dir)

    first = verify.long_probes(counter, "big:31b")
    second = verify.long_probes(counter, "big:31b")

    assert first == second


def test_long_leaf_probes_are_the_apps_own_leaf_requests(models_dir: Path) -> None:
    long = {probe.name: probe for probe in verify.long_probes(_counter(models_dir), "big:31b")}

    for name in ("long-leaf-2k", "long-leaf-4k", "long-leaf-6k"):
        probe = long[name]
        assert "You extract structured information from one region of a longer document" in probe.system
        assert "BEGIN" in probe.user and "END" in probe.user
        # The Ollama wire schema the app sends, not the shared hosted one: it adds
        # constraints the hosted subset cannot carry.
        assert probe.schema is not None and "properties" in probe.schema
        assert set(probe.schema["properties"]) >= {"summary", "provenance", "quotations"}
        assert probe.schema["properties"]["summary"]["minLength"] == 1
        assert probe.schema["properties"]["provenance"]["minItems"] == 1
        assert probe.schema != leaf_summary_schema()
    assert [p.schema is not None for p in long.values()].count(True) == 3


def test_long_clean_and_marker_probes_have_no_whitespace_at_a_message_edge(models_dir: Path) -> None:
    for probe in verify.long_probes(_counter(models_dir), "big:31b"):
        if probe.kind in ("clean", "marker"):
            assert _edge_whitespace(probe.system) == [] and _edge_whitespace(probe.user) == [], probe.name


def test_the_long_boundary_probes_add_one_blank_line_run_to_one_edge_of_the_same_passage(
    models_dir: Path,
) -> None:
    long = {probe.name: probe for probe in verify.long_probes(_counter(models_dir), "big:31b")}

    trailing, leading = long["long-trailing-blank-lines"], long["long-leading-blank-lines"]

    assert _edge_whitespace(trailing.user) == [("trailing", "\n\n")] and _edge_whitespace(trailing.system) == []
    assert _edge_whitespace(leading.user) == [("leading", "\n\n")] and _edge_whitespace(leading.system) == []
    assert trailing.user.strip() == leading.user.strip()


def test_the_long_marker_probe_carries_markers_inside_a_passage(models_dir: Path) -> None:
    marker = next(p for p in verify.long_probes(_counter(models_dir), "big:31b") if p.kind == "marker")

    assert all(token in marker.user for token in ("<bos>", "<|turn>", "<eos>"))
    assert len(marker.user) > 1_000


def test_no_two_long_leaf_probes_are_adjacent(models_dir: Path) -> None:
    # Neighbouring requests with the same instructions would share a cached prefix.
    long = verify.long_probes(_counter(models_dir), "big:31b")
    names = [probe.name for probe in long]

    for first, second in zip(names, names[1:], strict=False):
        assert not (first.startswith("long-leaf") and second.startswith("long-leaf"))
    assert not (verify.WARM_UP.system.split(":")[0] == long[0].system.split(":")[0])


def test_the_long_mode_sends_the_summarizers_request_shape(models_dir: Path) -> None:
    code, ollama = _run(models_dir, long=True)

    chats = ollama.chats()
    long = verify.long_probes(_counter(models_dir), "big:31b")
    assert code == verify.EXIT_PASSED
    assert len(chats) == 1 + len(long)
    for body in chats:
        assert body["stream"] is False and body["think"] is False
        assert body["options"] == {"num_predict": 1, "temperature": 0, "seed": 0}
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert [body.get("format") is not None for body in chats[1:]] == [p.schema is not None for p in long]
    assert [body["messages"][1]["content"] for body in chats[1:]] == [p.user for p in long]


def test_the_long_mode_passes_when_ollama_trims_the_blank_lines(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, long=True, trim=True)

    out = capsys.readouterr().out
    assert code == verify.EXIT_PASSED
    assert "long-trailing-blank-lines: Ollama counted" in out and "long-leading-blank-lines: Ollama counted" in out


def test_the_long_mode_fails_a_clean_probe_that_disagrees(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A reused cache prefix that lowered one count would show up here.
    code, _ = _run(models_dir, long=True, extra={"long-leaf-4k": -300})

    out = capsys.readouterr().out
    assert code == verify.EXIT_FAILED
    assert "not constant" in out and "long-leaf-4k" in out


def test_the_long_mode_fails_a_boundary_probe_counted_higher(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(models_dir, long=True, extra={"long-trailing-blank-lines": 3})

    assert code == verify.EXIT_FAILED
    assert "long-trailing-blank-lines: Ollama counted 3 more" in capsys.readouterr().out


def test_the_long_mode_marker_probe_may_count_fewer_never_more(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fewer, _ = _run(models_dir, long=True, extra={"long-markers": -5})
    more, _ = _run(models_dir, long=True, extra={"long-markers": 2})

    assert (fewer, more) == (verify.EXIT_PASSED, verify.EXIT_FAILED)


def test_a_long_dry_run_prints_the_sizes_and_sends_nothing(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"dry run sent {request.url}")

    client = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(refuse))
    code = verify.main(
        ["--model", "big:31b", "--models-dir", str(models_dir), "--set", "long", "--dry-run"],
        client=client,
    )

    out = capsys.readouterr().out
    assert code == verify.EXIT_PASSED
    assert "long-leaf-6k" in out and "Dry run: nothing was sent to Ollama." in out


def test_missing_sample_documents_make_the_long_mode_incomplete(
    models_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"sent {request.url}")

    client = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(refuse))
    empty = tmp_path / "no-documents"
    empty.mkdir()
    code = verify.main(
        ["--model", "big:31b", "--models-dir", str(models_dir), "--set", "long", "--documents-dir", str(empty)],
        client=client,
    )

    assert code == verify.EXIT_INCOMPLETE
    assert "sample document" in capsys.readouterr().err


def test_the_default_set_is_still_the_short_one(models_dir: Path) -> None:
    _, ollama = _run(models_dir)

    assert len(ollama.chats()) == 1 + len(verify.probes())


def test_long_passages_end_at_a_sentence_and_start_at_a_paragraph(models_dir: Path) -> None:
    long = {probe.name: probe for probe in verify.long_probes(_counter(models_dir), "big:31b")}

    for name in ("long-plain-4k", "long-trailing-blank-lines", "long-leading-blank-lines"):
        text = long[name].user.strip()
        assert text[-1] in ".!?\"”’)", (name, text[-40:])
        assert text[0].isupper() or text[0] in "\"“‘(", (name, text[:40])


# --- the reserve ----------------------------------------------------------------


def test_the_reserve_is_the_apps_smallest_schema_term_plus_its_default_safety_margin(
    models_dir: Path,
) -> None:
    counter = _counter(models_dir)

    reserve = verify.app_reserve(counter)
    terms = verify.schema_terms(counter)

    assert reserve.margin == StrategyConfig().safety_margin_tokens
    assert reserve.schema_term == min(terms.values()) and terms[reserve.schema_stage] == reserve.schema_term
    assert reserve.tokens == reserve.schema_term + reserve.margin
    # Every stage whose request budget charges a schema is listed.
    assert {"leaf and merge", "editorial", "compression"} <= set(terms)
    assert all(term > 0 for term in terms.values())


def test_the_schema_terms_are_counted_the_way_the_budgets_count_them(models_dir: Path) -> None:
    counter = _counter(models_dir)
    terms = verify.schema_terms(counter)

    assert terms["editorial"] == counter.count(json.dumps(final_draft_schema(), separators=(",", ":")))
    assert terms["compression"] == counter.count(json.dumps(compression_draft_schema(), separators=(",", ":")))
    # The leaf and merge budget adds the quote-candidate bytes and the Ollama schema reserve.
    assert terms["leaf and merge"] > counter.count(
        json.dumps(leaf_summary_schema(), separators=(",", ":"), sort_keys=True)
    )


def _reserve(tokens: int = 289) -> object:
    return verify.Reserve(tokens=tokens, schema_term=33, schema_stage="editorial", margin=tokens - 33)


def _clean(overhead: int) -> list:
    return [_m(name, "clean", 5 + i, 5 + i + overhead) for i, name in enumerate("abc")]


def test_judge_passes_an_overhead_at_or_below_the_reserve() -> None:
    for overhead in (18, 288, 289):
        verdict = verify.judge(_clean(overhead), reserve=_reserve())

        assert verdict.passed is True, overhead
        assert verdict.overhead == overhead


def test_judge_fails_an_overhead_above_the_reserve_and_prints_both_numbers() -> None:
    verdict = verify.judge(_clean(290), reserve=_reserve())

    assert verdict.passed is False
    [problem] = verdict.problems
    assert "290" in problem and "289" in problem
    assert "33" in problem and "editorial" in problem and "256" in problem


def test_judge_without_a_reserve_does_not_apply_the_check() -> None:
    assert verify.judge(_clean(10_000)).passed is True


def test_the_run_prints_the_observed_overhead_beside_the_reserve(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reserve = verify.app_reserve(_counter(models_dir))

    code, _ = _run(models_dir)

    out = capsys.readouterr().out
    assert code == verify.EXIT_PASSED
    assert f"observed {TEMPLATE} tokens" in out and f"reserve {reserve.tokens}" in out
    assert f"{reserve.schema_term} {reserve.schema_stage} schema term" in out and f"{reserve.margin} default safety margin" in out


def test_a_run_whose_overhead_equals_the_reserve_passes(models_dir: Path) -> None:
    reserve = verify.app_reserve(_counter(models_dir))

    code, _ = _run(models_dir, template=reserve.tokens)

    assert code == verify.EXIT_PASSED


def test_a_run_whose_overhead_is_one_above_the_reserve_fails(
    models_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reserve = verify.app_reserve(_counter(models_dir))

    code, _ = _run(models_dir, template=reserve.tokens + 1)

    out = capsys.readouterr().out
    assert code == verify.EXIT_FAILED
    assert f"exceeds the smallest reserve the app applies ({reserve.tokens}" in out
    assert "FAILED" in out and "PASSED" not in out


def test_a_dry_run_has_no_overhead_to_check(models_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"dry run sent {request.url}")

    client = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(refuse))
    code = verify.main(["--model", "big:31b", "--models-dir", str(models_dir), "--dry-run"], client=client)

    assert code == verify.EXIT_PASSED and "reserve" not in capsys.readouterr().out
