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
    ) -> None:
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
        [probe] = [p for p in [verify.WARM_UP, *verify.probes()] if p.system == system]
        extra = self.extra.get(probe.name, 0)
        if extra is None:
            return httpx.Response(200, json={"message": {"content": "x"}})
        if self.trim:
            system, user = system.strip(), user.strip()
        count = self.counter.count(system) + self.counter.count(user) + self.template + extra
        return httpx.Response(200, json={"prompt_eval_count": count, "message": {"content": "x"}})

    def chats(self) -> list[dict]:
        return [body for path, body in self.requests if path == "/api/chat" and body is not None]


def _run(
    models_dir: Path,
    *,
    extra: dict[str, int | None] | None = None,
    template: int = TEMPLATE,
    trim: bool = False,
    output: Path | None = None,
) -> tuple[int, ScriptedOllama]:
    counter = TokenizerJsonCounter.for_ollama_model(
        "big:31b", models_dir=models_dir, allow_unverified=True
    )
    ollama = ScriptedOllama(counter, template=template, extra=extra, trim=trim)
    argv = ["--model", "big:31b", "--models-dir", str(models_dir)]
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
