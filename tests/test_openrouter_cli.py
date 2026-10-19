"""The CLI with `--provider openrouter`.

The end-to-end tests drive the real `main`, pipeline, `OpenRouterProvider` and
estimate counter. Only the wire is replaced, by a mock transport that answers
in OpenRouter's documented response format (`tests/fixtures/openrouter/`).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from summarizer import cli
from summarizer.cli import build_provider, main, parse_args
from summarizer.config import AppConfig
from summarizer.providers.base import ModelProvider
from summarizer.providers.openrouter import OpenRouterProvider
from summarizer.tokenization import ScaledTiktokenCounter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "openrouter"
KEY = "sk-or-v1-testsentinel0123456789abcdef0123456789abcdef"
MODEL = "meta-llama/llama-3.3-70b-instruct"
SOURCE = (
    "The committee met on Tuesday to review the harbor proposal. "
    "Members agreed the dredging schedule was realistic."
)


class _FakeEncoding:
    name = "o200k_base"

    def encode_ordinary(self, text: str) -> list[int]:
        return list(range((len(text) + 3) // 4))


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tiktoken

    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: _FakeEncoding())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text(SOURCE, encoding="utf-8")


def _completion(content: str, *, number: int) -> dict[str, object]:
    body = json.loads((FIXTURES / "chat_completion_structured_ok.json").read_text())
    body["id"] = f"gen-{number:04d}-fixture"
    body["choices"][0]["message"]["content"] = content
    body["model"] = MODEL
    return body


class Wire:
    """Answer chat and listing requests the way OpenRouter documents them."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200, json=json.loads((FIXTURES / "models_listing.json").read_text())
            )
        body = json.loads(request.content)
        schema_name = body["response_format"]["json_schema"]["name"]
        text = body["messages"][1]["content"]
        if schema_name == "final_editorial_draft":
            content = {"text": "Members agreed the dredging schedule was realistic."}
        elif schema_name == "verification_claim_anchors":
            spans = dict.fromkeys(re.findall(r'"span_id":"(V\d+S\d+)"', text))
            content = {"spans": [{"span_id": span, "anchors": []} for span in spans]}
        elif schema_name == "verification_claim_findings":
            evidence = re.search(r'"segment_id":"([DS]\d+)"', text)
            assert evidence is not None
            claims = dict.fromkeys(re.findall(r'"claim_id":"(V\d+C\d+)"', text))
            content = {
                "findings": [
                    {
                        "claim_id": claim,
                        "verdict": "supported",
                        "evidence": [
                            {
                                "segment_id": evidence.group(1),
                                "exact_quote": "Members agreed the dredging schedule was realistic.",
                            }
                        ],
                    }
                    for claim in claims
                ]
            }
        else:
            segment = re.search(r"\b[DS]\d{6}\b", body["messages"][0]["content"])
            assert segment is not None
            content = {
                "summary": "The committee reviewed the harbor proposal.",
                "content_units": [],
                "entities": [],
                "qualifications": [],
                "contradictions": [],
                "quotations": [],
                "provenance": [segment.group(0)],
                "level": 0,
            }
        return httpx.Response(
            200, json=_completion(json.dumps(content), number=len(self.requests))
        )

    @property
    def chat(self) -> list[httpx.Request]:
        return [request for request in self.requests if request.method == "POST"]


def run(
    argv: list[str], wire: Wire, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str, list[OpenRouterProvider]]:
    built: list[OpenRouterProvider] = []

    def openrouter_factory(**kwargs: object) -> ModelProvider:
        provider = OpenRouterProvider(
            **kwargs,  # type: ignore[arg-type]
            client_factory=lambda **client_kwargs: httpx.Client(
                transport=httpx.MockTransport(wire), **client_kwargs
            ),
        )
        built.append(provider)
        return provider

    code = main(
        argv,
        provider_factory=lambda app: build_provider(app, openrouter_factory=openrouter_factory),
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err, built


# --- argument parsing ----------------------------------------------------------


def test_openrouter_is_a_provider_choice_and_carries_model_and_upstream() -> None:
    parsed = parse_args(
        ["--provider", "openrouter", "--model", MODEL, "--openrouter-upstream", "deepinfra"]
    )

    assert parsed.app.provider == "openrouter"
    assert parsed.app.model == MODEL
    assert parsed.app.openrouter_upstream == "deepinfra"


def test_an_openrouter_run_carries_no_ollama_host() -> None:
    # A default Ollama host here would make the cache identity reject the run.
    assert parse_args(["--provider", "openrouter", "--model", MODEL]).app.ollama_host == ""
    assert parse_args([]).app.ollama_host == "http://localhost:11434"
    assert parse_args(["--provider", "ollama"]).app.ollama_host == "http://localhost:11434"


def test_model_flag_is_required_for_openrouter(capsys: pytest.CaptureFixture[str]) -> None:
    # The default model name is an OpenAI one and would be sent to OpenRouter.
    with pytest.raises(SystemExit):
        parse_args(["--provider", "openrouter"])

    assert "--model" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--provider", "openai", "--openrouter-upstream", "deepinfra"],
        ["--provider", "openrouter", "--model", MODEL, "--openrouter-upstream", "bad slug"],
    ],
)
def test_upstream_must_be_valid_and_only_for_openrouter(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_existing_providers_keep_their_default_model() -> None:
    assert parse_args([]).app.model == "gpt-4o-mini"
    assert parse_args(["--provider", "ollama"]).app.model == "gpt-4o-mini"


def test_build_provider_passes_the_pinned_upstream_and_builds_nothing_else() -> None:
    constructions: list[tuple[str, object]] = []
    expected = object()

    provider = build_provider(
        AppConfig(provider="openrouter", model=MODEL, openrouter_upstream="deepinfra"),
        openai_factory=lambda: constructions.append(("openai", None)),
        ollama_factory=lambda **kw: constructions.append(("ollama", kw)),
        openrouter_factory=lambda **kw: constructions.append(("openrouter", kw)) or expected,
    )

    assert provider is expected
    assert constructions == [("openrouter", {"upstream": "deepinfra"})]


def test_build_counter_gives_a_labeled_estimate_for_openrouter() -> None:
    counter = cli.build_counter(AppConfig(provider="openrouter", model=MODEL))

    assert isinstance(counter, ScaledTiktokenCounter)
    assert counter.exact is False


def test_help_documents_the_openrouter_flag_and_provider() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert "--openrouter-upstream" in result.stdout
    assert "openrouter" in result.stdout


# --- dry run ---------------------------------------------------------------------


def test_dry_run_reports_provider_model_upstream_window_source_and_estimates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        ["--provider", "openrouter", "--model", MODEL, "--openrouter-upstream", "deepinfra",
         "--context-window", "65536", "--dry-run"],
        provider_factory=lambda app: pytest.fail("provider constructed during dry run"),
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "Context window: 65536 tokens" in captured.out
    assert "Provider: openrouter" in captured.out
    assert f"Model: {MODEL}" in captured.out
    assert "Upstream: deepinfra (pinned)" in captured.out
    assert "Context window source: explicit" in captured.out
    assert "Token counts: estimate (estimate:o200k_base-x1.5)" in captured.out
    assert "estimates" in captured.err
    assert KEY not in captured.out + captured.err


def test_dry_run_without_a_window_says_it_is_assumed_and_how_to_set_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["--provider", "openrouter", "--model", MODEL, "--dry-run"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Upstream: not pinned" in out
    assert re.search(r"Context window source: assumed.*--context-window", out)


def test_dry_run_output_for_existing_providers_is_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--dry-run", "--context-window", "10000"], counter_factory=lambda app: _Exact())

    out = capsys.readouterr()
    assert out.out.splitlines() == [
        "Strategy: direct",
        "Context window: 10000 tokens",
        "Usable input capacity: 315 tokens",
    ]
    assert out.err == ""


class _Exact:
    identity = "test:characters"
    exact = True
    monotonic = True

    def count(self, text: str) -> int:
        return len(text)


# --- end to end ------------------------------------------------------------------


def test_a_run_publishes_through_verification_and_audit_with_openrouter_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wire = Wire()

    code, out, err, built = run(
        ["--provider", "openrouter", "--model", MODEL, "--openrouter-upstream", "deepinfra/turbo",
         "--verify", "--audit", "audit.json"],
        wire,
        capsys,
    )

    assert code == 0, err
    assert (tmp_path / "output.txt").read_text(encoding="utf-8").strip() == (
        "Members agreed the dredging schedule was realistic."
    )
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["configuration"]["app"]["provider"] == "openrouter"
    assert audit["configuration"]["app"]["model"] == MODEL
    assert audit["configuration"]["app"]["openrouter_upstream"] == "deepinfra/turbo"
    assert audit["configuration"]["budget"]["counter_exact"] is False
    assert audit["configuration"]["budget"]["context_window_assumed"] is False
    assert {usage["provider"] for usage in audit["usage"]} == {"openrouter"}
    assert audit["verification"]["enabled"] is True

    # Every chat request pinned the upstream and required schema support.
    assert wire.chat
    for request in wire.chat:
        body = json.loads(request.content)
        assert body["provider"]["order"] == ["deepinfra/turbo"]
        assert body["provider"]["allow_fallbacks"] is False
        assert body["provider"]["require_parameters"] is True
        assert request.headers["authorization"] == f"Bearer {KEY}"

    # The window came from the listing, counts are labeled, requests are logged.
    assert "Context window: 131072 tokens (OpenRouter model listing)" in err
    assert "Token counts are estimates" in err
    assert f"OpenRouter chat requests: {len(wire.chat)}" in err
    assert built[0].request_count == len(wire.chat)


def test_the_key_never_reaches_any_output_the_run_produces(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wire = Wire()

    code, out, err, _ = run(
        ["--provider", "openrouter", "--model", MODEL, "--audit", "audit.json",
         "--cache-dir", "cache", "--run-id", "keyless"],
        wire,
        capsys,
    )

    assert code == 0, err
    produced = [out, err]
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.name != "input.txt":
            produced.append(path.read_text(encoding="utf-8", errors="replace"))
    assert not [text for text in produced if KEY in text or "sk-or-v1-" in text]


def test_an_explicit_window_above_the_listed_maximum_fails_before_any_chat_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    wire = Wire()

    code, _, err, _ = run(
        ["--provider", "openrouter", "--model", MODEL, "--context-window", "999999"],
        wire,
        capsys,
    )

    assert code == 1
    assert "exceeds the model's listed maximum 131072" in err
    assert wire.chat == []


def test_a_missing_key_is_reported_without_a_traceback_or_any_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    wire = Wire()

    code, _, err, _ = run(["--provider", "openrouter", "--model", MODEL], wire, capsys)

    assert code == 1
    assert "OPENROUTER_API_KEY" in err
    assert "Traceback" not in err
    assert wire.requests == []
    assert not (tmp_path / "output.txt").exists()


def test_a_failed_run_still_reports_how_many_requests_were_made(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return Wire()(request)
        return httpx.Response(
            402, json=json.loads((FIXTURES / "error_402_key_limit.json").read_text())
        )

    code, _, err, _ = run(["--provider", "openrouter", "--model", MODEL], refuse, capsys)  # type: ignore[arg-type]

    assert code == 1
    assert "credit limit" in err
    assert "OpenRouter chat requests: 1" in err
    assert not (tmp_path / "output.txt").exists()


def test_readme_documents_openrouter_credentials_flags_and_caveats() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "OPENROUTER_API_KEY" in readme
    assert "--provider openrouter" in readme
    assert "--openrouter-upstream" in readme
    assert "estimate" in readme
    assert "OPENAI_BASE_URL" in readme
