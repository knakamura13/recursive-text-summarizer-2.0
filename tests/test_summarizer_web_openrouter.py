"""OpenRouter in the web app: config, key status, preflight, create, and the worker.

The OpenRouter key is read from the server's environment only. These tests set
a sentinel key and assert it never reaches a response, a stored Run
configuration, a setting, an audit or a failure record.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import summarizer_web.worker.runner as runner
from summarizer.providers.base import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRequestError,
)
from summarizer.providers.openrouter import OpenRouterProvider
from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database
from summarizer_web.models.api import RunConfig
from summarizer_web.services import openrouter_service, runs_service
from summarizer_web.services.events_service import start_attempt
from summarizer_web.worker.runner import classify_failure
from tests.support.web_runs import ScriptedProvider
from tests.support.web_runs import seed_document as seed_run_document
from tests.support.web_runs import seed_run
from tests.support.web_views import csrf_headers, seed_document, web_client

class _FakeEncoding:
    name = "o200k_base"

    def encode_ordinary(self, text: str) -> list[int]:
        return list(range((len(text) + 3) // 4))


KEY = "sk-or-v1-testsentinel0123456789abcdef0123456789abcdef"
MODEL = "meta-llama/llama-3.3-70b-instruct"
FIXTURES = Path(__file__).parent / "fixtures" / "openrouter"
SHORT_TEXT = "The bridge deck spans 200 feet across the river. Engineers inspected it in 1998."


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: _FakeEncoding())


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with web_client(tmp_path, monkeypatch) as test_client:
        yield test_client


def listing(monkeypatch: pytest.MonkeyPatch, response: httpx.Response | None = None) -> list[httpx.Request]:
    """Serve OpenRouter's model listing to the web app's provider factory."""
    seen: list[httpx.Request] = []
    body = json.loads((FIXTURES / "models_listing.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response if response is not None else httpx.Response(200, json=body)

    def create_provider(upstream: str) -> OpenRouterProvider:
        return OpenRouterProvider(
            upstream=upstream,
            client_factory=lambda **kwargs: httpx.Client(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        )

    monkeypatch.setattr(openrouter_service, "create_provider", create_provider)
    return seen


def preflight(client: TestClient, **config: object) -> dict:
    response = client.post(
        "/api/v1/preflight",
        json={"document_id": "doc-1", "config": {"provider": "openrouter", "model": MODEL, **config}},
        headers=csrf_headers(client),
    )
    assert response.status_code == 200
    return response.json()


def codes(notices: list[dict]) -> list[str]:
    return [notice["code"] for notice in notices]


# --- configuration -------------------------------------------------------------------


def test_the_default_provider_is_ollama_so_stored_runs_and_defaults_are_unchanged() -> None:
    assert RunConfig().provider == "ollama"
    assert RunConfig().openrouter_upstream == ""
    # A Run stored before the field existed loads with the old behavior.
    stored = RunConfig.model_construct(**{"model": "llama3.2:3b", "target_words": 300})
    assert stored.provider == "ollama"


@pytest.mark.parametrize(
    "values",
    [
        {"provider": "ollama", "openrouter_upstream": "deepinfra"},
        {"provider": "openrouter", "openrouter_upstream": "two words"},
        {"provider": "other"},
    ],
)
def test_run_config_rejects_an_unknown_provider_or_a_misplaced_upstream(values: dict) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(values)


def test_openrouter_status_reports_only_whether_a_key_is_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_response = client.get("/api/v1/openrouter/health")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    unset_response = client.get("/api/v1/openrouter/health")

    assert set_response.json()["key_set"] is True
    assert unset_response.json()["key_set"] is False
    assert "OPENROUTER_API_KEY" in unset_response.json()["message"]
    assert KEY not in set_response.text + unset_response.text


def test_default_provider_and_upstream_persist_through_settings(client: TestClient) -> None:
    response = client.patch(
        "/api/v1/settings",
        json={"defaults": {"provider": "openrouter", "model": MODEL, "openrouter_upstream": "deepinfra"}},
        headers=csrf_headers(client),
    )

    assert response.status_code == 200
    defaults = client.get("/api/v1/settings").json()["defaults"]
    assert (defaults["provider"], defaults["openrouter_upstream"]) == ("openrouter", "deepinfra")


def test_switching_the_default_back_to_ollama_drops_the_upstream(client: TestClient) -> None:
    headers = csrf_headers(client)
    client.patch(
        "/api/v1/settings",
        json={"defaults": {"provider": "openrouter", "openrouter_upstream": "deepinfra"}},
        headers=headers,
    )

    response = client.patch(
        "/api/v1/settings", json={"defaults": {"provider": "ollama"}}, headers=headers
    )

    assert response.status_code == 200
    assert response.json()["defaults"]["openrouter_upstream"] == ""


def test_an_invalid_upstream_in_settings_is_a_422(client: TestClient) -> None:
    response = client.patch(
        "/api/v1/settings",
        json={"defaults": {"provider": "openrouter", "openrouter_upstream": "bad slug"}},
        headers=csrf_headers(client),
    )

    assert response.status_code == 422


def test_the_key_is_never_stored_or_returned_by_settings(client: TestClient) -> None:
    client.patch(
        "/api/v1/settings",
        json={"defaults": {"provider": "openrouter", "model": MODEL}},
        headers=csrf_headers(client),
    )

    body = client.get("/api/v1/settings").text
    stored = [dict(row) for row in get_database().fetchall("SELECT * FROM settings")]

    assert KEY not in body and "sk-or-v1-" not in body
    assert KEY not in json.dumps(stored)


# --- preflight ------------------------------------------------------------------------


def test_preflight_uses_the_listed_window_estimated_counts_and_never_touches_ollama(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_document(text=SHORT_TEXT)
    seen = listing(monkeypatch)

    result = preflight(client, verify=False)

    assert result["ok"] is True, result
    assert result["provider"] == "openrouter"
    assert result["model_installed"] is None
    assert (result["context_window_tokens"], result["context_window_source"]) == (131_072, "model")
    assert result["token_counts_estimated"] is True
    assert result["selected_strategy"] == "direct"
    assert (result["estimated_leaf_count"], result["estimated_model_calls"]) == (1, 2)
    assert "token_counts_estimated" in codes(result["warnings"])
    assert "third_party_processing" in codes(result["warnings"])
    assert "ollama_unreachable" not in codes(result["warnings"] + result["errors"])
    assert [request.url.path for request in seen] == ["/api/v1/models"]
    assert KEY not in json.dumps(result)


def test_the_estimate_warning_states_the_counter(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_document(text=SHORT_TEXT)
    listing(monkeypatch)

    warning = next(
        notice
        for notice in preflight(client)["warnings"]
        if notice["code"] == "token_counts_estimated"
    )

    assert "estimate:o200k_base-x1.5" in warning["message"]


def test_a_missing_key_is_a_coded_error_before_any_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_document(text=SHORT_TEXT)
    seen = listing(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY")

    result = preflight(client)

    assert result["ok"] is False
    assert "openrouter_key_missing" in codes(result["errors"])
    assert "OPENROUTER_API_KEY" in result["errors"][0]["message"]
    assert seen == []


def test_a_rejected_key_is_a_coded_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_document(text=SHORT_TEXT)
    listing(monkeypatch, httpx.Response(401, json={"error": {"code": 401, "message": "bad"}}))

    result = preflight(client)

    assert result["ok"] is False
    assert "openrouter_key_rejected" in codes(result["errors"])


def test_an_explicit_window_above_the_listed_maximum_is_invalid_config(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_document(text=SHORT_TEXT)
    listing(monkeypatch)

    result = preflight(client, context_window=200_000)

    assert result["ok"] is False
    assert codes(result["errors"]) == ["invalid_config"]
    assert "131072" in result["errors"][0]["message"]


def test_an_explicit_window_is_used_as_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_document(text=SHORT_TEXT)
    listing(monkeypatch)

    result = preflight(client, context_window=50_000)

    assert (result["context_window_tokens"], result["context_window_source"]) == (50_000, "configured")


def test_an_unusable_listing_falls_back_to_the_assumed_window_with_a_warning(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_document(text=SHORT_TEXT)
    listing(monkeypatch, httpx.Response(500, json={"error": {"code": 500, "message": "x"}}))

    result = preflight(client)

    assert result["context_window_source"] == "assumed"
    assert "context_window_assumed" in codes(result["warnings"])


def test_preflight_still_validates_the_config_for_openrouter(client: TestClient) -> None:
    seed_document(text=SHORT_TEXT)

    result = preflight(client, openrouter_upstream="bad slug")

    assert result["ok"] is False
    assert codes(result["errors"]) == ["invalid_config"]


def test_a_pinned_upstream_is_noted_because_the_window_is_model_wide(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_document(text=SHORT_TEXT)
    listing(monkeypatch)

    result = preflight(client, openrouter_upstream="deepinfra")

    assert "context_window_note" in codes(result["warnings"])


# --- creating a Run ---------------------------------------------------------------------


class _QueueOnlySupervisor:
    def enqueue(self, _run_id: str) -> None:
        pass


def test_a_run_without_a_key_is_refused_up_front(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runs_service, "get_supervisor", _QueueOnlySupervisor)
    seed_document(text=SHORT_TEXT)
    monkeypatch.delenv("OPENROUTER_API_KEY")

    response = client.post(
        "/api/v1/runs",
        json={"document_id": "doc-1", "config": {"provider": "openrouter", "model": MODEL}},
        headers=csrf_headers(client),
    )

    assert response.status_code == 409
    assert response.json()["code"] == "openrouter_key_missing"


def test_a_stored_run_config_carries_provider_and_upstream_but_never_the_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runs_service, "get_supervisor", _QueueOnlySupervisor)
    seed_document(text=SHORT_TEXT)

    response = client.post(
        "/api/v1/runs",
        json={
            "document_id": "doc-1",
            "config": {
                "provider": "openrouter",
                "model": MODEL,
                "openrouter_upstream": "deepinfra",
            },
        },
        headers=csrf_headers(client),
    )

    assert response.status_code == 201
    row = get_database().fetchone("SELECT config_json FROM runs")
    stored = json.loads(row["config_json"])
    assert (stored["provider"], stored["openrouter_upstream"]) == ("openrouter", "deepinfra")
    assert KEY not in row["config_json"] and KEY not in response.text
    assert response.json()["config"]["provider"] == "openrouter"


def test_an_ollama_run_does_not_need_an_openrouter_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runs_service, "get_supervisor", _QueueOnlySupervisor)
    seed_document(text=SHORT_TEXT)
    monkeypatch.delenv("OPENROUTER_API_KEY")

    response = client.post(
        "/api/v1/runs",
        json={"document_id": "doc-1", "config": {"model": "llama3.2:3b"}},
        headers=csrf_headers(client),
    )

    assert response.status_code == 201
    assert response.json()["config"]["provider"] == "ollama"


# --- the worker ---------------------------------------------------------------------------


def _worker_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, object]
) -> tuple[str, list]:
    from tests.support.web_runs import use_data_dir

    use_data_dir(tmp_path, monkeypatch)
    apps: list = []
    provider = ScriptedProvider(context_window=100_000)

    def build_provider(app):
        apps.append(app)
        return provider

    monkeypatch.setattr(runner, "build_provider", build_provider)
    document_id, revision_id = seed_run_document()
    run_id = seed_run(
        document_id,
        revision_id,
        {"model": MODEL, "verify": False, "citations": False, "target_words": 40, **config},
    )
    assert start_attempt(run_id) is not None
    assert runner.run_job(run_id) == "completed"
    return run_id, apps


def test_the_worker_builds_an_openrouter_app_with_the_upstream_and_no_ollama_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id, apps = _worker_run(
        tmp_path, monkeypatch, {"provider": "openrouter", "openrouter_upstream": "deepinfra"}
    )

    [app] = apps
    assert (app.provider, app.model, app.openrouter_upstream) == ("openrouter", MODEL, "deepinfra")
    assert app.ollama_host == ""
    assert run_id


def test_the_worker_still_builds_an_ollama_app_for_stored_runs_without_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, apps = _worker_run(tmp_path, monkeypatch, {"model": "llama3.2:3b"})

    [app] = apps
    assert (app.provider, app.openrouter_upstream) == ("ollama", "")
    assert app.ollama_host.startswith("http")


def test_an_openrouter_run_records_the_estimate_in_the_audit_and_the_run_notices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id, _ = _worker_run(
        tmp_path, monkeypatch, {"provider": "openrouter", "openrouter_upstream": "deepinfra"}
    )

    audit = json.loads((load_paths().runs / run_id / "audit.json").read_text(encoding="utf-8"))

    assert audit["configuration"]["app"]["provider"] == "openrouter"
    assert audit["configuration"]["app"]["openrouter_upstream"] == "deepinfra"
    assert audit["configuration"]["budget"]["counter_exact"] is False
    assert "token_counts_estimated" in audit["warnings"]
    assert KEY not in json.dumps(audit)


def test_the_run_summary_view_shows_the_estimate_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from summarizer_web.main import create_app

    run_id, _ = _worker_run(tmp_path, monkeypatch, {"provider": "openrouter"})

    with TestClient(create_app()) as client:
        notices = client.get(f"/api/v1/runs/{run_id}/summary").json()["notices"]

    estimate = [notice for notice in notices if notice["code"] == "token_counts_estimated"]
    assert estimate and "estimate" in estimate[0]["message"]


def test_an_ollama_run_adds_no_estimate_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id, _ = _worker_run(tmp_path, monkeypatch, {"model": "llama3.2:3b"})

    audit = json.loads((load_paths().runs / run_id / "audit.json").read_text(encoding="utf-8"))

    assert "token_counts_estimated" not in audit["warnings"]
    assert "openrouter_upstream" not in audit["configuration"]["app"]


# --- failure classification --------------------------------------------------------------


def _classify(error: BaseException, provider: str = "openrouter"):
    return classify_failure(
        error,
        host="",
        model=MODEL,
        timeout_seconds=60,
        stage="summarizing",
        item="Segment 1",
        provider=provider,
    )


def test_a_connection_failure_names_openrouter_not_ollama() -> None:
    failure = _classify(ProviderConnectionError("OpenRouter connection failed"))

    assert failure.code == "openrouter_unreachable"
    assert "Ollama" not in failure.message + (failure.hint or "")


def test_a_rejected_key_is_classified_with_a_hint_to_fix_the_environment() -> None:
    failure = _classify(ProviderAuthenticationError("OpenRouter authentication failed"))

    assert failure.code == "openrouter_key_invalid"
    assert "OPENROUTER_API_KEY" in (failure.hint or "")


def test_other_openrouter_errors_keep_the_provider_message() -> None:
    failure = _classify(ProviderRequestError("OpenRouter account has insufficient credits"))

    assert failure.code == "provider_error"
    assert "insufficient credits" in failure.message


def test_ollama_classification_is_unchanged_by_the_provider_argument() -> None:
    failure = _classify(ProviderConnectionError("down"), provider="ollama")

    assert failure.code == "ollama_unreachable"


def test_a_failure_record_never_contains_the_key() -> None:
    error = ProviderRequestError(f"rejected (status 400) with {'x' * 5}")
    failure = _classify(error)

    assert KEY not in failure.model_dump_json()


# --- real provider, failing wire -------------------------------------------------------------


def _failing_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handler,
    capfd: pytest.CaptureFixture[str],
) -> tuple[dict, str]:
    from tests.support.web_runs import use_data_dir

    use_data_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "build_provider",
        lambda app: OpenRouterProvider(
            upstream=app.openrouter_upstream,
            client_factory=lambda **kwargs: httpx.Client(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        ),
    )
    document_id, revision_id = seed_run_document()
    run_id = seed_run(
        document_id,
        revision_id,
        {"provider": "openrouter", "model": MODEL, "verify": False, "citations": False,
         "context_window": 100_000, "max_retries": 1},
    )
    assert start_attempt(run_id) is not None
    assert runner.run_job(run_id) == "failed"
    attempt = get_database().fetchone("SELECT * FROM run_attempts WHERE run_id = ?", (run_id,))
    return json.loads(attempt["failure_json"]), capfd.readouterr().err


@pytest.mark.parametrize(
    ("handler", "code"),
    [
        (lambda request: httpx.Response(401, json={"error": {"code": 401, "message": f"bad {KEY}"}}),
         "openrouter_key_invalid"),
        (lambda request: (_ for _ in ()).throw(httpx.ConnectError(f"refused {KEY}")),
         "openrouter_unreachable"),
        (lambda request: httpx.Response(402, json={"error": {"code": 402, "message": f"x {KEY}",
                                                              "metadata": {"limit_source": "openrouter_key_limit"}}}),
         "provider_error"),
        (lambda request: httpx.Response(400, json={"error": {"code": 400, "message": f"x {KEY}"}}),
         "provider_error"),
    ],
    ids=["401", "connect", "402", "400"],
)
def test_a_failed_openrouter_run_is_classified_without_the_key_anywhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    handler,
    code: str,
) -> None:
    failure, stderr = _failing_run(tmp_path, monkeypatch, handler, capfd)

    assert failure["code"] == code
    stored = " ".join(
        str(value)
        for table in ("runs", "run_attempts", "run_events")
        for row in get_database().fetchall(f"SELECT * FROM {table}")
        for value in dict(row).values()
    )
    on_disk = " ".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in load_paths().runs.rglob("*")
        if path.is_file()
    )
    for place in (json.dumps(failure), stderr, stored, on_disk):
        assert KEY not in place and "sk-or-v1-" not in place


# --- adversarial review findings ----------------------------------------------------------------


def test_a_malformed_key_is_reported_as_unusable_not_as_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "bad key with spaces")

    status = client.get("/api/v1/openrouter/health").json()

    assert status["key_set"] is False
    assert "cannot be used" in status["message"]
    assert "bad key" not in json.dumps(status)


def test_preflight_and_run_creation_refuse_a_malformed_key_with_the_same_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runs_service, "get_supervisor", _QueueOnlySupervisor)
    seed_document(text=SHORT_TEXT)
    monkeypatch.setenv("OPENROUTER_API_KEY", "bad key with spaces")

    result = preflight(client)
    created = client.post(
        "/api/v1/runs",
        json={"document_id": "doc-1", "config": {"provider": "openrouter", "model": MODEL}},
        headers=csrf_headers(client),
    )

    assert codes(result["errors"]) == ["openrouter_key_missing"]
    assert "cannot be used" in result["errors"][0]["message"]
    assert created.status_code == 409 and "cannot be used" in created.json()["message"]


@pytest.mark.parametrize("model", ["a/b/c/d", "a:b:c", "has space inside", "m" * 129])
def test_an_openrouter_model_the_audit_could_not_record_is_refused_up_front(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setattr(runs_service, "get_supervisor", _QueueOnlySupervisor)
    seed_document(text=SHORT_TEXT)

    result = preflight(client, model=model)
    created = client.post(
        "/api/v1/runs",
        json={"document_id": "doc-1", "config": {"provider": "openrouter", "model": model}},
        headers=csrf_headers(client),
    )

    assert result["ok"] is False and codes(result["errors"]) == ["invalid_config"]
    assert created.status_code == 422


def test_an_openrouter_model_is_trimmed_before_it_is_stored(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runs_service, "get_supervisor", _QueueOnlySupervisor)
    seed_document(text=SHORT_TEXT)

    response = client.post(
        "/api/v1/runs",
        json={"document_id": "doc-1", "config": {"provider": "openrouter", "model": f"  {MODEL}\n"}},
        headers=csrf_headers(client),
    )

    assert response.status_code == 201 and response.json()["config"]["model"] == MODEL


def test_a_not_found_message_on_an_openrouter_run_is_not_a_missing_ollama_model() -> None:
    failure = _classify(ProviderRequestError("OpenRouter rejected: model not found"))

    assert failure.code == "provider_error"
    assert _classify(ProviderRequestError("model not found"), provider="ollama").code == "model_not_found"


def test_the_failure_message_is_scrubbed_even_for_an_unclassified_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    def explode(request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"unexpected {KEY} in an error")

    failure, stderr = _failing_run(tmp_path, monkeypatch, explode, capfd)

    assert failure["code"] == "internal_error"
    assert KEY not in json.dumps(failure) and KEY not in stderr
    assert "[REDACTED]" in failure["message"]


def test_the_scaled_estimate_uses_its_factor_per_output_token_in_preflight() -> None:
    from summarizer.tokenization import ScaledTiktokenCounter
    from summarizer_web.services.preflight_service import _output_units_per_token

    assert _output_units_per_token(ScaledTiktokenCounter(_FakeEncoding(), 1.5)) == 2
    assert _output_units_per_token(ScaledTiktokenCounter(_FakeEncoding(), 1.0)) == 1


def test_an_estimate_counter_never_uses_the_ollama_schema_reserve(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same document, window and limits must size merges without the Ollama
    # decoder's schema reserve, which a hosted adapter does not send.
    from summarizer_web.services import preflight_service

    seen: list[int] = []
    original = preflight_service.plan_merge_request

    def spy(limits, *, level, provider_schema_reserve):
        seen.append(provider_schema_reserve)
        return original(limits, level=level, provider_schema_reserve=provider_schema_reserve)

    monkeypatch.setattr(preflight_service, "plan_merge_request", spy)
    long_text = "\n\n".join(f"Paragraph {i}. " + "The pier was inspected carefully. " * 40 for i in range(120))
    seed_document(text=long_text)
    listing(monkeypatch)

    result = preflight(client, strategy="hierarchical", chunk_tokens=600, context_window=60_000)

    assert result["estimated_leaf_count"] > 1
    assert seen and set(seen) == {0}
