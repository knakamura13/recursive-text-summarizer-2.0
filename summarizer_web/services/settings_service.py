"""Persistent application settings: the Ollama host and the default run configuration.

Settings live in the ``settings`` table as JSON values under the keys
``ollama_host`` and ``defaults``. Stored values are read tolerantly, so a row
written by an older version (unknown or no longer valid fields) never breaks
the Settings page: unknown fields are ignored and invalid ones fall back to
their defaults.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from summarizer_web.db.connection import get_database
from summarizer_web.errors import ApiError
from summarizer_web.models.api import (
    RunConfig,
    RunConfigPatch,
    SettingsResponse,
    SettingsUpdate,
)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_HOST_KEY = "ollama_host"
_DEFAULTS_KEY = "defaults"
_UPSERT = (
    "INSERT INTO settings (key, value_json) VALUES (?, ?) "
    "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json"
)


def get_settings() -> SettingsResponse:
    rows = get_database().fetchall(
        "SELECT key, value_json FROM settings WHERE key IN (?, ?)",
        (_HOST_KEY, _DEFAULTS_KEY),
    )
    return _settings_from_rows(rows)


def update_settings(payload: SettingsUpdate) -> SettingsResponse:
    """Apply a partial update atomically and return the stored result.

    ``ollama_host`` must be an http(s) URL. ``defaults`` is merged into the
    stored defaults: omitted or null fields keep their values and ``clear``
    resets nullable fields to null. The merged configuration must validate.
    """
    host = None if payload.ollama_host is None else normalize_ollama_host(payload.ollama_host)
    with get_database().transaction() as connection:
        current = _settings_from_rows(
            connection.execute(
                "SELECT key, value_json FROM settings WHERE key IN (?, ?)",
                (_HOST_KEY, _DEFAULTS_KEY),
            ).fetchall()
        )
        updated = SettingsResponse(
            ollama_host=host if host is not None else current.ollama_host,
            defaults=(
                current.defaults
                if payload.defaults is None
                else merge_run_config(current.defaults, payload.defaults)
            ),
        )
        connection.executemany(
            _UPSERT,
            (
                (_HOST_KEY, json.dumps(updated.ollama_host)),
                (_DEFAULTS_KEY, updated.defaults.model_dump_json()),
            ),
        )
    return updated


def normalize_ollama_host(value: str) -> str:
    """Return the host without a trailing slash, or raise 422 unless it is an http(s) URL."""
    candidate = value.strip()
    parsed = urlsplit(candidate)
    try:
        port_ok = parsed.port is None or parsed.port > 0
    except ValueError:
        port_ok = False
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or not port_ok
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in candidate)
    ):
        raise ApiError(
            422,
            "invalid_request",
            "The Ollama host must be an http:// or https:// URL, "
            "for example http://localhost:11434.",
            details={"field": "ollama_host"},
        )
    return candidate.rstrip("/")


def merge_run_config(current: RunConfig, patch: RunConfigPatch) -> RunConfig:
    """Merge a patch into a configuration; raise 422 when the result is invalid."""
    updates = patch.model_dump(exclude={"clear"}, exclude_none=True)
    conflicting = sorted(set(updates) & set(patch.clear))
    if conflicting:
        raise ApiError(
            422,
            "invalid_request",
            f"{', '.join(conflicting)} cannot be set and cleared in the same update.",
            details={"fields": conflicting},
        )
    merged = {
        **current.model_dump(),
        **updates,
        **{field: None for field in patch.clear},
    }
    try:
        return RunConfig.model_validate(merged)
    except ValidationError as error:
        raise ApiError(
            422,
            "invalid_request",
            "The default run configuration is invalid: " + _validation_summary(error),
            details={"errors": _validation_details(error)},
        ) from error


def _settings_from_rows(rows: list[sqlite3.Row]) -> SettingsResponse:
    stored = {row["key"]: row["value_json"] for row in rows}
    return SettingsResponse(
        ollama_host=_stored_host(stored.get(_HOST_KEY)),
        defaults=_stored_defaults(stored.get(_DEFAULTS_KEY)),
    )


def _stored_host(raw: str | None) -> str:
    if raw is None:
        return DEFAULT_OLLAMA_HOST
    try:
        value = json.loads(raw)
        return normalize_ollama_host(value) if isinstance(value, str) else DEFAULT_OLLAMA_HOST
    except (ValueError, ApiError):
        return DEFAULT_OLLAMA_HOST


def _stored_defaults(raw: str | None) -> RunConfig:
    if raw is None:
        return RunConfig()
    try:
        value = json.loads(raw)
    except ValueError:
        return RunConfig()
    if not isinstance(value, dict):
        return RunConfig()
    candidate = {key: item for key, item in value.items() if key in RunConfig.model_fields}
    # Each round drops the fields that failed; a failure no field owns (a
    # cross-field rule) or a clean validation ends the loop.
    while True:
        try:
            return RunConfig.model_validate(candidate)
        except ValidationError as error:
            invalid = {
                str(item["loc"][0]) for item in error.errors() if item.get("loc")
            } & candidate.keys()
            if not invalid:
                return RunConfig()
            for key in invalid:
                del candidate[key]


def _validation_summary(error: ValidationError) -> str:
    parts = []
    for item in error.errors()[:5]:
        message = str(item.get("msg", "invalid")).removeprefix("Value error, ")
        location = ".".join(str(part) for part in item.get("loc", ()))
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "invalid value"


def _validation_details(error: ValidationError) -> list[dict[str, Any]]:
    return [
        {
            "field": ".".join(str(part) for part in item.get("loc", ())) or None,
            "message": str(item.get("msg", "invalid")).removeprefix("Value error, "),
        }
        for item in error.errors()
    ]
