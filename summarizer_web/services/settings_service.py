"""Persistent application settings."""

from __future__ import annotations

import json

from summarizer_web.db.connection import get_database
from summarizer_web.models.api import RunConfig, SettingsResponse, SettingsUpdate

_DEFAULTS = SettingsResponse(
    ollama_host="http://localhost:11434",
    defaults=RunConfig(model=""),
)


def get_settings() -> SettingsResponse:
    db = get_database()
    host_row = db.fetchone("SELECT value_json FROM settings WHERE key = ?", ("ollama_host",))
    defaults_row = db.fetchone("SELECT value_json FROM settings WHERE key = ?", ("defaults",))
    ollama_host = _DEFAULTS.ollama_host
    defaults = _DEFAULTS.defaults
    if host_row is not None:
        ollama_host = json.loads(host_row["value_json"])
    if defaults_row is not None:
        defaults = RunConfig.model_validate_json(defaults_row["value_json"])
    return SettingsResponse(ollama_host=ollama_host, defaults=defaults)


def update_settings(payload: SettingsUpdate) -> SettingsResponse:
    current = get_settings()
    ollama_host = payload.ollama_host if payload.ollama_host is not None else current.ollama_host
    defaults = payload.defaults if payload.defaults is not None else current.defaults
    db = get_database()
    db.execute(
        "INSERT INTO settings (key, value_json) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
        ("ollama_host", json.dumps(ollama_host)),
    )
    db.execute(
        "INSERT INTO settings (key, value_json) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
        ("defaults", defaults.model_dump_json()),
    )
    return SettingsResponse(ollama_host=ollama_host, defaults=defaults)
