"""Small, deterministic safeguards for values that may be retained or shown."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any


_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|secret|password|token|authorization|credential|auth)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{36,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"(?i)\bauthorization\s*:\s*basic\s+[^\s,;]+"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"://[^\s/@:]+:[^\s/@]+@"),
)


def redact_text(value: str) -> str:
    """Replace common credential forms without changing unrelated prose."""
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def safe_json_value(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-compatible, recursively redacted value.

    Audit artifacts intentionally have no arbitrary object encoder. Converting
    paths and enums here keeps their representation stable while avoiding a
    later serialization fallback that could accidentally expose credentials.
    """
    if key is not None and _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(item_key): safe_json_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [safe_json_value(item) for item in value]
    if isinstance(value, Enum):
        return safe_json_value(value.value, key=key)
    if isinstance(value, Path):
        return redact_text(str(value))
    if isinstance(value, str):
        return redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value))
