from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

# Identifies the record shape for cache keys and audit artifacts. Bump it
# whenever a change would make a previously stored record invalid or mean
# something different.
LEAF_SCHEMA_VERSION = "leaf/3"

# A quote is a pull-quote, not a transcription. Both caps bound how much a
# single node's quotations can inflate `serialize_child` (summarizer/merge.py)
# and trip the merge budget check. The per-quote cap applies everywhere an
# `EvidenceItem.quote` appears, since any one of them serializes into the
# node; the per-node cap applies only to `SummaryNode.quotations`, the
# "salient quotations" the master plan bounds. Shared here so the leaf and
# merge prompts state the same limit the validators enforce.
MAX_QUOTE_CHARS = 500
MAX_QUOTATIONS_PER_NODE = 5


class ContentKind(str, Enum):
    """What kind of thing a content unit asserts.

    Deliberately small and genre-neutral: the same vocabulary has to describe
    an article, a transcript, a report, and a narrative.
    """

    FACT = "fact"
    CLAIM = "claim"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    EXAMPLE = "example"
    OTHER = "other"


class _Record(BaseModel):
    """Base for every structured record built from model output.

    Two settings are load-bearing rather than stylistic. `extra="forbid"` is
    what makes the generated schema emit `additionalProperties: false`, and
    declaring fields without defaults is what makes the generated schema mark
    them required. Together they make `model_json_schema()` directly usable as
    an OpenAI strict schema, so the schema sent to a provider cannot drift from
    the validator that checks the response. Do not add field defaults here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _empty_when_null(value: Any) -> Any:
    """Treat a null collection as an absent one.

    Only the strict OpenAI path guarantees a present, non-null key. Ollama's
    format argument constrains decoding on a best-effort basis, so a model with
    nothing to say about a field may emit null instead of an empty array.
    """
    return () if value is None else value


class EvidenceItem(_Record):
    """A pointer from an assertion back to the segment that supports it."""

    segment_id: str
    quote: str | None

    @field_validator("segment_id", mode="before")
    @classmethod
    def _clean_segment_id(cls, value: Any) -> str:
        if isinstance(value, str):
            value = value.strip()
        if not isinstance(value, str) or not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("quote", mode="before")
    @classmethod
    def _validate_quote(cls, value: str | None) -> str | None:
        """Absent means null, not empty. Also enforces a length cap.

        Normalizes blank or whitespace-only quotes to null (None) so that models
        emitting empty string quotes (e.g. "") pass validation cleanly as null.
        The length cap bounds one quote's contribution to its node's serialized size.
        """
        if value is not None:
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    return None
                if len(cleaned) > MAX_QUOTE_CHARS:
                    raise ValueError(f"quote must not exceed {MAX_QUOTE_CHARS} characters")
                return cleaned
            if not str(value).strip():
                return None
        return value


class ContentUnit(_Record):
    """One assertion carried forward for later merging."""

    text: str
    kind: ContentKind
    evidence: tuple[EvidenceItem, ...]
    qualification: str | None
    uncertain: bool

    _reject_blank_text = field_validator("text")(_reject_blank)
    _empty_evidence = field_validator("evidence", mode="before")(_empty_when_null)

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, value: Any) -> ContentKind:
        if isinstance(value, str):
            normalized = value.strip().lower()
            try:
                return ContentKind(normalized)
            except ValueError:
                return ContentKind.OTHER
        if isinstance(value, ContentKind):
            return value
        return ContentKind.OTHER

    @field_validator("qualification", mode="before")
    @classmethod
    def _normalize_qualification(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned if cleaned else None
        return str(value).strip() if str(value).strip() else None


class GroundedAnnotation(_Record):
    """A retained qualification or contradiction with explicit support."""

    text: str
    evidence: tuple[EvidenceItem, ...]

    _reject_blank_text = field_validator("text")(_reject_blank)
    _empty_evidence = field_validator("evidence", mode="before")(_empty_when_null)


class SummaryNode(_Record):
    """A summary of one region of the source.

    `level` is zero for a leaf. Later merge levels reuse this record rather
    than introducing a parallel one, so the hierarchy has a single central
    type.
    """

    summary: str
    content_units: tuple[ContentUnit, ...]
    entities: tuple[str, ...]
    qualifications: tuple[GroundedAnnotation, ...]
    contradictions: tuple[GroundedAnnotation, ...]
    quotations: tuple[EvidenceItem, ...]
    provenance: tuple[str, ...]
    level: int

    _reject_blank_summary = field_validator("summary")(_reject_blank)
    _empty_collections = field_validator(
        "content_units",
        "entities",
        "qualifications",
        "contradictions",
        "quotations",
        "provenance",
        mode="before",
    )(_empty_when_null)

    @field_validator("level")
    @classmethod
    def _reject_negative_level(cls, value: int) -> int:
        if value < 0:
            raise ValueError("level must not be negative")
        return value

    @field_validator("quotations")
    @classmethod
    def _validate_quotation_count(cls, value: tuple[EvidenceItem, ...]) -> tuple[EvidenceItem, ...]:
        if len(value) > MAX_QUOTATIONS_PER_NODE:
            raise ValueError(f"must not exceed {MAX_QUOTATIONS_PER_NODE} quotations")
        return value


# A verbatim quotation is the one field a schema alone cannot describe: its
# legal values are spans of the source, and a model asked to copy one has no
# way to be held to it. Enumerating the spans a response may quote turns that
# into a decodable constraint. The cap is the maximum bytes added to the
# compact ASCII-escaped schema serialization used for request measurement. A
# byte-fallback tokenizer cannot emit more tokens than input bytes. 256 bytes
# hold one maximum-length ASCII candidate plus the enum syntax without making
# small context windows reserve source-sized schema space.
MAX_QUOTE_CANDIDATE_JSON_BYTES = 256

# Ollama adds local-only nonempty and leaf-identity constraints after request
# construction. This cap keeps that deterministic adapter expansion inside the
# same context guarantee; the adapter test pins its serialized delta below it.
MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES = 1_024

# `"enum":[],` is the fixed compact-schema cost before candidate strings.
_QUOTE_ENUM_JSON_BYTES = len(b'"enum":[],')

# Shorter spans carry little evidence; longer spans duplicate too much source
# text in the request schema. Evidence records still permit longer quotes when
# parsed from providers that do not use the candidate-constrained schema.
MIN_QUOTE_CANDIDATE_CHARS = 20
MAX_QUOTE_CANDIDATE_CHARS = 240

# Sentence terminators accepted as candidate boundaries. Scanning boundaries
# once avoids retrying every suffix of a long unpunctuated source.
_QUOTE_TERMINATOR = re.compile(r"[.!?](?=\s|\Z)")


def _quote_spans(text: str) -> Iterator[str]:
    start = 0
    for terminal in _QUOTE_TERMINATOR.finditer(text):
        span = text[start : terminal.end()].strip()
        if span:
            yield span
        start = terminal.end()
    tail = text[start:].strip()
    if tail:
        yield tail


def _fit_quote_candidate(
    span: str,
    *,
    available_json_bytes: int,
    separator_bytes: int,
    min_chars: int,
) -> tuple[str, int] | None:
    """Return the longest exact prefix that fits the remaining enum budget."""
    bounded = span[:MAX_QUOTE_CANDIDATE_CHARS].rstrip()
    if len(bounded) < min_chars:
        return None
    low = min_chars
    high = len(bounded)
    fitted: tuple[str, int] | None = None
    while low <= high:
        length = (low + high) // 2
        candidate = bounded[:length].rstrip()
        cost = len(json.dumps(candidate).encode("utf-8")) + separator_bytes
        if len(candidate) >= min_chars and cost <= available_json_bytes:
            fitted = (candidate, cost)
            low = length + 1
        else:
            high = length - 1
    return fitted


def quote_candidates(
    sources: Sequence[str],
    *,
    preferred: Sequence[str] = (),
    budget_json_bytes: int = MAX_QUOTE_CANDIDATE_JSON_BYTES,
) -> tuple[str, ...]:
    """Enumerate exact source prefixes within a compact-schema byte budget."""
    candidates: list[str] = []
    seen: set[str] = set()
    remaining = budget_json_bytes - _QUOTE_ENUM_JSON_BYTES

    def add(span: str, *, min_chars: int, truncate: bool) -> None:
        nonlocal remaining
        if truncate:
            fitted = _fit_quote_candidate(
                span,
                available_json_bytes=remaining,
                separator_bytes=1 if candidates else 0,
                min_chars=min_chars,
            )
        else:
            candidate = span.rstrip()
            cost = len(json.dumps(candidate).encode("utf-8")) + (
                1 if candidates else 0
            )
            fitted = (candidate, cost) if cost <= remaining else None
        if fitted is None:
            return
        candidate, cost = fitted
        if candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)
        remaining -= cost

    for quote in preferred:
        add(quote, min_chars=1, truncate=False)
    for text in sources:
        for span in _quote_spans(text):
            add(span, min_chars=MIN_QUOTE_CANDIDATE_CHARS, truncate=True)
    return tuple(candidates)


def summary_schema(
    *, candidates: Sequence[str] | None = None
) -> dict[str, Any]:
    """Return a summary schema, optionally narrowed to authoritative quotes."""
    schema = SummaryNode.model_json_schema()
    if candidates is None:
        return schema
    quote = schema["$defs"]["EvidenceItem"]["properties"]["quote"]
    if not candidates:
        quote["anyOf"] = [
            variant for variant in quote["anyOf"] if variant.get("type") == "null"
        ]
        return schema
    for variant in quote["anyOf"]:
        if variant.get("type") == "string":
            variant["enum"] = list(candidates)
    return schema


def leaf_summary_schema() -> dict[str, Any]:
    """Return the unconstrained summary schema."""
    return summary_schema()
