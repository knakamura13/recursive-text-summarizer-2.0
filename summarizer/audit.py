"""Versioned, secret-safe audit artifacts for completed summary runs.

An audit records structure, identifiers, verdicts, and usage, never request
prompts or provider responses. Reader-facing prose appears only under
`publication`: published sentences repeat the credential-redacted summary,
and quotations and removed sentences are credential-redacted on the way in.

JSON paths read by the web application (offsets are code points, i.e. Python
`str` indices):

- `source_segments[]`: every source segment and verification passage, with
  `segment_id`, `order`, and `core_start`/`core_end` plus
  `context_start`/`context_end` into the canonical document text. When a
  segment's core cannot fit one evidence bundle, verification splits it into
  passages numbered after the last leaf segment (`S000087`, `S000088`, ...;
  `S000001`, ... for the direct strategy's `D000001`). A passage carries
  `parent_segment_id`, the leaf segment it lies in; leaf segments have no
  `parent_segment_id` key, and only leaf segments appear in `tree_nodes`.
- `publication`, absent when nothing was published and in older audits:
  - `kind`: `editorial` (the editorial draft passed verification, or
    verification was off), `verified_subset` (the editorial draft with its
    failing sentences removed; warning `verified_sentence_subset`), or
    `content_unit_fallback` (verified root content units; warning
    `verified_content_unit_fallback`).
  - `sentences[]`: the published text in order, with `index`, `paragraph`
    (0-based, blank-line separated), `start`/`end` into the published summary
    (`summary.txt` without its `Sources:` trailer), `text`, `verdict`
    (`supported`, `not_meaningfully_verifiable`, or `unchecked` when
    verification was off), and `evidence[]` with `segment_id` (a
    `source_segments[]` id), the verifier's `quote`, and `start`/`end` of that
    quote in the canonical document text (null when it cannot be located).
  - Sentence evidence is the direct mapping for each published sentence: each
    entry comes from a supported verifier finding for that sentence; its
    `segment_id` joins to `source_segments[]`, and its quote offsets locate the
    source text. WebViews can render evidence from this field without trying to
    reconstruct sentence-to-claim links.
  - `removed_sentences[]`: sentences dropped from the published candidate, in
    the order verification first saw them, with `text`, `verdict`
    (`contradicted`, `insufficiently_supported`, `not_meaningfully_verifiable`,
    `unverified`, or `unfinished` for a last draft sentence that stops before
    its end and is never verified), and a readable `reason`.
  - `substitutions[]`, absent when none was proposed: with `strict_numbers`
    on, each rejected draft sentence swapped for a source sentence, with
    `original_text`, `original_verdict`, `replacement_text`, the verifier's
    `replacement_verdict` from a pass over the swapped draft (`unverified` when
    that pass could not run), and `action` (`replaced` only when supported and
    published, else `removed`).
- `sections[]`, present only when the run summarized by section (the schema
  version is unchanged: a run without section mode writes no `sections` key,
  so its audit is byte-identical to before). One record per section of the
  tree, in document order: `section_id` (`s1`, `s2`, ...), `heading` (as the
  import stored it; null for the untitled opening section), `level`,
  `page_start`/`page_end` (the section's own text; null without a page map),
  `parent_id`, `child_ids`, `folded_headings` (headings of undersized sections
  merged into it), `segment_ids` (the leaf segments of its own text),
  `node_id` (the tree node summarizing its whole subtree, null when it has no
  text and no summarized subsection) and `publication`, absent without a node
  unless the section is heading-only. A section publication is extra output
  beside the root `publication`, written from the section's own text only:
  `status` (`verified`, `unverified` when verification was off, `empty` with a
  `reason`, or `heading_only` with a `reason` when the section has no own text
  or under one sentence's share of the target, so its heading stands alone and
  no model was called), `kind`, `sentences[]` and `removed_sentences[]` as
  above but checked only against the section's own segments, `citations[]`
  (`segment_id`, `order`, `page_start`, `page_end`), `segment_ids` (the
  section's own segments, the only evidence it could cite), `page_start`/
  `page_end` of the own text, `target_words` (the own share of the run target
  by words, headings excluded, at least one sentence) and `words`. Source
  prose never appears here: a
  heading is the only source text, and sentence quotations are redacted as in
  `publication`.
- `warnings[]`: `verified_sentence_subset`, `verified_content_unit_fallback`,
  or, in a failure audit written without a publication,
  `verified_content_unit_fallback_failed`.
- `verification.passes[]`: the claim verdicts (`assessments[].claim_id`, the
  work id of the claim item events) and evidence identifiers of every
  completed check, ending with the check that produced the published text. An assessment
  may carry `reassessment`: the claim's first verdict (`original_verdict`,
  `original_findings`) before the verifier looked again because its only
  flagged difference was an allowed `tolerance` (`rounded_number` or
  `shortened_name`); the assessment's own verdict is that second look.
  `selections[].retrieval_method` is one of the methods verification defines.
  A pass may carry `unresolved[]`, absent when empty: each span
  (`phase` `decomposition`) or claim (`phase` `classification`) the verifier
  never finished after its bounded re-asks, by `item_id`, with `reason`
  `omitted`, `invalid_response`, or `capacity`. Such a pass is never
  `complete`, and its unresolved sentences are never published. When a
  verified subset publishes, its last pass keeps the unresolved entries for
  the spans and claims it withheld, which no longer appear in that pass.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)

from summarizer.hierarchy import TreeNode
from summarizer.providers.base import GenerationResult
from summarizer.safety import redact_text
from summarizer.sections import SectionTree
from summarizer.segmentation import SourceSegment
from summarizer.verification import RETRIEVAL_METHODS, ClaimVerdict

if TYPE_CHECKING:
    from summarizer.verification import VerificationResult


AUDIT_SCHEMA_VERSION = "audit/2"
AUDIT_SCHEMA_VERSION_V3 = "audit/3"
AUDIT_SCHEMA_VERSION_V4 = "audit/4"


class AuditError(ValueError):
    """An audit record is incomplete, inconsistent, or cannot be written."""


_CLOSED_CODE = re.compile(r"^[a-z][a-z0-9_]*$")
_AUDIT_WORK_ID = re.compile(
    r"^(?:[DS]\d{6}|L\d+N\d{4}|M(?:\d{6}|\d+N\d{4})|V\d{2}(?:[CS]\d{6})?|"
    r"editorial-final|segmentation|"
    r"Qs[1-9][0-9]*-(?:L\d+N\d{4}|editorial|V\d{2}|T?C\d{2}K\d{6}))$"
)
_VERIFICATION_SPAN_ID = re.compile(r"^V\d{2}S\d{6}$")
_VERIFICATION_CLAIM_ID = re.compile(r"^V\d{2}C\d{6}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEGMENT_ID = re.compile(r"^[DS]\d{6}$")
_NODE_ID = re.compile(r"^L\d+N\d{4}$")
_SECTION_ID = re.compile(r"^s[1-9][0-9]*$")
_SAFE_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_MODEL_IDENTITY = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}(?::[A-Za-z0-9][A-Za-z0-9._-]{0,127})?$"
)
_VERIFICATION_PROMPT_VERSION = frozenset(
    {
        "verification-decomposition/1",
        "verification-decomposition/2",
        "verification-decomposition/3",
        "verification-classification/1",
        "verification-classification/2",
        "verification-classification/3",
        "verification-classification/4",
        "verification-classification/5",
        "verification-classification/6",
        "verification-classification/7",
        "verification-reassessment/1",
        "verification-repair/1",
    }
)
_FINISH_STATUSES = frozenset(
    {
        "completed",
        "stop",
        "length",
        "content_filter",
        "tool_calls",
        "cancelled",
        "error",
        "unknown",
    }
)
_CACHE_HIT_CODES = frozenset({"hit"})
_CACHE_MISS_CODES = frozenset({"missing", "corrupt", "wrong_version", "incompatible"})
_CACHE_INVALIDATION_CODES = frozenset(
    {
        "source_changed",
        "prompt_changed",
        "schema_changed",
        "model_changed",
        "behavior_changed",
    }
)
_RETRY_FAILURE_CODES = frozenset(
    {"timeout", "rate_limit", "connection", "server", "transient"}
)


def _audit_identity(value: str) -> str:
    if _SAFE_IDENTITY.fullmatch(value) and redact_text(value) == value:
        return value
    raise ValueError("audit identity must be a non-secret identifier")


def _audit_optional_identity(value: str | None) -> str | None:
    return None if value is None else _audit_identity(value)


def _audit_model_identity(value: str) -> str:
    if _SAFE_MODEL_IDENTITY.fullmatch(value) and redact_text(value) == value:
        return value
    raise ValueError("audit model must be a non-secret model identity")


def _audit_optional_model_identity(value: str | None) -> str | None:
    return None if value is None else _audit_model_identity(value)


def _audit_finish_status(value: str | None) -> str | None:
    if value is None or value in _FINISH_STATUSES:
        return value
    raise ValueError("audit finish_status must be a closed status")


def _audit_prompt_version(value: str) -> str:
    if value in _VERIFICATION_PROMPT_VERSION:
        return value
    raise ValueError("audit prompt_version must be a supported version")


def _audit_source_id(value: str) -> str:
    if _SHA256.fullmatch(value):
        return value
    raise ValueError("audit source_id must be a sha256 identity")


def _audit_segment_id(value: str) -> str:
    if _SEGMENT_ID.fullmatch(value):
        return value
    raise ValueError("audit segment_id must be a segment identity")


def _audit_segment_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_audit_segment_id(value) for value in values)


def _audit_optional_segment_id(value: str | None) -> str | None:
    return None if value is None else _audit_segment_id(value)


def _audit_prose(value: object) -> str:
    """Keep reader-facing text with any credential-like substring replaced."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("audit publication text must be nonblank text")
    return redact_text(value)


def _audit_published_prose(value: object) -> str:
    """Accept published summary text only when it is already credential-free."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("audit publication text must be nonblank text")
    if redact_text(value) != value:
        raise ValueError("published summary text must be credential-redacted")
    return value


def _audit_node_id(value: str) -> str:
    if _NODE_ID.fullmatch(value):
        return value
    raise ValueError("audit node_id must be a tree identity")


def _audit_node_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_audit_node_id(value) for value in values)


def _audit_closed_codes(
    values: object, *, allowed: frozenset[str], label: str
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(value, str) or value not in allowed for value in values
    ):
        raise ValueError(f"{label} must contain supported closed codes")
    return tuple(values)


def _audit_work_id(value: str) -> str:
    if _AUDIT_WORK_ID.fullmatch(value):
        return value
    raise ValueError("audit work_id must be a safe stable identifier")


def _audit_integer(value: object, *, minimum: int, label: str) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be a non-boolean integer of at least {minimum}")
    return value


class _AuditRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AuditSegment(_AuditRecord):
    """A source segment, or a verification passage split from one.

    A passage names the leaf segment it lies in with `parent_segment_id`;
    the key is omitted for leaf segments.
    """

    segment_id: str
    source_id: str
    order: int
    core_start: int
    core_end: int
    context_start: int
    context_end: int
    core_token_count: int
    token_count: int
    leading_overlap_tokens: int
    trailing_overlap_tokens: int
    boundary_kind: Literal[
        "heading", "paragraph", "list", "sentence", "hard", "document", "code_fence"
    ]
    parent_segment_id: str | None = None

    _valid_segment_id = field_validator("segment_id")(_audit_segment_id)
    _valid_source_id = field_validator("source_id")(_audit_source_id)
    _valid_parent = field_validator("parent_segment_id")(_audit_optional_segment_id)

    @model_serializer(mode="wrap")
    def _omit_absent_parent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if data.get("parent_segment_id") is None:
            data.pop("parent_segment_id", None)
        return data


class AuditCitation(_AuditRecord):
    segment_id: str
    source_id: str
    order: int

    _valid_segment_id = field_validator("segment_id")(_audit_segment_id)
    _valid_source_id = field_validator("source_id")(_audit_source_id)


class AuditUsage(_AuditRecord):
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    finish_status: str | None

    _valid_provider = field_validator("provider")(_audit_identity)
    _valid_model = field_validator("model")(_audit_model_identity)
    _valid_finish_status = field_validator("finish_status")(_audit_finish_status)


class AuditVerificationSpan(_AuditRecord):
    span_id: str
    ordinal: int
    start: int
    end: int
    content_hash: str


class AuditVerificationClaim(_AuditRecord):
    claim_id: str
    span_id: str
    ordinal: int
    is_fallback: bool


class AuditVerificationSelection(_AuditRecord):
    claim_id: str
    selected_ids: tuple[str, ...]
    examined_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    token_cost: int
    retrieval_method: str
    retrieval_complete: bool

    @field_validator("retrieval_method")
    @classmethod
    def _known_retrieval_method(cls, value: str) -> str:
        if value not in RETRIEVAL_METHODS:
            raise ValueError("audit retrieval_method must be a supported method")
        return value


class AuditVerificationFinding(_AuditRecord):
    claim_id: str
    verdict: Literal[
        "supported",
        "contradicted",
        "insufficiently_supported",
        "not_meaningfully_verifiable",
    ]
    evidence_ids: tuple[str, ...]


class AuditVerificationReassessment(_AuditRecord):
    """The first verdict on a rejected claim the verifier looked at again.

    Its only flagged difference was a rounded number or a shortened name the
    run allows; the owning assessment holds the verifier's second verdict.
    """

    tolerance: Literal["rounded_number", "shortened_name"]
    original_verdict: Literal["insufficiently_supported"]
    original_findings: tuple[AuditVerificationFinding, ...]


class AuditVerificationAssessment(_AuditRecord):
    claim_id: str
    verdict: Literal[
        "supported",
        "contradicted",
        "insufficiently_supported",
        "not_meaningfully_verifiable",
    ]
    verifier_provider: str
    verifier_model: str
    prompt_version: str
    findings: tuple[AuditVerificationFinding, ...]
    reassessment: AuditVerificationReassessment | None = None

    _valid_provider = field_validator("verifier_provider")(_audit_identity)
    _valid_model = field_validator("verifier_model")(_audit_model_identity)
    _valid_prompt_version = field_validator("prompt_version")(_audit_prompt_version)

    @model_serializer(mode="wrap")
    def _omit_absent_reassessment(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data = handler(self)
        if data.get("reassessment") is None:
            data.pop("reassessment", None)
        return data


class AuditVerificationUnresolved(_AuditRecord):
    """A span or claim verification never finished; it never publishes."""

    item_id: str
    phase: Literal["decomposition", "classification"]
    reason: Literal["omitted", "invalid_response", "capacity"]


class AuditVerificationPass(_AuditRecord):
    pass_index: int
    complete: bool
    spans: tuple[AuditVerificationSpan, ...]
    claims: tuple[AuditVerificationClaim, ...]
    selections: tuple[AuditVerificationSelection, ...]
    assessments: tuple[AuditVerificationAssessment, ...]
    unresolved: tuple[AuditVerificationUnresolved, ...] = ()

    @model_serializer(mode="wrap")
    def _omit_absent_unresolved(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data = handler(self)
        if not data.get("unresolved"):
            data.pop("unresolved", None)
        return data


class AuditVerificationRepair(_AuditRecord):
    span_id: str
    original_hash: str
    triggering_claim_ids: tuple[str, ...]
    action: Literal["qualify", "replace", "remove"]


class AuditVerificationUsage(_AuditRecord):
    phase: Literal["decomposition", "classification", "repair"]
    pass_index: int
    prompt_version: str
    provider: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    finish_status: str | None

    _valid_provider = field_validator("provider")(_audit_optional_identity)
    _valid_model = field_validator("model")(_audit_optional_model_identity)
    _valid_prompt_version = field_validator("prompt_version")(_audit_prompt_version)
    _valid_finish_status = field_validator("finish_status")(_audit_finish_status)


class AuditVerification(_AuditRecord):
    enabled: bool
    pass_count: int
    exhausted: bool
    failed: bool
    passes: tuple[AuditVerificationPass, ...]
    repairs: tuple[AuditVerificationRepair, ...]
    usage: tuple[AuditVerificationUsage, ...]
    warning_codes: tuple[str, ...]
    limitation_codes: tuple[str, ...]
    failure_codes: tuple[str, ...]


class AuditCacheMetadata(_AuditRecord):
    """Closed cache outcome and descriptor-invalidation codes for audit/3."""

    cache_hits: tuple[str, ...] = ()
    cache_misses: tuple[str, ...] = ()
    invalidation_reasons: tuple[str, ...] = ()

    @field_validator("cache_hits", mode="before")
    @classmethod
    def _validate_cache_hits(cls, value: object) -> tuple[str, ...]:
        return _audit_closed_codes(value, allowed=_CACHE_HIT_CODES, label="cache hits")

    @field_validator("cache_misses", mode="before")
    @classmethod
    def _validate_cache_misses(cls, value: object) -> tuple[str, ...]:
        return _audit_closed_codes(
            value, allowed=_CACHE_MISS_CODES, label="cache misses"
        )

    @field_validator("invalidation_reasons", mode="before")
    @classmethod
    def _validate_invalidation_reasons(cls, value: object) -> tuple[str, ...]:
        return _audit_closed_codes(
            value,
            allowed=_CACHE_INVALIDATION_CODES,
            label="cache invalidation reasons",
        )


class AuditRetryAttempt(_AuditRecord):
    """Attempt metadata for a single work item in audit/3."""

    work_id: str
    attempt_count: int
    failure_reasons: tuple[str, ...] = ()

    _valid_work_id = field_validator("work_id")(_audit_work_id)

    @field_validator("attempt_count", mode="before")
    @classmethod
    def _validate_positive_count(cls, value: object) -> int:
        return _audit_integer(value, minimum=1, label="attempt_count")

    @field_validator("failure_reasons", mode="before")
    @classmethod
    def _validate_failure_reasons(cls, value: object) -> tuple[str, ...]:
        return _audit_closed_codes(
            value, allowed=_RETRY_FAILURE_CODES, label="retry failure reasons"
        )


class AuditReliability(_AuditRecord):
    """Aggregate reliability metadata for audit/3."""

    cache: AuditCacheMetadata | None = None
    resumed: bool | None = None
    reused_count: int | None = None
    recomputed_count: int | None = None
    attempts: tuple[AuditRetryAttempt, ...] = ()

    @field_validator("reused_count", "recomputed_count", mode="before")
    @classmethod
    def _validate_resume_counts(cls, value: object) -> int | None:
        if value is None:
            return None
        return _audit_integer(value, minimum=0, label="resume counts")

    @model_validator(mode="after")
    def _resume_fields_are_complete(self) -> AuditReliability:
        resume_fields = (self.resumed, self.reused_count, self.recomputed_count)
        if any(value is not None for value in resume_fields) and any(
            value is None for value in resume_fields
        ):
            raise ValueError("resume state must include resumed and both counts")
        return self


class AuditEvidence(_AuditRecord):
    """A traceable evidence edge without copying a possibly sensitive quote."""

    segment_id: str
    quoted: bool

    _valid_segment_id = field_validator("segment_id")(_audit_segment_id)


class AuditContentUnit(_AuditRecord):
    kind: Literal["fact", "claim", "definition", "procedure", "example", "other"]
    evidence: tuple[AuditEvidence, ...]
    qualified: bool
    uncertain: bool


class AuditAnnotation(_AuditRecord):
    evidence: tuple[AuditEvidence, ...]


class AuditSummary(_AuditRecord):
    """The structural/evidence view of a summary, never reader-facing text.

    A summary, entity, content-unit, or quotation string can reproduce a
    source credential. The audit needs their links and classifications, not a
    second durable copy of the generated prose, so this intentionally stores
    no free-form source-derived text.
    """

    level: int
    provenance: tuple[str, ...]
    content_units: tuple[AuditContentUnit, ...]
    qualification_evidence: tuple[AuditAnnotation, ...]
    contradiction_evidence: tuple[AuditAnnotation, ...]
    quotation_evidence: tuple[AuditEvidence, ...]

    _valid_provenance = field_validator("provenance")(_audit_segment_ids)


class AuditGroundingSelection(_AuditRecord):
    selected_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    reserve_tokens: int | None = Field(default=None, ge=1)
    request_capacity_tokens: int | None = Field(default=None, ge=1)
    omission_reason: Literal["budget"]

    _valid_selected_ids = field_validator("selected_ids")(_audit_segment_ids)
    _valid_omitted_ids = field_validator("omitted_ids")(_audit_segment_ids)


class AuditNode(_AuditRecord):
    node_id: str
    level: int
    order: int
    children: tuple[str, ...]
    covered_segments: tuple[str, ...]
    summary: AuditSummary
    _valid_node_id = field_validator("node_id")(_audit_node_id)
    _valid_children = field_validator("children")(_audit_node_ids)
    _valid_covered_segments = field_validator("covered_segments")(_audit_segment_ids)


class AuditNodeV4(AuditNode):
    grounding: AuditGroundingSelection | None


PublicationKind = Literal["editorial", "verified_subset", "content_unit_fallback"]
PUBLICATION_WARNINGS: Mapping[str, str] = {
    "verified_subset": "verified_sentence_subset",
    "content_unit_fallback": "verified_content_unit_fallback",
}


class AuditSentenceEvidence(_AuditRecord):
    """One verifier quotation that supports a published sentence."""

    segment_id: str
    quote: str
    start: int | None = None
    end: int | None = None

    _valid_segment_id = field_validator("segment_id")(_audit_segment_id)
    _valid_quote = field_validator("quote", mode="before")(_audit_prose)

    @model_validator(mode="after")
    def _offsets_are_a_range(self) -> AuditSentenceEvidence:
        if (self.start is None) != (self.end is None) or (
            self.start is not None
            and self.end is not None
            and not 0 <= self.start < self.end
        ):
            raise ValueError("evidence offsets must be an ordered pair or absent")
        return self


class AuditPublishedSentence(_AuditRecord):
    index: int = Field(ge=0)
    paragraph: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int
    text: str
    verdict: Literal["supported", "not_meaningfully_verifiable", "unchecked"]
    evidence: tuple[AuditSentenceEvidence, ...]

    _valid_text = field_validator("text", mode="before")(_audit_published_prose)

    @model_validator(mode="after")
    def _range_matches_text(self) -> AuditPublishedSentence:
        if self.end - self.start != len(self.text):
            raise ValueError("published sentence range must match its text")
        if self.verdict == "unchecked" and self.evidence:
            raise ValueError("an unchecked sentence cannot cite evidence")
        if self.verdict == "supported" and not self.evidence:
            raise ValueError("a supported sentence must cite evidence")
        return self


class AuditRemovedSentence(_AuditRecord):
    text: str
    verdict: Literal[
        "contradicted",
        "insufficiently_supported",
        "not_meaningfully_verifiable",
        "unverified",
        "unfinished",
    ]
    reason: str

    _valid_text = field_validator("text", "reason", mode="before")(_audit_prose)


class AuditSubstitution(_AuditRecord):
    """A rejected draft sentence, the source sentence proposed for it, and the outcome.

    The replacement publishes only when a verification pass over the new draft
    supports it; otherwise it is removed with that pass's verdict, or
    `unverified` when the pass could not run.
    """

    original_text: str
    original_verdict: Literal["insufficiently_supported", "not_meaningfully_verifiable"]
    replacement_text: str
    replacement_verdict: Literal[
        "supported",
        "contradicted",
        "insufficiently_supported",
        "not_meaningfully_verifiable",
        "unverified",
    ]
    action: Literal["replaced", "removed"]

    _valid_text = field_validator("original_text", "replacement_text", mode="before")(
        _audit_prose
    )

    @model_validator(mode="after")
    def _action_matches_verdict(self) -> AuditSubstitution:
        if (self.action == "replaced") != (self.replacement_verdict == "supported"):
            raise ValueError("only a supported replacement may be published")
        return self


def _audit_section_id(value: str) -> str:
    if _SECTION_ID.fullmatch(value):
        return value
    raise ValueError("audit section id must be a section identity")


def _audit_section_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_audit_section_id(value) for value in values)


def _audit_heading(value: str | None) -> str | None:
    """Keep a heading as the import stored it, with credential-like text replaced."""
    return None if value is None else redact_text(value)


MAX_HEADING_LEVEL = 6


class AuditHeading(_AuditRecord):
    """One heading line of the assembled final summary.

    `start:end` is the whole line, the `#` marks included, in the published
    text. `text` is the outline heading after the marks, as a section record
    keeps it: credential-like text is replaced, so a redacted heading's text
    can differ in length from its line.
    """

    section_id: str
    level: int = Field(ge=1, le=MAX_HEADING_LEVEL)
    text: str
    start: int = Field(ge=0)
    end: int

    _valid_section_id = field_validator("section_id")(_audit_section_id)
    _valid_text = field_validator("text", mode="before")(
        lambda value: _audit_heading(_audit_prose(value))
    )

    @model_validator(mode="after")
    def _range_holds_the_line(self) -> AuditHeading:
        if self.end - self.start <= self.level:
            raise ValueError("heading range must hold its marks and text")
        return self


class AuditSectionWords(_AuditRecord):
    """The words a section was asked for and the words of its published prose.

    Heading lines are not counted on either side.
    """

    section_id: str
    requested: int = Field(ge=1)
    published: int = Field(ge=0)

    _valid_section_id = field_validator("section_id")(_audit_section_id)


class AuditPublication(_AuditRecord):
    """How the published summary was chosen and what supports each sentence.

    A section mode summary is assembled from section prose under source
    headings: `headings` places each heading line in the published text and
    `section_words` gives each section's requested and published words. Both
    are empty for a summary written whole. Sentence offsets index the whole
    published text, heading lines included.
    """

    kind: PublicationKind
    sentences: tuple[AuditPublishedSentence, ...]
    removed_sentences: tuple[AuditRemovedSentence, ...]
    substitutions: tuple[AuditSubstitution, ...] = ()
    headings: tuple[AuditHeading, ...] = ()
    section_words: tuple[AuditSectionWords, ...] = ()

    @model_validator(mode="after")
    def _sentences_are_ordered(self) -> AuditPublication:
        if not self.sentences:
            raise ValueError("a publication must contain sentences")
        previous: AuditPublishedSentence | None = None
        for index, sentence in enumerate(self.sentences):
            if sentence.index != index:
                raise ValueError("published sentences must be numbered in order")
            if previous is not None and (
                sentence.start < previous.end or sentence.paragraph < previous.paragraph
            ):
                raise ValueError("published sentences must be ordered and disjoint")
            previous = sentence
        published = " ".join(" ".join(sentence.text.split()) for sentence in self.sentences)
        for substitution in self.substitutions:
            if substitution.action == "replaced" and (
                " ".join(substitution.replacement_text.split()) not in published
            ):
                raise ValueError("a replaced sentence must be published")
        previous_end = -1
        for heading in self.headings:
            if heading.start < previous_end:
                raise ValueError("headings must be ordered and disjoint")
            previous_end = heading.end
            if any(
                sentence.start < heading.end and heading.start < sentence.end
                for sentence in self.sentences
            ):
                raise ValueError("a heading must not overlap a published sentence")
        words_ids = tuple(item.section_id for item in self.section_words)
        if len(set(words_ids)) != len(words_ids):
            raise ValueError("section words must name each section once")
        if not {heading.section_id for heading in self.headings} <= set(words_ids):
            raise ValueError("every heading's section must record its words")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_optional_records(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data = handler(self)
        for key in ("substitutions", "headings", "section_words"):
            if not data.get(key):
                data.pop(key, None)
        return data


def _audit_page_range(start: int | None, end: int | None) -> None:
    if (start is None) != (end is None) or (
        start is not None and end is not None and not 1 <= start <= end
    ):
        raise ValueError("a page range must be an ordered pair of pages or absent")


SectionStatus = Literal["verified", "unverified", "empty", "heading_only"]


class AuditSectionCitation(_AuditRecord):
    """A source segment a section's prose cites, with the pages it spans."""

    segment_id: str
    order: int = Field(ge=0)
    page_start: int | None = None
    page_end: int | None = None

    _valid_segment_id = field_validator("segment_id")(_audit_segment_id)

    @model_validator(mode="after")
    def _pages_are_a_range(self) -> AuditSectionCitation:
        _audit_page_range(self.page_start, self.page_end)
        return self


class AuditSectionPublication(_AuditRecord):
    """One section's reader-facing prose, checked against that section's source only."""

    status: SectionStatus
    kind: PublicationKind | None = None
    reason: str | None = None
    sentences: tuple[AuditPublishedSentence, ...]
    removed_sentences: tuple[AuditRemovedSentence, ...]
    citations: tuple[AuditSectionCitation, ...]
    segment_ids: tuple[str, ...]
    page_start: int | None = None
    page_end: int | None = None
    target_words: int = Field(ge=1)
    words: int = Field(ge=0)

    _valid_segment_ids = field_validator("segment_ids")(_audit_segment_ids)
    _valid_reason = field_validator("reason", mode="before")(
        lambda value: None if value is None else _audit_prose(value)
    )

    @model_validator(mode="after")
    def _status_matches_content(self) -> AuditSectionPublication:
        _audit_page_range(self.page_start, self.page_end)
        if self.status == "heading_only":
            if (
                self.sentences
                or self.citations
                or self.removed_sentences
                or self.kind is not None
                or self.words
                or not self.reason
            ):
                raise ValueError(
                    "a heading-only section publication has a reason and no prose"
                )
        elif self.status == "empty":
            if self.sentences or self.citations or self.kind is not None or not self.reason:
                raise ValueError("an empty section publication has a reason and no prose")
        else:
            if not self.sentences or self.reason is not None:
                raise ValueError("a published section has sentences and no reason")
            unchecked = {sentence.verdict == "unchecked" for sentence in self.sentences}
            if unchecked != {self.status == "unverified"}:
                raise ValueError("section sentence verdicts must match the status")
            if (self.kind is None) != (self.status == "unverified") or (
                self.status == "unverified" and self.removed_sentences
            ):
                raise ValueError("a section publication kind must match its status")
        if tuple(sentence.index for sentence in self.sentences) != tuple(
            range(len(self.sentences))
        ):
            raise ValueError("published sentences must be numbered in order")
        if not {item.segment_id for item in self.citations} <= set(self.segment_ids):
            raise ValueError("section citations must lie in the section's segments")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_kind_and_reason(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data = handler(self)
        for key in ("kind", "reason"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class AuditSection(_AuditRecord):
    """One section of the tree a section mode run summarized by."""

    section_id: str
    heading: str | None
    level: int = Field(ge=1)
    page_start: int | None
    page_end: int | None
    parent_id: str | None
    child_ids: tuple[str, ...]
    folded_headings: tuple[str, ...]
    segment_ids: tuple[str, ...]
    node_id: str | None
    publication: AuditSectionPublication | None = None

    _valid_section_id = field_validator("section_id")(_audit_section_id)
    _valid_parent = field_validator("parent_id")(
        lambda value: None if value is None else _audit_section_id(value)
    )
    _valid_child_ids = field_validator("child_ids")(_audit_section_ids)
    _valid_heading = field_validator("heading")(_audit_heading)
    _valid_folded = field_validator("folded_headings")(
        lambda values: tuple(redact_text(value) for value in values)
    )
    _valid_segment_ids = field_validator("segment_ids")(_audit_segment_ids)
    _valid_node_id = field_validator("node_id")(
        lambda value: None if value is None else _audit_node_id(value)
    )

    @model_validator(mode="after")
    def _pages_and_publication_agree(self) -> AuditSection:
        _audit_page_range(self.page_start, self.page_end)
        if (
            self.publication is not None
            and self.node_id is None
            and self.publication.status != "heading_only"
        ):
            raise ValueError("a section publication needs the node that summarizes it")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_publication(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data = handler(self)
        if data.get("publication") is None:
            data.pop("publication", None)
        return data


def _sections_link_resolve(
    sections: tuple[AuditSection, ...],
    *,
    verification: AuditVerification,
    nodes: set[str],
    segment_ids: set[str],
) -> None:
    """Tie the section records to each other, to the tree, and to the sources."""
    ids = {section.section_id: section for section in sections}
    if len(ids) != len(sections) or not sections:
        raise ValueError("section identifiers must be unique and present")
    for section in sections:
        parent = ids.get(section.parent_id) if section.parent_id is not None else None
        if section.parent_id is not None and (
            parent is None or section.section_id not in parent.child_ids
        ):
            raise ValueError("a section's parent must list it as a child")
        if any(
            child not in ids or ids[child].parent_id != section.section_id
            for child in section.child_ids
        ):
            raise ValueError("a section's children must name it as their parent")
        if section.node_id is not None and section.node_id not in nodes:
            raise ValueError("a section's node must resolve to a tree node")
        if not set(section.segment_ids) <= segment_ids:
            raise ValueError("section segments must resolve to source segments")
        publication = section.publication
        if publication is None:
            continue
        if not set(publication.segment_ids) <= segment_ids:
            raise ValueError("section segments must resolve to source segments")
        if publication.status == "heading_only":
            pass
        elif publication.status != "unverified" and not verification.enabled:
            raise ValueError("verified section prose requires verification")
        elif publication.status == "unverified" and verification.enabled:
            raise ValueError("an unverified section requires verification to be off")
        if any(
            evidence.segment_id not in publication.segment_ids
            for sentence in publication.sentences
            for evidence in sentence.evidence
        ):
            raise ValueError("section evidence must lie in the section's segments")


class _AuditArtifactBase(_AuditRecord):
    source_id: str
    strategy: Literal["auto", "direct", "hierarchical"]
    model: str
    configuration: dict[str, Any]
    source_segments: tuple[AuditSegment, ...]
    tree_nodes: tuple[AuditNode, ...]
    root_node_id: str
    citations: tuple[AuditCitation, ...]
    usage: tuple[AuditUsage, ...]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    verification: AuditVerification
    publication: AuditPublication | None = None
    sections: tuple[AuditSection, ...] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_publication(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data = handler(self)
        for key in ("publication", "sections"):
            if data.get(key) is None:
                data.pop(key, None)
        return data

    @field_validator("configuration", mode="before")
    @classmethod
    def _safe_configuration(cls, value: object) -> dict[str, object]:
        return _validate_audit_configuration(value)

    @field_validator("source_id")
    @classmethod
    def _valid_source_id(cls, value: str) -> str:
        return _audit_source_id(value)

    _valid_model = field_validator("model")(_audit_model_identity)
    _valid_root_node_id = field_validator("root_node_id")(_audit_node_id)

    @model_validator(mode="after")
    def _links_resolve(self) -> _AuditArtifactBase:
        segments = {segment.segment_id: segment for segment in self.source_segments}
        if len(segments) != len(self.source_segments):
            raise ValueError("source segment identifiers must be unique")
        if tuple(segment.segment_id for segment in self.source_segments) != tuple(
            segment.segment_id
            for segment in sorted(self.source_segments, key=lambda item: item.order)
        ):
            raise ValueError("source segments must be in source order")
        if any(segment.source_id != self.source_id for segment in self.source_segments):
            raise ValueError("every source segment must belong to the audit source")
        passage_ids: set[str] = set()
        for segment in self.source_segments:
            if segment.parent_segment_id is None:
                continue
            parent = segments.get(segment.parent_segment_id)
            if parent is None or parent.parent_segment_id is not None:
                raise ValueError("a verification passage parent must be a leaf segment")
            if (
                segment.core_start < parent.context_start
                or segment.core_end > parent.context_end
            ):
                raise ValueError("a verification passage must lie within its parent segment")
            passage_ids.add(segment.segment_id)

        nodes = {node.node_id: node for node in self.tree_nodes}
        if len(nodes) != len(self.tree_nodes):
            raise ValueError("tree node identifiers must be unique")
        root = nodes.get(self.root_node_id)
        if root is None:
            raise ValueError("root_node_id must resolve to a tree node")

        for node in self.tree_nodes:
            unknown_covered = set(node.covered_segments) - (set(segments) - passage_ids)
            if unknown_covered:
                raise ValueError("tree coverage must resolve to leaf source segments")
            grounding = getattr(node, "grounding", None)
            if isinstance(node, AuditNodeV4):
                if len(node.children) > 1 and grounding is None:
                    raise ValueError("executed merges must record grounding")
                if len(node.children) <= 1 and grounding is not None:
                    raise ValueError("only executed merges may record grounding")
            if grounding is not None:
                selected = set(grounding.selected_ids)
                omitted = set(grounding.omitted_ids)
                if (
                    len(selected) != len(grounding.selected_ids)
                    or len(omitted) != len(grounding.omitted_ids)
                    or selected & omitted
                ):
                    raise ValueError("grounding selection identifiers must not overlap")
                if (selected | omitted) - set(node.covered_segments):
                    raise ValueError(
                        "grounding selection must resolve to tree coverage"
                    )
                if (grounding.reserve_tokens is None) == (
                    grounding.request_capacity_tokens is None
                ):
                    raise ValueError("grounding selection must record one budget mode")
            for child_id in node.children:
                child = nodes.get(child_id)
                if child is None:
                    raise ValueError("tree child must resolve to a tree node")
                if child.level >= node.level:
                    raise ValueError("tree children must be at a lower level")
            references = _summary_references(node.summary)
            if not references <= set(segments):
                raise ValueError("summary evidence must resolve to source segments")

        citation_ids = tuple(citation.segment_id for citation in self.citations)
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("citation identifiers must be unique")
        expected_order = tuple(
            segment.segment_id
            for segment in self.source_segments
            if segment.segment_id in citation_ids
        )
        if citation_ids != expected_order:
            raise ValueError("citations must be unique and in source order")
        for citation in self.citations:
            segment = segments.get(citation.segment_id)
            if segment is None or (
                citation.source_id != segment.source_id
                or citation.order != segment.order
            ):
                raise ValueError("citation mappings must resolve to segment metadata")
        if any(
            not _CLOSED_CODE.fullmatch(code)
            for code in (*self.warnings, *self.failures)
        ):
            raise ValueError("audit diagnostics must be closed identifiers")
        _verification_links_resolve(self.verification, set(segments))
        if self.publication is not None:
            _publication_links_resolve(
                self.publication,
                verification=self.verification,
                warnings=self.warnings,
                segment_ids=set(segments),
            )
        if self.sections is not None:
            _sections_link_resolve(
                self.sections,
                verification=self.verification,
                nodes=set(nodes),
                segment_ids=set(segments) - passage_ids,
            )
            if self.publication is not None:
                published = {
                    section.section_id: section.publication
                    for section in self.sections
                    if section.publication is not None
                }
                for item in self.publication.section_words:
                    record = published.get(item.section_id)
                    if (
                        record is None
                        or record.target_words != item.requested
                        or record.words != item.published
                    ):
                        raise ValueError(
                            "publication section words must match the section records"
                        )
        return self


class AuditArtifactV2(_AuditArtifactBase):
    """The unchanged audit/2 wire contract."""

    schema_version: Literal["audit/2"]


class AuditArtifactV3(_AuditArtifactBase):
    """The audit/3 contract with reliability-only metadata."""

    schema_version: Literal["audit/3"]
    reliability: AuditReliability


class AuditArtifactV4(_AuditArtifactBase):
    """The audit/4 contract with per-merge grounding metadata."""

    schema_version: Literal["audit/4"]
    tree_nodes: tuple[AuditNodeV4, ...]
    reliability: AuditReliability | None = None


_AuditArtifactRecord = Annotated[
    AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4,
    Field(discriminator="schema_version"),
]
_AUDIT_ARTIFACT_ADAPTER = TypeAdapter(_AuditArtifactRecord)


class AuditArtifact:
    """Version-discriminated audit artifact reader compatibility facade."""

    def __new__(cls, **value: object) -> AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4:
        return _AUDIT_ARTIFACT_ADAPTER.validate_python(value)

    @staticmethod
    def model_validate(value: object) -> AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4:
        return _AUDIT_ARTIFACT_ADAPTER.validate_python(value)

    @staticmethod
    def model_validate_json(value: str | bytes) -> AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4:
        return _AUDIT_ARTIFACT_ADAPTER.validate_json(value)


def _unresolved_work_resolves(
    record: AuditVerificationPass,
    spans: Mapping[str, AuditVerificationSpan],
    claims: Mapping[str, AuditVerificationClaim],
    selections: Mapping[str, AuditVerificationSelection],
    assessments: Mapping[str, AuditVerificationAssessment],
) -> bool:
    """Unresolved work lists exactly the pass's unfinished spans and claims.

    An item still in the pass is unfinished there: a span without claims or a
    claim without an assessment, and every such span and claim is listed. An
    item absent from the pass was withheld when a verified subset was
    published. Returns whether any listed item is still in the pass.
    """
    if record.complete:
        raise ValueError("a verification pass with unresolved work is not complete")
    item_ids = [item.item_id for item in record.unresolved]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("verification unresolved work must be unique")
    if any(int(item_id[1:3]) != record.pass_index for item_id in item_ids):
        raise ValueError("verification unresolved work must belong to its pass")
    claimed_spans = {claim.span_id for claim in claims.values()}
    present_spans = {
        item.item_id
        for item in record.unresolved
        if item.phase == "decomposition" and item.item_id in spans
    }
    present_claims = {
        item.item_id
        for item in record.unresolved
        if item.phase == "classification" and item.item_id in claims
    }
    if (
        present_spans != set(spans) - claimed_spans
        or present_claims != set(claims) - set(assessments)
        or set(selections) != set(claims)
    ):
        raise ValueError("verification unresolved work must list exactly the unfinished pass work")
    return bool(present_spans or present_claims)


def _verification_links_resolve(
    verification: AuditVerification, segment_ids: set[str]
) -> None:
    """Validate audit-only verification links without retaining prose."""
    if not verification.enabled:
        if (
            verification.pass_count
            or verification.passes
            or verification.repairs
            or verification.usage
            or verification.exhausted
            or verification.failed
            or verification.warning_codes
            or verification.limitation_codes
            or verification.failure_codes
        ):
            raise ValueError("verification disabled record must be empty")
        return
    if verification.pass_count != len(verification.passes):
        raise ValueError("verification pass count must match pass records")
    pass_indexes = tuple(item.pass_index for item in verification.passes)
    if len(set(pass_indexes)) != len(pass_indexes) or pass_indexes != tuple(
        sorted(pass_indexes)
    ):
        raise ValueError("verification passes must be uniquely ordered")
    if any(
        not _CLOSED_CODE.fullmatch(code)
        for codes in (
            verification.warning_codes,
            verification.limitation_codes,
            verification.failure_codes,
        )
        for code in codes
    ):
        raise ValueError("verification diagnostic codes must be closed identifiers")

    spans_by_id: dict[str, AuditVerificationSpan] = {}
    claims_by_id: dict[str, AuditVerificationClaim] = {}
    for record_position, record in enumerate(verification.passes):
        if not 1 <= record.pass_index <= 99:
            raise ValueError("verification pass index must be between 1 and 99")
        prefix = f"V{record.pass_index:02d}"
        spans = {span.span_id: span for span in record.spans}
        claims = {claim.claim_id: claim for claim in record.claims}
        selections = {selection.claim_id: selection for selection in record.selections}
        assessments = {
            assessment.claim_id: assessment for assessment in record.assessments
        }
        if len(spans) != len(record.spans) or len(claims) != len(record.claims):
            raise ValueError("verification pass identifiers must be unique")
        if any(
            not _VERIFICATION_SPAN_ID.fullmatch(span.span_id)
            or not span.span_id.startswith(f"{prefix}S")
            or not _SHA256.fullmatch(span.content_hash)
            for span in record.spans
        ) or any(
            not _VERIFICATION_CLAIM_ID.fullmatch(claim.claim_id)
            or not claim.claim_id.startswith(f"{prefix}C")
            for claim in record.claims
        ):
            raise ValueError("verification identifiers must belong to their pass")
        if tuple(span.ordinal for span in record.spans) != tuple(
            range(1, len(record.spans) + 1)
        ):
            raise ValueError("verification spans must be in ordinal order")
        claims_by_span: dict[str, list[AuditVerificationClaim]] = {}
        for claim in record.claims:
            claims_by_span.setdefault(claim.span_id, []).append(claim)
        if any(
            tuple(claim.ordinal for claim in span_claims)
            != tuple(range(1, len(span_claims) + 1))
            for span_claims in claims_by_span.values()
        ):
            raise ValueError("verification claims must be in span ordinal order")
        if any(
            span.start < 0
            or span.end <= span.start
            or (index and span.start < record.spans[index - 1].end)
            for index, span in enumerate(record.spans)
        ):
            raise ValueError("verification span ranges must be ordered and nonempty")
        if not set(selections) <= set(claims) or len(selections) != len(
            record.selections
        ):
            raise ValueError("verification selections must resolve pass claims")
        if not set(assessments) <= set(claims) or len(assessments) != len(
            record.assessments
        ):
            raise ValueError("verification assessments must resolve pass claims")
        coverage_complete = set(selections) == set(claims) and set(assessments) == set(
            claims
        )
        if record.complete and not coverage_complete:
            raise ValueError(
                "verification pass completion must match recorded coverage"
            )
        is_last = record_position == len(verification.passes) - 1
        if record.unresolved:
            # An earlier pass may leave work unfinished when a later pass
            # publishes. The last pass may hold unfinished work only when
            # verification failed; a published subset lists only work it withheld.
            still_in_pass = _unresolved_work_resolves(
                record, spans, claims, selections, assessments
            )
            partial_needs_failure = is_last and still_in_pass
        else:
            partial_needs_failure = not record.complete
        if partial_needs_failure and (not verification.failed or not is_last):
            raise ValueError(
                "only a terminal failed verification may have a partial pass"
            )
        for claim in record.claims:
            if claim.span_id not in spans:
                raise ValueError("verification claim must resolve to a pass span")
        for selection in record.selections:
            selected = set(selection.selected_ids)
            examined = set(selection.examined_ids)
            omitted = set(selection.omitted_ids)
            if len(selected) != len(selection.selected_ids) or len(
                examined | omitted
            ) != len(selection.examined_ids) + len(selection.omitted_ids):
                raise ValueError("verification evidence identifiers must be unique")
            if not selected <= examined or not (examined | omitted) <= segment_ids:
                raise ValueError(
                    "verification evidence must resolve to source segments"
                )
            if selection.retrieval_complete != (not omitted):
                raise ValueError(
                    "verification retrieval completeness must match omissions"
                )
        for assessment in record.assessments:
            finding_ids = [finding.claim_id for finding in assessment.findings]
            if not assessment.findings or any(
                item != assessment.claim_id for item in finding_ids
            ):
                raise ValueError(
                    "verification findings must resolve to their assessment"
                )
            selection = selections[assessment.claim_id]
            reassessed = (
                assessment.reassessment.original_findings
                if assessment.reassessment is not None
                else ()
            )
            for finding in (*assessment.findings, *reassessed):
                if finding.claim_id != assessment.claim_id:
                    raise ValueError(
                        "verification findings must resolve to their assessment"
                    )
                if not set(finding.evidence_ids) <= set(selection.examined_ids):
                    raise ValueError("verification finding evidence must be examined")
        if set(spans_by_id) & set(spans) or set(claims_by_id) & set(claims):
            raise ValueError("verification identifiers must be unique across passes")
        spans_by_id.update(spans)
        claims_by_id.update(claims)

    for repair in verification.repairs:
        span = spans_by_id.get(repair.span_id)
        if span is None or repair.original_hash != span.content_hash:
            raise ValueError(
                "verification repair must resolve to its original pass span"
            )
        if not repair.triggering_claim_ids or any(
            claim_id not in claims_by_id
            or claims_by_id[claim_id].span_id != repair.span_id
            for claim_id in repair.triggering_claim_ids
        ):
            raise ValueError(
                "verification repair triggers must resolve to its pass span"
            )
    if any(usage.pass_index not in pass_indexes for usage in verification.usage):
        raise ValueError("verification usage must resolve to a pass")


def _publication_links_resolve(
    publication: AuditPublication,
    *,
    verification: AuditVerification,
    warnings: Sequence[str],
    segment_ids: set[str],
) -> None:
    """Tie the publication to its verification outcome, warning, and sources."""
    if verification.failed:
        raise ValueError("a failed verification cannot record a publication")
    for kind, code in PUBLICATION_WARNINGS.items():
        if (code in warnings) != (publication.kind == kind):
            raise ValueError("publication kind must match its audit warning")
    for sentence in publication.sentences:
        if (sentence.verdict == "unchecked") == verification.enabled:
            raise ValueError("published sentence verdicts must match verification")
        if any(evidence.segment_id not in segment_ids for evidence in sentence.evidence):
            raise ValueError("sentence evidence must resolve to source segments")
    if not verification.enabled and (
        publication.kind != "editorial"
        or publication.removed_sentences
        or publication.substitutions
    ):
        raise ValueError("an unverified publication must be the unchanged editorial draft")


@dataclass(frozen=True)
class Citation:
    segment_id: str
    source_id: str
    order: int


def _summary_references(node: AuditSummary) -> set[str]:
    identifiers = set(node.provenance)
    identifiers.update(
        evidence.segment_id for unit in node.content_units for evidence in unit.evidence
    )
    identifiers.update(
        evidence.segment_id
        for annotation in (
            *node.qualification_evidence,
            *node.contradiction_evidence,
        )
        for evidence in annotation.evidence
    )
    identifiers.update(evidence.segment_id for evidence in node.quotation_evidence)
    return identifiers


def resolve_citations(
    provenance: Sequence[str], *, source_id: str, segments: Sequence[SourceSegment]
) -> tuple[Citation, ...]:
    by_id = {segment.segment_id: segment for segment in segments}
    if len(by_id) != len(segments):
        raise AuditError("source segment identifiers must be unique")
    if any(segment.source_id != source_id for segment in segments):
        raise AuditError("every source segment must belong to the supplied source")
    unknown = set(provenance) - set(by_id)
    if unknown:
        raise AuditError("citation provenance contains an unknown source segment")
    cited = set(provenance)
    return tuple(
        Citation(segment.segment_id, segment.source_id, segment.order)
        for segment in sorted(segments, key=lambda item: item.order)
        if segment.segment_id in cited
    )


def citation_provenance_for_summary(
    root_provenance: Sequence[str],
    verification: VerificationResult | None,
    *,
    verification_enabled: bool,
) -> Sequence[str]:
    """Prefer verifier evidence segments over whole-document provenance when available."""
    if not verification_enabled or verification is None or verification.failed:
        return root_provenance
    if not verification.pass_results:
        return root_provenance
    final_pass = verification.pass_results[-1]
    if final_pass.failed:
        return root_provenance
    ordered: list[str] = []
    seen: set[str] = set()
    for assessment in final_pass.assessments:
        if assessment.verdict is not ClaimVerdict.SUPPORTED:
            continue
        for finding in assessment.findings:
            if finding.verdict is not ClaimVerdict.SUPPORTED:
                continue
            for evidence_id in finding.evidence_ids:
                if evidence_id in seen:
                    continue
                seen.add(evidence_id)
                ordered.append(evidence_id)
    if not ordered:
        return root_provenance
    sub_segments = [segment_id for segment_id in ordered if segment_id.startswith("S")]
    if sub_segments:
        return tuple(sub_segments)
    return tuple(ordered)


def render_citations(text: str, citations: Sequence[Citation]) -> str:
    """Return a readable opt-in source list; the default caller skips this."""
    if not citations:
        return text
    identifiers = ", ".join(citation.segment_id for citation in citations)
    return f"{text}\n\nSources: {identifiers}"


def _audit_segment(segment: SourceSegment, *, parent_segment_id: str | None) -> AuditSegment:
    return AuditSegment(
        segment_id=segment.segment_id,
        source_id=segment.source_id,
        order=segment.order,
        core_start=segment.core_start,
        core_end=segment.core_end,
        context_start=segment.context_start,
        context_end=segment.context_end,
        core_token_count=segment.core_token_count,
        token_count=segment.token_count,
        leading_overlap_tokens=segment.leading_overlap_tokens,
        trailing_overlap_tokens=segment.trailing_overlap_tokens,
        boundary_kind=segment.boundary_kind.value,
        parent_segment_id=parent_segment_id,
    )


def _audit_node(node: TreeNode, *, include_grounding: bool) -> AuditNode | AuditNodeV4:
    summary = node.summary
    audit_summary = AuditSummary(
        level=summary.level,
        provenance=summary.provenance,
        content_units=tuple(
            AuditContentUnit(
                kind=unit.kind.value,
                evidence=tuple(
                    AuditEvidence(
                        segment_id=evidence.segment_id,
                        quoted=evidence.quote is not None,
                    )
                    for evidence in unit.evidence
                ),
                qualified=unit.qualification is not None,
                uncertain=unit.uncertain,
            )
            for unit in summary.content_units
        ),
        qualification_evidence=tuple(
            AuditAnnotation(
                evidence=tuple(
                    AuditEvidence(
                        segment_id=evidence.segment_id,
                        quoted=evidence.quote is not None,
                    )
                    for evidence in annotation.evidence
                )
            )
            for annotation in summary.qualifications
        ),
        contradiction_evidence=tuple(
            AuditAnnotation(
                evidence=tuple(
                    AuditEvidence(
                        segment_id=evidence.segment_id,
                        quoted=evidence.quote is not None,
                    )
                    for evidence in annotation.evidence
                )
            )
            for annotation in summary.contradictions
        ),
        quotation_evidence=tuple(
            AuditEvidence(
                segment_id=evidence.segment_id,
                quoted=evidence.quote is not None,
            )
            for evidence in summary.quotations
        ),
    )
    values = {
        "node_id": node.node_id,
        "level": node.level,
        "order": node.order,
        "children": node.children,
        "covered_segments": node.covered_segments,
        "summary": audit_summary,
    }
    if not include_grounding:
        return AuditNode(**values)
    return AuditNodeV4(
        **values,
        grounding=(
            AuditGroundingSelection(
                selected_ids=node.grounding.selection.selected_ids,
                omitted_ids=node.grounding.selection.omitted_ids,
                reserve_tokens=node.grounding.reserve_tokens,
                request_capacity_tokens=node.grounding.request_capacity_tokens,
                omission_reason="budget",
            )
            if node.grounding is not None
            else None
        ),
    )


def _usage(generation: GenerationResult) -> AuditUsage:
    return AuditUsage(
        provider=redact_text(generation.provider),
        model=redact_text(generation.model),
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        finish_status=(
            redact_text(generation.finish_status)
            if generation.finish_status is not None
            else None
        ),
    )


_SAFE_CONFIGURATION = frozenset(
    {
        "provider",
        "model",
        "timeout_seconds",
        "strategy",
        "context_window",
        "max_output_tokens",
        "safety_margin_tokens",
        "safety_margin_fraction",
        "max_direct_tokens",
        "target_words",
        "max_merge_children",
        "include_citations",
        "evidence_tokens",
        "request_tokens",
        "output_reserve_tokens",
        "max_repair_passes",
        "enabled",
        "strict_numbers",
        "strict_names",
        "context_window_tokens",
        "context_window_assumed",
        "counter_exact",
        "reserved_output_tokens",
        "usable_input_capacity",
        "document_tokens",
        "fits",
        "prompt_version",
    }
)
_SAFE_CONFIGURATION_SECTIONS = frozenset(
    {"app", "strategy", "pipeline", "verification", "budget"}
)
_CONFIG_PROMPT_VERSION = re.compile(r"^[a-z][a-z0-9_-]{0,63}/[1-9][0-9]*$")
_CONFIG_BOOLEAN_FIELDS = frozenset(
    {"include_citations", "enabled", "context_window_assumed", "counter_exact", "fits", "strict_numbers", "strict_names"}
)
_CONFIG_POSITIVE_INTEGER_FIELDS = frozenset(
    {
        "context_window",
        "max_output_tokens",
        "max_direct_tokens",
        "target_words",
        "max_merge_children",
        "evidence_tokens",
        "request_tokens",
        "output_reserve_tokens",
        "context_window_tokens",
        "usable_input_capacity",
    }
)
_CONFIG_NONNEGATIVE_INTEGER_FIELDS = frozenset(
    {
        "safety_margin_tokens",
        "max_repair_passes",
        "reserved_output_tokens",
        "document_tokens",
    }
)
_CONFIG_NULLABLE_POSITIVE_INTEGER_FIELDS = frozenset(
    {"context_window", "max_direct_tokens", "max_merge_children"}
)


def _configuration_error(message: str) -> ValueError:
    return ValueError(f"audit configuration {message}")


def _safe_configuration_value(
    key: str, value: object
) -> str | int | float | bool | None:
    if isinstance(value, Enum):
        value = value.value
    if key in _CONFIG_BOOLEAN_FIELDS:
        if isinstance(value, bool):
            return value
        raise _configuration_error(f"{key} must be boolean")
    if key == "provider":
        if value in {"openai", "ollama"}:
            return value
        raise _configuration_error("provider must be an allowed identity")
    if key == "strategy":
        if value in {"auto", "direct", "hierarchical"}:
            return value
        raise _configuration_error("strategy must be an allowed identity")
    if key == "prompt_version":
        if isinstance(value, str) and _CONFIG_PROMPT_VERSION.fullmatch(value):
            return value
        raise _configuration_error("prompt_version must be a version identity")
    if key == "model":
        if (
            isinstance(value, str)
            and _SAFE_MODEL_IDENTITY.fullmatch(value)
            and redact_text(value) == value
        ):
            return value
        raise _configuration_error("model must be a non-secret model identity")
    if key == "timeout_seconds":
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and value > 0
        ):
            return value
        raise _configuration_error("timeout_seconds must be finite and positive")
    if key == "safety_margin_fraction":
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and 0 <= value < 1
        ):
            return value
        raise _configuration_error("safety_margin_fraction must be in [0, 1)")
    if key in _CONFIG_NULLABLE_POSITIVE_INTEGER_FIELDS and value is None:
        return None
    if key in _CONFIG_POSITIVE_INTEGER_FIELDS:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        raise _configuration_error(f"{key} must be a positive integer")
    if key in _CONFIG_NONNEGATIVE_INTEGER_FIELDS:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            and (key != "max_repair_passes" or value <= 49)
        ):
            return value
        raise _configuration_error(f"{key} must be a permitted nonnegative integer")
    raise _configuration_error(f"contains unsupported field {key!r}")


def _validate_configuration_entries(
    configuration: Mapping[object, object],
) -> dict[str, object]:
    unknown = set(configuration) - _SAFE_CONFIGURATION
    if unknown:
        raise _configuration_error("contains unsupported fields")
    return {
        key: _safe_configuration_value(key, value)
        for key, value in configuration.items()
    }


def _validate_audit_configuration(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise _configuration_error("must be a mapping")
    sections = set(value) & _SAFE_CONFIGURATION_SECTIONS
    if not sections:
        return _validate_configuration_entries(value)
    if set(value) != sections:
        raise _configuration_error("must not mix configuration sections and fields")
    validated: dict[str, object] = {}
    for section in sorted(sections):
        entries = value[section]
        if not isinstance(entries, Mapping):
            raise _configuration_error("section must be a mapping")
        validated[section] = _validate_configuration_entries(entries)
    return validated


def _allowlisted_configuration(
    configuration: Mapping[str, object],
) -> dict[str, object]:
    """Retain only declared safe runtime metadata; never redact-and-keep paths."""
    if not isinstance(configuration, Mapping):
        raise AuditError("audit configuration must be a mapping")
    sections = set(configuration) & _SAFE_CONFIGURATION_SECTIONS
    if not sections:
        return _validate_audit_configuration(
            {
                key: value
                for key, value in configuration.items()
                if key in _SAFE_CONFIGURATION
            }
        )
    allowed: dict[str, object] = {}
    for section in sorted(sections):
        value = configuration[section]
        if not isinstance(value, Mapping):
            raise AuditError("audit configuration section must be a mapping")
        retained = {
            key: item for key, item in value.items() if key in _SAFE_CONFIGURATION
        }
        if retained:
            allowed[section] = retained
    return _validate_audit_configuration(allowed)


def _audit_verification(
    verification: VerificationResult | None, *, enabled: bool
) -> AuditVerification:
    """Project rich verification runtime data without copying any prose fields."""
    if verification is None:
        return AuditVerification(
            enabled=enabled,
            pass_count=0,
            exhausted=False,
            failed=False,
            passes=(),
            repairs=(),
            usage=(),
            warning_codes=(),
            limitation_codes=(),
            failure_codes=(),
        )
    if not enabled:
        if (
            verification.pass_results
            or verification.passes
            or verification.selections
            or verification.repairs
            or verification.generations
            or verification.diagnostic_codes
            or verification.exhausted
            or verification.failed
            or verification.phase_generations
            or verification.warning_codes
            or verification.limitation_codes
            or verification.failure_codes
        ):
            raise AuditError("disabled verification result must not contain activity")
        return AuditVerification(
            enabled=False,
            pass_count=0,
            exhausted=False,
            failed=False,
            passes=(),
            repairs=(),
            usage=(),
            warning_codes=(),
            limitation_codes=(),
            failure_codes=(),
        )
    pass_results = verification.pass_results
    if verification.passes and len(pass_results) != len(verification.passes):
        raise AuditError("verification audit projection requires complete pass records")
    records: list[AuditVerificationPass] = []
    for item in pass_results:
        pass_indexes = {
            generation.pass_index for generation in item.phase_generations
        } or {assessment.pass_index for assessment in item.assessments}
        if not pass_indexes:
            pass_indexes = {int(span.span_id[1:3]) for span in item.spans}
        if len(pass_indexes) != 1:
            raise AuditError("verification pass record does not match its pass index")
        pass_index = pass_indexes.pop()
        records.append(
            AuditVerificationPass(
                pass_index=pass_index,
                complete=(
                    not item.failed
                    and not item.unresolved
                    and {selection.claim_id for selection in item.selections}
                    == {claim.claim_id for claim in item.claims}
                    and {assessment.claim_id for assessment in item.assessments}
                    == {claim.claim_id for claim in item.claims}
                ),
                spans=tuple(
                    AuditVerificationSpan(
                        span_id=span.span_id,
                        ordinal=span.ordinal,
                        start=span.start,
                        end=span.end,
                        content_hash=span.content_hash,
                    )
                    for span in item.spans
                ),
                claims=tuple(
                    AuditVerificationClaim(
                        claim_id=claim.claim_id,
                        span_id=claim.span_id,
                        ordinal=claim.ordinal,
                        is_fallback=claim.is_fallback,
                    )
                    for claim in item.claims
                ),
                selections=tuple(
                    AuditVerificationSelection(
                        claim_id=selection.claim_id,
                        selected_ids=selection.selected_ids,
                        examined_ids=selection.examined_ids,
                        omitted_ids=selection.omitted_ids,
                        token_cost=selection.token_cost,
                        retrieval_method=selection.retrieval_method,
                        retrieval_complete=selection.retrieval_complete,
                    )
                    for selection in item.selections
                ),
                assessments=tuple(
                    AuditVerificationAssessment(
                        claim_id=assessment.claim_id,
                        verdict=assessment.verdict.value,
                        verifier_provider=redact_text(assessment.verifier_provider),
                        verifier_model=redact_text(assessment.verifier_model),
                        prompt_version=assessment.prompt_version,
                        findings=tuple(
                            AuditVerificationFinding(
                                claim_id=finding.claim_id,
                                verdict=finding.verdict.value,
                                evidence_ids=finding.evidence_ids,
                            )
                            for finding in assessment.findings
                        ),
                        reassessment=(
                            AuditVerificationReassessment(
                                tolerance=assessment.reassessment.tolerance.value,
                                original_verdict=assessment.reassessment.original_verdict.value,
                                original_findings=tuple(
                                    AuditVerificationFinding(
                                        claim_id=finding.claim_id,
                                        verdict=finding.verdict.value,
                                        evidence_ids=finding.evidence_ids,
                                    )
                                    for finding in assessment.reassessment.original_findings
                                ),
                            )
                            if assessment.reassessment is not None
                            else None
                        ),
                    )
                    for assessment in item.assessments
                ),
                unresolved=tuple(
                    AuditVerificationUnresolved(
                        item_id=work.item_id, phase=work.phase.value, reason=work.reason
                    )
                    for work in item.unresolved
                ),
            )
        )
    phase_generations = verification.phase_generations or tuple(
        phase_generation
        for item in pass_results
        for phase_generation in item.phase_generations
    )
    return AuditVerification(
        enabled=True,
        pass_count=len(records),
        exhausted=verification.exhausted,
        failed=verification.failed,
        passes=tuple(records),
        repairs=tuple(
            AuditVerificationRepair(
                span_id=repair.span_id,
                original_hash=repair.original_hash,
                triggering_claim_ids=repair.triggering_claim_ids,
                action=repair.action.value,
            )
            for repair in verification.repairs
        ),
        usage=tuple(
            AuditVerificationUsage(
                phase=item.phase.value,
                pass_index=item.pass_index,
                prompt_version=item.prompt_version,
                provider=(
                    redact_text(item.generation.provider)
                    if item.generation is not None
                    else None
                ),
                model=(
                    redact_text(item.generation.model)
                    if item.generation is not None
                    else None
                ),
                input_tokens=(
                    item.generation.input_tokens
                    if item.generation is not None
                    else None
                ),
                output_tokens=(
                    item.generation.output_tokens
                    if item.generation is not None
                    else None
                ),
                finish_status=(
                    redact_text(item.generation.finish_status)
                    if item.generation is not None
                    and item.generation.finish_status is not None
                    else None
                ),
            )
            for item in phase_generations
        ),
        warning_codes=tuple(
            dict.fromkeys((*verification.warning_codes, *verification.diagnostic_codes))
        ),
        limitation_codes=verification.limitation_codes,
        failure_codes=verification.failure_codes,
    )


def build_audit_artifact(
    *,
    source_id: str,
    strategy: str,
    model: str,
    configuration: Mapping[str, object],
    segments: Sequence[SourceSegment],
    nodes: Sequence[TreeNode],
    root_node_id: str,
    citations: Sequence[Citation],
    generations: Sequence[GenerationResult] = (),
    warnings: Sequence[str] = (),
    failures: Sequence[str] = (),
    verification: VerificationResult | None = None,
    verification_enabled: bool = False,
    publication: AuditPublication | None = None,
    segment_parents: Mapping[str, str] | None = None,
    reliability_cache: Mapping[str, Sequence[str]] | None = None,
    reliability_resume: Mapping[str, object] | None = None,
    reliability_attempts: Sequence[Mapping[str, object]] | None = None,
) -> AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4:
    """Build a validated artifact without retaining source text or request data.

    Only `publication` holds reader-facing prose. `segment_parents` maps each
    verification passage in `segments` to the leaf segment it lies in.
    """
    parents = segment_parents or {}
    # Determine if we have reliability metadata
    has_reliability = (
        reliability_cache is not None
        or reliability_resume is not None
        or reliability_attempts is not None
    )

    # Build reliability object if present
    reliability_obj = None
    if has_reliability:
        cache_obj = None
        if reliability_cache is not None:
            cache_obj = AuditCacheMetadata(
                cache_hits=tuple(reliability_cache.get("cache_hits", ())),
                cache_misses=tuple(reliability_cache.get("cache_misses", ())),
                invalidation_reasons=tuple(
                    reliability_cache.get("invalidation_reasons", ())
                ),
            )

        resume_values: dict[str, object] = {}
        if reliability_resume is not None:
            resume_values = {
                "resumed": reliability_resume.get("resumed", False),
                "reused_count": reliability_resume.get("reused_count", 0),
                "recomputed_count": reliability_resume.get("recomputed_count", 0),
            }

        attempts = ()
        if reliability_attempts is not None:
            attempts = tuple(
                AuditRetryAttempt(
                    work_id=attempt["work_id"],
                    attempt_count=attempt["attempt_count"],
                    failure_reasons=attempt.get("failure_reasons", ()),
                )
                for attempt in reliability_attempts
            )

        reliability_obj = AuditReliability(
            cache=cache_obj, attempts=attempts, **resume_values
        )

    has_merges = any(len(node.children) > 1 for node in nodes)
    has_grounding = any(node.grounding is not None for node in nodes)
    uses_audit_v4 = has_merges or has_grounding
    common = {
        "source_id": source_id,
        "strategy": strategy,
        "model": redact_text(model),
        "configuration": _allowlisted_configuration(configuration),
        "source_segments": tuple(
            _audit_segment(segment, parent_segment_id=parents.get(segment.segment_id))
            for segment in segments
        ),
        "tree_nodes": tuple(
            _audit_node(node, include_grounding=uses_audit_v4) for node in nodes
        ),
        "root_node_id": root_node_id,
        "citations": tuple(
            AuditCitation(
                segment_id=citation.segment_id,
                source_id=citation.source_id,
                order=citation.order,
            )
            for citation in citations
        ),
        "usage": tuple(_usage(generation) for generation in generations),
        "warnings": tuple(warnings),
        "failures": tuple(failures),
        "verification": _audit_verification(
            verification,
            enabled=verification_enabled
            or bool(
                verification
                and (
                    verification.pass_results
                    or verification.passes
                    or verification.repairs
                    or verification.phase_generations
                )
            ),
        ),
        "publication": publication,
    }
    if uses_audit_v4:
        return AuditArtifactV4(
            schema_version=AUDIT_SCHEMA_VERSION_V4,
            reliability=reliability_obj,
            **common,
        )
    if reliability_obj is None:
        return AuditArtifactV2(schema_version=AUDIT_SCHEMA_VERSION, **common)
    return AuditArtifactV3(
        schema_version=AUDIT_SCHEMA_VERSION_V3,
        reliability=reliability_obj,
        **common,
    )


def with_sections(
    artifact: AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4,
    sections: Sequence[AuditSection],
) -> AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4:
    """The artifact with its section records, validated against the rest of the audit."""
    data = artifact.model_dump(mode="json")
    data["sections"] = [section.model_dump(mode="json") for section in sections]
    try:
        return AuditArtifact.model_validate(data)
    except ValueError as error:
        raise AuditError("audit section records failed validation") from error


def serialize_audit(artifact: AuditArtifactV2 | AuditArtifactV3 | AuditArtifactV4) -> bytes:
    """Serialize canonically and prove the written representation is valid."""
    encoded = json.dumps(
        artifact.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        AuditArtifact.model_validate_json(encoded)
    except ValueError as error:
        raise AuditError("audit serialization failed validation") from error
    return encoded


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def write_audit(path: Path, artifact: AuditArtifact) -> None:
    """Atomically replace an artifact only after canonical validation succeeds."""
    payload = serialize_audit(artifact)
    _atomic_replace(path, payload)
