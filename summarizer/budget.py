from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal

from summarizer.config import StrategyConfig, StrategyName
from summarizer.ingestion import SourceDocument
from summarizer.leaf import build_leaf_request
from summarizer.providers.base import GenerationRequest
from summarizer.reask import MAX_REASON_BYTES, reask_request
from summarizer.segmentation import BoundaryKind, SourceSegment
from summarizer.summaries import (
    MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES,
    MAX_QUOTE_CANDIDATE_JSON_BYTES,
    leaf_summary_schema,
)
from summarizer.tokenization import TokenCounter

# Context windows have no offline source of truth, and for OpenAI no online one
# either: the installed openai client exposes only {id, created, object,
# owned_by, shutdown_date} per model, and tiktoken maps names to encodings
# rather than to sizes. Ollama does report an architectural length through
# show(), but the key is architecture-prefixed rather than tag-named (the tag
# `qwen3.8` reports `qwen35.context_length`) and reaching it is a network call
# that is unavailable before a pull. So this table is maintained by hand, and
# anything missing from it is reported as assumed rather than as knowledge.
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
}

# Consulted by longest prefix after the exact table. Only model families listed
# here accept suffixed names; repeating an exact entry here explicitly opts its
# dated snapshots into family resolution.
_MODEL_PREFIX_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-5": 400_000,
    "o1": 200_000,
    # Nested under "o1" with a different window, which is what makes the
    # longest-prefix rule observable rather than incidental.
    "o1-mini": 128_000,
    "o3": 200_000,
    "o4": 200_000,
}

# Deliberately small: an assumed window should not let an unknown model gamble
# a large request. Selection routes an assumed window to hierarchical rather
# than trusting it.
ASSUMED_CONTEXT_WINDOW = 8_192


class BudgetError(ValueError):
    """A configuration leaves no room to send a request safely."""


@dataclass(frozen=True)
class ContextWindow:
    """A model's total context size, and whether that size is actually known."""

    tokens: int
    assumed: bool

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError("context window tokens must be positive")


def resolve_context_window(
    *,
    provider: str,
    model: str,
    explicit: int | None = None,
) -> ContextWindow:
    """Resolve a model's context window without contacting a provider.

    An explicit value is authoritative. For OpenAI, the exact-name table is
    consulted first and then the longest matching known family prefix. Other
    providers skip those OpenAI tables. An unresolved model yields an assumed
    window flagged as such, so a caller can decline to rely on it.
    """
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("context window must be positive")
        return ContextWindow(tokens=explicit, assumed=False)

    if provider.strip().lower() == "openai":
        name = model.strip()
        exact = _MODEL_CONTEXT_WINDOWS.get(name)
        if exact is not None:
            return ContextWindow(tokens=exact, assumed=False)

        matches = [
            prefix
            for prefix in _MODEL_PREFIX_CONTEXT_WINDOWS
            if name.startswith(prefix)
        ]
        if matches:
            longest = max(matches, key=len)
            return ContextWindow(
                tokens=_MODEL_PREFIX_CONTEXT_WINDOWS[longest], assumed=False
            )

    return ContextWindow(tokens=ASSUMED_CONTEXT_WINDOW, assumed=True)


@dataclass(frozen=True)
class OverheadMeasurement:
    """What a request costs before any source text is added."""

    instructions: int
    schema: int
    fencing: int

    @property
    def total(self) -> int:
        return self.instructions + self.schema + self.fencing


def measure_overhead(
    counter: TokenCounter,
    *,
    with_overlap: bool,
    provider_schema_reserve: int = 0,
) -> OverheadMeasurement:
    """Measure per-request overhead rather than assuming a constant.

    A hard-coded figure would be wrong the moment the prompt or the record
    changes, and the schema dominates: it is roughly two thirds of the total,
    because the record's own docstrings are rendered into it and shipped on
    every call.

    The schema is counted in its compact serialization. That approximates what
    a provider's tokenizer sees; an indented form would overstate it by about
    two thirds. Approximating downward is why a safety margin exists.
    """
    # The probe uses a region framing, which is the longer of the two by three
    # tokens, so the measurement covers a whole-document request as well.
    probe = _overhead_probe_segment(with_overlap=with_overlap)
    request = build_leaf_request(probe, model="probe", timeout_seconds=1)

    # The schema a real request carries also enumerates verbatim quote
    # candidates, which this probe is too small to produce. Their compact,
    # ASCII-escaped JSON contribution is capped in bytes. Byte-fallback
    # tokenizers cannot emit more tokens than bytes, so the reserve is safe.
    schema = (
        counter.count(
            json.dumps(leaf_summary_schema(), separators=(",", ":"), sort_keys=True)
        )
        + MAX_QUOTE_CANDIDATE_JSON_BYTES
        + provider_schema_reserve
    )
    fencing = counter.count(request.input_text) - counter.count(probe.text)
    return OverheadMeasurement(
        instructions=counter.count(request.instructions),
        schema=schema,
        fencing=max(fencing, 0),
    )


def _overhead_probe_segment(*, with_overlap: bool) -> SourceSegment:
    """Build the smallest segment that exercises the real request builder.

    Measuring the assembled request rather than its parts means the fences and
    the overlap instruction block are counted exactly as they are sent.
    """
    text = "probe"
    if not with_overlap:
        return SourceSegment(
            segment_id="S000001",
            source_id="0" * 64,
            order=0,
            text=text,
            core_start=0,
            core_end=len(text),
            context_start=0,
            context_end=len(text),
            core_token_count=1,
            token_count=1,
            leading_overlap_tokens=0,
            trailing_overlap_tokens=0,
            boundary_kind=BoundaryKind.PARAGRAPH,
        )

    padded = f"a{text}b"
    return SourceSegment(
        segment_id="S000001",
        source_id="0" * 64,
        order=0,
        text=padded,
        core_start=1,
        core_end=1 + len(text),
        context_start=0,
        context_end=len(padded),
        core_token_count=1,
        token_count=1,
        leading_overlap_tokens=1,
        trailing_overlap_tokens=1,
        boundary_kind=BoundaryKind.PARAGRAPH,
    )


def safety_margin(window_tokens: int, config: StrategyConfig) -> int:
    """Reserve the larger of a fixed floor and a proportional share.

    A fixed margin alone is negligible against a very large window; a
    proportional one alone discards thousands of tokens there while
    under-protecting a small window.
    """
    proportional = int(window_tokens * config.safety_margin_fraction)
    return max(config.safety_margin_tokens, proportional)


def correction_headroom(counter: TokenCounter) -> int:
    """Bound what one re-ask adds to a request's instructions.

    A re-ask appends one note to the original instructions and never
    accumulates notes, so the bound is the longer note template plus the
    longest reason `rejection_reason` can return. That reason is capped in
    UTF-8 bytes, and no counter here charges more than one token per byte.
    """
    probe = GenerationRequest(
        model="probe", instructions="probe", input_text="probe", timeout_seconds=1
    )
    structured = replace(probe, response_schema={}, schema_name="probe")
    template = max(
        counter.count(reask_request(request, "").instructions)
        - counter.count(request.instructions)
        for request in (probe, structured)
    )
    return max(template, 0) + MAX_REASON_BYTES


RequestStage = Literal["direct", "leaf", "merge", "editorial", "compression"]


class BudgetFailure(str, Enum):
    """Why one request cannot be sent at its budgeted size.

    Every member describes the request configuration: the context window and
    the reserves charged against it. None of them is a finding that the
    requested summary target is impossible; the error carries that target's
    output allowance unchanged.
    """

    # The output allowance takes the whole usable context or more.
    OUTPUT_EXCEEDS_CONTEXT = "output_exceeds_context"
    # The allowance fits, but prompt, schema, evidence and correction reserves
    # leave no source input.
    NO_INPUT_CAPACITY = "no_input_capacity"
    # The request's measured input is larger than its input capacity.
    INPUT_EXCEEDS_CAPACITY = "input_exceeds_capacity"
    # The input capacity cannot hold the two largest-sized children a merge
    # needs to make progress.
    MERGE_PAIR_EXCEEDS_CAPACITY = "merge_pair_exceeds_capacity"


@dataclass(frozen=True)
class RequestBudget:
    """Every term one stage's request is charged against its context window.

    `input_capacity` is what remains for source text or child summaries:
    the window less the safety margin, the output allowance, the measured
    instructions, schema and fencing, the evidence reserve, and room for one
    correction note. It may be non-positive only inside a `RequestBudgetError`.
    """

    stage: RequestStage
    context_window_tokens: int
    overhead: OverheadMeasurement
    evidence_tokens: int
    correction_headroom_tokens: int
    output_allowance_tokens: int
    safety_margin_tokens: int

    @property
    def usable_context_tokens(self) -> int:
        return self.context_window_tokens - self.safety_margin_tokens

    @property
    def input_capacity(self) -> int:
        return (
            self.usable_context_tokens
            - self.output_allowance_tokens
            - self.overhead.total
            - self.evidence_tokens
            - self.correction_headroom_tokens
        )

    def arithmetic(self) -> str:
        """Show the subtraction that produced `input_capacity`."""
        return (
            f"context window {self.context_window_tokens} - safety margin "
            f"{self.safety_margin_tokens} - output allowance "
            f"{self.output_allowance_tokens} - instructions "
            f"{self.overhead.instructions} - schema {self.overhead.schema} - "
            f"fencing {self.overhead.fencing} - evidence {self.evidence_tokens} "
            f"- correction headroom {self.correction_headroom_tokens} = "
            f"{self.input_capacity} input tokens"
        )

    def require_input(self, tokens: int) -> None:
        """Refuse a request whose measured input exceeds the capacity."""
        if tokens > self.input_capacity:
            raise RequestBudgetError(
                BudgetFailure.INPUT_EXCEEDS_CAPACITY,
                self,
                f"an input of {tokens} tokens exceeds the input capacity of "
                f"{self.input_capacity}",
            )

    def require_merge_pair(self, largest_child_tokens: int) -> None:
        """Refuse a merge budget that cannot hold two of its largest child.

        Sized from the largest child, as fanout is, so a pair is never
        admitted that fits only on average.
        """
        pair = 2 * largest_child_tokens
        if pair > self.input_capacity:
            raise RequestBudgetError(
                BudgetFailure.MERGE_PAIR_EXCEEDS_CAPACITY,
                self,
                f"a largest child of {largest_child_tokens} tokens makes a pair of "
                f"{pair} against an input capacity of {self.input_capacity}",
            )

    @property
    def request_capacity(self) -> int:
        """What one whole assembled request may measure before a re-ask.

        The input capacity plus the prompt, schema, fencing and evidence it
        was reduced by, for stages that measure the request they assemble.
        """
        return self.input_capacity + self.overhead.total + self.evidence_tokens

    def require_request(self, tokens: int) -> None:
        """Refuse an assembled request whose measured size exceeds the budget."""
        if tokens > self.request_capacity:
            raise RequestBudgetError(
                BudgetFailure.INPUT_EXCEEDS_CAPACITY,
                self,
                f"an assembled request of {tokens} tokens exceeds the request "
                f"capacity of {self.request_capacity}",
            )


class RequestBudgetError(BudgetError):
    """A request cannot be sent at its budgeted size, with the arithmetic."""

    def __init__(
        self, failure: BudgetFailure, budget: RequestBudget, detail: str
    ) -> None:
        self.failure = failure
        self.budget = budget
        super().__init__(
            f"{budget.stage} request is infeasible ({failure.value}): {detail}; "
            f"{budget.arithmetic()}. This is a limit of the request "
            f"configuration, not of the requested output target"
        )


def plan_request(
    stage: RequestStage,
    *,
    window: ContextWindow,
    overhead: OverheadMeasurement,
    output_allowance: int,
    correction_headroom: int,
    config: StrategyConfig,
    evidence: int = 0,
) -> RequestBudget:
    """Budget one stage's request, or raise `RequestBudgetError`.

    The output allowance is taken as given and never reduced to make room:
    lowering it would silently shorten the requested target. A budget that
    cannot be met is refused with its classification and arithmetic.
    """
    if output_allowance <= 0:
        raise ValueError("output allowance must be positive")
    if correction_headroom < 0 or evidence < 0:
        raise ValueError("correction headroom and evidence must not be negative")
    budget = RequestBudget(
        stage=stage,
        context_window_tokens=window.tokens,
        overhead=overhead,
        evidence_tokens=evidence,
        correction_headroom_tokens=correction_headroom,
        output_allowance_tokens=output_allowance,
        safety_margin_tokens=safety_margin(window.tokens, config),
    )
    if budget.usable_context_tokens <= 0:
        raise RequestBudgetError(
            BudgetFailure.NO_INPUT_CAPACITY,
            budget,
            f"a safety margin of {budget.safety_margin_tokens} tokens leaves no "
            f"usable context in a window of {window.tokens}",
        )
    if output_allowance >= budget.usable_context_tokens:
        raise RequestBudgetError(
            BudgetFailure.OUTPUT_EXCEEDS_CONTEXT,
            budget,
            f"an output allowance of {output_allowance} tokens does not fit a "
            f"usable context of {budget.usable_context_tokens}",
        )
    if budget.input_capacity <= 0:
        raise RequestBudgetError(
            BudgetFailure.NO_INPUT_CAPACITY,
            budget,
            "the reserves leave no room for input",
        )
    return budget


def measure_request_tokens(
    request: GenerationRequest,
    counter: TokenCounter,
    *,
    provider_schema_reserve: int = 0,
) -> int:
    """Measure an assembled request: instructions, input and compact schema."""
    schema = (
        counter.count(json.dumps(request.response_schema, separators=(",", ":")))
        if request.response_schema is not None
        else 0
    )
    return (
        counter.count(request.instructions)
        + counter.count(request.input_text)
        + schema
        + provider_schema_reserve
    )


# The chapter-comparison study's allowance (experiments/chapter_comparison):
# three tokens per requested word leaves room for tokenization, JSON escaping
# and the editorial prompt's permitted overshoot, plus a fixed envelope.
_EDITORIAL_TOKENS_PER_TARGET_WORD = 3
_EDITORIAL_ENVELOPE_TOKENS = 1_024


def editorial_output_allowance(target_words: int, config: StrategyConfig) -> int:
    """Return the editorial output allowance for a requested length.

    Derived from the target rather than from the per-summary allowance, so a
    long target is budgeted in full or refused, never cut short at transport.
    """
    return max(
        config.max_output_tokens,
        _EDITORIAL_TOKENS_PER_TARGET_WORD * target_words + _EDITORIAL_ENVELOPE_TOKENS,
    )


@dataclass(frozen=True)
class RequestLimits:
    """The run-wide terms every stage plans its request budget from."""

    window: ContextWindow
    config: StrategyConfig
    counter: TokenCounter
    correction_headroom: int

    @classmethod
    def from_report(
        cls, report: BudgetReport, counter: TokenCounter, config: StrategyConfig
    ) -> RequestLimits:
        return cls(
            window=ContextWindow(
                tokens=report.context_window_tokens,
                assumed=report.context_window_assumed,
            ),
            config=config,
            counter=counter,
            correction_headroom=report.correction_headroom_tokens,
        )

    def plan(
        self,
        stage: RequestStage,
        *,
        overhead: OverheadMeasurement,
        output_allowance: int | None = None,
        evidence: int = 0,
        correctable: bool = True,
    ) -> RequestBudget:
        """Plan one request; the allowance defaults to the per-summary one.

        `correctable` is false for a stage that is never re-asked, which
        needs no room for a correction note.
        """
        return plan_request(
            stage,
            window=self.window,
            overhead=overhead,
            output_allowance=(
                self.config.max_output_tokens
                if output_allowance is None
                else output_allowance
            ),
            correction_headroom=self.correction_headroom if correctable else 0,
            config=self.config,
            evidence=evidence,
        )


@dataclass(frozen=True)
class BudgetReport:
    """Everything a strategy decision saw, and why it decided.

    Returned to callers as run metadata, and safe to hash for later cache keys
    because identical inputs produce an identical report.
    """

    strategy: StrategyName
    reason: str
    context_window_tokens: int
    context_window_assumed: bool
    counter_identity: str
    counter_exact: bool
    overhead: OverheadMeasurement
    reserved_output_tokens: int
    safety_margin_tokens: int
    correction_headroom_tokens: int
    usable_input_capacity: int
    document_tokens: int
    fits: bool


def select_strategy(
    document: SourceDocument,
    counter: TokenCounter,
    *,
    provider: str,
    model: str,
    config: StrategyConfig,
) -> BudgetReport:
    """Choose an execution path from a measured budget.

    The document is measured with `counter.count(document.text)` - exactly the
    text a direct request sends - rather than by summing segment counts, which
    differs under a byte-pair encoder and diverges further under overlap.
    """
    window = resolve_context_window(
        provider=provider, model=model, explicit=config.context_window
    )
    if config.strategy == "direct" and window.assumed:
        raise BudgetError(
            f"direct summarization was requested but the context window for "
            f"model {model!r} is not known, so a fit cannot be established; "
            f"pass an explicit context window to proceed. This matters most "
            f"on a provider that truncates an oversized prompt silently "
            f"rather than rejecting it."
        )

    # A direct request carries no overlap. A stage that sends overlap-carrying
    # requests must measure its own overhead: the overlap variant is about 120
    # tokens larger, and sizing against this figure would under-reserve.
    overhead = measure_overhead(
        counter,
        with_overlap=False,
        provider_schema_reserve=(
            MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES
            if provider.strip().lower() == "ollama"
            else 0
        ),
    )
    budget = plan_request(
        "direct",
        window=window,
        overhead=overhead,
        output_allowance=config.max_output_tokens,
        correction_headroom=correction_headroom(counter),
        config=config,
    )
    capacity = budget.input_capacity
    document_tokens = counter.count(document.text)
    fits = document_tokens <= capacity

    capped = (
        config.max_direct_tokens is not None
        and document_tokens > config.max_direct_tokens
    )

    if config.strategy == "direct":
        budget.require_input(document_tokens)

    strategy, reason = _decide(
        config=config,
        fits=fits,
        capped=capped,
        assumed=window.assumed,
        document_tokens=document_tokens,
        capacity=capacity,
    )

    return BudgetReport(
        strategy=strategy,
        reason=reason,
        context_window_tokens=window.tokens,
        context_window_assumed=window.assumed,
        counter_identity=counter.identity,
        counter_exact=counter.exact,
        overhead=overhead,
        reserved_output_tokens=budget.output_allowance_tokens,
        safety_margin_tokens=budget.safety_margin_tokens,
        correction_headroom_tokens=budget.correction_headroom_tokens,
        usable_input_capacity=capacity,
        document_tokens=document_tokens,
        fits=fits,
    )


def _decide(
    *,
    config: StrategyConfig,
    fits: bool,
    capped: bool,
    assumed: bool,
    document_tokens: int,
    capacity: int,
) -> tuple[StrategyName, str]:
    if config.strategy == "direct":
        return "direct", (
            f"direct was requested and {document_tokens} tokens fit a capacity "
            f"of {capacity}"
        )
    if config.strategy == "hierarchical":
        return "hierarchical", "hierarchical was requested explicitly"

    if not fits:
        return "hierarchical", (
            f"{document_tokens} tokens exceed a usable input capacity of "
            f"{capacity}"
        )
    if assumed:
        return "hierarchical", (
            "the context window is assumed rather than known, so a direct "
            "request cannot be shown to fit"
        )
    if capped:
        return "hierarchical", (
            f"{document_tokens} tokens exceed the configured direct cap of "
            f"{config.max_direct_tokens}"
        )
    return "direct", (
        f"{document_tokens} tokens fit a usable input capacity of {capacity}"
    )
