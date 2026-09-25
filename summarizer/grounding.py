"""Deterministic selection of authoritative source passages for a merge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

from summarizer.budget import BudgetError
from summarizer.summaries import EvidenceItem, SummaryNode
from summarizer.tokenization import TokenCounter


# Shorter excerpts leave too little context to ground a merged summary.
MIN_EXCERPT_CHARS = 400


@dataclass(frozen=True)
class GroundingPolicy:
    """The input space reserved for original source text at each merge."""

    max_tokens: int

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("grounding max_tokens must be positive")


@dataclass(frozen=True)
class SourcePassage:
    """A citable source core selected for a merge request.

    Normally the complete core. When excerpts are allowed and no complete
    core fits, one contiguous excerpt of a core, so every quote the model
    takes from it is still verbatim source text.
    """

    segment_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.segment_id.strip() or not self.text:
            raise ValueError("a source passage needs a segment identifier and text")


@dataclass(frozen=True)
class GroundingSelection:
    passages: tuple[SourcePassage, ...]
    selected_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]


def serialize_source_passage(passage: SourcePassage) -> str:
    """Use one compact, deterministic representation for budget and payload."""
    return json.dumps(
        {"segment_id": passage.segment_id, "text": passage.text},
        separators=(",", ":"),
        sort_keys=True,
    )


def _evidence_ids(items: Sequence[EvidenceItem]) -> tuple[str, ...]:
    return tuple(item.segment_id for item in items)


def _candidates(children: Sequence[SummaryNode]) -> tuple[tuple[str, bool], ...]:
    """Return unique source IDs in priority order, with mandatory support first."""
    ranked: list[tuple[str, bool]] = []
    for child in children:
        for annotation in child.contradictions:
            ranked.extend((identifier, True) for identifier in _evidence_ids(annotation.evidence))
    for child in children:
        for annotation in child.qualifications:
            ranked.extend((identifier, True) for identifier in _evidence_ids(annotation.evidence))
        for unit in child.content_units:
            if unit.uncertain or unit.qualification is not None:
                ranked.extend((identifier, True) for identifier in _evidence_ids(unit.evidence))
    for child in children:
        ranked.extend((identifier, False) for identifier in _evidence_ids(child.quotations))
    for child in children:
        for unit in child.content_units:
            ranked.extend((identifier, False) for identifier in _evidence_ids(unit.evidence))
    for child in children:
        ranked.extend((identifier, False) for identifier in child.provenance)

    selected: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for identifier, mandatory in ranked:
        if identifier not in seen:
            selected.append((identifier, mandatory))
            seen.add(identifier)
    return tuple(selected)


def _child_quotes(children: Sequence[SummaryNode], identifier: str) -> tuple[str, ...]:
    quotes: list[str] = []
    for child in children:
        evidence = [*child.quotations]
        for unit in child.content_units:
            evidence.extend(unit.evidence)
        for annotation in (*child.qualifications, *child.contradictions):
            evidence.extend(annotation.evidence)
        quotes.extend(
            item.quote for item in evidence if item.segment_id == identifier and item.quote
        )
    return tuple(quotes)


def _excerpt(text: str, anchor: int, length: int) -> str:
    """The `length`-character window around `anchor`, trimmed to whole words."""
    start = max(0, min(anchor - length // 2, len(text) - length))
    end = min(len(text), start + length)
    if start > 0:
        space = text.find(" ", start, end)
        start = space + 1 if space != -1 else start
    if end < len(text):
        space = text.rfind(" ", start, end)
        end = space if space > start else end
    return text[start:end].strip()


def _fit_excerpt(
    identifier: str,
    text: str,
    quotes: Sequence[str],
    passages: Sequence[SourcePassage],
    cost: Callable[[tuple[SourcePassage, ...]], int],
    max_tokens: int,
) -> SourcePassage | None:
    """The longest excerpt of `text` that fits beside `passages`, if any.

    Centred on the first child quote found in the core, so the excerpt keeps
    the evidence the children already cite.
    """
    if len(text) <= MIN_EXCERPT_CHARS:
        return None
    anchor = 0
    for quote in quotes:
        position = text.find(quote)
        if position != -1:
            anchor = position + len(quote) // 2
            break
    best = None
    low, high = MIN_EXCERPT_CHARS, len(text) - 1
    while low <= high:
        length = (low + high) // 2
        excerpt = _excerpt(text, anchor, length)
        candidate = SourcePassage(identifier, excerpt) if excerpt else None
        if candidate is not None and cost((*passages, candidate)) <= max_tokens:
            best = candidate
            low = length + 1
        else:
            high = length - 1
    return best


def select_source_passages(
    children: Sequence[SummaryNode],
    *,
    source: Mapping[str, str],
    counter: TokenCounter,
    policy: GroundingPolicy,
    selection_cost: Callable[[tuple[SourcePassage, ...]], int] | None = None,
    allow_excerpts: bool = False,
) -> GroundingSelection:
    """Pack complete authoritative source passages into a fixed token reserve.

    Model output supplies only candidate identifiers. The source mapping is the
    authority for both text and eligibility, so an unknown candidate fails
    before a request is built. Ambiguous or qualified support is mandatory;
    ordinary support may be omitted only after the reserve is exhausted.

    With `allow_excerpts`, a mandatory passage that does not fit whole, or the
    first candidate when no passage fits whole, is replaced by the longest
    contiguous excerpt that fits instead of failing the merge.
    """
    if not children:
        raise ValueError("source grounding requires at least one child")

    def measure(tentative: tuple[SourcePassage, ...]) -> int:
        if selection_cost is not None:
            return selection_cost(tentative)
        return counter.count("\n".join(serialize_source_passage(item) for item in tentative))

    def excerpt_of(identifier: str) -> SourcePassage | None:
        if not allow_excerpts:
            return None
        return _fit_excerpt(
            identifier,
            source[identifier],
            _child_quotes(children, identifier),
            passages,
            measure,
            policy.max_tokens,
        )

    candidates = _candidates(children)
    passages: list[SourcePassage] = []
    selected_ids: list[str] = []
    rejected: list[tuple[int, str]] = []
    for identifier, mandatory in candidates:
        if identifier not in source:
            raise ValueError(f"source text is missing for segment {identifier}")
        passage = SourcePassage(identifier, source[identifier])
        cost = measure(tuple((*passages, passage)))
        if cost <= policy.max_tokens:
            passages.append(passage)
            selected_ids.append(identifier)
            continue
        if mandatory:
            excerpt = excerpt_of(identifier)
            if excerpt is None:
                raise BudgetError(
                    f"grounding reserve of {policy.max_tokens} tokens cannot hold "
                    f"mandatory source passage {identifier} costing {cost} tokens"
                )
            passages.append(excerpt)
            selected_ids.append(identifier)
            continue
        rejected.append((cost, identifier))

    if not passages and rejected:
        first_id = rejected[0][1]
        excerpt = excerpt_of(first_id)
        if excerpt is not None:
            passages.append(excerpt)
            selected_ids.append(first_id)
    if not passages:
        smallest_cost, smallest_id = min(rejected)
        raise BudgetError(
            f"grounding reserve of {policy.max_tokens} tokens cannot hold source "
            f"passage {smallest_id} costing {smallest_cost} tokens"
        )
    omitted_ids = tuple(identifier for identifier, _ in candidates if identifier not in selected_ids)
    return GroundingSelection(
        passages=tuple(passages),
        selected_ids=tuple(selected_ids),
        omitted_ids=omitted_ids,
    )
