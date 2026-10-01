"""Build a section tree from a text and its outline.

A pure function: `build_section_tree` turns a text plus heading entries into
a tree of sections whose own spans partition the text. Nothing here reads a
file, a database or a provider, so the same code serves the web importer
(`OutlineEntry`) and the CLI (`MarkdownHeading`).
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

# About one sentence: a typical English sentence runs 15-25 words. A section
# that earns less than this from the target is too small to stand as its own
# section in the summary.
SENTENCE_WORDS = 20


class HeadingLike(Protocol):
    """A heading placed in the text.

    `summarizer.segmentation.MarkdownHeading` and the web importer's
    `OutlineEntry` both satisfy this. `start` is where the heading begins
    (for `OutlineEntry`, where its title begins). Levels are 1 or more.
    """

    @property
    def title(self) -> str: ...

    @property
    def level(self) -> int: ...

    @property
    def start(self) -> int: ...


class PageExtent(Protocol):
    """Where a page's text sits in the text (end exclusive).

    The web importer's `PageSpan` satisfies this.
    """

    @property
    def page(self) -> int: ...

    @property
    def start(self) -> int: ...

    @property
    def end(self) -> int: ...


@dataclass(frozen=True, slots=True)
class SectionOutline:
    """What a run needs to summarize by section: the outline, and page extents if known."""

    headings: tuple[HeadingLike, ...]
    pages: tuple[PageExtent, ...] | None = None


@dataclass(frozen=True, slots=True)
class SectionNode:
    """One section. `text[start:end]` is its own text, excluding its children.

    `heading` is None for the untitled opening section. Pages are 1-based and
    inclusive, and None without a page map. `folded_headings` lists the
    headings of undersized sections merged into this one, in document order.
    `own_words` counts the words of the own text, leaving out the heading
    lines (this section's and any folded into it).
    """

    id: str
    heading: str | None
    level: int
    start: int
    end: int
    page_start: int | None
    page_end: int | None
    parent_id: str | None
    child_ids: tuple[str, ...]
    folded_headings: tuple[str, ...] = ()
    own_words: int = 0


@dataclass(frozen=True, slots=True)
class SectionTree:
    """Sections in document order (a parent precedes its children).

    The own spans of `nodes` are contiguous, ordered and cover the whole text.
    """

    nodes: tuple[SectionNode, ...]

    def get(self, section_id: str) -> SectionNode:
        for node in self.nodes:
            if node.id == section_id:
                return node
        raise KeyError(section_id)

    @property
    def roots(self) -> tuple[SectionNode, ...]:
        return tuple(node for node in self.nodes if node.parent_id is None)

    def subtree(self, section_id: str) -> tuple[SectionNode, ...]:
        """The section and all its descendants, in document order."""
        ids = {section_id}
        members: list[SectionNode] = []
        for node in self.nodes:
            if node.id in ids or node.parent_id in ids:
                ids.add(node.id)
                members.append(node)
        if not members:
            raise KeyError(section_id)
        return tuple(members)

    def subtree_pages(self, section_id: str) -> tuple[int, int] | None:
        """First and last page of the section's whole subtree, or None without a page map."""
        pages = [
            page
            for node in self.subtree(section_id)
            for page in (node.page_start, node.page_end)
            if page is not None
        ]
        return (min(pages), max(pages)) if pages else None

    def own_share_words(self, section_id: str, total_words: int) -> int:
        """The section's own text as a share of `total_words`, by words, before any minimum.

        A section's prose covers its own text only, so the share is its own
        words over the whole source's own words, heading lines excluded on
        both sides. A source with no words beyond headings gives every section 0.
        """
        total = sum(node.own_words for node in self.nodes)
        return round(total_words * self.get(section_id).own_words / total) if total else 0

    def target_words(self, section_id: str, total_words: int) -> int:
        """The section's share of `total_words`, by own words, at least one sentence."""
        return max(SENTENCE_WORDS, self.own_share_words(section_id, total_words))

    def is_heading_only(self, section_id: str, total_words: int) -> bool:
        """Whether the section has no own text worth a sentence: its prose is its heading alone.

        True for a section whose own text is only its heading, and for one whose
        share of `total_words` is under `SENTENCE_WORDS`.
        """
        return self.own_share_words(section_id, total_words) < SENTENCE_WORDS


class PageLookup:
    """The page holding an offset, from the pages' extents."""

    def __init__(self, pages: Sequence[PageExtent] | None) -> None:
        self._spans = sorted(
            (span for span in pages or () if span.end > span.start), key=lambda s: s.start
        )
        self._starts = [span.start for span in self._spans]

    def at(self, offset: int) -> int | None:
        if not self._spans:
            return None
        return self._spans[max(bisect.bisect_right(self._starts, offset) - 1, 0)].page


class _Draft:
    """A mutable node while the tree is built and folded."""

    __slots__ = ("heading", "level", "start", "end", "parent", "children", "folded")

    def __init__(self, heading: str | None, level: int, start: int, end: int) -> None:
        self.heading = heading
        self.level = level
        self.start = start
        self.end = end
        self.parent: _Draft | None = None
        self.children: list[_Draft] = []
        self.folded: list[str] = []


def build_section_tree(
    text: str,
    outline: Sequence[HeadingLike],
    *,
    target_words: int,
    pages: Sequence[PageExtent] | None = None,
) -> SectionTree:
    """Split `text` into sections at the outline's headings, then fold small ones.

    Own span: a section starts at the start of the line holding its heading,
    so heading markers such as `#` belong to the section they open and no
    section starts mid-line. It ends at the next heading in document order,
    which is its first child, or the next heading at the same or a higher
    level, or the end of the text. A level that skips (an L1 then an L3) makes
    the L3 a child of the L1.

    Text before the first heading is an untitled level-1 opening section. When
    that text is only whitespace there is no opening section, and the whitespace
    stays in the first heading's section, so every character belongs to exactly
    one section. An empty outline gives one untitled section for the whole text.

    Folding: a section's share of the target is its subtree's characters over
    all characters, times `target_words`. A childless section with a share under
    `SENTENCE_WORDS` merges into the preceding childless sibling, else the
    following one, else its parent's own text when it is the parent's first
    child (the only merge that keeps the parent's own span one range). A section
    with children is never folded, and nothing folds across parents. Folding
    repeats until nothing changes. The survivor keeps the first merged section's
    heading (the later heading when the first is untitled), and lists the
    other headings in `folded_headings`.

    Page ranges come from `pages` (start and last character of the own span).
    """
    total = len(text)
    starts: list[tuple[int, HeadingLike]] = []
    previous = 0
    for entry in sorted(outline, key=lambda item: item.start):
        line_start = text.rfind("\n", 0, min(max(entry.start, 0), total)) + 1
        start = max(line_start, previous)
        starts.append((start, entry))
        previous = start

    drafts: list[_Draft] = []
    if not starts:
        drafts.append(_Draft(None, 1, 0, total))
    else:
        if text[: starts[0][0]].strip():
            drafts.append(_Draft(None, 1, 0, starts[0][0]))
        else:
            starts[0] = (0, starts[0][1])
        for index, (start, entry) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else total
            drafts.append(_Draft(" ".join(entry.title.split()), entry.level, start, end))

    open_drafts: list[_Draft] = []
    roots: list[_Draft] = []
    for draft in drafts:
        while open_drafts and open_drafts[-1].level >= draft.level:
            open_drafts.pop()
        if open_drafts:
            draft.parent = open_drafts[-1]
            open_drafts[-1].children.append(draft)
        else:
            roots.append(draft)
        open_drafts.append(draft)

    _fold(roots, total, target_words)
    return _freeze(roots, pages, text)


def _body_words(text: str, draft: _Draft) -> int:
    """Words of the draft's own text, leaving out its heading lines.

    The heading is the first non-blank line, because a section starts at the
    line holding its heading. A folded heading is a later line whose letters
    and digits equal the heading's, dropped once per folded heading.
    """
    folded = [_letters(title) for title in draft.folded]
    words = 0
    heading_pending = draft.heading is not None
    for line in text[draft.start : draft.end].splitlines():
        if not line.strip():
            continue
        if heading_pending:
            heading_pending = False
            continue
        key = _letters(line)
        if key in folded:
            folded.remove(key)
            continue
        words += len(line.split())
    return words


def _letters(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum())


def _has_body(text: str, node: SectionNode) -> bool:
    """Whether a section's own span holds anything besides its heading lines.

    Compared on letters and digits only, so heading markup (`#`, underlines,
    numbering punctuation) does not count as text.
    """
    remaining = _letters(text[node.start : node.end])
    headings = ([node.heading] if node.heading else []) + list(node.folded_headings)
    for heading in headings:
        remaining = remaining.replace(_letters(heading), "", 1)
    return bool(remaining)


def own_text_spans(text: str, tree: SectionTree) -> tuple[tuple[str, int, int], ...]:
    """Return `(section id, start, end)` for each section with text of its own.

    A section whose own span is only its heading (a parent that goes straight
    to its subsections) has nothing to segment or summarize. If no section has
    any text beyond headings, every non-blank own span counts, so the run still
    has something to summarize.
    """
    spans = tuple(
        (node.id, node.start, node.end)
        for node in tree.nodes
        if _has_body(text, node)
    )
    if spans:
        return spans
    return tuple(
        (node.id, node.start, node.end)
        for node in tree.nodes
        if text[node.start : node.end].strip()
    )


def _siblings(draft: _Draft, roots: list[_Draft]) -> list[_Draft]:
    return draft.parent.children if draft.parent is not None else roots


def _fold_once(roots: list[_Draft], total: int, target_words: int) -> bool:
    def walk(nodes: list[_Draft]):
        for node in nodes:
            yield node
            yield from walk(node.children)

    for node in walk(roots):
        if node.children or (node.end - node.start) * target_words >= SENTENCE_WORDS * total:
            continue
        siblings = _siblings(node, roots)
        index = siblings.index(node)
        before = siblings[index - 1] if index > 0 else None
        after = siblings[index + 1] if index + 1 < len(siblings) else None
        if before is not None and not before.children:
            before.end = node.end
            before.folded += ([node.heading] if node.heading is not None else []) + node.folded
        elif after is not None and not after.children:
            after.start = node.start
            if node.heading is not None:
                after.folded = node.folded + ([after.heading] if after.heading else []) + after.folded
                after.heading, after.level = node.heading, node.level
            else:
                after.folded = node.folded + after.folded
        elif index == 0 and node.parent is not None:
            parent = node.parent
            parent.end = node.end
            parent.folded += ([node.heading] if node.heading is not None else []) + node.folded
        else:
            continue
        siblings.pop(index)
        return True
    return False


def _fold(roots: list[_Draft], total: int, target_words: int) -> None:
    while _fold_once(roots, total, target_words):
        pass


def _freeze(
    roots: list[_Draft], pages: Sequence[PageExtent] | None, text: str
) -> SectionTree:
    page_at = PageLookup(pages).at

    ordered: list[_Draft] = []

    def collect(nodes: list[_Draft]) -> None:
        for node in nodes:
            ordered.append(node)
            collect(node.children)

    collect(roots)
    ids = {id(node): f"s{number}" for number, node in enumerate(ordered, start=1)}
    nodes = tuple(
        SectionNode(
            id=ids[id(node)],
            heading=node.heading,
            level=node.level,
            start=node.start,
            end=node.end,
            page_start=page_at(node.start),
            page_end=page_at(max(node.end - 1, node.start)),
            parent_id=ids[id(node.parent)] if node.parent is not None else None,
            child_ids=tuple(ids[id(child)] for child in node.children),
            folded_headings=tuple(node.folded),
            own_words=_body_words(text, node),
        )
        for node in ordered
    )
    return SectionTree(nodes)
