from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from summarizer.sections import SENTENCE_WORDS, SectionTree, build_section_tree
from summarizer.segmentation import detect_markdown_headings

BIG = 10**9  # with this target no section is undersized, so nothing folds


@dataclass(frozen=True)
class Heading:
    title: str
    level: int
    start: int


@dataclass(frozen=True)
class Page:
    page: int
    start: int
    end: int


def words(count: int) -> str:
    return " ".join(["word"] * count) + "\n\n"


def doc(*parts: tuple[str, int, str]) -> tuple[str, list[Heading]]:
    """Join (title, level, body) parts into Markdown text and its outline."""
    text = ""
    outline = []
    for title, level, body in parts:
        outline.append(Heading(title, level, len(text) + level + 1))
        text += f"{'#' * level} {title}\n\n{body}"
    return text, outline


def check_invariants(text: str, tree: SectionTree, pages: list[Page] | None = None) -> None:
    nodes = tree.nodes
    position = 0
    for node in nodes:
        assert node.start == position
        assert node.end >= node.start
        position = node.end
    assert position == len(text)
    assert "".join(text[n.start : n.end] for n in nodes) == text
    by_id = {n.id: n for n in nodes}
    assert len(by_id) == len(nodes)
    for node in nodes:
        for child_id in node.child_ids:
            assert by_id[child_id].parent_id == node.id
        if node.parent_id is not None:
            assert node.id in by_id[node.parent_id].child_ids
            assert int(node.parent_id[1:]) < int(node.id[1:])
        else:
            assert node in tree.roots
        kids = [by_id[c] for c in node.child_ids]
        # A node's own text sits right before its first child.
        if kids:
            assert kids[0].start == node.end
        for left, right in zip(kids, kids[1:], strict=False):
            assert left.start <= right.start
    if pages:
        for node in nodes:
            assert node.page_start == expected_page(pages, node.start)
            assert node.page_end == expected_page(pages, max(node.end - 1, node.start))
    else:
        assert all(n.page_start is None and n.page_end is None for n in nodes)


def expected_page(pages: list[Page], offset: int) -> int:
    best = pages[0].page
    for page in pages:
        if page.start <= offset:
            best = page.page
    return best


def test_nesting_with_skipped_level_and_nothing_folds():
    text, outline = doc(
        ("A", 1, words(30)),
        ("B", 3, words(30)),
        ("C", 2, words(30)),
        ("D", 1, words(30)),
    )
    tree = build_section_tree(text, outline, target_words=BIG)
    check_invariants(text, tree)
    a, b, c, d = tree.nodes
    assert [n.heading for n in tree.nodes] == ["A", "B", "C", "D"]
    assert b.parent_id == a.id and c.parent_id == a.id
    assert a.child_ids == (b.id, c.id)
    assert d.parent_id is None
    assert text[a.start : a.end] == "# A\n\n" + words(30)  # the marker belongs to A


def test_empty_section_between_headings_is_kept_when_it_has_children():
    text, outline = doc(("A", 1, ""), ("B", 2, words(30)), ("C", 2, words(30)))
    tree = build_section_tree(text, outline, target_words=BIG)
    check_invariants(text, tree)
    a = tree.nodes[0]
    assert text[a.start : a.end] == "# A\n\n"
    assert len(a.child_ids) == 2


def test_empty_childless_section_folds_into_its_neighbour():
    text, outline = doc(("A", 1, words(30)), ("B", 1, ""), ("C", 1, words(30)))
    tree = build_section_tree(text, outline, target_words=200)
    check_invariants(text, tree)
    assert [(n.heading, n.folded_headings) for n in tree.nodes] == [
        ("A", ("B",)),
        ("C", ()),
    ]


def test_opening_section_before_first_heading():
    text, outline = doc(("A", 1, words(30)))
    text = "Preface text here. " * 5 + "\n\n" + text
    outline = [Heading("A", 1, text.index("# A") + 2)]
    tree = build_section_tree(text, outline, target_words=BIG)
    check_invariants(text, tree)
    opening, a = tree.nodes
    assert opening.heading is None and opening.level == 1 and opening.start == 0
    assert text[opening.end :].startswith("# A")
    assert a.parent_id is None or a.parent_id == opening.id


def test_whitespace_only_leading_text_has_no_opening_section():
    text, outline = doc(("A", 1, words(30)))
    text = "  \n\n" + text
    outline = [Heading("A", 1, text.index("# A") + 2)]
    tree = build_section_tree(text, outline, target_words=BIG)
    check_invariants(text, tree)
    assert len(tree.nodes) == 1
    assert tree.nodes[0].heading == "A" and tree.nodes[0].start == 0


def test_empty_outline_gives_one_untitled_section():
    text = "Just some text.\n"
    tree = build_section_tree(text, [], target_words=BIG)
    check_invariants(text, tree)
    (node,) = tree.nodes
    assert node.heading is None and (node.start, node.end) == (0, len(text))


def test_empty_text_and_outline():
    tree = build_section_tree("", [], target_words=100)
    check_invariants("", tree)
    assert len(tree.nodes) == 1


def test_folds_into_preceding_sibling():
    text, outline = doc(("P", 1, ""), ("A", 2, words(60)), ("B", 2, "tiny\n\n"), ("C", 2, words(60)))
    tree = build_section_tree(text, outline, target_words=200)
    check_invariants(text, tree)
    a = next(n for n in tree.nodes if n.heading == "A")
    assert a.folded_headings == ("B",)
    assert "tiny" in text[a.start : a.end]
    assert [n.heading for n in tree.nodes] == ["P", "A", "C"]


def test_folds_into_following_sibling_when_first_child_cannot_merge_backward():
    # Siblings: X (has a child), T (tiny, childless), then U. T's preceding
    # sibling has children, so it folds forward into U and keeps its own heading.
    text, outline = doc(
        ("P", 1, ""),
        ("X", 2, words(60)),
        ("Xc", 3, words(60)),
        ("T", 2, "tiny\n\n"),
        ("U", 2, words(60)),
    )
    tree = build_section_tree(text, outline, target_words=300)
    check_invariants(text, tree)
    merged = next(n for n in tree.nodes if n.heading == "T")
    assert merged.folded_headings == ("U",)
    assert text[merged.start : merged.end].startswith("## T")
    assert "U" not in [n.heading for n in tree.nodes]


def test_untitled_opening_folds_forward_keeping_the_heading():
    text = "Hi.\n\n" + "# A\n\n" + words(60) + "# B\n\n" + words(60)
    outline = [Heading("A", 1, text.index("# A") + 2), Heading("B", 1, text.index("# B") + 2)]
    tree = build_section_tree(text, outline, target_words=200)
    check_invariants(text, tree)
    assert [n.heading for n in tree.nodes] == ["A", "B"]
    assert tree.nodes[0].start == 0


def test_folds_into_parent_when_it_is_the_first_child():
    text, outline = doc(("P", 1, words(60)), ("T", 2, "tiny\n\n"), ("Q", 1, words(60)))
    tree = build_section_tree(text, outline, target_words=200)
    check_invariants(text, tree)
    p = tree.nodes[0]
    assert p.heading == "P" and p.folded_headings == ("T",) and p.child_ids == ()
    assert "tiny" in text[p.start : p.end]


def test_later_child_without_childless_neighbour_stays():
    text, outline = doc(
        ("P", 1, words(60)),
        ("X", 2, words(60)),
        ("Xc", 3, words(60)),
        ("T", 2, "tiny\n\n"),
    )
    tree = build_section_tree(text, outline, target_words=300)
    check_invariants(text, tree)
    assert [n.heading for n in tree.nodes] == ["P", "X", "Xc", "T"]


def test_parent_with_children_is_never_folded_even_when_empty():
    text, outline = doc(("P", 1, ""), ("A", 2, words(60)), ("B", 2, words(60)))
    tree = build_section_tree(text, outline, target_words=200)
    check_invariants(text, tree)
    assert tree.nodes[0].heading == "P" and len(tree.nodes[0].child_ids) == 2


def test_parent_folds_in_a_later_round_once_childless():
    text, outline = doc(
        ("R", 1, words(100)),
        ("P", 1, "tiny\n\n"),
        ("c", 2, "tiny\n\n"),
        ("S", 1, words(100)),
    )
    tree = build_section_tree(text, outline, target_words=300)
    check_invariants(text, tree)
    # c folds into P, then P (childless, tiny) folds into R, its preceding sibling.
    assert [n.heading for n in tree.nodes] == ["R", "S"]
    assert tree.nodes[0].folded_headings == ("P", "c")


def test_no_folding_across_parents():
    text, outline = doc(
        ("A", 1, ""),
        ("a1", 2, words(60)),
        ("B", 1, ""),
        ("b1", 2, "tiny\n\n"),
    )
    tree = build_section_tree(text, outline, target_words=300)
    check_invariants(text, tree)
    a1 = next(n for n in tree.nodes if n.heading == "a1")
    assert a1.folded_headings == ()
    b = next(n for n in tree.nodes if n.heading == "B")
    assert b.folded_headings == ("b1",)  # into its own parent, never into a1


def test_nothing_folds_with_a_large_target():
    # A tiny share folds; with a huge target even tiny sections stand.
    text, outline = doc(("A", 1, "x\n\n"), ("B", 1, "y\n\n"), ("C", 1, "z\n\n"))
    big = build_section_tree(text, outline, target_words=10**6)
    assert [n.heading for n in big.nodes] == ["A", "B", "C"]
    small = build_section_tree(text, outline, target_words=SENTENCE_WORDS)
    assert len(small.nodes) < 3
    check_invariants(text, big)
    check_invariants(text, small)


def test_page_ranges_follow_the_page_map():
    text, outline = doc(("A", 1, words(30)), ("B", 1, words(30)))
    cut = text.index("# B")
    pages = [Page(1, 0, cut - 5), Page(2, cut - 5, len(text))]
    tree = build_section_tree(text, outline, target_words=10**6, pages=pages)
    check_invariants(text, tree, pages)
    assert (tree.nodes[0].page_start, tree.nodes[0].page_end) == (1, 2)
    assert (tree.nodes[1].page_start, tree.nodes[1].page_end) == (2, 2)


def test_markdown_headings_feed_the_builder():
    text = "# Title\n\nbody " * 1 + words(50) + "Setext\n======\n\n" + words(50) + "## Sub\n\n" + words(50)
    headings = detect_markdown_headings(text)
    tree = build_section_tree(text, headings, target_words=10**6)
    check_invariants(text, tree)
    assert [n.heading for n in tree.nodes] == ["Title", "Setext", "Sub"]
    assert tree.nodes[2].parent_id == tree.nodes[1].id


@pytest.mark.parametrize("seed", range(60))
def test_random_outlines_keep_every_invariant(seed: int):
    rng = random.Random(seed)
    parts = []
    for index in range(rng.randint(0, 14)):
        body = rng.choice(["", "tiny\n\n", words(rng.randint(1, 80))])
        parts.append((f"H{index}", rng.randint(1, 4), body))
    text, outline = doc(*parts)
    if rng.random() < 0.5:
        text = "lead " * rng.randint(0, 3) + "\n\n" + text
        shift = len(text) - len(doc(*parts)[0])
        outline = [Heading(h.title, h.level, h.start + shift) for h in outline]
    pages = None
    if rng.random() < 0.5 and text:
        cuts = sorted(rng.sample(range(len(text) + 1), min(3, len(text) + 1)))
        edges = [0, *cuts, len(text)]
        pages = [Page(i + 1, a, b) for i, (a, b) in enumerate(zip(edges, edges[1:], strict=False)) if b > a]
    target = rng.choice([10, 100, 400, 5000])
    tree = build_section_tree(text, outline, target_words=target, pages=pages)
    check_invariants(text, tree, pages)
    again = build_section_tree(text, outline, target_words=target, pages=pages)
    assert again == tree
