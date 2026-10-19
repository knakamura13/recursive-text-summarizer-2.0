import re

from summarizer_web.ingestion.folding import FoldedText, compat_fold

SOURCE = "ﬁrst ﬀ and ﬃ of café Ωﬁ"


def test_text_without_compatibility_forms_is_not_copied_or_remapped() -> None:
    source = "Plain text, café.\nSecond line."

    folded = FoldedText.of(source)

    assert folded.text is source
    assert (folded.to_folded(7), folded.to_source(7)) == (7, 7)
    assert folded.search(re.compile("(Second)"), 0, len(source)) == (
        source.index("Second"),
        source.index("Second") + len("Second"),
    )


def test_one_for_one_folds_keep_offsets_and_need_no_entries() -> None:
    source = "a bΩc" * 1000 + "ﬁx"
    folded = FoldedText.of(source)

    assert folded.text == "a bΩc" * 1000 + "fix"
    # Only the ligature changes a length; the no-break spaces and the Ohm sign do not.
    assert (list(folded.source_at), list(folded.folded_at), list(folded.folded_len)) == ([5000], [5000], [2])
    assert (folded.to_folded(4999), folded.to_source(4999)) == (4999, 4999)
    assert folded.search(re.compile("(a bΩc)"), 0, 5) == (0, 5)


def test_text_with_only_one_for_one_folds_is_still_folded() -> None:
    folded = FoldedText.of("Ｒｅｓｕｌｔｓ")

    assert folded.text == "Results"
    assert folded.search(re.compile("(?i)(results)"), 0, 7) == (0, 7)


def test_each_character_is_folded_alone() -> None:
    assert compat_fold("oﬃce ﬁx") == "office fix"
    # A combining sequence is not composed: that would tie one folded character to two source ones.
    assert compat_fold("é") == "é"


def test_offsets_map_both_ways_across_expanded_characters() -> None:
    folded = FoldedText.of(SOURCE)

    assert folded.text == "first ff and ffi of café Ωfi"
    for offset in range(len(SOURCE) + 1):
        assert folded.to_source(folded.to_folded(offset)) == offset
    # Folded characters of one expansion all came from its source character.
    assert [folded.to_source(i) for i in range(5)] == [0, 0, 1, 2, 3]
    assert folded.to_source(folded.text.index("ffi")) == SOURCE.index("ﬃ")
    assert folded.to_source(folded.text.index("ffi") + 2) == SOURCE.index("ﬃ")


def test_a_match_is_a_span_of_whole_source_characters() -> None:
    folded = FoldedText.of(SOURCE)

    assert folded.search(re.compile("(first)"), 0, len(SOURCE)) == (0, 4)
    assert SOURCE[slice(*folded.search(re.compile("(ff and)"), 0, len(SOURCE)))] == "ﬀ and"


def test_a_match_that_cuts_a_source_character_is_skipped() -> None:
    office = FoldedText.of("oﬃce")

    assert office.search(re.compile("(ffi)"), 0, 4) == (1, 2)
    assert office.search(re.compile("(fi)"), 0, 4) is None  # Begins inside the ligature.
    assert office.search(re.compile("(of)"), 0, 4) is None  # Ends inside it.
    assert office.match(re.compile("(ffi)"), 1) == (1, 2)
    assert office.match(re.compile("(fi)"), 1) is None

    # The search goes on to a later match that does not cut one.
    half = FoldedText.of("½ then 2")
    assert half.text == "1⁄2 then 2"
    assert half.search(re.compile("(2)"), 0, 8) == (7, 8)


def test_a_search_starts_and_ends_at_source_offsets() -> None:
    source = "ﬁx ﬁx ﬁx"
    folded = FoldedText.of(source)
    pattern = re.compile("(fix)")

    assert folded.search(pattern, 0, len(source)) == (0, 2)
    assert folded.search(pattern, 1, len(source)) == (3, 5)
    assert folded.search(pattern, 0, 2) == (0, 2)
    assert folded.search(pattern, 0, 1) is None
