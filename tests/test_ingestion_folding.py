from summarizer_web.ingestion.folding import FoldedText, compat_fold

SOURCE = "ﬁrst ﬀ and ﬃ of café Ωﬁ"


def test_text_without_compatibility_forms_folds_to_itself() -> None:
    folded = FoldedText.of("Plain text, café.\nSecond line.")

    assert folded.text == "Plain text, café.\nSecond line."
    assert (folded.to_folded(7), folded.to_source(7), folded.span_to_source(6, 10)) == (7, 7, (6, 10))


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


def test_a_span_maps_to_whole_source_characters() -> None:
    folded = FoldedText.of(SOURCE)
    start = folded.text.index("ff and")

    assert folded.span_to_source(0, 5) == (0, 4)
    assert SOURCE[slice(*folded.span_to_source(start, start + 6))] == "ﬀ and"
    # A span that ends inside an expansion still takes the whole source character.
    assert SOURCE[slice(*folded.span_to_source(start, start + 1))] == "ﬀ"
