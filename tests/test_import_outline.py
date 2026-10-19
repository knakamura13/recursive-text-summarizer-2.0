import json
import sqlite3
from pathlib import Path

from summarizer_web.db.migrations import LATEST_VERSION, MIGRATIONS, apply_migrations
from summarizer_web.ingestion.common import Extraction, OutlineHint, PageText
from summarizer_web.ingestion import extract
from summarizer_web.ingestion.extract import assemble, build_report
from summarizer_web.worker.imports import insert_revision

_SCHEMA = Path(__file__).resolve().parents[1] / "summarizer_web" / "db" / "schema.sql"

BOOK = (
    "Part One\n\n"
    "Opening words.\n\n"
    "Early Days\n\n"
    "The first chapter mentions Later Years in passing.\n\n"
    "Later Years\n\n"
    "The second chapter.\n\n"
    "Part Two\n\n"
    "Closing words."
)


def outline(extraction: Extraction) -> list[tuple[str, int, str]]:
    imported = assemble(extraction)
    return [
        (entry.title, entry.level, imported.text[entry.start : entry.end])
        for entry in imported.outline
    ]


def test_hints_become_nested_sections_without_changing_the_text() -> None:
    hints = [
        OutlineHint("Part One", 1),
        OutlineHint("Early Days", 2),
        OutlineHint("Later Years", 2),
        OutlineHint("Part Two", 1),
    ]
    plain = assemble(Extraction("txt", text=BOOK))
    with_outline = assemble(Extraction("txt", text=BOOK, outline_hints=hints))

    assert (with_outline.text, with_outline.source_id) == (plain.text, plain.source_id)
    assert outline(Extraction("txt", text=BOOK, outline_hints=hints)) == [
        ("Part One", 1, BOOK[: BOOK.index("Part Two")]),
        ("Early Days", 2, BOOK[BOOK.index("Early Days") : BOOK.index("Later Years\n\nThe second")]),
        ("Later Years", 2, BOOK[BOOK.index("Later Years\n\nThe second") : BOOK.index("Part Two")]),
        ("Part Two", 1, BOOK[BOOK.index("Part Two") :]),
    ]
    assert [(entry.page_start, entry.page_end) for entry in with_outline.outline] == [(None, None)] * 4


def test_a_title_is_placed_where_it_begins_a_line_and_whitespace_and_case_may_differ() -> None:
    hints = [OutlineHint("early   DAYS", 1), OutlineHint("later years", 1)]

    titles = [(title, text.split("\n")[0]) for title, _level, text in outline(Extraction("txt", text=BOOK, outline_hints=hints))]

    # "Later Years" also occurs mid-sentence first; the heading line wins.
    assert titles == [("Early Days", "Early Days"), ("Later Years", "Later Years")]


LIGATURES = "First ﬁne ﬂow of the oﬃce.\n\n"


def test_a_title_matches_a_text_that_spells_it_with_ligatures() -> None:
    text = LIGATURES + "Reﬁning the Estimate\n\nBody.\n\nEﬀort\n\nMore body."
    hints = [OutlineHint("Refining the Estimate", 1), OutlineHint("Effort", 1)]

    imported = assemble(Extraction("txt", text=text, outline_hints=hints))

    # Offsets and titles are the canonical text's; the ligatures stay in it.
    assert imported.text == text
    assert imported.unplaced_headings == 0
    assert [(entry.title, imported.text[entry.start : entry.end]) for entry in imported.outline] == [
        ("Reﬁning the Estimate", "Reﬁning the Estimate\n\nBody.\n\n"),
        ("Eﬀort", "Eﬀort\n\nMore body."),
    ]


def test_a_title_spelled_with_ligatures_matches_a_text_that_is_not() -> None:
    text = "Refining the Estimate\n\nBody."

    imported = assemble(Extraction("txt", text=text, outline_hints=[OutlineHint("Reﬁning the Estimate", 1)]))

    assert [imported.text[entry.start : entry.end] for entry in imported.outline] == [text]


def test_a_title_inside_a_ligature_word_is_not_a_whole_word() -> None:
    text = "Oﬀer\n\nBody."

    assert assemble(Extraction("txt", text=text, outline_hints=[OutlineHint("Of", 1)])).unplaced_headings == 1
    assert assemble(Extraction("txt", text=text, outline_hints=[OutlineHint("Offer", 1)])).unplaced_headings == 0


def test_a_hint_with_an_offset_is_placed_there_after_earlier_ligatures() -> None:
    text = LIGATURES + "Reﬁning\n\nBody."
    offset = text.index("Reﬁning")

    imported = assemble(Extraction("md", text=text, outline_hints=[OutlineHint("Refining", 1, offset=offset)]))

    [entry] = imported.outline
    assert (entry.start, imported.text[entry.start : entry.end]) == (offset, "Reﬁning\n\nBody.")


def test_a_page_hint_is_searched_on_its_page_after_earlier_ligatures() -> None:
    pages = [
        # Enough ligatures that, unmapped, page 3's start would fall back onto page 2's heading.
        PageText(1, " ".join(["oﬃce"] * 20)),
        PageText(2, "Reﬁning\n\nBody."),
        PageText(3, "Reﬁning\n\nAgain."),
    ]

    imported = assemble(Extraction("pdf", pages=pages, outline_hints=[OutlineHint("Refining", 1, page=3)]))

    [entry] = imported.outline
    assert imported.text[entry.start :] == "Reﬁning\n\nAgain."
    assert (entry.page_start, entry.page_end) == (3, 3)


def test_a_text_is_folded_only_when_a_title_is_not_found_as_written(monkeypatch) -> None:
    def fail(_text: str):
        raise AssertionError("folded a text whose titles are all found as written")

    monkeypatch.setattr(extract.FoldedText, "of", fail)

    no_hints = assemble(Extraction("txt", text=LIGATURES + "Body."))
    found = assemble(Extraction("txt", text="Intro\n\n" + LIGATURES, outline_hints=[OutlineHint("Intro", 1)]))

    assert (no_hints.outline, no_hints.unplaced_headings) == ((), 0)
    assert [entry.title for entry in found.outline] == ["Intro"]


def test_a_title_found_as_written_is_placed_there_whatever_its_folded_form_matches() -> None:
    # Folding turns the no-break space into a space, so the first line would be a whole-line match.
    text = "Results \n\nThe Results\n\nResults\n\nBody"
    pages = [
        PageText(1, "Contents."),
        PageText(2, "Getting Started with Kindle™\n\nCharge the device first."),
        PageText(3, "Networks for Classiﬁcation\n\n4.1.1 Classification\nTo start, a simple problem."),
    ]

    by_text = assemble(Extraction("txt", text=text, outline_hints=[OutlineHint("Results", 1)]))
    by_page = assemble(
        Extraction(
            "pdf",
            pages=pages,
            outline_hints=[
                # "Kindle" ends where the trademark sign begins; folding would read it as "TM".
                OutlineHint("Getting Started with Kindle", 1, page=2),
                # The page header says the word with a ligature, ahead of the heading that does not.
                OutlineHint("Classification", 3, page=3),
            ],
        )
    )

    assert [by_text.text[entry.start :] for entry in by_text.outline] == ["Results\n\nBody"]
    assert by_page.unplaced_headings == 0
    assert [by_page.text[entry.start :].split("\n")[0] for entry in by_page.outline] == [
        "Getting Started with Kindle™",
        "Classification",
    ]


def test_an_offset_hint_whose_title_begins_with_a_spacing_accent_is_placed() -> None:
    text = "Intro text.\n\n# ´Twas the Night\n\nBody."
    offset = text.index("´Twas")

    imported = assemble(Extraction("md", text=text, outline_hints=[OutlineHint("´Twas the Night", 1, offset=offset)]))

    assert [(entry.start, entry.title) for entry in imported.outline] == [(offset, "´Twas the Night")]


def test_a_title_never_lands_on_part_of_a_character_it_folds_from() -> None:
    text = "Add ½ cup of flour.\n\nThen Step 2 begins here."

    # "½" folds to "1⁄2", but a "2" heading is not half of it.
    only_the_half = assemble(Extraction("txt", text=text.split("\n")[0], outline_hints=[OutlineHint("2", 1)]))
    step = assemble(Extraction("txt", text=text, outline_hints=[OutlineHint("2", 1)]))

    assert only_the_half.unplaced_headings == 1
    assert [step.text[entry.start :] for entry in step.outline] == ["2 begins here."]


def test_a_ligature_at_the_start_of_a_hint_is_placed() -> None:
    text = "ﬁnancial Results\n\nBody."
    pages = [PageText(1, "Intro."), PageText(2, text)]

    by_offset = assemble(Extraction("md", text=text, outline_hints=[OutlineHint("financial Results", 1, offset=0)]))
    by_page = assemble(Extraction("pdf", pages=pages, outline_hints=[OutlineHint("financial Results", 1, page=2)]))

    assert [(entry.start, entry.title) for entry in by_offset.outline] == [(0, "ﬁnancial Results")]
    [entry] = by_page.outline
    assert by_page.text[entry.start :] == text


def test_a_title_matches_a_text_that_has_only_one_for_one_folds() -> None:
    text = "Ｒｅｓｕｌｔｓ\n\nBody."

    imported = assemble(Extraction("txt", text=text, outline_hints=[OutlineHint("Results", 1)]))

    assert [imported.text[entry.start :] for entry in imported.outline] == [text]


def test_a_title_matches_a_whole_heading_line_before_a_line_it_only_begins() -> None:
    text = "Results from the pilot were mixed.\n\nIntroduction\n\nResults\n\nThe final numbers."
    hints = [OutlineHint("Results", 1), OutlineHint("Intro", 1)]
    extraction = Extraction("txt", text=text, outline_hints=hints)

    imported = assemble(extraction)

    [results] = imported.outline
    assert imported.text[results.start :] == "Results\n\nThe final numbers."
    # "Intro" is only a prefix of "Introduction", so it is not placed.
    assert imported.unplaced_headings == 1


def test_a_repeated_title_lands_on_its_next_occurrence() -> None:
    text = "Summary\n\nFirst part.\n\nSummary\n\nSecond part."
    hints = [OutlineHint("Summary", 1), OutlineHint("Summary", 1)]

    assert [section for _title, _level, section in outline(Extraction("txt", text=text, outline_hints=hints))] == [
        "Summary\n\nFirst part.\n\n",
        "Summary\n\nSecond part.",
    ]


def test_a_heading_line_whose_space_moved_or_was_added_is_placed_in_its_spelling() -> None:
    text = "Bounds for p> 1\n\nBody.\n\nPart 1 1 Overview\n\nMore."
    hints = [OutlineHint("Bounds for p >1", 1), OutlineHint("Part 11 Overview", 1)]

    imported = assemble(Extraction("txt", text=text, outline_hints=hints))

    assert imported.unplaced_headings == 0
    assert [(entry.title, imported.text[entry.start : entry.end]) for entry in imported.outline] == [
        ("Bounds for p> 1", "Bounds for p> 1\n\nBody.\n\n"),
        ("Part 1 1 Overview", "Part 1 1 Overview\n\nMore."),
    ]


def test_a_heading_line_with_merged_words_and_a_ligature_is_placed_whichever_side_has_the_ligature() -> None:
    text = "Staﬀ Ratesfor p> 1\n\nBody.\n\nStaff Ratesfor p> 1\n\nMore."
    hints = [OutlineHint("Staff Rates for p >1", 1), OutlineHint("Staﬀ Rates for p >1", 1)]

    imported = assemble(Extraction("txt", text=text, outline_hints=hints))

    assert imported.unplaced_headings == 0
    assert [imported.text[entry.start : entry.end] for entry in imported.outline] == [
        "Staﬀ Ratesfor p> 1\n\nBody.\n\n",
        "Staff Ratesfor p> 1\n\nMore.",
    ]


def test_title_words_run_together_inside_a_sentence_are_not_a_heading() -> None:
    text = "Read the keypoints below.\n\nSee anotherTitle for more."
    hints = [OutlineHint("Key Points", 1), OutlineHint("Another Title", 1)]

    imported = assemble(Extraction("txt", text=text, outline_hints=hints))

    assert (imported.outline, imported.unplaced_headings) == ((), 2)


def test_a_line_that_only_begins_with_the_title_without_its_spaces_is_not_a_heading() -> None:
    text = "Input Devices\n\nKeyPoints and Tips\n\nBody."
    hints = [OutlineHint("In Put", 1), OutlineHint("Key Points", 1)]

    imported = assemble(Extraction("txt", text=text, outline_hints=hints))

    assert (imported.outline, imported.unplaced_headings) == ((), 2)


def test_the_title_as_written_wins_over_an_earlier_line_that_matches_without_its_spaces() -> None:
    text = "KeyPoints\n\nBody that lists the key points below.\n\nKey Points\n\nMore."

    imported = assemble(Extraction("txt", text=text, outline_hints=[OutlineHint("Key Points", 1)]))

    [entry] = imported.outline
    assert imported.text[entry.start :] == "Key Points\n\nMore."


def test_a_title_as_written_on_the_page_wins_over_a_respaced_line_after_the_cursor() -> None:
    pages = [PageText(1, "Intro."), PageText(2, "Key Points\n\nBody.\n\nOther\n\nKeyPoints")]
    hints = [OutlineHint("Other", 1, page=2), OutlineHint("Key Points", 1, page=2)]

    imported = assemble(Extraction("pdf", pages=pages, outline_hints=hints))

    # Nothing after "Other" holds the title as written, but the page does, so the line
    # that lost its space is not used.
    assert [(entry.title, imported.text[entry.start : entry.end]) for entry in imported.outline] == [
        ("Key Points", "Key Points\n\nBody.\n\n"),
        ("Other", "Other\n\nKeyPoints"),
    ]


def test_a_repeated_respaced_title_lands_on_its_next_occurrence() -> None:
    text = "KeyPoints\n\nFirst.\n\nKeyPoints\n\nSecond."
    hints = [OutlineHint("Key Points", 1), OutlineHint("Key Points", 1)]

    assert [section for _title, _level, section in outline(Extraction("txt", text=text, outline_hints=hints))] == [
        "KeyPoints\n\nFirst.\n\n",
        "KeyPoints\n\nSecond.",
    ]


def test_a_respaced_title_is_searched_on_its_page() -> None:
    pages = [PageText(1, "Contents\n\nKeyPoints"), PageText(2, "KeyPoints\n\nBody.")]

    imported = assemble(Extraction("pdf", pages=pages, outline_hints=[OutlineHint("Key Points", 1, page=2)]))

    # The contents page also has the line; the page hint skips it.
    [entry] = imported.outline
    assert imported.text[entry.start :] == "KeyPoints\n\nBody."
    assert (entry.page_start, entry.page_end) == (2, 2)


def test_an_offset_hint_is_not_placed_on_a_line_that_lost_a_space() -> None:
    text = "Intro.\n\nKeyPoints\n\nBody."

    imported = assemble(
        Extraction("md", text=text, outline_hints=[OutlineHint("Key Points", 1, offset=text.index("KeyPoints"))])
    )

    assert (imported.outline, imported.unplaced_headings) == ((), 1)


def test_a_title_not_in_the_text_is_left_out_and_reported() -> None:
    extraction = Extraction(
        "txt", text=BOOK, outline_hints=[OutlineHint("Part One", 1), OutlineHint("Missing Chapter", 2)]
    )
    imported = assemble(extraction)
    report = build_report(extraction, imported, duration_seconds=0)

    assert [entry.title for entry in imported.outline] == ["Part One"]
    assert (report.heading_count, report.unplaced_headings) == (1, 1)
    assert [notice.code for notice in report.notices] == ["headings_unplaced"]


def test_no_headings_is_an_empty_outline() -> None:
    extraction = Extraction("txt", text=BOOK)
    imported = assemble(extraction)
    report = build_report(extraction, imported, duration_seconds=0)

    assert imported.outline == ()
    assert (report.heading_count, report.unplaced_headings, report.notices) == (0, 0, [])


def test_a_page_hint_is_searched_on_its_page_and_sections_carry_page_ranges() -> None:
    pages = [
        PageText(1, "Contents\n\nMethods ... 2"),
        PageText(2, "Methods\n\nWe measured."),
        PageText(3, "More measuring."),
        PageText(4, "Results\n\nIt worked."),
    ]
    hints = [OutlineHint("Methods", 1, page=2), OutlineHint("Results", 1, page=4)]

    imported = assemble(Extraction("pdf", pages=pages, outline_hints=hints))

    methods, results = imported.outline
    # The contents page also says "Methods"; the page hint skips it.
    assert imported.text[methods.start :].startswith("Methods\n\nWe measured.")
    assert (methods.page_start, methods.page_end) == (2, 3)
    assert (results.page_start, results.page_end) == (4, 4)


def test_existing_revisions_keep_no_outline_and_new_ones_store_theirs(tmp_path) -> None:
    connection = sqlite3.connect(tmp_path / "app.db")
    connection.executescript(_SCHEMA.read_text(encoding="utf-8"))
    for statement in dict(MIGRATIONS)[2]:
        connection.execute(statement)
    connection.execute("PRAGMA user_version = 2")
    connection.execute(
        "INSERT INTO documents (document_id, title, filename, format, size_bytes, import_state, created_at, updated_at) "
        "VALUES ('d', 't', 'f.txt', 'txt', 1, 'ready', 'now', 'now')"
    )
    connection.execute(
        "INSERT INTO source_revisions (revision_id, document_id, source_sha256, extraction_version, canonical_path, created_at) "
        "VALUES ('r', 'd', 'sha', 'import/3', 'c.txt', 'now')"
    )
    connection.commit()

    apply_migrations(connection)

    assert connection.execute("PRAGMA user_version").fetchone()[0] == LATEST_VERSION

    for revision, hints in (("with", [OutlineHint("Part Two", 1)]), ("without", [])):
        extraction = Extraction("txt", text=BOOK, outline_hints=hints)
        imported = assemble(extraction)
        connection.execute(
            "INSERT INTO documents (document_id, title, filename, format, size_bytes, import_state, created_at, updated_at) "
            "VALUES (?, 't', 'f.txt', 'txt', 1, 'ready', 'now', 'now')",
            (revision,),
        )
        insert_revision(
            connection,
            document_id=revision,
            imported=imported,
            report=build_report(extraction, imported, duration_seconds=0),
            canonical=tmp_path / "c.txt",
            original=tmp_path / "o.txt",
            created_at="now",
        )
    stored = dict(connection.execute("SELECT document_id, outline_json FROM source_revisions").fetchall())

    assert stored["d"] is None
    assert stored["without"] == "[]"
    [entry] = json.loads(stored["with"])
    assert (entry["title"], entry["level"], BOOK[entry["start"] : entry["end"]]) == (
        "Part Two",
        1,
        BOOK[BOOK.index("Part Two") :],
    )
