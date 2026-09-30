import json
import sqlite3
from pathlib import Path

from summarizer_web.db.migrations import LATEST_VERSION, MIGRATIONS, apply_migrations
from summarizer_web.ingestion.common import Extraction, OutlineHint, PageText
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
