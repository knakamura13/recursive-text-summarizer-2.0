"""Node and segment projections of a Run for the web UI.

The worker applies pipeline item events to `node_projections`: every change is
recorded in `run_events` first and the node row is updated in the same
transaction with `updated_event_id` set to that event, so a client holding
cursor N fetches exactly the rows that changed after N. `run_segments` holds
the code point offsets of each segment plus the pages its core range spans.

Readers (the run stream and the tree view) use `tree_items` and
`max_event_id`. Legacy completed Runs have rows without kind/state history,
child ids, or segments; readers fall back to column defaults and parent links.
"""

from __future__ import annotations

import bisect
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime

from summarizer.hierarchy import TreeNode
from summarizer.runtime.observers import ItemEvent, SegmentInfo
from summarizer_web.db.connection import get_database
from summarizer_web.models.api import NodeTreeItem

TREE_KINDS = frozenset({"leaf", "merge", "passthrough"})

PageRange = tuple[int, int] | None
EventWriter = Callable[[sqlite3.Connection, str, str, Mapping[str, object]], int]


# --- Pages and labels ---------------------------------------------------------


class PageMap:
    """The pages of a Document as code point ranges of its canonical text."""

    def __init__(self, entries: Iterable[tuple[int, int, int]] = ()) -> None:
        ordered = sorted(entries, key=lambda entry: (entry[1], entry[2], entry[0]))
        self._pages = [entry[0] for entry in ordered]
        self._starts = [entry[1] for entry in ordered]
        self._ends = [entry[2] for entry in ordered]

    @classmethod
    def from_json(cls, page_map_json: str | None) -> PageMap:
        """Parse a revision's `page_map_json`: `[{page, start, end, ...}]`, end exclusive."""
        if not page_map_json:
            return cls()
        entries = []
        for item in json.loads(page_map_json):
            if isinstance(item, Mapping) and {"page", "start", "end"} <= item.keys():
                entries.append((int(item["page"]), int(item["start"]), int(item["end"])))
        return cls(entries)

    def __bool__(self) -> bool:
        return bool(self._pages)

    def pages_for(self, start: int, end: int) -> PageRange:
        """First and last page overlapping the half-open range [start, end)."""
        if not self._pages or end <= start:
            return None
        # Pages are sequential and non-overlapping, so both bounds bisect.
        first = bisect.bisect_right(self._ends, start)
        last = bisect.bisect_left(self._starts, end) - 1
        if first > last:
            return None
        return self._pages[first], self._pages[last]


def segment_pages(segments: Sequence[SegmentInfo], page_map: PageMap) -> dict[str, PageRange]:
    """Pages spanned by each segment's core range (its attributable text)."""
    return {
        segment.segment_id: page_map.pages_for(segment.core_start, segment.core_end)
        or page_map.pages_for(segment.start, segment.end)
        for segment in segments
    }


def span_pages(segment_ids: Iterable[str], pages: Mapping[str, PageRange]) -> PageRange:
    starts: list[int] = []
    ends: list[int] = []
    for segment_id in segment_ids:
        found = pages.get(segment_id)
        if found is not None:
            starts.append(found[0])
            ends.append(found[1])
    if not starts:
        return None
    return min(starts), max(ends)


def pages_suffix(pages: PageRange) -> str:
    if pages is None:
        return ""
    first, last = pages
    return f" · p. {first}" if first == last else f" · pp. {first}–{last}"


def is_direct(covered_segment_ids: Sequence[str]) -> bool:
    """The direct strategy's only segment is `D000001`; hierarchical ones are `S…`."""
    return bool(covered_segment_ids) and all(
        segment_id.startswith("D") for segment_id in covered_segment_ids
    )


def node_label(
    kind: str, level: int, order: int, covered_segment_ids: Sequence[str], pages: PageRange
) -> str:
    if kind == "leaf":
        if is_direct(covered_segment_ids):
            return "Whole document"
        return f"Segment {order + 1}{pages_suffix(pages)}"
    return f"Level {level} · Group {order + 1}{pages_suffix(pages)}"


def root_label(level: int) -> str:
    return f"Root · Level {level}"


def parse_node_id(node_id: str) -> tuple[int, int] | None:
    """Level and zero-based order from a tree node id such as `L1N0003`."""
    if not node_id.startswith("L") or "N" not in node_id:
        return None
    level, _, order = node_id[1:].partition("N")
    if not (level.isdigit() and order.isdigit()) or int(order) < 1:
        return None
    return int(level), int(order) - 1


class NodeLabeler:
    """Labels tree items from their position and the pages their segments span."""

    def __init__(self, page_map: PageMap) -> None:
        self._page_map = page_map
        self._pages: dict[str, PageRange] = {}

    def set_segments(self, segments: Sequence[SegmentInfo]) -> dict[str, PageRange]:
        self._pages = segment_pages(segments, self._page_map)
        return dict(self._pages)

    def pages_for(self, segment_ids: Sequence[str]) -> PageRange:
        return span_pages(segment_ids, self._pages)

    def label(self, kind: str, level: int, order: int, covered_segment_ids: Sequence[str]) -> str:
        return node_label(kind, level, order, covered_segment_ids, self.pages_for(covered_segment_ids))

    def label_event(self, event: ItemEvent) -> str:
        """Human label of any item; the failing item of a RunFailure uses the same text."""
        if event.kind in TREE_KINDS:
            level, order = event_position(event)
            return self.label(event.kind, level, order, event.covered_segment_ids)
        if event.kind == "editorial":
            return "Final summary draft"
        if event.order is not None and event.total is not None:
            return f"Claim {event.order + 1} of {event.total}"
        return f"Claim {event.work_id}"


def event_position(event: ItemEvent) -> tuple[int, int]:
    parsed = parse_node_id(event.work_id)
    level = event.level if event.level is not None else (parsed[0] if parsed else 0)
    order = event.order if event.order is not None else (parsed[1] if parsed else 0)
    return level, order


# --- Writes (inside the caller's transaction) --------------------------------


def replace_segments(
    connection: sqlite3.Connection,
    run_id: str,
    segments: Sequence[SegmentInfo],
    pages: Mapping[str, PageRange],
) -> None:
    connection.execute("DELETE FROM run_segments WHERE run_id = ?", (run_id,))
    connection.executemany(
        """
        INSERT INTO run_segments (
            run_id, segment_id, order_index, start_offset, end_offset,
            core_start, core_end, token_count, page_start, page_end
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                segment.segment_id,
                segment.order,
                segment.start,
                segment.end,
                segment.core_start,
                segment.core_end,
                segment.token_count,
                *(pages.get(segment.segment_id) or (None, None)),
            )
            for segment in segments
        ],
    )


def summary_columns(summary: Mapping[str, object]) -> tuple[str | None, str, str, str]:
    """`summary_text`, `content_units_json`, `detail_json`, `evidence_refs_json`."""
    units = list(summary.get("content_units") or ())
    detail = {
        key: list(summary.get(key) or ())
        for key in ("entities", "qualifications", "contradictions", "quotations")
    }
    refs = [
        {"segment_id": evidence.get("segment_id"), "quote": evidence.get("quote")}
        for unit in units
        if isinstance(unit, Mapping)
        for evidence in (unit.get("evidence") or ())
        if isinstance(evidence, Mapping)
    ]
    text = summary.get("summary")
    return (
        text if isinstance(text, str) else None,
        json.dumps(units),
        json.dumps(detail),
        json.dumps(refs),
    )


_INSERT_SQL = """
    INSERT INTO node_projections (
        projection_id, run_id, node_id, parent_id, level, order_index, label, summary_text,
        provisional, covered_segment_ids_json, evidence_refs_json, kind, state,
        content_units_json, detail_json, child_ids_json, started_at, completed_at, error,
        updated_event_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Planned events never downgrade a completed node: Resume re-plans every node.
_ON_PLANNED = """
    ON CONFLICT(run_id, node_id) DO UPDATE SET
        level = excluded.level,
        order_index = excluded.order_index,
        label = excluded.label,
        kind = excluded.kind,
        covered_segment_ids_json = excluded.covered_segment_ids_json,
        child_ids_json = COALESCE(excluded.child_ids_json, node_projections.child_ids_json),
        state = 'pending',
        started_at = NULL,
        completed_at = NULL,
        error = NULL,
        updated_event_id = excluded.updated_event_id
    WHERE node_projections.state != 'completed'
"""

# A re-ask keeps the start time of the item's first model call.
_ON_ACTIVE = """
    ON CONFLICT(run_id, node_id) DO UPDATE SET
        state = 'active',
        started_at = CASE
            WHEN node_projections.state = 'active' AND node_projections.started_at IS NOT NULL
            THEN node_projections.started_at ELSE excluded.started_at END,
        completed_at = NULL,
        error = NULL,
        updated_event_id = excluded.updated_event_id
"""


def _on_finished(*, reused: bool) -> str:
    # A reused node keeps an earlier Attempt's times; otherwise it has none.
    started = (
        "CASE WHEN node_projections.state = 'completed' THEN node_projections.started_at END"
        if reused
        else "node_projections.started_at"
    )
    completed = (
        "CASE WHEN node_projections.state = 'completed' "
        "THEN COALESCE(node_projections.completed_at, excluded.completed_at) "
        "ELSE excluded.completed_at END"
        if reused
        else "excluded.completed_at"
    )
    return f"""
    ON CONFLICT(run_id, node_id) DO UPDATE SET
        state = 'completed',
        provisional = 0,
        summary_text = COALESCE(excluded.summary_text, node_projections.summary_text),
        content_units_json = COALESCE(excluded.content_units_json, node_projections.content_units_json),
        detail_json = COALESCE(excluded.detail_json, node_projections.detail_json),
        evidence_refs_json = CASE
            WHEN excluded.summary_text IS NULL THEN node_projections.evidence_refs_json
            ELSE excluded.evidence_refs_json END,
        child_ids_json = COALESCE(excluded.child_ids_json, node_projections.child_ids_json),
        started_at = {started},
        completed_at = {completed},
        error = NULL,
        updated_event_id = excluded.updated_event_id
"""


_ON_COMPLETED = _on_finished(reused=False)
_ON_REUSED = _on_finished(reused=True)

_ON_FAILED = """
    ON CONFLICT(run_id, node_id) DO UPDATE SET
        state = 'failed',
        completed_at = NULL,
        error = excluded.error,
        updated_event_id = excluded.updated_event_id
"""


def apply_item_event(
    connection: sqlite3.Connection,
    run_id: str,
    event_id: int,
    event: ItemEvent,
    label: str,
    now: str,
) -> None:
    """Apply one tree item event recorded as `event_id`; other kinds are ignored.

    A completed node keeps the start time of its model call; a reused node was
    not computed in this Attempt, so it keeps no duration unless an earlier
    Attempt of this Run already finished it.
    """
    if event.kind not in TREE_KINDS:
        return
    level, order = event_position(event)
    summary_text = units_json = detail_json = None
    refs_json = "[]"
    if event.state in ("completed", "reused") and event.summary is not None:
        summary_text, units_json, detail_json, refs_json = summary_columns(event.summary)
    finished = event.state in ("completed", "reused")
    if event.state == "planned":
        state, conflict = "pending", _ON_PLANNED
    elif event.state in ("active", "retrying"):
        state, conflict = "active", _ON_ACTIVE
    elif event.state == "completed":
        state, conflict = "completed", _ON_COMPLETED
    elif event.state == "reused":
        state, conflict = "completed", _ON_REUSED
    else:
        state, conflict = "failed", _ON_FAILED
    connection.execute(
        _INSERT_SQL + conflict,
        (
            str(uuid.uuid4()),
            run_id,
            event.work_id,
            None,
            level,
            order,
            label,
            summary_text,
            0 if finished else 1,
            json.dumps(list(event.covered_segment_ids)),
            refs_json,
            event.kind,
            state,
            units_json,
            detail_json,
            json.dumps(list(event.child_ids)),
            now if state == "active" else None,
            now if finished else None,
            (event.message or "Failed") if state == "failed" else None,
            event_id,
        ),
    )
    if event.child_ids:
        adopt_children(connection, run_id, event.work_id, event.child_ids, event_id)


def adopt_children(
    connection: sqlite3.Connection,
    run_id: str,
    parent_id: str,
    child_ids: Sequence[str],
    event_id: int,
) -> None:
    """Point children at their parent; only rows whose link changes are touched."""
    placeholders = ", ".join("?" * len(child_ids))
    connection.execute(
        f"""
        UPDATE node_projections SET parent_id = ?, updated_event_id = ?
        WHERE run_id = ? AND node_id IN ({placeholders})
          AND (parent_id IS NULL OR parent_id != ?)
        """,
        (parent_id, event_id, run_id, *child_ids, parent_id),
    )


def label_root(connection: sqlite3.Connection, run_id: str, write_event: EventWriter) -> None:
    """Label the single top-level node as the root once merging completes."""
    top = connection.execute(
        "SELECT MAX(level) FROM node_projections WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    if top is None or top < 1:
        return
    rows = connection.execute(
        "SELECT node_id, label, state, kind FROM node_projections WHERE run_id = ? AND level = ?",
        (run_id, top),
    ).fetchall()
    if len(rows) != 1:
        return
    row = rows[0]
    label = root_label(top)
    if row["label"] == label:
        return
    event_id = write_event(
        connection,
        run_id,
        "item",
        {
            "work_id": row["node_id"],
            "kind": row["kind"],
            "state": "completed" if row["state"] == "completed" else "planned",
            "stage": "merging",
            "label": label,
            "mutation": "root_label",
        },
    )
    connection.execute(
        "UPDATE node_projections SET label = ?, updated_event_id = ? WHERE run_id = ? AND node_id = ?",
        (label, event_id, run_id, row["node_id"]),
    )

def reset_active_nodes(
    connection: sqlite3.Connection, run_id: str, write_event: EventWriter
) -> int:
    """Return unfinished nodes to pending, with an item cursor for each changed row."""
    rows = connection.execute(
        "SELECT node_id, kind, label FROM node_projections WHERE run_id = ? AND state = 'active'",
        (run_id,),
    ).fetchall()
    changed = 0
    for row in rows:
        event_id = write_event(
            connection,
            run_id,
            "item",
            {
                "work_id": row["node_id"],
                "kind": row["kind"],
                "state": "planned",
                "stage": "summarizing" if row["kind"] == "leaf" else "merging",
                "label": row["label"],
                "mutation": "attempt_ended",
            },
        )
        changed += connection.execute(
            """
            UPDATE node_projections
            SET state = 'pending', started_at = NULL, updated_event_id = ?
            WHERE run_id = ? AND node_id = ? AND state = 'active'
            """,
            (event_id, run_id, row["node_id"]),
        ).rowcount
    return changed

def reconcile_final_tree(
    connection: sqlite3.Connection,
    run_id: str,
    write_event: EventWriter,
    root: TreeNode,
    nodes: Sequence[TreeNode],
    labeler: NodeLabeler,
    now: str,
) -> None:
    """Make the projection match the finished tree, cursor-stamping every changed row.

    Rows the tree does not contain are removed. Only rows whose values change
    are re-stamped, so the final stream update stays small.
    """
    parent_of = {child: node.node_id for node in nodes for child in node.children}
    existing = {
        row["node_id"]: row
        for row in connection.execute(
            "SELECT * FROM node_projections WHERE run_id = ?", (run_id,)
        ).fetchall()
    }
    for node in nodes:
        kind = "leaf" if not node.children else "passthrough" if len(node.children) == 1 else "merge"
        if node.node_id == root.node_id and node.children:
            label = root_label(node.level)
        else:
            label = labeler.label(kind, node.level, node.order, node.covered_segments)
        summary_text, units_json, detail_json, refs_json = summary_columns(
            node.summary.model_dump(mode="json")
        )
        values = {
            "parent_id": parent_of.get(node.node_id),
            "level": node.level,
            "order_index": node.order,
            "label": label,
            "kind": kind,
            "state": "completed",
            "provisional": 0,
            "summary_text": summary_text,
            "content_units_json": units_json,
            "detail_json": detail_json,
            "evidence_refs_json": refs_json,
            "child_ids_json": json.dumps(list(node.children)),
            "covered_segment_ids_json": json.dumps(list(node.covered_segments)),
            "error": None,
        }
        row = existing.pop(node.node_id, None)
        stage = "summarizing" if kind == "leaf" else "merging"
        payload = {
            "work_id": node.node_id,
            "kind": kind,
            "state": "completed",
            "stage": stage,
            "label": label,
            "mutation": "final_tree",
        }
        if row is None:
            event_id = write_event(connection, run_id, "item", payload)
            connection.execute(
                _INSERT_SQL,
                (
                    str(uuid.uuid4()),
                    run_id,
                    node.node_id,
                    values["parent_id"],
                    node.level,
                    node.order,
                    label,
                    summary_text,
                    0,
                    values["covered_segment_ids_json"],
                    refs_json,
                    kind,
                    "completed",
                    units_json,
                    detail_json,
                    values["child_ids_json"],
                    None,
                    now,
                    None,
                    event_id,
                ),
            )
            continue
        changed = {key: value for key, value in values.items() if row[key] != value}
        if row["completed_at"] is None:
            changed["completed_at"] = now
        if not changed:
            continue
        event_id = write_event(connection, run_id, "item", payload)
        assignments = ", ".join(f"{key} = ?" for key in changed)
        connection.execute(
            f"UPDATE node_projections SET {assignments}, updated_event_id = ? "
            "WHERE run_id = ? AND node_id = ?",
            (*changed.values(), event_id, run_id, node.node_id),
        )
    if existing:
        stale = list(existing)
        connection.execute(
            f"DELETE FROM node_projections WHERE run_id = ? AND node_id IN ({', '.join('?' * len(stale))})",
            (run_id, *stale),
        )


# --- Reads ----------------------------------------------------------------------


def max_event_id(run_id: str) -> int:
    """The cursor of a Run: its latest run_events id, or 0 without events."""
    row = get_database().fetchone(
        "SELECT COALESCE(MAX(event_id), 0) AS cursor FROM run_events WHERE run_id = ?",
        (run_id,),
    )
    return int(row["cursor"]) if row is not None else 0


def node_duration_seconds(row: sqlite3.Row) -> float | None:
    """Seconds a completed node's model call took; null for reused nodes or missing times."""
    started_at, completed_at = row["started_at"], row["completed_at"]
    if row["state"] != "completed" or not started_at or not completed_at:
        return None
    try:
        seconds = (
            datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)
        ).total_seconds()
    except ValueError:
        return None
    return round(seconds, 3) if seconds >= 0 else None


def tree_items(
    run_id: str,
    *,
    after_event_id: int = 0,
    through_event_id: int | None = None,
    connection: sqlite3.Connection | None = None,
) -> list[NodeTreeItem]:
    """Tree rows changed after `after_event_id`, ordered by level then order.

    `after_event_id` 0 returns every row, including legacy rows stamped 0.
    `through_event_id` bounds the read to a cursor taken beforehand, so rows
    committed later are left for the next read instead of being skipped.
    Pass `connection` to read inside the caller's read transaction.
    """

    def fetchall(query: str, params: tuple[object, ...]) -> list[sqlite3.Row]:
        if connection is not None:
            return connection.execute(query, params).fetchall()
        return get_database().fetchall(query, params)

    query = """
        SELECT n.node_id, n.parent_id, n.level, n.order_index, n.kind, n.label, n.state,
               n.child_ids_json, n.covered_segment_ids_json, n.started_at, n.completed_at,
               CASE WHEN n.child_ids_json IS NULL THEN (
                   SELECT COUNT(*) FROM node_projections AS c
                   WHERE c.run_id = n.run_id AND c.parent_id = n.node_id
               ) END AS legacy_child_count
        FROM node_projections AS n
        WHERE n.run_id = ?
    """
    params: list[object] = [run_id]
    if after_event_id > 0:
        query += " AND n.updated_event_id > ?"
        params.append(after_event_id)
    if through_event_id is not None:
        query += " AND n.updated_event_id <= ?"
        params.append(through_event_id)
    rows = fetchall(query + " ORDER BY n.level, n.order_index, n.node_id", tuple(params))
    if not rows:
        return []
    pages: dict[str, PageRange] = {
        row["segment_id"]: (row["page_start"], row["page_end"])
        for row in fetchall(
            "SELECT segment_id, page_start, page_end FROM run_segments "
            "WHERE run_id = ? AND page_start IS NOT NULL",
            (run_id,),
        )
    }
    items = []
    for row in rows:
        covered = json.loads(row["covered_segment_ids_json"] or "[]")
        page_range = span_pages(covered, pages)
        if row["child_ids_json"] is not None:
            child_count = len(json.loads(row["child_ids_json"]))
        else:
            child_count = int(row["legacy_child_count"] or 0)
        items.append(
            NodeTreeItem(
                node_id=row["node_id"],
                parent_id=row["parent_id"],
                level=row["level"],
                order=row["order_index"],
                kind=row["kind"],
                label=row["label"],
                state=row["state"],
                child_count=child_count,
                page_start=page_range[0] if page_range else None,
                page_end=page_range[1] if page_range else None,
                duration_seconds=node_duration_seconds(row),
            )
        )
    return items
