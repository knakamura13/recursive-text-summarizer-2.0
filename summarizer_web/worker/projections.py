"""Persist tree projections for the web UI."""

from __future__ import annotations

import json
import uuid

from summarizer.hierarchy import TreeNode
from summarizer_web.db.connection import get_database


def _label_for_node(node: TreeNode) -> str:
    if node.level == 0 and not node.children:
        return "Full document"
    if node.children:
        return f"Merge group {node.order + 1}"
    return f"Segment {node.order + 1}"


def persist_projections(
    run_id: str,
    root: TreeNode,
    nodes: tuple[TreeNode, ...],
    source_text: str,
) -> None:
    db = get_database()
    db.execute("DELETE FROM node_projections WHERE run_id = ?", (run_id,))
    parent_map = {node.node_id: node for node in nodes}
    child_parent: dict[str, str | None] = {root.node_id: None}
    for node in nodes:
        for child_id in node.children:
            child_parent[child_id] = node.node_id
    for node in nodes:
        summary_text = node.summary.summary
        evidence_refs = [
            {"segment_id": item.segment_id, "quote": item.quote}
            for unit in node.summary.content_units
            for item in unit.evidence
        ]
        db.execute(
            """
            INSERT INTO node_projections (
                projection_id, run_id, node_id, parent_id, level, order_index,
                label, summary_text, provisional, covered_segment_ids_json, evidence_refs_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                run_id,
                node.node_id,
                child_parent.get(node.node_id),
                node.level,
                node.order,
                _label_for_node(node),
                summary_text,
                0,
                json.dumps(list(node.covered_segments)),
                json.dumps(evidence_refs),
            ),
        )
