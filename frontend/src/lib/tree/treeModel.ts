import type { TreeNode } from '$lib/api/types';

/** Child ids per parent id, sorted for display. The `null` key holds the displayed roots. */
export type ChildrenIndex = ReadonlyMap<string | null, readonly string[]>;

export interface FlatRow {
	id: string;
	/** Displayed parent (null for a displayed root). */
	parentId: string | null;
	depth: number;
	hasChildren: boolean;
	expanded: boolean;
	/** 1-based position among its displayed siblings, and the sibling count (aria-posinset/setsize). */
	position: number;
	siblings: number;
}

/**
 * Group nodes under their parents. A node whose parent is unknown (not planned yet, or missing
 * from the snapshot) is shown as a root so nothing disappears while a Run is still planning.
 */
export function buildChildrenIndex(nodes: ReadonlyMap<string, TreeNode>): ChildrenIndex {
	const groups = new Map<string | null, TreeNode[]>();
	for (const node of nodes.values()) {
		const parent =
			node.parent_id !== null && node.parent_id !== node.node_id && nodes.has(node.parent_id)
				? node.parent_id
				: null;
		let group = groups.get(parent);
		if (!group) {
			group = [];
			groups.set(parent, group);
		}
		group.push(node);
	}
	const index = new Map<string | null, string[]>();
	for (const [parent, group] of groups) {
		group.sort(
			parent === null
				? // Highest level first so the root of a finished tree sits on top; before merges
					// exist every leaf is a root and they list in document order.
					(a, b) => b.level - a.level || a.order - b.order || (a.node_id < b.node_id ? -1 : 1)
				: (a, b) => a.order - b.order || a.level - b.level || (a.node_id < b.node_id ? -1 : 1)
		);
		index.set(
			parent,
			group.map((node) => node.node_id)
		);
	}
	if (!index.has(null)) index.set(null, []);
	return index;
}

/** True when an upsert changes where nodes appear, so the children index must be rebuilt. */
export function affectsStructure(previous: TreeNode | undefined, next: TreeNode): boolean {
	return (
		previous === undefined ||
		previous.parent_id !== next.parent_id ||
		previous.order !== next.order ||
		previous.level !== next.level
	);
}

/**
 * Depth-first list of the rows that are visible given the collapsed set. Iterative so a deep
 * chain of passthrough nodes cannot overflow the call stack; a cycle in bad data is cut off.
 */
export function flattenTree(index: ChildrenIndex, collapsed: ReadonlySet<string>): FlatRow[] {
	const rows: FlatRow[] = [];
	const seen = new Set<string>();
	const roots = index.get(null) ?? [];
	type Frame = { ids: readonly string[]; next: number; depth: number; parentId: string | null };
	const stack: Frame[] = [{ ids: roots, next: 0, depth: 0, parentId: null }];
	while (stack.length > 0) {
		const frame = stack[stack.length - 1];
		if (frame.next >= frame.ids.length) {
			stack.pop();
			continue;
		}
		const position = frame.next;
		const id = frame.ids[frame.next++];
		if (seen.has(id)) continue;
		seen.add(id);
		const children = index.get(id) ?? [];
		const hasChildren = children.length > 0;
		const expanded = hasChildren && !collapsed.has(id);
		rows.push({
			id,
			parentId: frame.parentId,
			depth: frame.depth,
			hasChildren,
			expanded,
			position: position + 1,
			siblings: frame.ids.length
		});
		if (expanded) stack.push({ ids: children, next: 0, depth: frame.depth + 1, parentId: id });
	}
	return rows;
}

/** Every node that has children; used for "Collapse all". */
export function parentIds(index: ChildrenIndex): string[] {
	const ids: string[] = [];
	for (const [parent, children] of index) {
		if (parent !== null && children.length > 0) ids.push(parent);
	}
	return ids;
}

/** Ancestors of a node from its displayed parent upward, used to reveal a selected node. */
export function ancestorIds(nodes: ReadonlyMap<string, TreeNode>, id: string): string[] {
	const result: string[] = [];
	const seen = new Set<string>([id]);
	let current = nodes.get(id)?.parent_id ?? null;
	while (current !== null && nodes.has(current) && !seen.has(current)) {
		result.push(current);
		seen.add(current);
		current = nodes.get(current)?.parent_id ?? null;
	}
	return result;
}

export interface RowWindow {
	/** First rendered row index (inclusive). */
	start: number;
	/** Last rendered row index (exclusive). */
	end: number;
}

/** Rows to render for a scroll position, with `overscan` extra rows above and below. */
export function windowRows(
	scrollTop: number,
	viewportHeight: number,
	rowHeight: number,
	rowCount: number,
	overscan = 6
): RowWindow {
	if (rowCount <= 0 || rowHeight <= 0) return { start: 0, end: 0 };
	const first = Math.floor(Math.max(0, scrollTop) / rowHeight);
	const visible = Math.ceil(Math.max(0, viewportHeight) / rowHeight) + 1;
	const start = Math.max(0, Math.min(rowCount - 1, first - overscan));
	const end = Math.min(rowCount, first + visible + overscan);
	return { start, end: Math.max(start, end) };
}

/** Scroll offset that brings row `index` fully into view, or null when it already is. */
export function scrollToReveal(
	index: number,
	scrollTop: number,
	viewportHeight: number,
	rowHeight: number
): number | null {
	const top = index * rowHeight;
	const bottom = top + rowHeight;
	if (top < scrollTop) return top;
	if (bottom > scrollTop + viewportHeight) return Math.max(0, bottom - viewportHeight);
	return null;
}

export type TreeKey = 'ArrowUp' | 'ArrowDown' | 'ArrowLeft' | 'ArrowRight' | 'Home' | 'End' | 'Enter';

const TREE_KEY_SET: Record<TreeKey, true> = {
	ArrowUp: true,
	ArrowDown: true,
	ArrowLeft: true,
	ArrowRight: true,
	Home: true,
	End: true,
	Enter: true
};

export function isTreeKey(key: string): key is TreeKey {
	return Object.hasOwn(TREE_KEY_SET, key);
}

export interface TreeKeyResult {
	/** Row that should hold the keyboard focus afterwards. */
	focus: string | null;
	/** Expand (true) or collapse (false) this node. */
	toggle?: { id: string; expanded: boolean };
	/** Open this node's detail. */
	select?: string;
}

/** WAI-ARIA tree keyboard behaviour over the visible rows. */
export function treeKeyAction(rows: readonly FlatRow[], focusId: string | null, key: TreeKey): TreeKeyResult {
	if (rows.length === 0) return { focus: null };
	const index = focusId === null ? -1 : rows.findIndex((row) => row.id === focusId);
	if (index < 0) {
		// Focus left the visible rows (collapsed away or removed): start from the top.
		return { focus: key === 'End' ? rows[rows.length - 1].id : rows[0].id };
	}
	const row = rows[index];
	switch (key) {
		case 'ArrowDown':
			return { focus: rows[Math.min(rows.length - 1, index + 1)].id };
		case 'ArrowUp':
			return { focus: rows[Math.max(0, index - 1)].id };
		case 'Home':
			return { focus: rows[0].id };
		case 'End':
			return { focus: rows[rows.length - 1].id };
		case 'ArrowRight':
			if (!row.hasChildren) return { focus: row.id };
			if (!row.expanded) return { focus: row.id, toggle: { id: row.id, expanded: true } };
			return { focus: rows[index + 1]?.id ?? row.id };
		case 'ArrowLeft':
			if (row.expanded) return { focus: row.id, toggle: { id: row.id, expanded: false } };
			return { focus: row.parentId ?? row.id };
		case 'Enter':
			return { focus: row.id, select: row.id };
	}
}
