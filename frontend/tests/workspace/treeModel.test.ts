import { describe, expect, it } from 'vitest';
import type { TreeNode } from '../../src/lib/api/types';
import {
	ancestorIds,
	buildChildrenIndex,
	flattenTree,
	scrollToReveal,
	treeKeyAction,
	windowRows,
	type FlatRow
} from '../../src/lib/tree/treeModel';

function node(id: string, level: number, order: number, parent: string | null): TreeNode {
	return {
		node_id: id,
		parent_id: parent,
		level,
		order,
		kind: level === 0 ? 'leaf' : 'merge',
		label: id,
		state: 'pending',
		child_count: 0,
		page_start: null,
		page_end: null,
		duration_seconds: null
	};
}

function mapOf(nodes: TreeNode[]): Map<string, TreeNode> {
	return new Map(nodes.map((item) => [item.node_id, item]));
}

/** root -> A (leaves a1, a2), B (leaf b1) */
const finished = mapOf([
	node('a2', 0, 1, 'A'),
	node('a1', 0, 0, 'A'),
	node('b1', 0, 2, 'B'),
	node('B', 1, 1, 'root'),
	node('A', 1, 0, 'root'),
	node('root', 2, 0, null)
]);

const ids = (rows: FlatRow[]) => rows.map((row) => row.id);

describe('buildChildrenIndex and flattenTree', () => {
	it('lists a finished tree depth-first from its root with siblings in order', () => {
		const rows = flattenTree(buildChildrenIndex(finished), new Set());
		expect(ids(rows)).toEqual(['root', 'A', 'a1', 'a2', 'B', 'b1']);
		expect(rows.map((row) => row.depth)).toEqual([0, 1, 2, 2, 1, 2]);
		expect(rows[1]).toMatchObject({ parentId: 'root', position: 1, siblings: 2, hasChildren: true });
	});

	it('shows leaves whose parents are not planned yet as roots in document order', () => {
		const planning = mapOf([node('l3', 0, 2, null), node('l1', 0, 0, null), node('l2', 0, 1, 'M-not-yet-sent')]);
		expect(ids(flattenTree(buildChildrenIndex(planning), new Set()))).toEqual(['l1', 'l2', 'l3']);
	});

	it('puts higher levels first among roots so a new merge level lands above the leaves', () => {
		const partial = mapOf([node('l1', 0, 0, 'M1'), node('l2', 0, 1, null), node('M1', 1, 0, null)]);
		expect(ids(flattenTree(buildChildrenIndex(partial), new Set()))).toEqual(['M1', 'l1', 'l2']);
	});

	it('hides the descendants of collapsed nodes', () => {
		const rows = flattenTree(buildChildrenIndex(finished), new Set(['A']));
		expect(ids(rows)).toEqual(['root', 'A', 'B', 'b1']);
		expect(rows[1]).toMatchObject({ id: 'A', expanded: false, hasChildren: true });
	});

	it('survives a parent cycle in bad data without looping', () => {
		const cyclic = mapOf([node('x', 1, 0, 'y'), node('y', 1, 1, 'x'), node('z', 0, 0, null)]);
		expect(ids(flattenTree(buildChildrenIndex(cyclic), new Set()))).toEqual(['z']);
	});

	it('flattens 2,000 nodes with every row present once', () => {
		const nodes: TreeNode[] = [node('root', 2, 0, null)];
		for (let g = 0; g < 40; g++) {
			nodes.push(node(`g${g}`, 1, g, 'root'));
			for (let l = 0; l < 49; l++) nodes.push(node(`g${g}l${l}`, 0, g * 49 + l, `g${g}`));
		}
		const rows = flattenTree(buildChildrenIndex(mapOf(nodes)), new Set());
		expect(rows).toHaveLength(nodes.length);
		expect(new Set(ids(rows)).size).toBe(nodes.length);
	});

	it('finds the ancestors to expand for a deep-linked node', () => {
		expect(ancestorIds(finished, 'a2')).toEqual(['A', 'root']);
		expect(ancestorIds(finished, 'root')).toEqual([]);
	});
});

describe('windowRows', () => {
	it('renders the visible rows plus overscan', () => {
		expect(windowRows(0, 440, 44, 2000, 5)).toEqual({ start: 0, end: 16 });
		expect(windowRows(44 * 100, 440, 44, 2000, 5)).toEqual({ start: 95, end: 116 });
	});

	it('clamps at the end of the list and handles empty lists', () => {
		expect(windowRows(44 * 1995, 440, 44, 2000, 5)).toEqual({ start: 1990, end: 2000 });
		expect(windowRows(0, 440, 44, 0)).toEqual({ start: 0, end: 0 });
	});

	it('scrolls just enough to reveal a row', () => {
		expect(scrollToReveal(10, 0, 440, 44)).toBe(44 * 11 - 440);
		expect(scrollToReveal(2, 200, 440, 44)).toBe(88);
		expect(scrollToReveal(5, 0, 440, 44)).toBeNull();
	});
});

describe('treeKeyAction', () => {
	const rows = flattenTree(buildChildrenIndex(finished), new Set());

	it('moves with Up, Down, Home, and End within bounds', () => {
		expect(treeKeyAction(rows, 'A', 'ArrowDown').focus).toBe('a1');
		expect(treeKeyAction(rows, 'A', 'ArrowUp').focus).toBe('root');
		expect(treeKeyAction(rows, 'root', 'ArrowUp').focus).toBe('root');
		expect(treeKeyAction(rows, 'b1', 'ArrowDown').focus).toBe('b1');
		expect(treeKeyAction(rows, 'a2', 'Home').focus).toBe('root');
		expect(treeKeyAction(rows, 'a2', 'End').focus).toBe('b1');
	});

	it('collapses with Left, then moves to the parent', () => {
		expect(treeKeyAction(rows, 'A', 'ArrowLeft')).toEqual({ focus: 'A', toggle: { id: 'A', expanded: false } });
		expect(treeKeyAction(rows, 'a2', 'ArrowLeft')).toEqual({ focus: 'A' });
	});

	it('expands with Right, then moves to the first child', () => {
		const collapsed = flattenTree(buildChildrenIndex(finished), new Set(['A']));
		expect(treeKeyAction(collapsed, 'A', 'ArrowRight')).toEqual({ focus: 'A', toggle: { id: 'A', expanded: true } });
		expect(treeKeyAction(rows, 'A', 'ArrowRight')).toEqual({ focus: 'a1' });
		expect(treeKeyAction(rows, 'a1', 'ArrowRight')).toEqual({ focus: 'a1' });
	});

	it('selects with Enter', () => {
		expect(treeKeyAction(rows, 'B', 'Enter')).toEqual({ focus: 'B', select: 'B' });
	});

	it('restarts from the top when the focused row is no longer visible', () => {
		expect(treeKeyAction(rows, 'gone', 'ArrowDown').focus).toBe('root');
		expect(treeKeyAction(rows, null, 'End').focus).toBe('b1');
		expect(treeKeyAction([], 'A', 'ArrowDown').focus).toBeNull();
	});
});
