// Data builders and helpers for the Document workspace specs (document-*.spec.ts).
import { expect, type Page } from '@playwright/test';
import type { FinalSummary, NodeDetail, SegmentRef, SourcePage, TreeNode } from '../../src/lib/api/types';

export type Layout = 'phone' | 'tablet' | 'desktop';

/** The layout the workspace picks for the page's viewport (contract 8: 640 and 1024 px). */
export function layoutOf(page: Page): Layout {
	const width = page.viewportSize()?.width ?? 1280;
	return width < 640 ? 'phone' : width < 1024 ? 'tablet' : 'desktop';
}

export function treeNode(overrides: Partial<TreeNode> & Pick<TreeNode, 'node_id'>): TreeNode {
	return {
		parent_id: null,
		level: 0,
		order: 0,
		kind: 'leaf',
		label: overrides.node_id,
		state: 'completed',
		child_count: 0,
		page_start: null,
		page_end: null,
		duration_seconds: 4,
		...overrides
	};
}

export function segment(order: number, start: number, end: number, pages: [number, number] | null = null): SegmentRef {
	return {
		segment_id: `S${String(order + 1).padStart(6, '0')}`,
		order,
		start,
		end,
		core_start: start,
		core_end: end,
		page_start: pages?.[0] ?? null,
		page_end: pages?.[1] ?? null
	};
}

export function nodeDetail(node: TreeNode, overrides: Partial<NodeDetail> = {}): NodeDetail {
	return {
		node_id: node.node_id,
		parent_id: node.parent_id,
		level: node.level,
		order: node.order,
		kind: node.kind,
		label: node.label,
		state: node.state,
		summary_text: `Summary of ${node.label}.`,
		content_units: [],
		annotations: [],
		entities: [],
		quotations: [],
		covered_segments: [],
		child_ids: [],
		started_at: '2026-09-23T10:00:00Z',
		completed_at: '2026-09-23T10:00:04Z',
		duration_seconds: 4,
		error: null,
		...overrides
	};
}

/** Three short paragraphs with phrases the specs look for. */
export const SMALL_SOURCE = [
	'The harbour commission met on Tuesday to review the dredging budget for the coming year.',
	'Members agreed that the northern channel needs work before the autumn storms arrive.',
	'A final vote on the contract is expected in March, after the public comment period closes.'
].join('\n\n');

/** Code-point range of `phrase` inside SMALL_SOURCE (ASCII, so code points equal indices). */
export function rangeOf(phrase: string): { start: number; end: number } {
	const start = SMALL_SOURCE.indexOf(phrase);
	if (start < 0) throw new Error(`Phrase not in source: ${phrase}`);
	return { start, end: start + phrase.length };
}

export function availableSummary(overrides: Partial<FinalSummary> = {}): FinalSummary {
	return {
		available: true,
		text: 'The commission reviewed the dredging budget. A vote is expected in March.',
		sentences: [],
		removed_sentences: [],
		citations: [],
		word_count: 12,
		target_words: 300,
		short_of_target: true,
		notices: [],
		verification_state: 'completed',
		publication: 'editorial',
		...overrides
	};
}

/** root -> 40 groups -> 1,959 leaves: 2,000 nodes. */
export function largeTree(): TreeNode[] {
	const nodes: TreeNode[] = [
		treeNode({ node_id: 'L2N0001', level: 2, kind: 'merge', label: 'Root · Level 2', child_count: 40 })
	];
	let leafOrder = 0;
	for (let group = 0; group < 40; group++) {
		const groupId = `L1N${String(group + 1).padStart(4, '0')}`;
		const leaves = group < 39 ? 49 : 1959 - 39 * 49;
		nodes.push(
			treeNode({
				node_id: groupId,
				parent_id: 'L2N0001',
				level: 1,
				order: group,
				kind: 'merge',
				label: `Level 1 · Group ${group + 1}`,
				child_count: leaves
			})
		);
		for (let i = 0; i < leaves; i++, leafOrder++) {
			nodes.push(
				treeNode({
					node_id: `L0N${String(leafOrder + 1).padStart(4, '0')}`,
					parent_id: groupId,
					order: leafOrder,
					label: `Segment ${leafOrder + 1}`
				})
			);
		}
	}
	return nodes;
}

// --- A 2.1 M-character Document, generated on demand ---------------------------------------

export const LARGE_LENGTH = 2_100_000;
const LINE = 64;
export const PAGE_CHARS = 3000;

function line(index: number): string {
	const head = `Line ${String(index).padStart(6, '0')} of the long document. `;
	return `${head}${'lorem ipsum '.repeat(6)}`.slice(0, LINE - 1) + '\n';
}

/** ASCII text, so code-point offsets equal string indices. */
export function largeSlice(offset: number, limit: number): string {
	const end = Math.min(LARGE_LENGTH, offset + limit);
	if (end <= offset) return '';
	let text = '';
	for (let index = Math.floor(offset / LINE); index * LINE < end; index++) text += line(index);
	const from = offset - Math.floor(offset / LINE) * LINE;
	return text.slice(from, from + (end - offset));
}

export function largePages(): SourcePage[] {
	const pages: SourcePage[] = [];
	for (let start = 0, page = 1; start < LARGE_LENGTH; start += PAGE_CHARS, page++) {
		pages.push({ page, start, end: Math.min(LARGE_LENGTH, start + PAGE_CHARS), ocr: false, blank: false });
	}
	return pages;
}

/** Selects a workspace tab; on phones the tab bar holds Source too. */
export async function openTab(page: Page, name: 'Summary' | 'Status' | 'Tree' | 'Source' | 'Runs') {
	const tab = page.getByRole('tablist', { name: 'Document views' }).getByRole('tab', { name: new RegExp(`^${name}`) });
	await tab.click();
	await expect(tab).toHaveAttribute('aria-selected', 'true');
}

/** The Source text region (a tab on phones, the side pane otherwise). */
export async function sourceRegion(page: Page) {
	if (layoutOf(page) === 'phone') await openTab(page, 'Source');
	const region = page.getByRole('region', { name: 'Document text' });
	await expect(region).toBeVisible();
	return region;
}
