import { describe, expect, it, vi } from 'vitest';
import type { SourcePage, SourceSlice } from '../../src/lib/api/types';
import {
	ChunkLayout,
	chunkPieces,
	LruCache,
	pageAtOffset,
	SOURCE_CHUNK_SIZE,
	SourceChunkLoader,
	type SourceChunk
} from '../../src/lib/source/sourceModel';

function chunkOf(text: string, start = 0, index = 0): SourceChunk {
	const length = Array.from(text).length;
	return {
		index,
		start,
		end: start + length,
		text,
		astral: /[\uD800-\uDFFF]/.test(text),
		totalLength: start + length
	};
}

function page(number: number, start: number, end: number): SourcePage {
	return { page: number, start, end, ocr: false, blank: false };
}

describe('LruCache', () => {
	it('evicts the least recently used entry', () => {
		const cache = new LruCache<number, string>(2);
		cache.set(1, 'a');
		cache.set(2, 'b');
		expect(cache.get(1)).toBe('a');
		cache.set(3, 'c');
		expect(cache.get(2)).toBeUndefined();
		expect(cache.get(1)).toBe('a');
		expect(cache.get(3)).toBe('c');
	});
});

describe('SourceChunkLoader', () => {
	function slices(total: number) {
		return vi.fn(
			async (offset: number, limit: number): Promise<SourceSlice> => ({
				text: 'x'.repeat(Math.max(0, Math.min(limit, total - offset))),
				offset,
				total_length: total,
				has_more: offset + limit < total
			})
		);
	}

	it('requests fixed chunks by code-point offset and shares concurrent requests', async () => {
		const fetch = slices(SOURCE_CHUNK_SIZE * 2 + 100);
		const loader = new SourceChunkLoader(fetch);
		const [first, again] = await Promise.all([loader.load(2), loader.load(2)]);
		expect(first).toBe(again);
		expect(fetch).toHaveBeenCalledTimes(1);
		expect(fetch).toHaveBeenCalledWith(SOURCE_CHUNK_SIZE * 2, SOURCE_CHUNK_SIZE);
		expect(first).toMatchObject({ start: SOURCE_CHUNK_SIZE * 2, end: SOURCE_CHUNK_SIZE * 2 + 100 });
	});

	it('serves cached chunks and refetches chunks evicted from the LRU cache', async () => {
		const fetch = slices(SOURCE_CHUNK_SIZE * 10);
		const loader = new SourceChunkLoader(fetch, 2);
		await loader.load(0);
		await loader.load(1);
		await loader.load(0);
		expect(fetch).toHaveBeenCalledTimes(2);
		await loader.load(2);
		expect(loader.peek(1)).toBeUndefined();
		expect(loader.peek(0)).toBeDefined();
		await loader.load(1);
		expect(fetch).toHaveBeenCalledTimes(4);
	});

	it('forgets a failed request so it can be retried', async () => {
		const fetch = vi
			.fn<(offset: number, limit: number) => Promise<SourceSlice>>()
			.mockRejectedValueOnce(new Error('offline'))
			.mockResolvedValueOnce({ text: 'ok', offset: 0, total_length: 2, has_more: false });
		const loader = new SourceChunkLoader(fetch);
		await expect(loader.load(0)).rejects.toThrow('offline');
		await expect(loader.load(0)).resolves.toMatchObject({ text: 'ok' });
	});

	it('measures chunk ends in code points when the text has astral characters', async () => {
		const loader = new SourceChunkLoader(async (offset) => ({
			text: 'a😀b',
			offset,
			total_length: 3,
			has_more: false
		}));
		expect(await loader.load(0)).toMatchObject({ start: 0, end: 3, astral: true });
	});
});

describe('chunkPieces', () => {
	it('marks a code-point range after astral characters without splitting surrogate pairs', () => {
		// Code points: 0 '😀', 1 'a', 2 '😀', 3 'b', 4 'c'
		const pieces = chunkPieces(chunkOf('😀a😀bc'), { highlight: { start: 2, end: 4 } });
		expect(pieces).toEqual([
			{ kind: 'text', text: '😀a' },
			{ kind: 'mark', text: '😀b' },
			{ kind: 'text', text: 'c' }
		]);
	});

	it('clips a highlight that spans several chunks to this chunk', () => {
		const chunk = chunkOf('0123456789', 100, 1);
		expect(chunkPieces(chunk, { highlight: { start: 50, end: 104 } })).toEqual([
			{ kind: 'mark', text: '0123' },
			{ kind: 'text', text: '456789' }
		]);
		expect(chunkPieces(chunk, { highlight: { start: 108, end: 500 } })).toEqual([
			{ kind: 'text', text: '01234567' },
			{ kind: 'mark', text: '89' }
		]);
		expect(chunkPieces(chunk, { highlight: { start: 0, end: 100 } })).toEqual([{ kind: 'text', text: '0123456789' }]);
	});

	it('inserts page markers for pages starting inside the chunk', () => {
		const pages = [page(1, 0, 95), page(2, 95, 104), page(3, 104, 130), page(4, 130, 200)];
		expect(chunkPieces(chunkOf('0123456789', 100), { pages })).toEqual([
			{ kind: 'text', text: '0123' },
			{ kind: 'page', page: 3, ocr: false, blank: false },
			{ kind: 'text', text: '456789' }
		]);
	});

	it('places a page marker, highlight, and scroll anchor at the same offset in reading order', () => {
		const pieces = chunkPieces(chunkOf('ab😀cd'), {
			pages: [page(1, 0, 2), page(2, 2, 5)],
			highlight: { start: 2, end: 3 },
			anchor: 2
		});
		expect(pieces).toEqual([
			{ kind: 'page', page: 1, ocr: false, blank: false },
			{ kind: 'text', text: 'ab' },
			{ kind: 'page', page: 2, ocr: false, blank: false },
			{ kind: 'anchor' },
			{ kind: 'mark', text: '😀' },
			{ kind: 'text', text: 'cd' }
		]);
	});
});

describe('pageAtOffset', () => {
	const pages = [page(1, 0, 10), page(2, 10, 20), page(5, 20, 40)];

	it('finds the page containing an offset', () => {
		expect(pageAtOffset(pages, 0)?.page).toBe(1);
		expect(pageAtOffset(pages, 10)?.page).toBe(2);
		expect(pageAtOffset(pages, 39)?.page).toBe(5);
		expect(pageAtOffset(pages, 400)?.page).toBe(5);
		expect(pageAtOffset([], 3)).toBeNull();
	});
});

describe('ChunkLayout', () => {
	const total = 2_100_000;

	it('covers a 2.1 M character Document with estimated chunk heights', () => {
		const layout = new ChunkLayout(total, 0.5);
		expect(layout.count).toBe(Math.ceil(total / SOURCE_CHUNK_SIZE));
		expect(layout.height(0)).toBe(SOURCE_CHUNK_SIZE * 0.5);
		expect(layout.height(layout.count - 1)).toBe((total - SOURCE_CHUNK_SIZE * (layout.count - 1)) * 0.5);
		expect(layout.totalHeight).toBeCloseTo(total * 0.5);
	});

	it('maps between pixels and offsets', () => {
		const layout = new ChunkLayout(total, 0.5);
		const offset = SOURCE_CHUNK_SIZE * 40 + SOURCE_CHUNK_SIZE / 2;
		const y = layout.positionOf(offset);
		expect(layout.chunkAt(y)).toBe(40);
		expect(layout.offsetAt(y)).toBe(offset);
		expect(layout.chunkAt(-10)).toBe(0);
		expect(layout.chunkAt(layout.totalHeight + 10)).toBe(layout.count - 1);
	});

	it('learns pixels per character from measured chunks and keeps measured heights', () => {
		const layout = new ChunkLayout(total, 0.5);
		expect(layout.measure([[3, SOURCE_CHUNK_SIZE * 2]])).toBe(true);
		expect(layout.pxPerChar).toBe(2);
		expect(layout.height(3)).toBe(SOURCE_CHUNK_SIZE * 2);
		expect(layout.height(4)).toBe(SOURCE_CHUNK_SIZE * 2);
		expect(layout.top(4)).toBe(SOURCE_CHUNK_SIZE * 2 * 4);
		// Sub-pixel noise is not a change.
		expect(layout.measure([[3, SOURCE_CHUNK_SIZE * 2 + 0.4]])).toBe(false);
	});

	it('renders at most three chunks however tall the viewport is', () => {
		const layout = new ChunkLayout(total, 0.01);
		const chunkHeight = SOURCE_CHUNK_SIZE * 0.01;
		const range = layout.visibleRange(chunkHeight * 50 + 1, chunkHeight * 10, 800, 3);
		expect(range.last - range.first + 1).toBe(3);
		expect(range.first).toBe(50);
		const normal = new ChunkLayout(total, 0.5).visibleRange(SOURCE_CHUNK_SIZE * 0.5 * 20 + 4000, 900, 800, 3);
		expect(normal).toEqual({ first: 20, last: 20 });
	});
});
