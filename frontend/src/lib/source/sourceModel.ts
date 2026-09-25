import type { SourcePage, SourceSlice } from '$lib/api/types';
import { pythonOffsetToUtf16Index } from '$lib/unicodeOffsets';

/** Code points per fetched chunk of Document text (the source API caps a slice at 65,536). */
export const SOURCE_CHUNK_SIZE = 16384;

/** A code-point range `[start, end)` into the canonical Document text. */
export interface TextRange {
	start: number;
	end: number;
}

/** What the Source viewer highlights and scrolls to. `key` changes on every request so asking
 * for the same range again scrolls to it again. */
export interface SourceFocus {
	range: TextRange;
	/** `span` tints a covered range (a node's segments); `quote` marks cited evidence. */
	tone: 'span' | 'quote';
	label: string;
	key: number;
}

export interface SourceChunk {
	index: number;
	/** Code-point offsets of the chunk inside the Document text. */
	start: number;
	end: number;
	text: string;
	/** True when the text holds surrogate pairs, so code points and UTF-16 indices differ. */
	astral: boolean;
	totalLength: number;
}

/** Least-recently-used cache with a fixed capacity. */
export class LruCache<K, V> {
	readonly capacity: number;
	#entries = new Map<K, V>();

	constructor(capacity: number) {
		this.capacity = Math.max(1, capacity);
	}

	/** Returns the value and marks it as most recently used. */
	get(key: K): V | undefined {
		if (!this.#entries.has(key)) return undefined;
		const value = this.#entries.get(key) as V;
		this.#entries.delete(key);
		this.#entries.set(key, value);
		return value;
	}

	set(key: K, value: V): void {
		this.#entries.delete(key);
		this.#entries.set(key, value);
		while (this.#entries.size > this.capacity) {
			const oldest = this.#entries.keys().next();
			if (oldest.done) break;
			this.#entries.delete(oldest.value);
		}
	}
}

export type SliceFetcher = (offset: number, limit: number) => Promise<SourceSlice>;

/**
 * Fetches Document text in fixed chunks, keeps recently used chunks in an LRU cache, and shares
 * one request per chunk between concurrent callers.
 */
export class SourceChunkLoader {
	readonly #fetchSlice: SliceFetcher;
	readonly #cache: LruCache<number, SourceChunk>;
	readonly #inflight = new Map<number, Promise<SourceChunk>>();

	constructor(fetchSlice: SliceFetcher, capacity = 12) {
		this.#fetchSlice = fetchSlice;
		this.#cache = new LruCache(capacity);
	}

	/** Cached chunk, if present (marks it as recently used). */
	peek(index: number): SourceChunk | undefined {
		return this.#cache.get(index);
	}

	load(index: number): Promise<SourceChunk> {
		const cached = this.#cache.get(index);
		if (cached) return Promise.resolve(cached);
		const pending = this.#inflight.get(index);
		if (pending) return pending;
		const request = this.#fetchSlice(index * SOURCE_CHUNK_SIZE, SOURCE_CHUNK_SIZE)
			.then((slice) => {
				const astral = /[\uD800-\uDFFF]/.test(slice.text);
				const length = astral ? Array.from(slice.text).length : slice.text.length;
				const chunk: SourceChunk = {
					index,
					start: slice.offset,
					end: Math.min(slice.total_length, slice.offset + length),
					text: slice.text,
					astral,
					totalLength: slice.total_length
				};
				this.#cache.set(index, chunk);
				return chunk;
			})
			.finally(() => this.#inflight.delete(index));
		this.#inflight.set(index, request);
		return request;
	}
}

/** Index of the first page whose start is >= offset (pages sorted by start). */
function firstPageAtOrAfter(pages: readonly SourcePage[], offset: number): number {
	let low = 0;
	let high = pages.length;
	while (low < high) {
		const mid = (low + high) >>> 1;
		if (pages[mid].start < offset) low = mid + 1;
		else high = mid;
	}
	return low;
}

/** The page containing a code-point offset (the last page starting at or before it). */
export function pageAtOffset(pages: readonly SourcePage[], offset: number): SourcePage | null {
	if (pages.length === 0) return null;
	let low = 0;
	let high = pages.length - 1;
	if (offset < pages[0].start) return pages[0];
	while (low < high) {
		const mid = (low + high + 1) >>> 1;
		if (pages[mid].start <= offset) low = mid;
		else high = mid - 1;
	}
	return pages[low];
}

export type ChunkPiece =
	| { kind: 'text'; text: string }
	| { kind: 'mark'; text: string }
	| { kind: 'page'; page: number; ocr: boolean; blank: boolean }
	| { kind: 'anchor' };

interface Boundary {
	at: number;
	order: number;
	piece?: ChunkPiece;
	markStart?: boolean;
	markEnd?: boolean;
}

/**
 * Split one chunk into text runs, highlight marks, page markers, and an optional zero-width
 * scroll anchor. Offsets are Document code points; they are converted to UTF-16 indices only
 * when the chunk holds astral characters.
 */
export function chunkPieces(
	chunk: SourceChunk,
	options: { highlight?: TextRange | null; pages?: readonly SourcePage[]; anchor?: number | null } = {}
): ChunkPiece[] {
	const { highlight = null, pages = [], anchor = null } = options;
	const length = chunk.end - chunk.start;
	const boundaries: Boundary[] = [];
	for (let i = firstPageAtOrAfter(pages, chunk.start); i < pages.length && pages[i].start < chunk.end; i++) {
		const page = pages[i];
		boundaries.push({
			at: page.start - chunk.start,
			order: 0,
			piece: { kind: 'page', page: page.page, ocr: page.ocr, blank: page.blank }
		});
	}
	const markFrom = highlight ? Math.max(highlight.start, chunk.start) - chunk.start : 0;
	const markTo = highlight ? Math.min(highlight.end, chunk.end) - chunk.start : 0;
	const hasMark = highlight !== null && markTo > markFrom;
	if (hasMark) {
		boundaries.push({ at: markFrom, order: 2, markStart: true });
		boundaries.push({ at: markTo, order: 1, markEnd: true });
	}
	if (anchor !== null && anchor >= chunk.start && anchor <= chunk.end) {
		boundaries.push({ at: anchor - chunk.start, order: 3, piece: { kind: 'anchor' } });
	}
	boundaries.sort((a, b) => a.at - b.at || a.order - b.order);

	const toIndex = (codePoint: number): number =>
		chunk.astral ? pythonOffsetToUtf16Index(chunk.text, codePoint) : codePoint;
	const pieces: ChunkPiece[] = [];
	let cursor = 0;
	let cursorIndex = 0;
	let marking = false;
	const emitUntil = (codePoint: number) => {
		if (codePoint <= cursor) return;
		const index = toIndex(codePoint);
		const text = chunk.text.slice(cursorIndex, index);
		if (text) pieces.push(marking ? { kind: 'mark', text } : { kind: 'text', text });
		cursor = codePoint;
		cursorIndex = index;
	};
	for (const boundary of boundaries) {
		const at = Math.min(Math.max(boundary.at, 0), length);
		emitUntil(at);
		if (boundary.markStart) marking = true;
		if (boundary.markEnd) marking = false;
		if (boundary.piece) pieces.push(boundary.piece);
	}
	emitUntil(length);
	return pieces;
}

/**
 * Pixel layout of every chunk for the virtualized viewer: measured heights for chunks that have
 * been rendered, estimates (characters x learned pixels per character) for the rest.
 */
export class ChunkLayout {
	readonly count: number;
	readonly totalLength: number;
	#heights: Float64Array;
	#measured: Uint8Array;
	#prefix: Float64Array;
	#pxPerChar: number;
	#measuredPx = 0;
	#measuredChars = 0;

	constructor(totalLength: number, pxPerChar: number) {
		this.totalLength = Math.max(0, totalLength);
		this.count = Math.ceil(this.totalLength / SOURCE_CHUNK_SIZE);
		this.#heights = new Float64Array(this.count);
		this.#measured = new Uint8Array(this.count);
		this.#prefix = new Float64Array(this.count + 1);
		this.#pxPerChar = pxPerChar;
		this.#estimateUnmeasured();
	}

	get pxPerChar(): number {
		return this.#pxPerChar;
	}

	get totalHeight(): number {
		return this.#prefix[this.count];
	}

	chunkLength(index: number): number {
		return Math.min(this.totalLength, (index + 1) * SOURCE_CHUNK_SIZE) - index * SOURCE_CHUNK_SIZE;
	}

	height(index: number): number {
		return this.#heights[index] ?? 0;
	}

	top(index: number): number {
		return this.#prefix[Math.max(0, Math.min(this.count, index))];
	}

	isMeasured(index: number): boolean {
		return this.#measured[index] === 1;
	}

	/** Chunk containing the pixel position `y` (clamped to the document). */
	chunkAt(y: number): number {
		if (this.count === 0) return 0;
		let low = 0;
		let high = this.count - 1;
		while (low < high) {
			const mid = (low + high + 1) >>> 1;
			if (this.#prefix[mid] <= y) low = mid;
			else high = mid - 1;
		}
		return low;
	}

	/** Approximate code-point offset shown at pixel position `y`. */
	offsetAt(y: number): number {
		if (this.count === 0) return 0;
		const index = this.chunkAt(y);
		const height = this.height(index) || 1;
		const fraction = Math.min(1, Math.max(0, (y - this.top(index)) / height));
		return Math.min(
			this.totalLength,
			index * SOURCE_CHUNK_SIZE + Math.floor(fraction * this.chunkLength(index))
		);
	}

	/** Approximate pixel position of a code-point offset. */
	positionOf(offset: number): number {
		if (this.count === 0) return 0;
		const index = Math.min(this.count - 1, Math.floor(Math.max(0, offset) / SOURCE_CHUNK_SIZE));
		const within = (offset - index * SOURCE_CHUNK_SIZE) / Math.max(1, this.chunkLength(index));
		return this.top(index) + Math.min(1, Math.max(0, within)) * this.height(index);
	}

	/**
	 * Record rendered heights. Returns true when any stored height changed. The per-character
	 * estimate for unmeasured chunks is re-learned from everything measured so far.
	 */
	measure(entries: Iterable<[index: number, height: number]>): boolean {
		let changed = false;
		for (const [index, height] of entries) {
			if (index < 0 || index >= this.count || !(height > 0)) continue;
			if (this.#measured[index] === 1) {
				if (Math.abs(this.#heights[index] - height) < 1) continue;
				this.#measuredPx -= this.#heights[index];
			} else {
				this.#measured[index] = 1;
				this.#measuredChars += this.chunkLength(index);
			}
			this.#heights[index] = height;
			this.#measuredPx += height;
			changed = true;
		}
		if (!changed) return false;
		if (this.#measuredChars > 0) this.#pxPerChar = this.#measuredPx / this.#measuredChars;
		this.#estimateUnmeasured();
		return true;
	}

	/**
	 * Chunks to render: those intersecting the viewport plus `overscan` pixels, never more than
	 * `maxChunks`, keeping the chunks under the viewport itself.
	 */
	visibleRange(scrollTop: number, viewportHeight: number, overscan: number, maxChunks = 3): { first: number; last: number } {
		if (this.count === 0) return { first: 0, last: -1 };
		const viewFirst = this.chunkAt(scrollTop);
		const viewLast = this.chunkAt(scrollTop + Math.max(0, viewportHeight - 1));
		let first = this.chunkAt(scrollTop - overscan);
		let last = this.chunkAt(scrollTop + viewportHeight + overscan);
		if (last - first + 1 <= maxChunks) return { first, last };
		first = viewFirst;
		last = Math.min(viewLast, viewFirst + maxChunks - 1);
		while (last - first + 1 < maxChunks) {
			if (last + 1 < this.count && (first === 0 || last - viewLast <= viewFirst - first)) last += 1;
			else if (first > 0) first -= 1;
			else break;
		}
		return { first, last };
	}

	#estimateUnmeasured(): void {
		for (let i = 0; i < this.count; i++) {
			if (this.#measured[i] !== 1) this.#heights[i] = Math.max(1, this.chunkLength(i) * this.#pxPerChar);
		}
		for (let i = 0; i < this.count; i++) this.#prefix[i + 1] = this.#prefix[i] + this.#heights[i];
	}
}
