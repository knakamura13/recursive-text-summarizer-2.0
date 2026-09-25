<script lang="ts">
	import { untrack } from 'svelte';
	import { errorMessage } from '$lib/api/client';
	import type { SourcePage } from '$lib/api/types';
	import { formatCount } from '$lib/format';
	import { sourceLoader } from '$lib/source/sourceCache';
	import { ChunkLayout, chunkPieces, pageAtOffset, type ChunkPiece, type SourceFocus } from '$lib/source/sourceModel';

	interface Props {
		documentId: string;
		/** Code points in the Document text when known (DocumentSummary.char_count). */
		totalLength: number | null;
		pages: readonly SourcePage[];
		focus: SourceFocus | null;
		onclearfocus?: () => void;
		/** Reading position (code point) to restore when the viewer mounts without a focus. */
		initialOffset?: number;
		onposition?: (offset: number) => void;
	}

	let { documentId, totalLength, pages, focus, onclearfocus, initialOffset = 0, onposition }: Props = $props();

	// Must match the .flow text metrics below; only used for the first height estimate, which
	// is replaced by measured pixels per character as soon as a chunk renders.
	const LINE_HEIGHT = 24;
	const AVERAGE_CHAR_WIDTH = 7.4;
	const PADDING_X = 48;
	/** At most three 16 K chunks are in the DOM at once. */
	const MAX_CHUNKS = 3;
	/** Room above a jump target for its "Page N" marker (see .page-marker). */
	const TOP_INSET = 44;

	type Jump = { offset: number; align: 'top' | 'reading' };
	type Rendered = { index: number; height: number; pieces: ChunkPiece[] | null; error: unknown };

	const loader = $derived(sourceLoader(documentId));
	let scroller = $state<HTMLDivElement | null>(null);
	let flow = $state<HTMLDivElement | null>(null);
	let topSpacer = $state<HTMLDivElement | null>(null);
	let bottomSpacer = $state<HTMLDivElement | null>(null);
	let scrollTop = $state(0);
	let viewportHeight = $state(0);
	let viewportWidth = $state(0);
	let learnedTotal = $state<number | null>(null);
	let layout = $state.raw<ChunkLayout | null>(null);
	let layoutVersion = $state(0);
	let loadedVersion = $state(0);
	let failures = $state.raw(new Map<number, unknown>());
	let jump = $state.raw<Jump | null>(untrack(() => (focus || initialOffset <= 0 ? null : { offset: initialOffset, align: 'top' })));
	let pageInput = $state<number | null>(null);
	let layoutWidth = 0;
	let pendingScroll: number | null = null;
	let lastFocusKey: number | null = null;

	const total = $derived(learnedTotal ?? totalLength);
	const lastPage = $derived(pages.length > 0 ? pages[pages.length - 1].page : null);
	const range = $derived.by(() => {
		void layoutVersion;
		if (!layout) return { first: 0, last: -1 };
		return layout.visibleRange(scrollTop, viewportHeight, Math.min(800, viewportHeight), MAX_CHUNKS);
	});
	const rendered = $derived.by((): Rendered[] => {
		void loadedVersion;
		const current = layout;
		if (!current) return [];
		const items: Rendered[] = [];
		for (let index = range.first; index <= range.last; index++) {
			const chunk = loader.peek(index);
			items.push({
				index,
				height: current.height(index),
				pieces: chunk
					? chunkPieces(chunk, { highlight: focus?.range ?? null, pages, anchor: jump?.offset ?? null })
					: null,
				error: failures.get(index) ?? null
			});
		}
		return items;
	});
	const spaceAbove = $derived.by(() => {
		void layoutVersion;
		return layout && rendered.length > 0 ? layout.top(rendered[0].index) : 0;
	});
	const spaceBelow = $derived.by(() => {
		void layoutVersion;
		if (!layout) return 0;
		const next = rendered.length > 0 ? rendered[rendered.length - 1].index + 1 : 0;
		return Math.max(0, layout.totalHeight - layout.top(next));
	});
	const position = $derived.by(() => {
		void layoutVersion;
		if (!layout || !total) return null;
		void rendered;
		const offset = layout.offsetAt(scrollTop + TOP_INSET);
		const marked = pageAtTop();
		const page = (marked !== null && pages.find((item) => item.page === marked)) || pageAtOffset(pages, offset);
		return { offset, percent: Math.min(100, Math.round((offset / total) * 100)), page };
	});

	/** The page whose marker is the last one at or above the viewport top; the offset estimate is
	 * too coarse inside a chunk to name the page reliably. */
	function pageAtTop(): number | null {
		if (!scroller || !flow || pages.length === 0) return null;
		const limit = scroller.getBoundingClientRect().top + TOP_INSET;
		let found: number | null = null;
		for (const marker of flow.querySelectorAll<HTMLElement>('.page-marker')) {
			if (marker.getBoundingClientRect().top > limit) break;
			found = Number(marker.dataset.page);
		}
		return found;
	}

	function setScroll(y: number) {
		if (!scroller) return;
		scroller.scrollTop = Math.max(0, y);
		scrollTop = scroller.scrollTop;
	}

	function jumpTo(offset: number, align: Jump['align'] = 'reading') {
		jump = { offset, align };
		if (layout && scroller) setScroll(layout.positionOf(offset) - (align === 'reading' ? viewportHeight / 3 : 0));
	}

	/** Chunks with a request in flight from this viewer (the loader shares the request itself). */
	const requested = new Set<number>();

	function requestChunk(index: number) {
		if (requested.has(index)) return;
		requested.add(index);
		loader.load(index).then(
			(chunk) => {
				requested.delete(index);
				if (chunk.totalLength !== total) learnedTotal = chunk.totalLength;
				loadedVersion += 1;
			},
			(error: unknown) => {
				requested.delete(index);
				failures = new Map(failures).set(index, error);
			}
		);
	}

	function retryChunk(index: number) {
		const next = new Map(failures);
		next.delete(index);
		failures = next;
	}

	// Without a known length, the first chunk tells it.
	$effect(() => {
		if (total === null && !failures.has(0)) requestChunk(0);
	});

	// (Re)build the pixel layout when the length is known or the width changes (text reflows).
	$effect(() => {
		const length = total;
		const width = viewportWidth;
		if (length === null || width <= 0) return;
		untrack(() => {
			const current = layout;
			if (current && current.totalLength === length && Math.abs(layoutWidth - width) < 2) return;
			const keep = current && scroller ? current.offsetAt(scroller.scrollTop) : null;
			const rate =
				current && layoutWidth > 0
					? current.pxPerChar * (layoutWidth / width)
					: LINE_HEIGHT / Math.max(16, (width - PADDING_X) / AVERAGE_CHAR_WIDTH);
			const next = new ChunkLayout(length, rate);
			layout = next;
			layoutWidth = width;
			layoutVersion += 1;
			if (jump) pendingScroll = next.positionOf(jump.offset) - (jump.align === 'reading' ? viewportHeight / 3 : 0);
			else if (keep !== null) pendingScroll = next.positionOf(keep);
		});
	});

	$effect(() => {
		for (const item of rendered) {
			if (item.pieces === null && item.error === null) requestChunk(item.index);
		}
	});

	$effect(() => {
		const current = focus;
		if (current === null || current.key === lastFocusKey) return;
		lastFocusKey = current.key;
		untrack(() => jumpTo(current.range.start));
	});

	$effect(() => {
		const offset = position?.offset;
		if (offset !== undefined) untrack(() => onposition?.(offset));
	});

	/** Write the new layout's spacer and placeholder heights now, so a scroll correction made
	 * in the same frame lands on the final geometry. */
	function syncSizes(current: ChunkLayout, items: NodeListOf<HTMLElement>) {
		if (items.length === 0 || !topSpacer || !bottomSpacer) return;
		const first = Number(items[0].dataset.chunk);
		const last = Number(items[items.length - 1].dataset.chunk);
		topSpacer.style.height = `${current.top(first)}px`;
		bottomSpacer.style.height = `${Math.max(0, current.totalHeight - current.top(last + 1))}px`;
		for (const item of items) {
			if (item.dataset.loaded === undefined) item.style.height = `${current.height(Number(item.dataset.chunk))}px`;
		}
	}

	/**
	 * After every render: measure rendered chunks, keep the chunk under the viewport top where it
	 * was when heights above it change, and finish a pending jump once its anchor is in the DOM.
	 */
	function afterRender() {
		const current = layout;
		const flowEl = flow;
		if (!scroller || !flowEl || !current) return;
		if (pendingScroll !== null) {
			setScroll(pendingScroll);
			pendingScroll = null;
		}
		const top = scroller.scrollTop;
		const anchorIndex = current.chunkAt(top);
		const anchorTop = current.top(anchorIndex);
		const items = flowEl.querySelectorAll<HTMLElement>('[data-chunk]');
		// Chunks flow inline so lines continue across chunk boundaries; a chunk's top is its
		// zero-height start marker and its height runs to the next chunk's top.
		const tops = Array.from(items, (item) =>
			item.dataset.loaded !== undefined ? (item.firstElementChild as HTMLElement).offsetTop : item.offsetTop
		);
		const entries: [number, number][] = [];
		items.forEach((item, k) => {
			if (item.dataset.loaded === undefined) return;
			const bottom = k + 1 < items.length ? tops[k + 1] : flowEl.offsetHeight;
			entries.push([Number(item.dataset.chunk), bottom - tops[k]]);
		});
		if (current.measure(entries)) {
			syncSizes(current, items);
			const delta = current.top(anchorIndex) - anchorTop;
			if (Math.abs(delta) >= 1) setScroll(top + delta);
			layoutVersion += 1;
		}
		const target = jump;
		const anchor = target ? flowEl.querySelector<HTMLElement>('[data-jump-anchor]') : null;
		const inset = target?.align === 'reading' ? Math.min(scroller.clientHeight / 3, 160) : TOP_INSET;
		if (target && anchor) {
			const y = anchor.getBoundingClientRect().top - scroller.getBoundingClientRect().top + scroller.scrollTop;
			jump = null;
			setScroll(y - inset);
		} else if (target) {
			// The first scroll used estimated heights; head for the target again with the measured ones
			// until its chunk renders.
			const y = Math.max(0, current.positionOf(target.offset) - inset);
			if (Math.abs(scroller.scrollTop - y) >= 1) setScroll(y);
		}
	}

	$effect(() => {
		void rendered;
		untrack(afterRender);
	});

	function onPageSubmit(event: SubmitEvent) {
		event.preventDefault();
		if (pageInput === null) return;
		const wanted = pageInput;
		const target = pages.find((page) => page.page >= wanted) ?? pages[pages.length - 1];
		if (target) jumpTo(target.start, 'top');
	}
</script>

<section class="source" aria-label="Source text">
	<div class="toolbar">
		{#if position}
			<p class="position">
				{#if position.page && lastPage !== null}{`p. ${position.page.page} of ${lastPage} · `}{/if}{position.percent}%{#if total}<span class="length">{` of ${formatCount(total)} characters`}</span>{/if}
			</p>
		{/if}
		{#if pages.length > 0}
			<form class="page-jump" onsubmit={onPageSubmit}>
				<label>
					<span class="visually-hidden">Go to page</span>
					<input
						type="number"
						inputmode="numeric"
						min={pages[0].page}
						max={lastPage}
						placeholder="Page"
						bind:value={pageInput}
					/>
				</label>
				<button class="button small" type="submit">Go</button>
			</form>
		{/if}
		{#if focus}
			<div class="focus" data-testid="source-focus">
				<span class="focus-label tone-{focus.tone}">{focus.label}</span>
				<button type="button" class="button small ghost" onclick={() => jumpTo(focus.range.start)}>Show</button>
				{#if onclearfocus}
					<button type="button" class="button small ghost" onclick={onclearfocus}>Clear</button>
				{/if}
			</div>
		{/if}
	</div>
	<!-- Focusable so keyboard users can scroll the text. -->
	<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
	<div
		class="scroller"
		bind:this={scroller}
		bind:clientHeight={viewportHeight}
		bind:clientWidth={viewportWidth}
		onscroll={(event) => (scrollTop = event.currentTarget.scrollTop)}
		tabindex="0"
		role="region"
		aria-label="Document text"
		data-testid="source-scroller"
	>
		{#if total === 0}
			<p class="note">This Document has no text.</p>
		{:else if !layout && failures.has(0)}
			<p class="note error" role="alert">
				Could not load the text: {errorMessage(failures.get(0))}
				<button type="button" class="button small" onclick={() => retryChunk(0)}>Try again</button>
			</p>
		{:else if !layout}
			<p class="note">Loading text…</p>
		{/if}
		<div bind:this={topSpacer} style:height="{spaceAbove}px"></div>
		<!-- One line on purpose: the text is pre-wrap, so template whitespace would render. -->
		<div class="flow tone-{focus?.tone ?? 'span'}" bind:this={flow}>{#each rendered as item (item.index)}{#if item.pieces}<span class="chunk" data-chunk={item.index} data-loaded=""><span class="chunk-start"></span>{#each item.pieces as piece, i (i)}{#if piece.kind === 'text'}{piece.text}{:else if piece.kind === 'mark'}<mark>{piece.text}</mark>{:else if piece.kind === 'page'}<span class="page-marker" data-page={piece.page} data-label="Page {piece.page}{piece.ocr ? ' · OCR' : ''}{piece.blank ? ' · blank' : ''}"></span>{:else}<span class="jump-anchor" data-jump-anchor=""></span>{/if}{/each}</span>{:else}<div class="placeholder" data-chunk={item.index} style:height="{item.height}px">
						{#if item.error}
							<span role="alert">Could not load this part of the text: {errorMessage(item.error)}</span>
							<button type="button" class="button small" onclick={() => retryChunk(item.index)}>Try again</button>
						{:else}
							<span class="loading">Loading…</span>
						{/if}
					</div>{/if}{/each}</div>
		<div bind:this={bottomSpacer} style:height="{spaceBelow}px"></div>
	</div>
</section>

<style>
	.source {
		display: flex;
		flex-direction: column;
		min-height: 0;
		height: 100%;
	}

	.toolbar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-2) var(--space-3);
		min-height: 3rem;
		padding: var(--space-1) var(--space-4);
		border-bottom: 1px solid var(--color-border);
		font-size: 0.8125rem;
	}

	.position {
		margin: 0;
		color: var(--color-text-muted);
		font-variant-numeric: tabular-nums;
	}

	.page-jump {
		display: flex;
		align-items: center;
		gap: var(--space-1);
	}

	.page-jump input {
		width: 5.5rem;
	}

	.focus {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: var(--space-1);
		min-width: 0;
	}

	.focus-label {
		padding: 0.1rem 0.4rem;
		border-radius: var(--radius-sm);
		overflow-wrap: anywhere;
	}

	.focus-label.tone-quote {
		background: var(--color-highlight);
	}

	.focus-label.tone-span {
		background: var(--color-surface);
	}

	.scroller {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		overflow-x: hidden;
		overflow-anchor: none;
		overscroll-behavior: contain;
		padding: 0 24px;
	}

	.note {
		margin: var(--space-4) 0;
		color: var(--color-text-muted);
	}

	.note.error {
		color: var(--color-danger);
	}

	.flow {
		position: relative;
		font-family: var(--font-serif, Georgia, 'Times New Roman', serif);
		font-size: 15px;
		line-height: 24px;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		color: var(--color-text);
	}

	.chunk-start {
		display: inline-block;
		width: 0;
		height: 0;
		vertical-align: top;
	}

	.flow mark {
		color: inherit;
		border-radius: 2px;
	}

	.flow.tone-span mark {
		background: var(--color-surface);
		box-shadow: -3px 0 0 var(--color-surface), 3px 0 0 var(--color-surface);
	}

	.flow.tone-quote mark {
		background: var(--color-highlight);
		box-shadow: -2px 0 0 var(--color-highlight), 2px 0 0 var(--color-highlight);
	}

	.page-marker {
		display: block;
		margin: 12px 0 4px;
		border-top: 1px dashed var(--color-border-strong);
		user-select: none;
		line-height: 20px;
	}

	.page-marker::before {
		content: attr(data-label);
		font-family: var(--font-sans);
		font-size: 0.75rem;
		color: var(--color-text-subtle);
	}

	.placeholder {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--space-2);
		padding-top: var(--space-4);
		white-space: normal;
		font-family: var(--font-sans);
		font-size: 0.85rem;
		color: var(--color-text-muted);
		overflow: hidden;
	}

	.jump-anchor {
		display: inline-block;
		width: 0;
		height: 0;
	}
</style>
