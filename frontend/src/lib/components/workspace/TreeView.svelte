<script lang="ts">
	import { SvelteSet } from 'svelte/reactivity';
	import { formatCount, formatDuration } from '$lib/format';
	import type { NodeState, TreeNode } from '$lib/api/types';
	import {
		ancestorIds,
		flattenTree,
		isTreeKey,
		parentIds,
		scrollToReveal,
		treeKeyAction,
		windowRows,
		type ChildrenIndex
	} from '$lib/tree/treeModel';

	interface Props {
		nodes: ReadonlyMap<string, TreeNode>;
		children: ChildrenIndex;
		selected: string | null;
		onselect: (nodeId: string) => void;
		label?: string;
	}

	let { nodes, children, selected, onselect, label = 'Summary tree' }: Props = $props();

	/** Fixed row height keeps windowing exact and rows at the 44 px touch-target minimum. */
	const ROW_HEIGHT = 44;
	const INDENT = 14;
	const MAX_INDENT_LEVELS = 8;

	const STATE_TEXT: Record<NodeState, string> = {
		pending: 'Pending',
		active: 'Working',
		completed: 'Done',
		failed: 'Failed'
	};

	const uid = $props.id();
	const collapsed = new SvelteSet<string>();
	let scroller = $state<HTMLDivElement | null>(null);
	let scrollTop = $state(0);
	let viewportHeight = $state(0);
	let focusId = $state<string | null>(null);
	let revealed: string | null = null;

	const rows = $derived(flattenTree(children, collapsed));
	const range = $derived(windowRows(scrollTop, viewportHeight, ROW_HEIGHT, rows.length));
	const visible = $derived(rows.slice(range.start, range.end));
	const counts = $derived.by(() => {
		let completed = 0;
		let failed = 0;
		let active = 0;
		for (const node of nodes.values()) {
			if (node.state === 'completed') completed += 1;
			else if (node.state === 'failed') failed += 1;
			else if (node.state === 'active') active += 1;
		}
		return { completed, failed, active, total: nodes.size };
	});
	const activeDescendant = $derived(
		focusId !== null && rows.some((row) => row.id === focusId) ? rowDomId(focusId) : undefined
	);

	function rowDomId(id: string): string {
		return `${uid}-row-${id.replace(/\s/g, '_')}`;
	}

	function reveal(id: string) {
		const index = rows.findIndex((row) => row.id === id);
		if (index < 0 || !scroller) return;
		const target = scrollToReveal(index, scroller.scrollTop, scroller.clientHeight, ROW_HEIGHT);
		if (target !== null) {
			scroller.scrollTop = target;
			scrollTop = target;
		}
	}

	// Show a node selected from outside the tree (URL on reload, detail links): expand its
	// ancestors and scroll to it once it exists.
	$effect(() => {
		const id = selected;
		if (id === null || id === revealed || !nodes.has(id)) return;
		revealed = id;
		for (const ancestor of ancestorIds(nodes, id)) collapsed.delete(ancestor);
		focusId = id;
		queueMicrotask(() => reveal(id));
	});

	function toggle(id: string, expanded: boolean) {
		if (expanded) collapsed.delete(id);
		else collapsed.add(id);
	}

	function onkeydown(event: KeyboardEvent) {
		if (!isTreeKey(event.key) || event.altKey || event.ctrlKey || event.metaKey) return;
		event.preventDefault();
		const result = treeKeyAction(rows, focusId ?? selected, event.key);
		if (result.toggle) toggle(result.toggle.id, result.toggle.expanded);
		focusId = result.focus;
		if (result.focus !== null) {
			const id = result.focus;
			queueMicrotask(() => reveal(id));
		}
		if (result.select) onselect(result.select);
	}

	function onRowClick(event: MouseEvent, id: string, hasChildren: boolean, expanded: boolean) {
		focusId = id;
		const target = event.target as HTMLElement;
		if (hasChildren && target.closest('[data-toggle]')) {
			toggle(id, !expanded);
			return;
		}
		onselect(id);
	}

	function collapseAll() {
		for (const id of parentIds(children)) collapsed.add(id);
		scroller?.scrollTo({ top: 0 });
	}
</script>

<div class="tree">
	<div class="toolbar">
		<p class="counts" aria-live="polite">
			{formatCount(counts.completed)} of {formatCount(counts.total)} nodes done{#if counts.active > 0}{` · ${formatCount(counts.active)} working`}{/if}{#if counts.failed > 0}{' · '}<span class="failed-count">{formatCount(counts.failed)} failed</span>{/if}
		</p>
		<div class="toolbar-actions">
			<button type="button" class="link" onclick={() => collapsed.clear()} disabled={collapsed.size === 0}>
				Expand all
			</button>
			<button type="button" class="link" onclick={collapseAll}>Collapse all</button>
		</div>
	</div>
	<div
		class="scroller"
		bind:this={scroller}
		bind:clientHeight={viewportHeight}
		onscroll={(event) => (scrollTop = event.currentTarget.scrollTop)}
		role="tree"
		tabindex="0"
		aria-label={label}
		aria-activedescendant={activeDescendant}
		{onkeydown}
		onfocus={() => {
			if (focusId === null && rows.length > 0) focusId = selected ?? rows[0].id;
		}}
	>
		<div class="track" style:height="{rows.length * ROW_HEIGHT}px">
			<div class="window" style:transform="translateY({range.start * ROW_HEIGHT}px)">
				{#each visible as row (row.id)}
					{@const node = nodes.get(row.id)}
					{#if node}
						<!-- Keyboard input and focus live on the tree container (aria-activedescendant). -->
						<!-- svelte-ignore a11y_click_events_have_key_events, a11y_interactive_supports_focus -->
						<div
							id={rowDomId(row.id)}
							class="row state-{node.state}"
							class:selected={row.id === selected}
							class:focused={row.id === focusId}
							role="treeitem"
							aria-level={row.depth + 1}
							aria-setsize={row.siblings}
							aria-posinset={row.position}
							aria-expanded={row.hasChildren ? row.expanded : undefined}
							aria-selected={row.id === selected}
							data-node-id={row.id}
							style:padding-left="{Math.min(row.depth, MAX_INDENT_LEVELS) * INDENT + 4}px"
							onclick={(event) => onRowClick(event, row.id, row.hasChildren, row.expanded)}
						>
							<span class="toggle" data-toggle aria-hidden="true">
								{#if row.hasChildren}<span class="chevron" class:open={row.expanded}>▸</span>{/if}
							</span>
							<span class="icon icon-{node.state}" aria-hidden="true"></span>
							<span class="label">{node.label}</span>
							<span class="meta">
								<span class="visually-hidden">{STATE_TEXT[node.state]}</span>
								{#if node.state === 'completed' && node.duration_seconds !== null}
									{formatDuration(node.duration_seconds)}
								{:else if node.state !== 'completed'}
									<span aria-hidden="true">{STATE_TEXT[node.state]}</span>
								{/if}
							</span>
						</div>
					{/if}
				{/each}
			</div>
		</div>
	</div>
</div>

<style>
	.tree {
		display: flex;
		flex-direction: column;
		min-height: 0;
		height: 100%;
	}

	.toolbar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-1) var(--space-3);
		padding: var(--space-2) var(--space-3);
		border-bottom: 1px solid var(--color-border);
	}

	.counts {
		margin: 0;
		font-size: 0.85rem;
		color: var(--color-text-muted);
	}

	.failed-count {
		color: var(--color-danger);
	}

	.toolbar-actions {
		display: flex;
		gap: var(--space-1);
	}

	.link {
		min-height: 44px;
		padding: 0 var(--space-2);
		border: none;
		background: none;
		color: var(--color-focus);
		font: inherit;
		font-size: 0.85rem;
		cursor: pointer;
	}

	.link:disabled {
		color: var(--color-text-subtle);
		cursor: default;
	}

	.scroller {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		overflow-x: hidden;
		overscroll-behavior: contain;
		position: relative;
	}

	.scroller:focus-visible {
		outline-offset: -2px;
	}

	.track {
		position: relative;
	}

	.window {
		position: absolute;
		inset: 0 0 auto 0;
		will-change: transform;
	}

	.row {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		height: 44px;
		padding-right: var(--space-3);
		border-bottom: 1px solid var(--color-border);
		cursor: pointer;
		font-size: 0.9rem;
		white-space: nowrap;
	}

	.row:hover {
		background: var(--color-surface);
	}

	.row.selected {
		background: var(--color-surface-elevated);
		box-shadow: inset 2px 0 0 var(--color-accent);
	}

	.scroller:focus-visible .row.focused {
		outline: 2px solid var(--color-focus);
		outline-offset: -2px;
	}

	.toggle {
		flex: 0 0 28px;
		align-self: stretch;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--color-text-muted);
	}

	.chevron {
		display: inline-block;
		transition: transform 120ms ease;
	}

	.chevron.open {
		transform: rotate(90deg);
	}

	.icon {
		flex: 0 0 12px;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		border: 2px solid var(--color-border-strong);
	}

	.icon-active {
		border-color: var(--color-amber);
		background: var(--color-amber);
		animation: pulse 1.2s ease-in-out infinite;
	}

	.icon-completed {
		border-color: var(--color-success);
		background: var(--color-success);
	}

	.icon-failed {
		border-color: var(--color-danger);
		background: var(--color-danger);
		border-radius: 2px;
	}

	@keyframes pulse {
		50% {
			opacity: 0.35;
		}
	}

	.label {
		flex: 1 1 auto;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.state-pending .label {
		color: var(--color-text-muted);
	}

	.state-failed .label {
		color: var(--color-danger);
	}

	.meta {
		flex: 0 0 auto;
		font-size: 0.8rem;
		color: var(--color-text-muted);
		font-variant-numeric: tabular-nums;
	}

	.visually-hidden {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip: rect(0 0 0 0);
		white-space: nowrap;
	}
</style>
