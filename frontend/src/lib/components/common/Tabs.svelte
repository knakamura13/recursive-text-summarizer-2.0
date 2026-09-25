<script lang="ts" module>
	export interface TabItem {
		id: string;
		label: string;
		badge?: string | number | null;
	}

	/** Id of the tab button for `id`; panels use it for aria-labelledby. */
	export function tabId(idPrefix: string, id: string): string {
		return `${idPrefix}-tab-${id}`;
	}

	/** Id the panel for `id` must carry (the tab's aria-controls). */
	export function tabPanelId(idPrefix: string, id: string): string {
		return `${idPrefix}-panel-${id}`;
	}
</script>

<script lang="ts">
	// Tab list with automatic activation: arrows, Home, and End move focus and select.
	// Render each panel yourself with role="tabpanel", id={tabPanelId(idPrefix, id)},
	// and aria-labelledby={tabId(idPrefix, id)}.
	let {
		items,
		active = $bindable(),
		label,
		idPrefix,
		fill = false,
		onchange
	}: {
		items: TabItem[];
		active?: string;
		/** Accessible name of the tab list. */
		label: string;
		idPrefix: string;
		/** Equal-width tabs that fill the row (phone tab bars). */
		fill?: boolean;
		onchange?: (id: string) => void;
	} = $props();

	let buttons: HTMLButtonElement[] = $state([]);
	const focusIndex = $derived(Math.max(0, items.findIndex((item) => item.id === active)));

	function select(index: number) {
		const item = items[index];
		if (!item) return;
		buttons[index]?.focus();
		if (item.id === active) return;
		active = item.id;
		onchange?.(item.id);
	}

	function onKeydown(event: KeyboardEvent, index: number) {
		const count = items.length;
		let next: number;
		if (event.key === 'ArrowRight') next = (index + 1) % count;
		else if (event.key === 'ArrowLeft') next = (index - 1 + count) % count;
		else if (event.key === 'Home') next = 0;
		else if (event.key === 'End') next = count - 1;
		else return;
		event.preventDefault();
		select(next);
	}
</script>

<div class="tabs" class:fill role="tablist" aria-label={label}>
	{#each items as item, index (item.id)}
		<button
			bind:this={buttons[index]}
			type="button"
			role="tab"
			id={tabId(idPrefix, item.id)}
			aria-selected={item.id === active}
			aria-controls={tabPanelId(idPrefix, item.id)}
			tabindex={index === focusIndex ? 0 : -1}
			onclick={() => select(index)}
			onkeydown={(event) => onKeydown(event, index)}
		>
			<span class="label">{item.label}</span>
			{#if item.badge !== undefined && item.badge !== null && item.badge !== ''}
				<span class="badge">{item.badge}</span>
			{/if}
		</button>
	{/each}
</div>

<style>
	.tabs {
		display: flex;
		gap: var(--space-1);
		overflow-x: auto;
		scrollbar-width: none;
		border-bottom: 1px solid var(--color-border);
	}

	.tabs::-webkit-scrollbar {
		display: none;
	}

	button {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-2);
		flex: none;
		min-height: var(--touch-target);
		padding: 0 var(--space-4);
		border: none;
		border-bottom: 2px solid transparent;
		margin-bottom: -1px;
		background: transparent;
		color: var(--color-text-muted);
		font: inherit;
		font-size: 0.9375rem;
		font-weight: 500;
		white-space: nowrap;
		cursor: pointer;
		touch-action: manipulation;
	}

	button:hover {
		color: var(--color-text);
	}

	button[aria-selected='true'] {
		color: var(--color-text);
		border-bottom-color: var(--color-accent);
	}

	button:focus-visible {
		outline-offset: -2px;
	}

	.fill button {
		flex: 1 1 0;
		min-width: 0;
		padding: 0 var(--space-1);
		font-size: 0.875rem;
	}

	.fill .label {
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.badge {
		min-width: 1.25rem;
		padding: 0 0.375rem;
		border-radius: 999px;
		background: var(--color-surface);
		font-size: 0.75rem;
		font-variant-numeric: tabular-nums;
		line-height: 1.25rem;
		text-align: center;
	}
</style>
