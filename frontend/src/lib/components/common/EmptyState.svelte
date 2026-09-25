<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		title,
		message,
		headingLevel = 2,
		children
	}: {
		title: string;
		message?: string;
		headingLevel?: 2 | 3;
		/** Actions below the message. */
		children?: Snippet;
	} = $props();
</script>

<div class="empty-state">
	<svelte:element this={`h${headingLevel}`} class="title">{title}</svelte:element>
	{#if message}
		<p>{message}</p>
	{/if}
	{#if children}
		<div class="actions">{@render children()}</div>
	{/if}
</div>

<style>
	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--space-3);
		padding: var(--space-8) var(--space-4);
		text-align: center;
		color: var(--color-text-muted);
	}

	.title {
		color: var(--color-text);
		font-size: 1.125rem;
	}

	p {
		max-width: 36rem;
	}

	.actions {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: var(--space-3);
		margin-top: var(--space-2);
	}
</style>
