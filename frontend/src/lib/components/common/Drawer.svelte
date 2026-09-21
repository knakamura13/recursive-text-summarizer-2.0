<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		open = $bindable(false),
		title,
		side = 'right',
		children
	}: {
		open?: boolean;
		title: string;
		side?: 'left' | 'right';
		children: Snippet;
	} = $props();
</script>

{#if open}
	<div class="backdrop" role="presentation" onclick={() => (open = false)}></div>
	<aside class="drawer {side}" aria-label={title}>
		<header>
			<h2>{title}</h2>
			<button type="button" aria-label="Close drawer" onclick={() => (open = false)}>×</button>
		</header>
		<div class="content">{@render children()}</div>
	</aside>
{/if}

<style>
	.backdrop {
		position: fixed;
		inset: 0;
		background: rgba(43, 43, 40, 0.25);
		z-index: 30;
	}

	.drawer {
		position: fixed;
		top: 0;
		bottom: 0;
		width: min(360px, 90vw);
		background: var(--color-bg);
		border: 1px solid var(--color-border);
		z-index: 31;
		display: flex;
		flex-direction: column;
	}

	.drawer.left {
		left: 0;
	}

	.drawer.right {
		right: 0;
	}

	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: var(--space-4);
		border-bottom: 1px solid var(--color-border);
	}

	header h2 {
		margin: 0;
		font-size: 0.95rem;
	}

	.content {
		overflow: auto;
		flex: 1;
	}
</style>
