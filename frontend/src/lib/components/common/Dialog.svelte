<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		open = $bindable(false),
		title,
		children,
		footer
	}: {
		open?: boolean;
		title: string;
		children: Snippet;
		footer?: Snippet;
	} = $props();

	function onKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			open = false;
		}
	}
</script>

{#if open}
	<div class="backdrop" role="presentation" onclick={() => (open = false)}></div>
	<div
		class="dialog"
		role="dialog"
		aria-modal="true"
		aria-label={title}
		onkeydown={onKeydown}
		tabindex="-1"
	>
		<header class="dialog-header">
			<h2>{title}</h2>
			<button type="button" class="icon-button" aria-label="Close dialog" onclick={() => (open = false)}>
				×
			</button>
		</header>
		<div class="dialog-body">{@render children()}</div>
		{#if footer}
			<footer class="dialog-footer">{@render footer()}</footer>
		{/if}
	</div>
{/if}

<style>
	.backdrop {
		position: fixed;
		inset: 0;
		background: rgba(43, 43, 40, 0.35);
		z-index: 40;
	}

	.dialog {
		position: fixed;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		width: min(560px, calc(100vw - 2rem));
		background: var(--color-bg);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-lg);
		box-shadow: var(--shadow-md);
		z-index: 50;
	}

	.dialog-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: var(--space-4) var(--space-5);
		border-bottom: 1px solid var(--color-border);
	}

	.dialog-header h2 {
		margin: 0;
		font-size: 1rem;
		font-weight: 600;
	}

	.dialog-body {
		padding: var(--space-5);
	}

	.dialog-footer {
		display: flex;
		justify-content: flex-end;
		gap: var(--space-3);
		padding: var(--space-4) var(--space-5);
		border-top: 1px solid var(--color-border);
	}

	.icon-button {
		border: none;
		background: transparent;
		font-size: 1.5rem;
		line-height: 1;
		cursor: pointer;
		color: var(--color-text-muted);
	}
</style>
