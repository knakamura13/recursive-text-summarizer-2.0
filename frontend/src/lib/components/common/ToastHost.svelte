<script lang="ts">
	import { toasts } from '$lib/stores/toasts.svelte';
</script>

<div class="toast-host" aria-live="polite" aria-relevant="additions">
	{#each toasts.items as toast (toast.id)}
		<div class="toast {toast.kind}" role={toast.kind === 'error' ? 'alert' : undefined}>
			<p>{toast.message}</p>
			<button
				type="button"
				class="button ghost icon"
				aria-label="Dismiss notification"
				onclick={() => toasts.dismiss(toast.id)}>×</button
			>
		</div>
	{/each}
</div>

<style>
	.toast-host {
		position: fixed;
		z-index: 60;
		left: var(--space-2);
		right: var(--space-2);
		bottom: calc(var(--space-2) + env(safe-area-inset-bottom, 0px));
		display: flex;
		flex-direction: column;
		align-items: stretch;
		gap: var(--space-2);
		pointer-events: none;
	}

	@media (min-width: 640px) {
		.toast-host {
			left: auto;
			right: var(--space-4);
			bottom: var(--space-4);
			width: 24rem;
		}
	}

	.toast {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		padding: var(--space-1) var(--space-1) var(--space-1) var(--space-4);
		border: 1px solid var(--color-border);
		border-left-width: 4px;
		border-radius: var(--radius-md);
		background: var(--color-bg);
		box-shadow: var(--shadow-lg);
		pointer-events: auto;
	}

	.toast p {
		flex: 1;
		min-width: 0;
	}

	.toast.error {
		border-left-color: var(--color-danger);
	}

	.toast.info {
		border-left-color: var(--color-info);
	}

	.toast.success {
		border-left-color: var(--color-sage);
	}
</style>
