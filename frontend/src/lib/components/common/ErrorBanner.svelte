<script lang="ts">
	import { ApiError, NetworkError, errorMessage } from '$lib/api/client';

	let {
		error,
		title,
		onretry,
		ondismiss
	}: {
		error: unknown;
		title?: string;
		onretry?: () => void;
		ondismiss?: () => void;
	} = $props();

	const message = $derived(errorMessage(error));

	// 409 run_active names the Run that blocks the request; link straight to it.
	const activeRun = $derived.by(() => {
		if (!(error instanceof ApiError) || error.code !== 'run_active' || !error.details) return null;
		const { document_id, run_id, document_title } = error.details;
		if (typeof document_id !== 'string' || typeof run_id !== 'string') return null;
		return {
			href: `/documents/${encodeURIComponent(document_id)}?run=${encodeURIComponent(run_id)}`,
			title: typeof document_title === 'string' ? document_title : null
		};
	});

	const heading = $derived(
		title ?? (error instanceof NetworkError ? 'Connection problem' : undefined)
	);
</script>

<div class="error-banner" role="alert">
	<div class="text">
		{#if heading}
			<strong>{heading}</strong>
		{/if}
		<p>{message}</p>
		{#if activeRun}
			<a href={activeRun.href}>
				Open the active Run{activeRun.title ? ` on “${activeRun.title}”` : ''}
			</a>
		{/if}
	</div>
	{#if onretry || ondismiss}
		<div class="actions">
			{#if onretry}
				<button type="button" class="button small" onclick={onretry}>Try again</button>
			{/if}
			{#if ondismiss}
				<button type="button" class="button ghost icon" aria-label="Dismiss" onclick={ondismiss}
					>×</button
				>
			{/if}
		</div>
	{/if}
</div>

<style>
	.error-banner {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		justify-content: space-between;
		gap: var(--space-3);
		padding: var(--space-3) var(--space-4);
		border: 1px solid var(--color-danger-border);
		border-radius: var(--radius-md);
		background: var(--color-danger-soft);
		color: var(--color-text);
	}

	.text {
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		min-width: 0;
		flex: 1 1 14rem;
	}

	strong {
		color: var(--color-danger);
	}

	.actions {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}
</style>
