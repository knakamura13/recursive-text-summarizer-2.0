<script lang="ts">
	import { ApiError, NetworkError, errorMessage } from '$lib/api/client';
	import ProgressBar from '$lib/components/common/ProgressBar.svelte';
	import { formatBytes, formatImportProgress } from '$lib/format';
	import { documents } from '$lib/stores/documents.svelte';
	import { imports, type ImportItem } from '$lib/stores/imports.svelte';

	let { items }: { items: readonly ImportItem[] } = $props();

	function documentHref(id: string): string {
		return `/documents/${encodeURIComponent(id)}`;
	}
</script>

<ul class="queue" aria-label="Imports">
	{#each items as item (item.id)}
		{@const document = item.documentId ? documents.get(item.documentId) : undefined}
		{@const retryable =
			item.error instanceof NetworkError || (item.error instanceof ApiError && item.error.retryable)}
		<li class="item" data-phase={item.phase}>
			<div class="head">
				<span class="name">{item.name}</span>
				<span class="size">{formatBytes(item.size)}</span>
			</div>

			{#if item.phase === 'waiting'}
				<p class="status">Waiting to upload</p>
			{:else if item.phase === 'uploading'}
				<ProgressBar
					value={item.loaded}
					max={item.total}
					label="Uploading {item.name}"
					valueText="{formatBytes(item.loaded)} of {formatBytes(item.total)}"
				/>
				<p class="status">
					Uploading · {item.total > 0 ? Math.floor((item.loaded / item.total) * 100) : 0}%
				</p>
			{:else if item.phase === 'failed'}
				<p class="status error" role="alert">{errorMessage(item.error)}</p>
			{:else if item.duplicate && item.documentId}
				<p class="status notice">
					Already in your library{document ? ` as “${document.title}”` : ''}.
					<a href={documentHref(item.documentId)}>Open it</a>
				</p>
			{:else if document?.import_state === 'importing'}
				<ProgressBar
					value={document.import_progress?.done ?? 0}
					max={document.import_progress?.total ?? null}
					label="Importing {item.name}"
					valueText={formatImportProgress(document.import_progress)}
				/>
				<p class="status">
					{formatImportProgress(document.import_progress)}
					{#if document.import_progress?.message}· {document.import_progress.message}{/if}
				</p>
			{:else if document?.import_state === 'failed'}
				<p class="status error">Import failed: {document.import_error ?? 'no reason given.'}</p>
			{:else if item.documentId}
				<p class="status done">
					{document?.import_state === 'ready' ? 'Imported.' : 'Received.'}
					<a href={documentHref(item.documentId)}>Open</a>
				</p>
			{/if}

			<div class="actions">
				{#if item.phase === 'failed' && retryable}
					<button type="button" class="button small" onclick={() => imports.retry(item.id)}>
						Try again
					</button>
				{/if}
				{#if item.phase === 'waiting' || item.phase === 'uploading'}
					<button
						type="button"
						class="button small"
						aria-label="Cancel importing {item.name}"
						onclick={() => imports.remove(item.id)}>Cancel</button
					>
				{:else}
					<button
						type="button"
						class="button small ghost"
						aria-label="Dismiss {item.name}"
						onclick={() => imports.remove(item.id)}>Dismiss</button
					>
				{/if}
			</div>
		</li>
	{/each}
</ul>

<style>
	.queue {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}

	.item {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		gap: var(--space-2) var(--space-3);
		align-items: center;
		padding: var(--space-3);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		background: var(--color-bg);
	}

	.head {
		display: flex;
		gap: var(--space-2);
		align-items: baseline;
		min-width: 0;
	}

	.name {
		font-weight: 600;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.size {
		flex: none;
		font-size: 0.875rem;
		color: var(--color-text-muted);
	}

	.item > :global(.progress),
	.status {
		grid-column: 1;
	}

	.actions {
		grid-column: 2;
		grid-row: 1 / span 3;
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		gap: var(--space-2);
	}

	.status {
		font-size: 0.9375rem;
		color: var(--color-text-muted);
	}

	.status.error {
		color: var(--color-danger);
	}

	.status.notice {
		color: var(--color-amber-strong);
	}

	.status.done {
		color: var(--color-success);
	}
</style>
