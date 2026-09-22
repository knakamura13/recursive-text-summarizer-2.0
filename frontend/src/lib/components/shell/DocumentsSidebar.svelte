<script lang="ts">
	import type { DocumentSummary } from '$lib/api/client';

	let {
		documents = [],
		selectedId = null,
		search = $bindable(''),
		onImport,
		onSelect,
		onRename,
		onDelete
	}: {
		documents?: DocumentSummary[];
		selectedId?: string | null;
		search?: string;
		onImport?: () => void;
		onSelect?: (id: string) => void;
		onRename?: (id: string, title: string) => void;
		onDelete?: (id: string) => void;
	} = $props();

	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}
</script>

<aside class="documents-sidebar" aria-label="Documents">
	<div class="header-row">
		<h2>Documents</h2>
		<button type="button" class="import-button" aria-label="Import documents" onclick={() => onImport?.()}>
			+
		</button>
	</div>
	<label class="search-label">
		<span class="visually-hidden">Search documents</span>
		<input type="search" placeholder="Search documents..." bind:value={search} />
	</label>
	<ul class="document-list" aria-label="Document list">
		{#each documents as document (document.document_id)}
			<li>
				<button
					type="button"
					class:selected={document.document_id === selectedId}
					aria-current={document.document_id === selectedId ? 'true' : undefined}
					onclick={() => onSelect?.(document.document_id)}
				>
					<span class="title">{document.title}</span>
					<span class="meta">
						{document.format.toUpperCase()} · {formatSize(document.size_bytes)}
						{#if document.latest_run_state}
							· {document.latest_run_state}
						{/if}
					</span>
				</button>
			</li>
		{/each}
	</ul>
</aside>

<style>
	.documents-sidebar {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
		padding: var(--space-5);
		border-right: 1px solid var(--color-border);
		background: var(--color-surface);
		min-width: var(--sidebar-left-width);
	}

	.header-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.header-row h2 {
		margin: 0;
		font-size: 0.95rem;
		font-weight: 600;
	}

	.import-button {
		width: 2rem;
		height: 2rem;
		border-radius: 999px;
		border: 1px solid var(--color-border-strong);
		background: var(--color-bg);
		font-size: 1.25rem;
		line-height: 1;
		cursor: pointer;
	}

	.search-label input {
		width: 100%;
		padding: var(--space-2) var(--space-3);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
		background: var(--color-bg);
	}

	.document-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
		overflow: auto;
	}

	.document-list button {
		width: 100%;
		text-align: left;
		border: 1px solid transparent;
		background: transparent;
		border-radius: var(--radius-md);
		padding: var(--space-3);
		cursor: pointer;
	}

	.document-list button.selected {
		background: var(--color-sage-soft);
		border-color: var(--color-sage-border);
	}

	.title {
		display: block;
		font-weight: 600;
		color: var(--color-text);
	}

	.meta {
		display: block;
		margin-top: var(--space-1);
		font-size: 0.8rem;
		color: var(--color-text-muted);
	}

	.visually-hidden {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}
</style>
