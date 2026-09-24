<script lang="ts">
	import { onMount } from 'svelte';
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import ProgressBar from '$lib/components/common/ProgressBar.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import StateBadge from '$lib/components/common/StateBadge.svelte';
	import { formatBytes, formatName } from '$lib/format';
	import { documents } from '$lib/stores/documents.svelte';
	import ImportDialog from './ImportDialog.svelte';
	import PasteDialog from './PasteDialog.svelte';

	// Library list beside the Document workspace on wide screens.
	let { currentId = null }: { currentId?: string | null } = $props();

	let importOpen = $state(false);
	let pasteOpen = $state(false);

	onMount(() => {
		if (!documents.loaded && !documents.loading) void documents.refresh();
	});
</script>

<aside class="sidebar" aria-label="Library">
	<div class="head">
		<h2><a href="/">Library</a></h2>
		<button type="button" class="button small" onclick={() => (importOpen = true)}>Import</button>
	</div>
	<input
		type="search"
		placeholder="Search Documents"
		aria-label="Search Documents"
		bind:value={documents.search}
	/>

	{#if documents.error && !documents.loaded}
		<ErrorBanner error={documents.error} onretry={() => documents.refresh()} />
	{:else if !documents.loaded}
		<div class="loading"><Spinner label="Loading Documents" /></div>
	{:else if documents.list.length === 0}
		<p class="empty">
			{documents.search.trim() ? 'No Documents match your search.' : 'No Documents yet.'}
		</p>
	{:else}
		<nav aria-label="Documents">
			<ul>
				{#each documents.list as document (document.document_id)}
					<li>
						<a
							href="/documents/{encodeURIComponent(document.document_id)}"
							aria-current={document.document_id === currentId ? 'page' : undefined}
							title={document.title}
						>
							<span class="title">{document.title}</span>
							<span class="meta">
								{formatName(document.format)} · {formatBytes(document.size_bytes)}
							</span>
							{#if document.import_state !== 'ready'}
								<StateBadge state={document.import_state} kind="import" size="sm" />
							{:else if document.latest_run}
								<StateBadge state={document.latest_run.state} size="sm" />
							{/if}
							{#if document.import_state === 'importing'}
								<ProgressBar
									size="sm"
									value={document.import_progress?.done ?? 0}
									max={document.import_progress?.total ?? null}
									label="Import progress of {document.title}"
								/>
							{/if}
						</a>
					</li>
				{/each}
			</ul>
		</nav>
	{/if}
</aside>

<ImportDialog bind:open={importOpen} onpaste={() => (pasteOpen = true)} />
<PasteDialog bind:open={pasteOpen} />

<style>
	.sidebar {
		display: flex;
		flex-direction: column;
		gap: var(--space-3);
		min-height: 0;
		padding: var(--space-4) var(--space-3);
		border-right: 1px solid var(--color-border);
		background: var(--color-surface);
		overflow: hidden;
	}

	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-2);
		padding: 0 var(--space-1);
	}

	.head a {
		color: inherit;
		text-decoration: none;
	}

	nav {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		margin: 0 calc(-1 * var(--space-1));
		padding: 0 var(--space-1);
	}

	ul {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
	}

	li a {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--space-1);
		padding: var(--space-2) var(--space-3);
		border: 1px solid transparent;
		border-radius: var(--radius-md);
		color: inherit;
		text-decoration: none;
	}

	li a:hover {
		background: var(--color-bg);
	}

	li a[aria-current='page'] {
		background: var(--color-sage-soft);
		border-color: var(--color-sage-border);
	}

	.title {
		max-width: 100%;
		font-weight: 600;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.meta {
		font-size: 0.8125rem;
		color: var(--color-text-muted);
	}

	.loading,
	.empty {
		padding: var(--space-4) var(--space-1);
		color: var(--color-text-muted);
	}
</style>
