<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$lib/api/client';
	import type { DocumentSummary } from '$lib/api/types';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import EmptyState from '$lib/components/common/EmptyState.svelte';
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import ProgressBar from '$lib/components/common/ProgressBar.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import StateBadge from '$lib/components/common/StateBadge.svelte';
	import ImportDialog from '$lib/components/shell/ImportDialog.svelte';
	import ImportQueue from '$lib/components/shell/ImportQueue.svelte';
	import PasteDialog from '$lib/components/shell/PasteDialog.svelte';
	import RenameDialog from '$lib/components/shell/RenameDialog.svelte';
	import {
		formatBytes,
		formatCount,
		formatEta,
		formatImportProgress,
		formatName,
		formatRelativeTime,
		stageLabel
	} from '$lib/format';
	import { activity } from '$lib/stores/activity.svelte';
	import { documents } from '$lib/stores/documents.svelte';
	import { imports } from '$lib/stores/imports.svelte';
	import { toasts } from '$lib/stores/toasts.svelte';

	let importDialog: ImportDialog | undefined = $state();
	let importOpen = $state(false);
	let pasteOpen = $state(false);
	let renameOpen = $state(false);
	let renameTarget = $state.raw<DocumentSummary | null>(null);
	let deleteOpen = $state(false);
	let deleteTarget = $state.raw<DocumentSummary | null>(null);
	let dragDepth = $state(0);
	let now = $state(Date.now());

	const uploads = $derived(
		imports.items.filter((item) => item.phase === 'waiting' || item.phase === 'uploading')
	);
	const searching = $derived(documents.search.trim() !== '');
	const activeRun = $derived(activity.activeRun);
	const activeEta = $derived(
		activeRun?.state === 'running' && activeRun.progress?.eta_seconds != null
			? formatEta(activeRun.progress.eta_seconds)
			: ''
	);

	onMount(() => {
		void documents.refresh();
		const clock = setInterval(() => (now = Date.now()), 60_000);
		return () => clearInterval(clock);
	});

	function carriesFiles(event: DragEvent): boolean {
		return event.dataTransfer?.types.includes('Files') ?? false;
	}

	function onDragEnter(event: DragEvent) {
		if (!carriesFiles(event)) return;
		event.preventDefault();
		dragDepth += 1;
	}

	function onDragOver(event: DragEvent) {
		if (!carriesFiles(event)) return;
		event.preventDefault();
		if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
	}

	function onDrop(event: DragEvent) {
		if (!carriesFiles(event)) return;
		event.preventDefault();
		dragDepth = 0;
		const files = Array.from(event.dataTransfer?.files ?? []);
		if (files.length === 0) return;
		importOpen = true;
		importDialog?.addFiles(files);
	}

	function askRename(document: DocumentSummary) {
		renameTarget = document;
		renameOpen = true;
	}

	function askDelete(document: DocumentSummary) {
		deleteTarget = document;
		deleteOpen = true;
	}

	async function confirmDelete() {
		const target = deleteTarget;
		if (!target) return;
		await api.deleteDocument(target.document_id);
		documents.remove(target.document_id);
		toasts.success(`Deleted “${target.title}”.`);
	}
</script>

<svelte:head>
	<title>Library · Recursive Summarizer</title>
</svelte:head>

<div
	class="library"
	role="region"
	aria-label="Library"
	ondragenter={onDragEnter}
	ondragover={onDragOver}
	ondragleave={(event) => {
		if (carriesFiles(event)) dragDepth = Math.max(0, dragDepth - 1);
	}}
	ondrop={onDrop}
>
	<header class="page-head">
		<div class="heading">
			<h1>Library</h1>
			{#if documents.loaded}
				<p class="count">
					{searching
						? `${formatCount(documents.list.length)} matching`
						: `${formatCount(documents.list.length)} ${documents.list.length === 1 ? 'Document' : 'Documents'}`}
				</p>
			{/if}
		</div>
		<div class="actions">
			<button type="button" class="button primary" onclick={() => (importOpen = true)}>
				Import files
			</button>
			<button type="button" class="button" onclick={() => (pasteOpen = true)}>Paste text</button>
		</div>
	</header>

	<div class="search">
		<input
			type="search"
			placeholder="Search by title or file name"
			aria-label="Search Documents"
			bind:value={documents.search}
		/>
		{#if documents.loading && documents.loaded}
			<Spinner size="sm" label="Searching" />
		{/if}
	</div>

	{#if uploads.length > 0}
		<section class="uploads" aria-label="Uploads in progress">
			<ImportQueue items={uploads} />
		</section>
	{/if}

	{#if documents.error && !documents.loaded}
		<ErrorBanner
			title="The library could not be loaded"
			error={documents.error}
			onretry={() => documents.refresh()}
		/>
	{:else if !documents.loaded}
		<div class="loading"><Spinner size="lg" label="Loading the library" /></div>
	{:else}
		{#if documents.error}
			<ErrorBanner error={documents.error} onretry={() => documents.refresh()} />
		{/if}
		{#if documents.list.length === 0}
			{#if searching}
				<EmptyState
					title="No Documents match “{documents.search.trim()}”"
					message="Search looks at titles and file names."
				>
					<button type="button" class="button" onclick={() => (documents.search = '')}>
						Clear search
					</button>
				</EmptyState>
			{:else}
				<EmptyState
					title="Your library is empty"
					message="Import a file or paste text to summarize it. You can also drop files anywhere on this page."
				>
					<button type="button" class="button primary" onclick={() => (importOpen = true)}>
						Import files
					</button>
					<button type="button" class="button" onclick={() => (pasteOpen = true)}>
						Paste text
					</button>
				</EmptyState>
			{/if}
		{:else}
			<ul class="documents" aria-label="Documents">
				{#each documents.list as document (document.document_id)}
					<li class="row" data-import-state={document.import_state}>
						<div class="summary">
							<a class="title" href="/documents/{encodeURIComponent(document.document_id)}">
								{document.title}
							</a>
							<p class="meta">
								{#if document.origin === 'paste'}
									<span>Pasted text</span>
								{:else if document.filename && document.filename.replace(/\.[^.]+$/, '') !== document.title}
									<span class="filename">{document.filename}</span>
								{/if}
								<span>{formatName(document.format)}</span>
								<span>{formatBytes(document.size_bytes)}</span>
								{#if document.page_count}
									<span>{`${formatCount(document.page_count)} ${document.page_count === 1 ? 'page' : 'pages'}`}</span>
								{/if}
								{#if document.char_count}
									<span>{formatCount(document.char_count)} characters</span>
								{/if}
							</p>
						</div>

						<div class="state">
							{#if document.import_state === 'importing'}
								<div class="import">
									<p class="status">
										<StateBadge state="importing" size="sm" />
										<span>{formatImportProgress(document.import_progress)}</span>
									</p>
									<ProgressBar
										value={document.import_progress?.done ?? 0}
										max={document.import_progress?.total ?? null}
										size="sm"
										label="Import progress of {document.title}"
										valueText={formatImportProgress(document.import_progress)}
									/>
								</div>
							{:else if document.import_state === 'failed'}
								<p class="status failed">
									<StateBadge state="failed" kind="import" size="sm" />
									<span>{document.import_error ?? 'The import failed.'}</span>
								</p>
							{:else if document.latest_run}
								<p class="status">
									<span class="visually-hidden">Latest Run:</span>
									<StateBadge state={document.latest_run.state} size="sm" />
									{#if activeRun?.run_id === document.latest_run.run_id && activeRun.progress?.stage}
										<span class="muted">{stageLabel(activeRun.progress.stage)}{activeEta ? ` · ${activeEta}` : ''}</span>
									{:else}
										<span class="muted">{formatRelativeTime(document.latest_run.updated_at, now)}</span>
									{/if}
								</p>
							{:else}
								<p class="status muted">Not summarized yet</p>
							{/if}
						</div>

						<div class="row-actions">
							<button
								type="button"
								class="button small ghost"
								aria-label="Rename “{document.title}”"
								onclick={() => askRename(document)}>Rename</button
							>
							<button
								type="button"
								class="button small ghost delete"
								aria-label="Delete “{document.title}”"
								onclick={() => askDelete(document)}>Delete</button
							>
						</div>
					</li>
				{/each}
			</ul>
		{/if}
	{/if}

	{#if dragDepth > 0}
		<div class="drop-overlay" aria-hidden="true"><p>Drop files to import them</p></div>
	{/if}
</div>

<ImportDialog bind:this={importDialog} bind:open={importOpen} onpaste={() => (pasteOpen = true)} />
<PasteDialog bind:open={pasteOpen} />
<RenameDialog bind:open={renameOpen} document={renameTarget} />
<ConfirmDialog
	bind:open={deleteOpen}
	title="Delete “{deleteTarget?.title ?? ''}”?"
	message={deleteTarget?.import_state === 'importing'
		? 'The import stops and the Document is removed from the library.'
		: 'This removes the Document with all its Runs and summaries. It cannot be undone.'}
	confirmLabel="Delete"
	danger
	onconfirm={confirmDelete}
/>

<style>
	.library {
		position: relative;
		display: flex;
		flex-direction: column;
		gap: var(--space-5);
		max-width: 68rem;
		min-height: 100%;
		margin: 0 auto;
		padding: var(--space-8) var(--space-6) var(--space-8);
	}

	.page-head {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		justify-content: space-between;
		gap: var(--space-3);
	}

	.page-head h1 {
		font-size: 2.25rem;
		line-height: 1.1;
	}

	.count {
		margin-top: var(--space-1);
		color: var(--color-text-muted);
		font-size: 0.9375rem;
	}

	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-2);
	}

	.search {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}

	.search input {
		max-width: 28rem;
	}

	.loading {
		display: flex;
		justify-content: center;
		padding: var(--space-8);
		color: var(--color-text-muted);
	}

	.documents {
		list-style: none;
		margin: 0;
		padding: 0;
		border-top: 1px solid var(--color-text);
	}

	.row {
		position: relative;
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(10rem, 14rem) auto;
		gap: var(--space-2) var(--space-5);
		align-items: center;
		padding: var(--space-4) var(--space-2);
		border-bottom: 1px solid var(--color-border);
		transition: background-color 120ms ease;
	}

	.row:hover {
		background: var(--color-surface-elevated);
	}

	.summary {
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		min-width: 0;
	}

	.title {
		align-self: flex-start;
		max-width: 100%;
		font-family: var(--font-serif);
		font-weight: 600;
		font-size: 1.1875rem;
		line-height: 1.3;
		color: var(--color-text);
		text-decoration: none;
		overflow-wrap: anywhere;
	}

	/* The whole row opens the Document; the actions sit above this layer. */
	.title::after {
		content: '';
		position: absolute;
		inset: 0;
	}

	.title:focus-visible {
		outline: none;
	}

	.title:focus-visible::after {
		outline: 2px solid var(--color-focus);
		outline-offset: -2px;
	}

	.row:hover .title {
		text-decoration: underline;
		text-decoration-color: var(--color-accent);
		text-decoration-thickness: 1px;
		text-underline-offset: 0.2em;
	}

	.meta {
		display: flex;
		flex-wrap: wrap;
		gap: 0 var(--space-2);
		font-size: 0.8125rem;
		color: var(--color-text-muted);
	}

	.meta span + span::before {
		content: '·';
		margin-right: var(--space-2);
		color: var(--color-text-subtle);
	}

	.filename {
		overflow-wrap: anywhere;
	}

	.state {
		min-width: 0;
	}

	.import {
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}

	.status {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-2);
		font-size: 0.875rem;
	}

	.status.failed {
		color: var(--color-danger);
	}

	.muted {
		color: var(--color-text-muted);
	}

	.row-actions {
		position: relative;
		z-index: 1;
		display: flex;
		gap: var(--space-1);
		opacity: 0.7;
		transition: opacity 120ms ease;
	}

	.row:hover .row-actions,
	.row:focus-within .row-actions {
		opacity: 1;
	}

	.row-actions .button {
		color: var(--color-text-muted);
	}

	.row-actions .button:hover {
		color: var(--color-text);
	}

	.row-actions .delete:hover {
		color: var(--color-danger);
		background: var(--color-danger-soft);
	}

	.drop-overlay {
		position: fixed;
		inset: 0;
		z-index: 30;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(245, 242, 235, 0.92);
		border: 3px dashed var(--color-accent);
		pointer-events: none;
		font-family: var(--font-serif);
		font-size: 1.5rem;
		font-weight: 600;
		color: var(--color-text);
	}

	@media (max-width: 767px) {
		.library {
			padding: var(--space-5) var(--space-3) var(--space-8);
			gap: var(--space-4);
		}

		.page-head h1 {
			font-size: 1.75rem;
		}

		.search input {
			max-width: none;
		}

		.row {
			grid-template-columns: minmax(0, 1fr) auto;
			padding: var(--space-3) var(--space-1);
		}

		.state {
			grid-column: 1;
			grid-row: 2;
		}

		.row-actions {
			grid-column: 2;
			grid-row: 1 / span 2;
			flex-direction: column;
			opacity: 1;
		}
	}
</style>
