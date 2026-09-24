<script lang="ts">
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import ProgressBar from '$lib/components/common/ProgressBar.svelte';
	import StateBadge from '$lib/components/common/StateBadge.svelte';
	import { api } from '$lib/api/client';
	import type { DocumentDetail } from '$lib/api/types';
	import { formatBytes, formatCount, importPhaseLabel } from '$lib/format';
	import { documents } from '$lib/stores/documents.svelte';

	interface Props {
		doc: DocumentDetail;
		/** Called after the Document was deleted (which also stops an import in progress). */
		ondeleted: () => void;
	}

	let { doc, ondeleted }: Props = $props();

	let confirming = $state(false);

	const progress = $derived(doc.import_progress);
	const unitLabel = $derived(
		progress?.unit === 'bytes'
			? `${formatBytes(progress.done)}${progress.total !== null ? ` of ${formatBytes(progress.total)}` : ''}`
			: progress && progress.total !== null
				? `${formatCount(progress.done)} of ${formatCount(progress.total)} ${progress.unit ?? 'steps'}`
				: null
	);

	async function remove() {
		await api.deleteDocument(doc.document_id);
		documents.remove(doc.document_id);
		ondeleted();
	}
</script>

<section class="import-panel" aria-labelledby="import-panel-title">
	<div class="head">
		<h2 id="import-panel-title">
			{doc.import_state === 'importing' ? 'Importing this Document' : 'The Import failed'}
		</h2>
		<StateBadge state={doc.import_state} kind="import" />
	</div>

	{#if doc.import_state === 'importing'}
		<div aria-live="polite">
			<p class="phase">{progress ? importPhaseLabel(progress.phase) : 'Waiting to import'}</p>
			{#if unitLabel}<p class="detail">{unitLabel}</p>{/if}
			{#if progress?.message}<p class="detail">{progress.message}</p>{/if}
		</div>
		<ProgressBar
			value={progress?.done ?? 0}
			max={progress && progress.total ? progress.total : null}
			label="Import progress"
			valueText={unitLabel ?? undefined}
		/>
		<p class="detail">
			You can leave this page; the Import keeps running. The Document opens here when its text is ready.
		</p>
		<div class="actions">
			<button type="button" class="button" onclick={() => (confirming = true)}>Cancel import</button>
		</div>
	{:else}
		<p class="error" role="alert">{doc.import_error ?? 'The file could not be imported.'}</p>
		<div class="actions">
			<button type="button" class="button danger" onclick={() => (confirming = true)}>Delete Document</button>
		</div>
	{/if}
</section>

<ConfirmDialog
	bind:open={confirming}
	title={doc.import_state === 'importing' ? 'Cancel this import?' : 'Delete this Document?'}
	message={doc.import_state === 'importing'
		? `The Import of “${doc.title}” stops and the uploaded file is deleted.`
		: `“${doc.title}” and its uploaded file are deleted.`}
	confirmLabel={doc.import_state === 'importing' ? 'Cancel import' : 'Delete'}
	cancelLabel={doc.import_state === 'importing' ? 'Keep importing' : 'Keep'}
	danger
	onconfirm={remove}
/>

<style>
	.import-panel {
		display: flex;
		flex-direction: column;
		gap: var(--space-3);
		max-width: 40rem;
		padding: var(--space-4);
	}

	.head {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-2);
	}

	h2 {
		margin: 0;
		font-size: 1.1rem;
	}

	.phase {
		margin: 0;
		font-weight: 600;
	}

	.detail {
		margin: 0;
		color: var(--color-text-muted);
		font-size: 0.9rem;
	}

	.error {
		margin: 0;
		color: var(--color-danger);
		overflow-wrap: anywhere;
	}

	.actions {
		display: flex;
		gap: var(--space-2);
	}
</style>
