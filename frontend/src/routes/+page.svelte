<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { api, type DocumentSummary } from '$lib/api/client';
	import DocumentsSidebar from '$lib/components/shell/DocumentsSidebar.svelte';
	import ImportDialog from '$lib/components/shell/ImportDialog.svelte';

	let documents = $state<DocumentSummary[]>([]);
	let search = $state('');
	let importOpen = $state(false);
	let error = $state<string | null>(null);

	async function loadDocuments() {
		const response = await api.listDocuments(search);
		documents = response.documents;
	}

	onMount(async () => {
		try {
			await loadDocuments();
			const lastId = localStorage.getItem('lastDocumentId');
			if (lastId && documents.some((doc) => doc.document_id === lastId)) {
				await goto(`/documents/${lastId}`);
			}
		} catch (loadError) {
			error = loadError instanceof Error ? loadError.message : 'Failed to load documents';
		}
	});

	async function handleUpload(files: FileList) {
		for (const file of Array.from(files)) {
			await api.uploadDocument(file);
		}
		await loadDocuments();
	}
</script>

<div class="workspace-layout single-sidebar">
	<DocumentsSidebar
		{documents}
		bind:search
		onImport={() => (importOpen = true)}
		onSelect={(id) => goto(`/documents/${id}`)}
	/>
	<section class="empty-state" aria-live="polite">
		{#if error}
			<p class="error">{error}</p>
		{:else}
			<h2>Import a document to begin</h2>
			<p>Use the plus button to import text, Markdown, or PDF files.</p>
		{/if}
	</section>
</div>

<ImportDialog bind:open={importOpen} onUpload={handleUpload} />

<style>
	.workspace-layout {
		display: grid;
		grid-template-columns: var(--sidebar-left-width) 1fr;
		min-height: calc(100vh - 4.5rem);
	}

	.empty-state {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		padding: var(--space-8);
		color: var(--color-text-muted);
	}

	.error {
		color: var(--color-danger);
	}
</style>
