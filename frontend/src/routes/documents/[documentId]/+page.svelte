<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { api, type DocumentSummary, type RunConfig } from '$lib/api/client';
	import DocumentsSidebar from '$lib/components/shell/DocumentsSidebar.svelte';
	import Drawer from '$lib/components/common/Drawer.svelte';
	import HierarchyPanel from '$lib/components/shell/HierarchyPanel.svelte';
	import ImportDialog from '$lib/components/shell/ImportDialog.svelte';
	import WorkspaceTabs from '$lib/components/shell/WorkspaceTabs.svelte';
	import type { HierarchyNode, ProgressStage, WorkspaceTab } from '$lib/stores/workspace';

	let documents = $state<DocumentSummary[]>([]);
	let document = $state<DocumentSummary | null>(null);
	let search = $state('');
	let importOpen = $state(false);
	let activeTab = $state<WorkspaceTab>('document');
	let selectedNodeId = $state<string | null>(null);
	let hierarchy = $state<HierarchyNode[]>([]);
	let stages = $state<ProgressStage[]>([]);
	let sourcePreview = $state('');
	let summaryPreview = $state('');
	let finalSummary = $state('');
	let runId = $state<string | null>(null);
	let runState = $state<string | null>(null);
	let runFailure = $state<string | null>(null);
	let showHierarchyDrawer = $state(false);
	let showDocumentsDrawer = $state(false);
	let summarizeOpen = $state(false);
	let runConfig = $state<RunConfig>({ model: '', target_words: 300, strategy: 'auto', verify: true });
	let preflightWarnings = $state<string[]>([]);
	let error = $state<string | null>(null);

	const documentId = $derived($page.params.documentId ?? '');

	async function loadAll() {
		const [list, detail, settings] = await Promise.all([
			api.listDocuments(search),
			api.getDocument(documentId),
			api.getSettings()
		]);
		documents = list.documents;
		document = detail;
		runConfig = { ...runConfig, ...settings.defaults };
		localStorage.setItem('lastDocumentId', documentId);
		const source = await api.getSourceSlice(documentId, 0, 1200);
		sourcePreview = source.text;
		if (detail.latest_run_id) {
			runId = detail.latest_run_id;
			await refreshRun();
		}
	}

	async function refreshRun() {
		if (!runId) return;
		const [tree, summary, run] = await Promise.all([
			api.getRunTree(runId),
			api.getRunSummary(runId),
			api.getRun(runId)
		]);
		hierarchy = tree.nodes;
		finalSummary = summary.available ? summary.text ?? '' : '';
		summaryPreview =
			hierarchy.find((node) => node.node_id === selectedNodeId)?.label ?? 'Select a node to inspect.';
		stages = mapStages(run.state);
		runState = run.state;
		runFailure = run.failure_reason ?? null;
	}

	async function pollRun() {
		if (!runId || !['queued', 'running', 'cancelling'].includes(runState ?? '')) return;
		const currentRunId = runId;
		const run = await api.getRun(currentRunId);
		if (runId !== currentRunId) return;
		runState = run.state;
		runFailure = run.failure_reason ?? null;
		stages = mapStages(run.state);
		if (!['queued', 'running', 'cancelling'].includes(run.state)) await refreshRun();
	}

	function mapStages(state: string): ProgressStage[] {
		if (state === 'failed' || state === 'cancelled' || state === 'interrupted') return [];
		const base = [
			{ stage: 'Preparing', state: 'completed' as const },
			{ stage: 'Segmenting', state: 'completed' as const },
			{ stage: 'Summarizing', state: 'active' as const },
			{ stage: 'Merging', state: 'pending' as const },
			{ stage: 'Writing', state: 'pending' as const },
			{ stage: 'Verifying', state: 'pending' as const },
			{ stage: 'Publishing', state: 'pending' as const }
		];
		if (state === 'completed') {
			return base.map((stage) => ({ ...stage, state: 'completed' as const }));
		}
		return base;
	}

	onMount(() => {
		loadAll().catch((loadError) => {
			error = loadError instanceof Error ? loadError.message : 'Failed to load workspace';
		});
		let polling = false;
		const interval = window.setInterval(async () => {
			if (polling) return;
			polling = true;
			try {
				await pollRun();
			} catch (pollError) {
				error = pollError instanceof Error ? pollError.message : 'Failed to refresh run';
			} finally {
				polling = false;
			}
		}, 2000);
		return () => window.clearInterval(interval);
	});

	async function handleUpload(files: FileList) {
		for (const file of Array.from(files)) {
			await api.uploadDocument(file);
		}
		await loadAll();
	}

	async function startSummarize() {
		const preflight = await api.preflight(documentId, runConfig);
		preflightWarnings = preflight.warnings ?? [];
		if (!preflight.fits) return;
		const created = await api.createRun(documentId, runConfig, crypto.randomUUID());
		runId = created.run_id;
		await refreshRun();
		summarizeOpen = false;
	}
</script>

<div class="workspace-layout">
	<div class="desktop-sidebar">
		<DocumentsSidebar
			{documents}
			selectedId={documentId}
			bind:search
			onImport={() => (importOpen = true)}
			onSelect={(id) => goto(`/documents/${id}`)}
		/>
	</div>

	{#if document}
		<WorkspaceTabs
			bind:activeTab
			documentTitle={document.title}
			{sourcePreview}
			{summaryPreview}
			{finalSummary}
			{stages}
			{runState}
			{runFailure}
			onSummarize={() => (summarizeOpen = true)}
		/>
	{:else if error}
		<p class="error">{error}</p>
	{/if}

	<div class="desktop-hierarchy">
		<HierarchyPanel
			nodes={hierarchy}
			selectedId={selectedNodeId}
			onSelect={(nodeId) => {
				selectedNodeId = nodeId;
			}}
		/>
	</div>

	<div class="mobile-controls">
		<button type="button" onclick={() => (showDocumentsDrawer = true)}>Documents</button>
		<button type="button" onclick={() => (showHierarchyDrawer = true)}>Hierarchy</button>
	</div>
</div>

<ImportDialog bind:open={importOpen} onUpload={handleUpload} />

<Drawer bind:open={showDocumentsDrawer} title="Documents" side="left">
	<DocumentsSidebar
		{documents}
		selectedId={documentId}
		bind:search
		onImport={() => (importOpen = true)}
		onSelect={(id) => {
			showDocumentsDrawer = false;
			goto(`/documents/${id}`);
		}}
	/>
</Drawer>

<Drawer bind:open={showHierarchyDrawer} title="Document Hierarchy" side="right">
	<HierarchyPanel
		nodes={hierarchy}
		selectedId={selectedNodeId}
		onSelect={(nodeId) => {
			selectedNodeId = nodeId;
			showHierarchyDrawer = false;
		}}
	/>
</Drawer>

{#if summarizeOpen}
	<div class="config-panel" role="dialog" aria-label="Summarize configuration">
		<h3>Summarize</h3>
		<label>
			Model
			<input bind:value={runConfig.model} required />
		</label>
		<label>
			Target words
			<input type="number" bind:value={runConfig.target_words} min="50" />
		</label>
		<label>
			Strategy
			<select bind:value={runConfig.strategy}>
				<option value="auto">Auto</option>
				<option value="direct">Direct</option>
				<option value="hierarchical">Hierarchical</option>
			</select>
		</label>
		<label>
			Context window (tokens)
			<input type="number" min="1" bind:value={runConfig.context_window} placeholder="Model default" />
		</label>
		<label>
			Output limit (tokens)
			<input type="number" min="256" step="256" bind:value={runConfig.max_output_tokens} />
		</label>
		<label>
			<input type="checkbox" bind:checked={runConfig.verify} />
			Claim verification
		</label>
		{#each preflightWarnings as warning}
			<p class="warning">{warning}</p>
		{/each}
		<div class="actions">
			<button type="button" onclick={() => (summarizeOpen = false)}>Cancel</button>
			<button type="button" class="primary" onclick={startSummarize}>Start</button>
		</div>
	</div>
{/if}

<style>
	.workspace-layout {
		display: grid;
		grid-template-columns: var(--sidebar-left-width) minmax(0, 1fr) var(--sidebar-right-width);
		min-height: calc(100vh - 4.5rem);
	}

	.mobile-controls {
		display: none;
	}

	.config-panel {
		position: fixed;
		right: var(--space-6);
		top: 5rem;
		width: min(360px, calc(100vw - 2rem));
		background: var(--color-bg);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-lg);
		padding: var(--space-5);
		box-shadow: var(--shadow-md);
		z-index: 20;
		display: flex;
		flex-direction: column;
		gap: var(--space-3);
	}

	.config-panel label {
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		font-size: 0.875rem;
	}

	.actions {
		display: flex;
		justify-content: flex-end;
		gap: var(--space-2);
	}

	.primary {
		background: var(--color-sage);
		color: white;
		border: none;
		border-radius: var(--radius-sm);
		padding: var(--space-2) var(--space-4);
	}

	.warning {
		color: var(--color-amber);
		font-size: 0.875rem;
	}

	.error {
		padding: var(--space-6);
		color: var(--color-danger);
	}

	@media (max-width: 1200px) {
		.desktop-hierarchy {
			display: none;
		}

		.workspace-layout {
			grid-template-columns: var(--sidebar-left-width) minmax(0, 1fr);
		}
	}

	@media (max-width: 900px) {
		.desktop-sidebar {
			display: none;
		}

		.workspace-layout {
			grid-template-columns: 1fr;
		}

		.mobile-controls {
			display: flex;
			gap: var(--space-2);
			padding: var(--space-3);
		}
	}
</style>
