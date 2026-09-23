<script lang="ts">
	import ProgressStepper from '$lib/components/shell/ProgressStepper.svelte';
	import type { ProgressStage, WorkspaceTab } from '$lib/stores/workspace';

	let {
		activeTab = $bindable<WorkspaceTab>('document'),
		documentTitle = '',
		sourcePreview = '',
		summaryPreview = '',
		finalSummary = '',
		stages = [],
		runState = null,
		runFailure = null,
		summaryWordCount = null,
		summaryTargetWords = null,
		summaryShortOfTarget = false,
		onSummarize
	}: {
		activeTab?: WorkspaceTab;
		documentTitle?: string;
		sourcePreview?: string;
		summaryPreview?: string;
		finalSummary?: string;
		stages?: ProgressStage[];
		runState?: string | null;
		runFailure?: string | null;
		summaryWordCount?: number | null;
		summaryTargetWords?: number | null;
		summaryShortOfTarget?: boolean;
		onSummarize?: () => void;
	} = $props();

	const tabs: { id: WorkspaceTab; label: string }[] = [
		{ id: 'document', label: 'Document' },
		{ id: 'sections', label: 'Section Summaries' },
		{ id: 'full', label: 'Full Summary' }
	];
</script>

<section class="workspace">
	<header class="workspace-header">
		<div>
			<h2>{documentTitle}</h2>
		</div>
		<button type="button" class="primary" onclick={() => onSummarize?.()}>Summarize</button>
	</header>

	<div class="tablist" role="tablist" aria-label="Workspace views">
		{#each tabs as tab}
			<button
				type="button"
				role="tab"
				aria-selected={activeTab === tab.id}
				class:selected={activeTab === tab.id}
				onclick={() => (activeTab = tab.id)}
			>
				{tab.label}
			</button>
		{/each}
	</div>

	{#if stages.length}
		<ProgressStepper {stages} />
	{/if}
	{#if runState === 'failed'}
		<p class="run-error" role="alert">Run failed. No summary was published.{runFailure ? ` ${runFailure}` : ''}</p>
	{:else if runState === 'cancelled'}
		<p role="status">Run cancelled.</p>
	{:else if runState === 'interrupted'}
		<p role="status">Run interrupted. It can be resumed.</p>
	{:else if runState === 'completed'}
		{#if summaryShortOfTarget && summaryWordCount !== null && summaryTargetWords !== null}
			<p class="run-short" role="status">
				Run completed. Published summary is {summaryWordCount} words, below the {summaryTargetWords}-word
				target.
			</p>
		{:else}
			<p role="status">Run completed.</p>
		{/if}
	{/if}

	{#if activeTab === 'document'}
		<div class="split-panels" role="tabpanel">
			<article class="panel source-panel">
				<h3>Source Section</h3>
				<p>{sourcePreview}</p>
			</article>
			<article class="panel summary-panel">
				<h3>Generated Summary</h3>
				<p>{summaryPreview}</p>
			</article>
		</div>
		{#if finalSummary}
			<aside class="final-summary">
				<h3>Final Summary</h3>
				<p>{finalSummary}</p>
			</aside>
		{/if}
	{:else if activeTab === 'sections'}
		<div class="panel" role="tabpanel">
			<p>{summaryPreview}</p>
		</div>
	{:else}
		<div class="panel" role="tabpanel">
			<p>{finalSummary || 'Final summary is not available yet.'}</p>
		</div>
	{/if}
</section>

<style>
	.workspace {
		display: flex;
		flex-direction: column;
		min-width: 0;
		padding: var(--space-5);
	}

	.workspace-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: var(--space-4);
	}

	.workspace-header h2 {
		margin: 0;
		font-size: 1.25rem;
	}

	.primary {
		background: var(--color-sage);
		color: white;
		border: none;
		border-radius: var(--radius-sm);
		padding: var(--space-2) var(--space-4);
		cursor: pointer;
		font-weight: 600;
	}

	.run-error {
		color: var(--color-error, #a7352f);
	}

	.run-short {
		color: var(--color-amber-text, #8a5a00);
	}

	.tablist {
		display: flex;
		gap: var(--space-2);
		border-bottom: 1px solid var(--color-border);
		margin-bottom: var(--space-4);
	}

	.tablist button {
		border: none;
		background: transparent;
		padding: var(--space-3) var(--space-4);
		cursor: pointer;
		color: var(--color-text-muted);
		border-bottom: 2px solid transparent;
	}

	.tablist button.selected {
		color: var(--color-text);
		border-bottom-color: var(--color-sage);
	}

	.split-panels {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: var(--space-4);
	}

	.panel {
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		padding: var(--space-4);
		background: var(--color-bg);
		min-height: 12rem;
	}

	.panel h3 {
		margin: 0 0 var(--space-3);
		font-size: 0.9rem;
		color: var(--color-text-muted);
	}

	.final-summary {
		margin-top: var(--space-4);
		border: 1px solid var(--color-amber-border);
		background: var(--color-amber-soft);
		border-radius: var(--radius-md);
		padding: var(--space-4);
	}

	@media (max-width: 900px) {
		.split-panels {
			grid-template-columns: 1fr;
		}
	}
</style>
