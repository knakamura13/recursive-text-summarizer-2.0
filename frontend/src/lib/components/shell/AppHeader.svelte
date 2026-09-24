<script lang="ts">
	import { page } from '$app/state';
	import type { Run } from '$lib/api/types';
	import { formatEta, runStateLabel, stageLabel } from '$lib/format';
	import { activity } from '$lib/stores/activity.svelte';
	import { imports } from '$lib/stores/imports.svelte';

	const run = $derived(activity.activeRun);
	const importingCount = $derived(activity.importing.length + imports.uploadingCount);
	const path = $derived(page.url.pathname);

	/** "Summarizing 12/40" for a running Run; the state label otherwise. */
	function stageText(active: Run): string {
		const progress = active.progress;
		if (active.state !== 'running' || !progress?.stage) return runStateLabel(active.state);
		const stage = progress.stages.find((item) => item.stage === progress.stage);
		const counts =
			stage && stage.total !== null && stage.total > 0
				? ` ${stage.completed ?? 0}/${stage.total}`
				: '';
		return `${stageLabel(progress.stage)}${counts}`;
	}

	const runStage = $derived(run ? stageText(run) : '');
	const runEta = $derived(
		run?.state === 'running' && run.progress ? formatEta(run.progress.eta_seconds) : ''
	);
	const runHref = $derived(
		run
			? `/documents/${encodeURIComponent(run.document_id)}?run=${encodeURIComponent(run.run_id)}`
			: '/'
	);
	// Announce state changes only; counts and ETA change too often to read aloud.
	const announcement = $derived(run ? `Run ${runStateLabel(run.state)}: ${run.document_title}` : '');
</script>

<header class="app-header">
	<span class="brand">Recursive Summarizer</span>
	<nav aria-label="Main">
		<a href="/" aria-current={path === '/' ? 'page' : undefined} class:section={path.startsWith('/documents')}
			>Library</a
		>
		<a href="/settings" aria-current={path.startsWith('/settings') ? 'page' : undefined}>Settings</a>
	</nav>
	{#if run || importingCount > 0}
		<div class="activity">
			{#if run}
				<a
					class="run"
					href={runHref}
					aria-label="Active Run: {run.document_title}. {runStage}{runEta ? `, ${runEta}` : ''}"
				>
					<span class="dot" class:stopping={run.state === 'stopping'} aria-hidden="true"></span>
					<span class="stage">{runStage}</span>
					<span class="title">{run.document_title}</span>
					{#if runEta}
						<span class="eta">{runEta}</span>
					{/if}
				</a>
			{/if}
			{#if importingCount > 0}
				<a class="importing" href="/">Importing {importingCount}</a>
			{/if}
		</div>
	{/if}
	<span class="visually-hidden" aria-live="polite">{announcement}</span>
</header>

<style>
	.app-header {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-1) var(--space-4);
		padding: var(--space-1) var(--space-4);
		border-bottom: 1px solid var(--color-border);
		background: var(--color-bg);
	}

	.brand {
		font-weight: 700;
		white-space: nowrap;
	}

	nav {
		display: flex;
		gap: var(--space-1);
	}

	nav a {
		display: inline-flex;
		align-items: center;
		min-height: var(--touch-target);
		padding: 0 var(--space-3);
		border-radius: var(--radius-sm);
		color: var(--color-text-muted);
		font-weight: 500;
		text-decoration: none;
	}

	nav a:hover {
		color: var(--color-text);
		background: var(--color-surface);
	}

	nav a[aria-current='page'],
	nav a.section {
		color: var(--color-sage);
		background: var(--color-sage-soft);
	}

	.activity {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin-left: auto;
		min-width: 0;
	}

	.activity a {
		display: inline-flex;
		align-items: center;
		gap: var(--space-2);
		min-height: 2.25rem;
		padding: var(--space-1) var(--space-3);
		border: 1px solid var(--color-info-border);
		border-radius: 999px;
		background: var(--color-info-soft);
		color: var(--color-info);
		font-size: 0.875rem;
		font-weight: 500;
		text-decoration: none;
		white-space: nowrap;
	}

	.run {
		min-width: 0;
		max-width: 32rem;
	}

	.run .title {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		color: var(--color-text);
	}

	.run .stage,
	.run .eta {
		flex: none;
	}

	.dot {
		flex: none;
		width: 0.5rem;
		height: 0.5rem;
		border-radius: 50%;
		background: currentColor;
		animation: pulse 1.6s ease-in-out infinite;
	}

	.dot.stopping {
		background: var(--color-amber);
	}

	@keyframes pulse {
		50% {
			opacity: 0.3;
		}
	}

	/* Phones: brand hidden, activity on its own full-width row. */
	@media (max-width: 639px) {
		.app-header {
			padding: var(--space-1) var(--space-2);
		}

		.brand {
			display: none;
		}

		.activity {
			flex: 1 0 100%;
			margin-left: 0;
			padding-bottom: var(--space-1);
		}

		.run {
			flex: 1 1 auto;
			max-width: none;
		}
	}
</style>
