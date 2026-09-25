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
	<a class="brand" href="/" aria-label="Recursive Summarizer, Library"><span class="mark" aria-hidden="true">¶</span><span><span class="brand-text">{'Recursive '}</span>Summarizer</span></a>
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
		gap: var(--space-1) var(--space-6);
		padding: 0 var(--space-5);
		border-bottom: 1px solid var(--color-border);
		background: var(--color-bg);
	}

	.brand {
		display: inline-flex;
		align-items: baseline;
		gap: 0.4rem;
		min-height: 3.25rem;
		align-self: center;
		padding-top: 0.85rem;
		font-family: var(--font-serif);
		font-size: 1.15rem;
		font-weight: 600;
		letter-spacing: -0.01em;
		white-space: nowrap;
		color: var(--color-text);
		text-decoration: none;
	}

	.mark {
		color: var(--color-accent);
		font-weight: 700;
	}

	nav {
		display: flex;
		align-self: stretch;
		gap: var(--space-4);
	}

	nav a {
		display: inline-flex;
		align-items: center;
		min-height: var(--touch-target);
		padding: 0 var(--space-1);
		border-bottom: 2px solid transparent;
		margin-bottom: -1px;
		color: var(--color-text-muted);
		font-size: 0.9375rem;
		font-weight: 500;
		text-decoration: none;
	}

	nav a:hover {
		color: var(--color-text);
	}

	nav a[aria-current='page'],
	nav a.section {
		color: var(--color-text);
		border-bottom-color: var(--color-accent);
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
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		background: var(--color-surface-elevated);
		color: var(--color-text-muted);
		font-size: 0.8125rem;
		font-weight: 500;
		text-decoration: none;
		white-space: nowrap;
		font-variant-numeric: tabular-nums;
	}

	.activity a:hover {
		border-color: var(--color-border-strong);
		color: var(--color-text);
	}

	.run {
		min-width: 0;
		max-width: 34rem;
	}

	.run .stage {
		color: var(--color-text);
		font-weight: 600;
	}

	.run .title {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		font-family: var(--font-serif);
		font-size: 0.875rem;
		font-weight: 400;
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
		background: var(--color-amber);
		animation: pulse 1.6s ease-in-out infinite;
	}

	.dot.stopping {
		background: var(--color-danger);
	}

	@keyframes pulse {
		50% {
			opacity: 0.35;
		}
	}

	/* Phones: compact brand, activity on its own full-width row. */
	@media (max-width: 639px) {
		.app-header {
			gap: 0 var(--space-4);
			padding: 0 var(--space-3);
		}

		.brand {
			min-height: var(--touch-target);
			padding-top: 0.7rem;
			font-size: 1rem;
		}

		.brand-text {
			display: none;
		}

		nav {
			margin-left: auto;
		}

		.activity {
			flex: 1 0 100%;
			margin-left: 0;
			padding-bottom: var(--space-2);
		}

		.run {
			flex: 1 1 auto;
			max-width: none;
		}
	}
</style>
