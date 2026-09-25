<script lang="ts">
	import ProgressBar from '$lib/components/common/ProgressBar.svelte';
	import StateBadge from '$lib/components/common/StateBadge.svelte';
	import type { Run, RunProgress, StageName } from '$lib/api/types';
	import { formatDuration, formatEta, formatRelativeTime, stageLabel } from '$lib/format';
	import { isActiveRunState } from '$lib/stores/runStream.svelte';
	import { ticker } from './clock.svelte';
	import { activeStageBar, describeConfig, liveElapsed, liveEta, reaskNote, stageRows, STAGE_ORDER } from './runDisplay';

	interface Props {
		run: Run;
		progress: RunProgress | null;
		/** Clock reading when `progress` arrived (see RunStream.progressAt). */
		progressAt: number;
	}

	let { run, progress, progressAt }: Props = $props();

	const live = $derived(isActiveRunState(run.state));
	const now = $derived(live ? ticker.now : progressAt);
	const wallNow = $derived.by(() => {
		if (live) void ticker.now;
		return Date.now();
	});
	const elapsed = $derived(liveElapsed(progress?.elapsed_seconds ?? null, progressAt, now, live && run.state !== 'queued'));
	const eta = $derived(run.state === 'running' ? liveEta(progress?.eta_seconds ?? null, progressAt, now, true) : null);
	// A completed Run never ran its pending stages (verification off): show them as skipped.
	const rows = $derived(
		stageRows(progress).map((row) =>
			run.state === 'completed' && row.state === 'pending' ? { ...row, state: 'skipped' as const } : row
		)
	);
	const bar = $derived(live ? activeStageBar(progress) : null);
	const items = $derived(live ? (progress?.current_items ?? []) : []);
	const failureStage = $derived(
		run.failure?.stage && (STAGE_ORDER as readonly string[]).includes(run.failure.stage)
			? stageLabel(run.failure.stage as StageName)
			: run.failure?.stage
	);
	const stageSummary = $derived.by(() => {
		if (run.state === 'queued') return 'Waiting for the worker to start this Run.';
		if (run.state === 'stopping') return 'Stopping: finished work is kept.';
		if (!live) return null;
		const active = rows.find((row) => row.state === 'active');
		return active ? `${active.label}${active.counts ? `: ${active.counts}` : ''}` : null;
	});
</script>

<section class="run-status" aria-label="Run status">
	<header class="head">
		<StateBadge state={run.state} />
		<span class="meta">
			Attempt {run.attempt?.attempt_number ?? run.attempt_count}{#if run.selected_strategy}{` · ${run.selected_strategy}`}{/if} · started {formatRelativeTime(run.created_at, wallNow)}
		</span>
	</header>

	<p class="config">{describeConfig(run.config)}</p>

	{#if stageSummary}
		<p class="summary" aria-live="polite">{stageSummary}</p>
	{/if}

	{#if bar}
		<ProgressBar
			value={bar.value}
			max={bar.max}
			label="{bar.label} progress"
			valueText={bar.max === null ? `${bar.label} in progress` : `${bar.value} of ${bar.max}`}
		/>
	{/if}

	{#if elapsed !== null || run.state === 'running'}
		<dl class="times">
			{#if elapsed !== null}
				<div>
					<dt>Elapsed</dt>
					<dd data-testid="run-elapsed">{formatDuration(elapsed)}</dd>
				</div>
			{/if}
			{#if run.state === 'running'}
				<div>
					<dt>Time left</dt>
					<dd data-testid="run-eta">{formatEta(eta)}</dd>
				</div>
			{/if}
		</dl>
		{#if run.state === 'running' && progress?.eta_basis}
			<p class="basis">{progress.eta_basis}</p>
		{/if}
	{/if}

	<ol class="stepper" aria-label="Stages">
		{#each rows as row (row.stage)}
			<li class="step step-{row.state}" aria-current={row.state === 'active' && live ? 'step' : undefined}>
				<span class="marker" aria-hidden="true"></span>
				<div class="step-body">
					<p class="step-title">
						<span class="step-label">{row.label}</span>
						<span class="visually-hidden">({row.state})</span>
						{#if row.state === 'skipped'}<span class="step-state">skipped</span>{/if}
						{#if row.counts && row.state !== 'pending'}<span class="step-counts">{row.counts}</span>{/if}
					</p>
					{#if row.state !== 'pending'}
						{#each row.notes as note}
							<p class="step-note">{note}</p>
						{/each}
					{/if}
				</div>
			</li>
		{/each}
	</ol>

	{#if items.length > 0}
		<h3 class="section-title">Working on</h3>
		<ul class="items">
			{#each items as item (item.work_id)}
				{@const note = reaskNote(item)}
				<li>
					<span class="item-label">{item.label}</span>
					{#if item.started_at}
						<span class="item-time">
							for {formatDuration((wallNow - Date.parse(item.started_at)) / 1000)}
						</span>
					{/if}
					{#if note}<span class="reask">{note}</span>{/if}
				</li>
			{/each}
		</ul>
	{/if}

	{#if run.failure}
		<div class="failure" role="alert">
			<h3 class="section-title">
				{run.state === 'interrupted' ? 'The Run was interrupted' : run.state === 'failed' ? 'The Run failed' : 'Last Attempt ended with an error'}
			</h3>
			<p class="failure-message">{run.failure.message}</p>
			{#if run.failure.item || failureStage}
				<p class="failure-where">
					{#if failureStage}Stage: {failureStage}{/if}{#if failureStage && run.failure.item}{' · '}{/if}{#if run.failure.item}Item: {run.failure.item}{/if}
				</p>
			{/if}
			{#if run.failure.hint}
				<p class="failure-hint">{run.failure.hint}</p>
			{/if}
			{#if run.failure.detail}
				<details>
					<summary>Error details</summary>
					<pre>{run.failure.detail}</pre>
				</details>
			{/if}
		</div>
	{:else if run.state === 'stopped'}
		<p class="note">Stopped. Finished work is kept; Resume continues with what is left.</p>
	{:else if run.state === 'interrupted'}
		<p class="note">The application stopped during this Run. Resume continues with what is left.</p>
	{/if}
</section>

<style>
	.run-status {
		display: flex;
		flex-direction: column;
		gap: var(--space-3);
	}

	.head {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-2);
	}

	.meta,
	.config,
	.basis {
		font-size: 0.85rem;
		color: var(--color-text-muted);
	}

	.config,
	.basis,
	.summary {
		margin: 0;
	}

	.summary {
		font-family: var(--font-serif);
		font-size: 1.25rem;
		font-weight: 600;
	}

	.times {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-2) var(--space-6);
		margin: 0;
	}

	.times div {
		display: flex;
		flex-direction: column;
	}

	.times dt {
		font-size: 0.72rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--color-text-subtle);
	}

	.times dd {
		margin: 0;
		font-family: var(--font-serif);
		font-size: 1.375rem;
		font-variant-numeric: tabular-nums;
	}

	.stepper {
		list-style: none;
		margin: 0;
		padding: 0;
	}

	.step {
		position: relative;
		display: flex;
		gap: var(--space-3);
		padding-bottom: var(--space-3);
	}

	.step:not(:last-child)::before {
		content: '';
		position: absolute;
		left: 6px;
		top: 18px;
		bottom: 0;
		width: 2px;
		background: var(--color-border);
	}

	.marker {
		flex: 0 0 14px;
		width: 14px;
		height: 14px;
		margin-top: 4px;
		border-radius: 50%;
		border: 2px solid var(--color-border-strong);
		background: var(--color-surface-elevated);
		z-index: 1;
	}

	.step-active .marker {
		border-color: var(--color-amber);
		background: var(--color-amber);
	}

	.step-completed .marker {
		border-color: var(--color-success);
		background: var(--color-success);
	}

	.step-skipped .marker {
		border-style: dashed;
	}

	.step-body {
		min-width: 0;
	}

	.step-title {
		display: flex;
		flex-wrap: wrap;
		gap: 0 var(--space-2);
		margin: 0;
	}

	.step-pending .step-label,
	.step-skipped .step-label {
		color: var(--color-text-muted);
	}

	.step-active .step-label {
		font-weight: 600;
	}

	.step-counts,
	.step-state {
		color: var(--color-text-muted);
		font-variant-numeric: tabular-nums;
	}

	.step-note {
		margin: 0.1rem 0 0;
		font-size: 0.85rem;
		color: var(--color-text-muted);
	}

	.section-title {
		margin: 0;
		font-size: 0.72rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--color-text-subtle);
	}

	.failure .section-title {
		font-size: 0.95rem;
		text-transform: none;
		letter-spacing: 0;
		color: var(--color-danger);
	}

	.items {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
	}

	.items li {
		display: flex;
		flex-wrap: wrap;
		gap: 0 var(--space-2);
		font-size: 0.9rem;
	}

	.item-label {
		overflow-wrap: anywhere;
	}

	.item-time {
		color: var(--color-text-muted);
		font-variant-numeric: tabular-nums;
	}

	.reask {
		flex-basis: 100%;
		color: var(--color-amber-strong, var(--color-amber));
		font-size: 0.85rem;
	}

	.failure {
		padding: var(--space-3);
		border: 1px solid var(--color-danger-border, var(--color-danger));
		border-radius: var(--radius-md);
		background: var(--color-danger-soft);
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}

	.failure p {
		margin: 0;
	}

	.failure-message {
		font-weight: 600;
		overflow-wrap: anywhere;
	}

	.failure-where,
	.failure-hint {
		font-size: 0.9rem;
	}

	.failure pre {
		margin: var(--space-2) 0 0;
		max-height: 16rem;
		overflow: auto;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		font-size: 0.8rem;
	}

	.failure summary {
		cursor: pointer;
		min-height: 44px;
		display: flex;
		align-items: center;
	}

	.note {
		margin: 0;
		color: var(--color-text-muted);
	}
</style>
