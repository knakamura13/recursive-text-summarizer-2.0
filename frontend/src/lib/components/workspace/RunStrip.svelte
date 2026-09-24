<script lang="ts">
	import ProgressBar from '$lib/components/common/ProgressBar.svelte';
	import StateBadge from '$lib/components/common/StateBadge.svelte';
	import type { Run, RunProgress } from '$lib/api/types';
	import { formatDuration, formatEta, formatRelativeTime } from '$lib/format';
	import { isActiveRunState } from '$lib/stores/runStream.svelte';
	import { ticker } from './clock.svelte';
	import RunControls from './RunControls.svelte';
	import { activeStageBar, liveElapsed, liveEta, stageRows } from './runDisplay';

	interface Props {
		run: Run;
		progress: RunProgress | null;
		progressAt: number;
		blockedBy: Run | null;
		stopPending: boolean;
		resumePending: boolean;
		onstop: () => void;
		onresume: () => void;
		onnew: () => void;
	}

	let { run, progress, progressAt, blockedBy, stopPending, resumePending, onstop, onresume, onnew }: Props = $props();

	const live = $derived(isActiveRunState(run.state));
	const now = $derived(live ? ticker.now : progressAt);
	const elapsed = $derived(liveElapsed(progress?.elapsed_seconds ?? null, progressAt, now, live && run.state !== 'queued'));
	const eta = $derived(run.state === 'running' ? liveEta(progress?.eta_seconds ?? null, progressAt, now, true) : null);
	const bar = $derived(live ? activeStageBar(progress) : null);
	const activeRow = $derived(live ? stageRows(progress).find((row) => row.state === 'active') : undefined);
</script>

<div class="strip" data-testid="run-strip">
	<div class="line">
		<StateBadge state={run.state} />
		{#if activeRow && run.state === 'running'}
			<span class="stage">{activeRow.label}{#if activeRow.counts}<span class="counts"> · {activeRow.counts}</span>{/if}</span>
		{:else if !live}
			<span class="stage muted">updated {formatRelativeTime(run.updated_at)}</span>
		{/if}
		{#if live && elapsed !== null}
			<span class="time">
				{formatDuration(elapsed)}{#if run.state === 'running'} · {formatEta(eta)}{/if}
			</span>
		{/if}
	</div>
	{#if bar}
		<ProgressBar value={bar.value} max={bar.max} size="sm" label="{bar.label} progress" />
	{/if}
	<RunControls {run} {blockedBy} {stopPending} {resumePending} {onstop} {onresume} {onnew} />
</div>

<style>
	.strip {
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
		min-width: 0;
	}

	.line {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-1) var(--space-2);
		font-size: 0.9rem;
		min-width: 0;
	}

	.stage {
		font-weight: 600;
		overflow-wrap: anywhere;
	}

	.stage.muted,
	.counts,
	.time {
		font-weight: 400;
		color: var(--color-text-muted);
	}

	.time {
		font-variant-numeric: tabular-nums;
	}
</style>
