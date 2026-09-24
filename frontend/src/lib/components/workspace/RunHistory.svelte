<script lang="ts">
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import EmptyState from '$lib/components/common/EmptyState.svelte';
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import StateBadge from '$lib/components/common/StateBadge.svelte';
	import type { Run } from '$lib/api/types';
	import { formatRelativeTime } from '$lib/format';
	import { isActiveRunState } from '$lib/stores/runStream.svelte';
	import { describeConfig } from './runDisplay';

	interface Props {
		runs: Run[] | null;
		error: unknown;
		selectedRunId: string | null;
		/** The app's active Run (any Document); it blocks Resume of every other Run. */
		activeRun: Run | null;
		resumingRunId: string | null;
		onretry: () => void;
		onopen: (runId: string) => void;
		onresume: (run: Run) => void;
		/** Deletes the Run; rejects with the API error (shown in the dialog). */
		ondelete: (run: Run) => Promise<void>;
		onnew: () => void;
	}

	let { runs, error, selectedRunId, activeRun, resumingRunId, onretry, onopen, onresume, ondelete, onnew }: Props =
		$props();

	let deleting = $state<Run | null>(null);
	let confirmOpen = $state(false);
</script>

<section class="history" aria-label="Runs">
	<div class="head">
		<h2>Runs</h2>
		<button type="button" class="button small" onclick={onnew}>Start new Run</button>
	</div>
	{#if error}
		<ErrorBanner {error} title="Could not load the Runs" {onretry} />
	{:else if runs === null}
		<p class="muted"><Spinner size="sm" label="" /> Loading…</p>
	{:else if runs.length === 0}
		<EmptyState title="No Runs yet" message="Start a Run to summarize this Document." headingLevel={3} />
	{:else}
		<ul class="runs">
			{#each runs as run (run.run_id)}
				{@const active = isActiveRunState(run.state)}
				{@const resumable = run.state === 'stopped' || run.state === 'failed' || run.state === 'interrupted'}
				{@const blocked = activeRun !== null && activeRun.run_id !== run.run_id}
				<li class:current={run.run_id === selectedRunId} data-run-id={run.run_id}>
					<div class="line">
						<StateBadge state={run.state} size="sm" />
						<span class="when">{formatRelativeTime(run.created_at)}</span>
						{#if run.attempt_count > 1}<span class="muted-text">{run.attempt_count} Attempts</span>{/if}
						{#if run.run_id === selectedRunId}<span class="muted-text">(shown)</span>{/if}
					</div>
					<p class="config">{describeConfig(run.config)}</p>
					{#if run.failure}<p class="failure">{run.failure.message}</p>{/if}
					<div class="actions">
						{#if run.run_id !== selectedRunId}
							<button type="button" class="button small" onclick={() => onopen(run.run_id)}>Open</button>
						{/if}
						{#if resumable}
							<button
								type="button"
								class="button small primary"
								disabled={blocked || resumingRunId === run.run_id}
								title={blocked ? 'Another Run is active' : undefined}
								onclick={() => onresume(run)}
							>
								{resumingRunId === run.run_id ? 'Resuming…' : 'Resume'}
							</button>
						{/if}
						{#if !active}
							<button
								type="button"
								class="button small ghost"
								onclick={() => {
									deleting = run;
									confirmOpen = true;
								}}
							>
								Delete
							</button>
						{/if}
					</div>
				</li>
			{/each}
		</ul>
	{/if}
</section>

<ConfirmDialog
	bind:open={confirmOpen}
	title="Delete this Run?"
	message="Its summary, tree, and progress are deleted. The Document stays."
	confirmLabel="Delete Run"
	danger
	onconfirm={async () => {
		if (deleting) await ondelete(deleting);
	}}
/>

<style>
	.history {
		display: flex;
		flex-direction: column;
		gap: var(--space-3);
	}

	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-2);
	}

	h2 {
		margin: 0;
		font-size: 1.05rem;
	}

	.muted {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin: 0;
		color: var(--color-text-muted);
	}

	.runs {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}

	.runs li {
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		padding: var(--space-3);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
	}

	.runs li.current {
		border-color: var(--color-sage-border);
		background: var(--color-sage-soft);
	}

	.line {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-2);
		font-size: 0.9rem;
	}

	.muted-text,
	.config {
		color: var(--color-text-muted);
		font-size: 0.85rem;
	}

	.config,
	.failure {
		margin: 0;
		overflow-wrap: anywhere;
	}

	.failure {
		color: var(--color-danger);
		font-size: 0.85rem;
	}

	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-1);
		margin-top: var(--space-1);
	}
</style>
