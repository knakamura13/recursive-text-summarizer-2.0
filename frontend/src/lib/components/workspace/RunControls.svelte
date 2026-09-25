<script lang="ts">
	import type { Run } from '$lib/api/types';

	interface Props {
		run: Run;
		/** Another Document's (or this Document's other) active Run, which blocks Resume. */
		blockedBy: Run | null;
		/** Stop was clicked and the server has not answered yet. */
		stopPending: boolean;
		resumePending: boolean;
		onstop: () => void;
		onresume: () => void;
		onnew: () => void;
	}

	let { run, blockedBy, stopPending, resumePending, onstop, onresume, onnew }: Props = $props();

	const stoppable = $derived(run.state === 'queued' || run.state === 'running');
	const stopping = $derived(run.state === 'stopping' || (stopPending && stoppable));
	const resumable = $derived(run.state === 'stopped' || run.state === 'failed' || run.state === 'interrupted');
	const blocker = $derived(blockedBy && blockedBy.run_id !== run.run_id ? blockedBy : null);
</script>

<div class="controls">
	{#if stoppable || run.state === 'stopping'}
		<button type="button" class="button danger-outline" disabled={stopping} onclick={onstop}>
			{stopping ? 'Stopping…' : 'Stop'}
		</button>
	{:else}
		{#if resumable}
			<button
				type="button"
				class="button primary"
				disabled={resumePending || blocker !== null}
				aria-describedby={blocker ? `resume-blocked-${run.run_id}` : undefined}
				onclick={onresume}
			>
				{resumePending ? 'Resuming…' : 'Resume'}
			</button>
		{/if}
		<button type="button" class="button" onclick={onnew}>Start new Run</button>
		{#if resumable && blocker}
			<p class="blocked" id="resume-blocked-{run.run_id}">
				Resume is available when the active Run on
				<a href="/documents/{encodeURIComponent(blocker.document_id)}?run={encodeURIComponent(blocker.run_id)}"
					>{blocker.document_title}</a
				> ends.
			</p>
		{/if}
	{/if}
</div>

<style>
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-2);
	}

	.blocked {
		flex-basis: 100%;
		margin: 0;
		font-size: 0.85rem;
		color: var(--color-text-muted);
	}
</style>
