<script lang="ts">
	import type { ProgressStage } from '$lib/stores/workspace';

	let { stages = [] }: { stages?: ProgressStage[] } = $props();

	function label(stage: ProgressStage): string {
		if (stage.total != null && stage.completed != null && stage.state === 'active') {
			const percent = Math.round((stage.completed / stage.total) * 100);
			return `${stage.stage} (${percent}%)`;
		}
		if (stage.total != null && stage.state === 'completed') {
			return `${stage.stage} (${stage.total})`;
		}
		if (stage.state === 'pending') {
			return `${stage.stage} (Waiting)`;
		}
		return stage.stage;
	}
</script>

<ol class="stepper" aria-label="Run progress">
	{#each stages as stage, index}
		<li class={stage.state}>
			<span class="marker" aria-hidden="true">
				{#if stage.state === 'completed'}✓{:else if stage.state === 'active'}●{:else}○{/if}
			</span>
			<span class="label">{label(stage)}</span>
			{#if index < stages.length - 1}
				<span class="connector" aria-hidden="true"></span>
			{/if}
		</li>
	{/each}
</ol>

<style>
	.stepper {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		list-style: none;
		margin: 0;
		padding: var(--space-4) 0;
		overflow-x: auto;
	}

	.stepper li {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		position: relative;
		white-space: nowrap;
		color: var(--color-text-muted);
		font-size: 0.875rem;
	}

	.stepper li.completed {
		color: var(--color-sage);
	}

	.stepper li.active {
		color: var(--color-amber);
	}

	.marker {
		display: inline-flex;
		width: 1.25rem;
		justify-content: center;
	}

	.connector {
		width: 2rem;
		height: 2px;
		background: var(--color-border);
		margin-left: var(--space-2);
	}
</style>
