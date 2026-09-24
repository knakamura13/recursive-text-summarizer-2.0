<script lang="ts" module>
	import type { ImportState, RunState } from '$lib/api/types';

	const TONES: Record<RunState | ImportState, 'active' | 'success' | 'warning' | 'danger'> = {
		queued: 'active',
		running: 'active',
		stopping: 'active',
		importing: 'active',
		completed: 'success',
		ready: 'success',
		stopped: 'warning',
		interrupted: 'warning',
		failed: 'danger'
	};
</script>

<script lang="ts">
	import { importStateLabel, runStateLabel } from '$lib/format';

	// `failed` exists for both Runs and Imports; pass kind="import" for "Import failed".
	let {
		state,
		kind,
		size = 'md'
	}: {
		state: RunState | ImportState;
		kind?: 'run' | 'import';
		size?: 'sm' | 'md';
	} = $props();

	const isImport = $derived(
		kind === 'import' || (kind === undefined && (state === 'importing' || state === 'ready'))
	);
	const label = $derived(
		isImport ? importStateLabel(state as ImportState) : runStateLabel(state as RunState)
	);
	const tone = $derived(TONES[state] ?? 'active');
	const live = $derived(state === 'running' || state === 'importing' || state === 'stopping');
</script>

<span class="badge {tone} {size}" data-state={state}>
	<span class="dot" class:live aria-hidden="true"></span>
	{label}
</span>

<style>
	.badge {
		display: inline-flex;
		align-items: center;
		gap: 0.375rem;
		padding: 0.125rem 0.5rem;
		border: 1px solid;
		border-radius: 999px;
		font-size: 0.8125rem;
		font-weight: 600;
		line-height: 1.4;
		white-space: nowrap;
	}

	.badge.sm {
		font-size: 0.75rem;
		padding: 0 0.4375rem;
	}

	.dot {
		width: 0.5rem;
		height: 0.5rem;
		border-radius: 50%;
		background: currentColor;
		flex: none;
	}

	.dot.live {
		animation: pulse 1.6s ease-in-out infinite;
	}

	@keyframes pulse {
		50% {
			opacity: 0.3;
		}
	}

	.active {
		color: var(--color-info);
		background: var(--color-info-soft);
		border-color: var(--color-info-border);
	}

	.success {
		color: var(--color-sage);
		background: var(--color-sage-soft);
		border-color: var(--color-sage-border);
	}

	.warning {
		color: var(--color-amber-strong);
		background: var(--color-amber-soft);
		border-color: var(--color-amber-border);
	}

	.danger {
		color: var(--color-danger);
		background: var(--color-danger-soft);
		border-color: var(--color-danger-border);
	}
</style>
