<script lang="ts">
	// Indeterminate (animated, no value announced) while `max` is null.
	let {
		value = 0,
		max,
		label,
		valueText,
		tone = 'default',
		size = 'md'
	}: {
		value?: number;
		max: number | null;
		/** Accessible name, e.g. "Import progress". */
		label: string;
		/** Spoken value, e.g. "12 of 40 pages". */
		valueText?: string;
		tone?: 'default' | 'success' | 'warning' | 'danger';
		size?: 'sm' | 'md';
	} = $props();

	const indeterminate = $derived(max === null);
	const clamped = $derived(max === null ? 0 : Math.min(Math.max(value, 0), Math.max(max, 0)));
	const percent = $derived(max === null || max <= 0 ? 0 : (clamped / max) * 100);
</script>

<div
	class="progress {tone} {size}"
	class:indeterminate
	role="progressbar"
	aria-label={label}
	aria-valuemin={indeterminate ? undefined : 0}
	aria-valuemax={indeterminate ? undefined : Math.max(max ?? 0, 0)}
	aria-valuenow={indeterminate ? undefined : clamped}
	aria-valuetext={valueText}
>
	<div class="fill" style:width={indeterminate ? undefined : `${percent}%`}></div>
</div>

<style>
	.progress {
		position: relative;
		width: 100%;
		height: 0.375rem;
		border-radius: 999px;
		background: var(--color-border);
		overflow: hidden;
	}

	.progress.sm {
		height: 0.25rem;
	}

	.fill {
		height: 100%;
		border-radius: inherit;
		background: var(--color-ink);
		transition: width 0.25s ease-out;
	}

	.success .fill {
		background: var(--color-success);
	}

	.warning .fill {
		background: var(--color-amber);
	}

	.danger .fill {
		background: var(--color-danger);
	}

	.indeterminate .fill {
		position: absolute;
		width: 35%;
		animation: slide 1.4s ease-in-out infinite;
	}

	@keyframes slide {
		from {
			left: -35%;
		}
		to {
			left: 100%;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.indeterminate .fill {
			left: 0;
			width: 100%;
			opacity: 0.35;
			animation: none;
		}
	}
</style>
