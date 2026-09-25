<script lang="ts">
	// With a label the spinner announces itself as a status; with label="" it is decorative.
	let { label = 'Loading', size = 'md' }: { label?: string; size?: 'sm' | 'md' | 'lg' } = $props();
</script>

{#if label}
	<span class="spinner {size}" role="status">
		<span class="ring" aria-hidden="true"></span>
		<span class="visually-hidden">{label}</span>
	</span>
{:else}
	<span class="spinner {size}" aria-hidden="true"><span class="ring"></span></span>
{/if}

<style>
	.spinner {
		display: inline-flex;
		flex: none;
		align-items: center;
		justify-content: center;
		vertical-align: middle;
	}

	.ring {
		width: 1.25rem;
		height: 1.25rem;
		border: 2px solid currentColor;
		border-right-color: transparent;
		border-radius: 50%;
		opacity: 0.7;
		animation: spin 0.8s linear infinite;
	}

	.sm .ring {
		width: 0.875rem;
		height: 0.875rem;
	}

	.lg .ring {
		width: 2rem;
		height: 2rem;
		border-width: 3px;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
