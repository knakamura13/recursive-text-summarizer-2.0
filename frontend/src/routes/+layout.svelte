<script lang="ts">
	import '$lib/tokens.css';
	import AppHeader from '$lib/components/shell/AppHeader.svelte';

	let { children } = $props();
</script>

<div class="app-shell">
	<AppHeader />
	<main>{@render children()}</main>
</div>

<style>
	:global(body) {
		margin: 0;
		font-family: var(--font-sans);
		color: var(--color-text);
		background: var(--color-bg);
	}

	:global(*, *::before, *::after) {
		box-sizing: border-box;
	}

	:global(:focus-visible) {
		outline: 2px solid var(--color-focus);
		outline-offset: 2px;
	}

	.app-shell {
		min-height: 100vh;
		display: flex;
		flex-direction: column;
	}

	main {
		flex: 1;
		min-height: 0;
	}
</style>
