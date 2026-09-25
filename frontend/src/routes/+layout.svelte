<script lang="ts">
	import '$lib/tokens.css';
	import '$lib/base.css';
	import { onMount } from 'svelte';
	import { MediaQuery } from 'svelte/reactivity';
	import { page } from '$app/state';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import ToastHost from '$lib/components/common/ToastHost.svelte';
	import AppHeader from '$lib/components/shell/AppHeader.svelte';
	import DocumentsSidebar from '$lib/components/shell/DocumentsSidebar.svelte';
	import { activity } from '$lib/stores/activity.svelte';

	let { children } = $props();

	const wide = new MediaQuery('min-width: 1024px');
	let online = $state(true);

	// Quick reconnects (a server restart, a stream the server ended) should not
	// flash a banner; show it once reconnecting lasts a moment.
	const RECONNECT_BANNER_DELAY_MS = 2_500;
	let reconnecting = $state(false);
	$effect(() => {
		if (activity.status !== 'reconnecting') {
			reconnecting = false;
			return;
		}
		const timer = setTimeout(() => (reconnecting = true), RECONNECT_BANNER_DELAY_MS);
		return () => clearTimeout(timer);
	});

	// Document pages manage their own scrolling panes inside a fixed-height main.
	const inWorkspace = $derived(page.route.id?.startsWith('/documents/') ?? false);
	const showSidebar = $derived(inWorkspace && wide.current);

	// A file dropped outside a drop zone must not replace the app with the file.
	function ignoreStrayFileDrop(event: DragEvent) {
		if (event.defaultPrevented || !event.dataTransfer?.types.includes('Files')) return;
		event.preventDefault();
		event.dataTransfer.dropEffect = 'none';
	}

	onMount(() => {
		const updateOnline = () => (online = navigator.onLine);
		updateOnline();
		window.addEventListener('online', updateOnline);
		window.addEventListener('offline', updateOnline);
		window.addEventListener('dragover', ignoreStrayFileDrop);
		window.addEventListener('drop', ignoreStrayFileDrop);
		const stopActivity = activity.start();
		return () => {
			stopActivity();
			window.removeEventListener('online', updateOnline);
			window.removeEventListener('offline', updateOnline);
			window.removeEventListener('dragover', ignoreStrayFileDrop);
			window.removeEventListener('drop', ignoreStrayFileDrop);
		};
	});
</script>

<div class="app">
	<a class="skip-link" href="#main">Skip to content</a>
	<AppHeader />
	<div class="connection" aria-live="polite">
		{#if !online}
			<p class="banner">You are offline. The app reconnects when the network is back.</p>
		{:else if reconnecting}
			<p class="banner">
				<Spinner size="sm" label="" />
				Lost the connection to the server. Reconnecting…
			</p>
		{/if}
	</div>
	<div class="body" class:with-sidebar={showSidebar}>
		{#if showSidebar}
			<DocumentsSidebar currentId={page.params.documentId ?? null} />
		{/if}
		<main id="main" tabindex="-1" class:workspace={inWorkspace}>
			{@render children()}
		</main>
	</div>
	<ToastHost />
</div>

<style>
	.app {
		display: flex;
		flex-direction: column;
		height: 100vh;
		height: 100dvh;
		overflow: hidden;
	}

	.skip-link {
		position: absolute;
		left: var(--space-2);
		top: -4rem;
		z-index: 70;
		padding: var(--space-2) var(--space-4);
		border-radius: var(--radius-sm);
		background: var(--color-bg);
		box-shadow: var(--shadow-md);
	}

	.skip-link:focus {
		top: var(--space-2);
	}

	.banner {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-2);
		padding: var(--space-2) var(--space-4);
		background: var(--color-amber-soft);
		border-bottom: 1px solid var(--color-amber-border);
		color: var(--color-amber-strong);
		font-size: 0.9375rem;
		text-align: center;
	}

	.body {
		flex: 1 1 auto;
		min-height: 0;
		display: flex;
	}

	.body.with-sidebar {
		display: grid;
		grid-template-columns: var(--sidebar-left-width) minmax(0, 1fr);
		grid-template-rows: minmax(0, 1fr);
	}

	main {
		flex: 1 1 auto;
		min-width: 0;
		min-height: 0;
		overflow: auto;
	}

	main:focus {
		outline: none;
	}

	main.workspace {
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}
</style>
