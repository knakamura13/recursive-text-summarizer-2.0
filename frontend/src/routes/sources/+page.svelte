<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';

	onMount(() => {
		const lastId = localStorage.getItem('lastDocumentId');
		if (lastId) {
			goto(`/documents/${lastId}?view=sources`);
			return;
		}
	});
</script>

<section class="empty-state">
	<h2>Sources</h2>
	<p>Select a document to inspect its extracted source text or PDF.</p>
</section>

<style>
	.empty-state {
		padding: var(--space-8);
		color: var(--color-text-muted);
	}
</style>
