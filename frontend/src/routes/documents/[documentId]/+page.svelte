<script lang="ts">
	import { page } from '$app/state';
	import DocumentWorkspace from '$lib/components/workspace/DocumentWorkspace.svelte';

	const documentId = $derived(page.params.documentId ?? '');
</script>

<!-- A different Document gets a fresh workspace: its Run stream, panes, and Source cache view. -->
{#key documentId}
	<DocumentWorkspace {documentId} />
{/key}
