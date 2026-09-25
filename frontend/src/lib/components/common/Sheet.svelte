<script lang="ts">
	import type { Snippet } from 'svelte';
	import Dialog from './Dialog.svelte';

	// A modal panel: full screen below 640 px, a side panel on the right otherwise.
	let {
		open = $bindable(false),
		title,
		onclose,
		children,
		footer
	}: {
		open?: boolean;
		title: string;
		onclose?: () => void;
		children: Snippet;
		footer?: Snippet;
	} = $props();
</script>

<Dialog bind:open {title} {onclose} {footer} variant="sheet">
	{@render children()}
</Dialog>
