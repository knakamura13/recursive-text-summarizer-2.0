<script lang="ts">
	import Dialog from '$lib/components/common/Dialog.svelte';

	let {
		open = $bindable(false),
		onUpload
	}: {
		open?: boolean;
		onUpload?: (files: FileList) => void;
	} = $props();

	let dragging = $state(false);

	function handleFiles(event: Event) {
		const input = event.target as HTMLInputElement;
		if (input.files?.length) {
			onUpload?.(input.files);
			open = false;
		}
	}

	function onDrop(event: DragEvent) {
		event.preventDefault();
		dragging = false;
		if (event.dataTransfer?.files?.length) {
			onUpload?.(event.dataTransfer.files);
			open = false;
		}
	}
</script>

<Dialog bind:open title="Import documents">
	<div
		class="dropzone"
		role="region"
		aria-label="File drop zone"
		class:dragging
		ondragover={(event) => {
			event.preventDefault();
			dragging = true;
		}}
		ondragleave={() => (dragging = false)}
		ondrop={onDrop}
	>
		<p>Drop .txt, .md, or .pdf files here, or choose files.</p>
		<label class="file-label">
			<span>Choose files</span>
			<input type="file" accept=".txt,.md,.pdf" multiple onchange={handleFiles} />
		</label>
	</div>
</Dialog>

<style>
	.dropzone {
		border: 1px dashed var(--color-border-strong);
		border-radius: var(--radius-md);
		padding: var(--space-8);
		text-align: center;
		background: var(--color-surface);
	}

	.dropzone.dragging {
		border-color: var(--color-sage);
		background: var(--color-sage-soft);
	}

	.file-label {
		display: inline-block;
		margin-top: var(--space-4);
		padding: var(--space-2) var(--space-4);
		border-radius: var(--radius-sm);
		background: var(--color-sage);
		color: white;
		cursor: pointer;
	}

	.file-label input {
		display: none;
	}
</style>
