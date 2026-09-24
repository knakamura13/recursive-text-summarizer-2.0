<script lang="ts">
	import { goto } from '$app/navigation';
	import Dialog from '$lib/components/common/Dialog.svelte';
	import { documents } from '$lib/stores/documents.svelte';
	import { imports, IMPORT_ACCEPT, SUPPORTED_FORMATS } from '$lib/stores/imports.svelte';
	import { toasts } from '$lib/stores/toasts.svelte';
	import ImportQueue from './ImportQueue.svelte';

	let {
		open = $bindable(false),
		onpaste
	}: {
		open?: boolean;
		/** Shows a "Paste text" action that closes this dialog and calls back. */
		onpaste?: () => void;
	} = $props();

	let dragging = $state(false);
	const finished = $derived(
		imports.items.some((item) => item.phase === 'sent' || item.phase === 'failed')
	);

	/**
	 * Starts importing the files. A single file that turns out to be a
	 * duplicate opens the existing Document, if this dialog is still open.
	 */
	export function addFiles(files: File[]): void {
		if (files.length === 0) return;
		const settled = imports.add(files);
		if (settled.length !== 1) return;
		void settled[0].then((item) => {
			if (!open || !item?.duplicate || !item.documentId) return;
			const existing = documents.get(item.documentId);
			toasts.info(
				`“${item.name}” is already in your library${existing ? ` as “${existing.title}”` : ''}. Opened it.`
			);
			imports.remove(item.id);
			open = false;
			void goto(`/documents/${encodeURIComponent(item.documentId)}`);
		});
	}

	function onPick(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		addFiles(Array.from(input.files ?? []));
		// Allow picking the same file again.
		input.value = '';
	}

	function onDrop(event: DragEvent) {
		event.preventDefault();
		dragging = false;
		addFiles(Array.from(event.dataTransfer?.files ?? []));
	}
</script>

<Dialog bind:open title="Import documents" size="lg">
	<div
		class="dropzone"
		class:dragging
		role="group"
		aria-label="Drop files to import"
		ondragenter={(event) => {
			event.preventDefault();
			dragging = true;
		}}
		ondragover={(event) => {
			event.preventDefault();
			if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
		}}
		ondragleave={(event) => {
			if (!event.currentTarget.contains(event.relatedTarget as Node | null)) dragging = false;
		}}
		ondrop={onDrop}
	>
		<p class="drop-hint">Drop files here, or</p>
		<label class="button primary picker">
			<input
				class="visually-hidden"
				type="file"
				multiple
				accept={IMPORT_ACCEPT}
				onchange={onPick}
				data-autofocus
			/>
			Choose files
		</label>
		<p class="formats">{SUPPORTED_FORMATS}. Up to 500 MiB each.</p>
	</div>

	{#if onpaste}
		<p class="paste">
			Have the text already?
			<button
				type="button"
				class="button small"
				onclick={() => {
					open = false;
					onpaste();
				}}>Paste text</button
			>
		</p>
	{/if}

	{#if imports.items.length > 0}
		<ImportQueue items={imports.items} />
	{/if}

	{#snippet footer()}
		{#if finished}
			<button type="button" class="button" onclick={() => imports.clearFinished()}>
				Clear finished
			</button>
		{/if}
		<button type="button" class="button primary" onclick={() => (open = false)}>Done</button>
	{/snippet}
</Dialog>

<style>
	.dropzone {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--space-3);
		padding: var(--space-6) var(--space-4);
		border: 2px dashed var(--color-border-strong);
		border-radius: var(--radius-md);
		background: var(--color-surface);
		text-align: center;
	}

	.dropzone.dragging {
		border-color: var(--color-sage);
		background: var(--color-sage-soft);
	}

	.picker {
		position: relative;
	}

	.picker:has(input:focus-visible) {
		outline: 2px solid var(--color-focus);
		outline-offset: 2px;
	}

	.formats {
		max-width: 32rem;
		font-size: 0.875rem;
		color: var(--color-text-muted);
	}

	.paste {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-2);
		color: var(--color-text-muted);
	}
</style>
