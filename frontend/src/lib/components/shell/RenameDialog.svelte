<script lang="ts">
	import { untrack } from 'svelte';
	import { api } from '$lib/api/client';
	import type { DocumentSummary } from '$lib/api/types';
	import Dialog from '$lib/components/common/Dialog.svelte';
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import { documents } from '$lib/stores/documents.svelte';
	import { toasts } from '$lib/stores/toasts.svelte';

	let {
		open = $bindable(false),
		document
	}: {
		open?: boolean;
		document: DocumentSummary | null;
	} = $props();

	const id = $props.id();
	let title = $state('');
	let busy = $state(false);
	let error = $state<unknown>(null);
	let submitted = $state(false);

	// Start from the current title each time the dialog opens; later updates of
	// the Document (import progress, Run state) must not overwrite the draft.
	let wasOpen = false;
	$effect(() => {
		const opening = open && !wasOpen;
		wasOpen = open;
		if (!opening) return;
		title = untrack(() => document?.title ?? '');
		error = null;
		submitted = false;
	});

	const titleError = $derived(submitted && title.trim() === '' ? 'Enter a title.' : null);

	async function submit(event: SubmitEvent) {
		event.preventDefault();
		submitted = true;
		if (!document || title.trim() === '') return;
		if (title.trim() === document.title) {
			open = false;
			return;
		}
		busy = true;
		error = null;
		try {
			const updated = await api.renameDocument(document.document_id, title.trim());
			documents.upsert(updated);
			toasts.success(`Renamed to “${updated.title}”.`);
			open = false;
		} catch (failure) {
			error = failure;
		} finally {
			busy = false;
		}
	}
</script>

<Dialog bind:open title="Rename Document" size="sm" dismissible={!busy}>
	<form id="{id}-form" class="form" novalidate onsubmit={submit}>
		<div class="field">
			<label for="{id}-title">Title</label>
			<input
				id="{id}-title"
				type="text"
				maxlength="300"
				autocomplete="off"
				bind:value={title}
				disabled={busy}
				data-autofocus
				aria-invalid={titleError ? 'true' : undefined}
				aria-describedby={titleError ? `${id}-error` : undefined}
			/>
			{#if titleError}
				<p class="field-error" id="{id}-error">{titleError}</p>
			{/if}
		</div>
		{#if error}
			<ErrorBanner {error} />
		{/if}
	</form>
	{#snippet footer()}
		<button type="button" class="button" disabled={busy} onclick={() => (open = false)}>
			Cancel
		</button>
		<button type="submit" form="{id}-form" class="button primary" disabled={busy}>
			{#if busy}<Spinner size="sm" label="" />{/if}
			Save
		</button>
	{/snippet}
</Dialog>

<style>
	.form {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
	}
</style>
