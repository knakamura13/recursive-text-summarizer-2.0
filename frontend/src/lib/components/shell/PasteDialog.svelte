<script lang="ts">
	import { goto } from '$app/navigation';
	import { api } from '$lib/api/client';
	import Dialog from '$lib/components/common/Dialog.svelte';
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import { formatCount } from '$lib/format';
	import { documents } from '$lib/stores/documents.svelte';
	import { toasts } from '$lib/stores/toasts.svelte';

	/** Server limit for pasted text (UTF-8 bytes). */
	const MAX_PASTE_BYTES = 20 * 1024 * 1024;

	// Imports pasted text as a Document and opens it. The draft survives closing
	// the dialog until an import succeeds.
	let { open = $bindable(false) }: { open?: boolean } = $props();

	const id = $props.id();
	let title = $state('');
	let text = $state('');
	let busy = $state(false);
	let submitted = $state(false);
	let error = $state<unknown>(null);

	const textError = $derived(
		submitted && text.trim() === '' ? 'Paste or type the text to import.' : null
	);

	async function submit(event: SubmitEvent) {
		event.preventDefault();
		submitted = true;
		error = null;
		if (text.trim() === '') return;
		if (new TextEncoder().encode(text).length > MAX_PASTE_BYTES) {
			error = new Error('The text is larger than 20 MiB. Import it as a file instead.');
			return;
		}
		busy = true;
		try {
			const created = await api.pasteText({ title: title.trim() || undefined, text });
			documents.upsert(created.document);
			toasts.success(
				created.already_imported
					? `Already in your library as “${created.document.title}”. Opened it.`
					: `Imported “${created.document.title}”.`
			);
			title = '';
			text = '';
			submitted = false;
			open = false;
			await goto(`/documents/${encodeURIComponent(created.document.document_id)}`);
		} catch (failure) {
			error = failure;
		} finally {
			busy = false;
		}
	}
</script>

<Dialog bind:open title="Paste text" size="lg" dismissible={!busy}>
	<form id="{id}-form" class="form" novalidate onsubmit={submit}>
		<div class="field">
			<label for="{id}-title">Title <span class="optional">(optional)</span></label>
			<input
				id="{id}-title"
				type="text"
				maxlength="300"
				autocomplete="off"
				placeholder="Defaults to the first line"
				bind:value={title}
				disabled={busy}
			/>
		</div>
		<div class="field">
			<label for="{id}-text">Text</label>
			<textarea
				id="{id}-text"
				rows="12"
				bind:value={text}
				disabled={busy}
				data-autofocus
				aria-invalid={textError ? 'true' : undefined}
				aria-describedby="{id}-count{textError ? ` ${id}-error` : ''}"
			></textarea>
			<p class="field-hint" id="{id}-count">{formatCount(text.length)} characters</p>
			{#if textError}
				<p class="field-error" id="{id}-error">{textError}</p>
			{/if}
		</div>
		{#if error}
			<ErrorBanner {error} />
		{/if}
	</form>

	{#snippet footer()}
		<button type="button" class="button" disabled={busy} onclick={() => (open = false)}>
			Close
		</button>
		<button type="submit" form="{id}-form" class="button primary" disabled={busy}>
			{#if busy}<Spinner size="sm" label="" />{/if}
			Import text
		</button>
	{/snippet}
</Dialog>

<style>
	.form {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
	}

	textarea {
		min-height: 12rem;
		font-family: var(--font-sans);
	}

	.optional {
		font-weight: 400;
		color: var(--color-text-muted);
	}
</style>
