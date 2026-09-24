<script lang="ts">
	import type { Snippet } from 'svelte';
	import Dialog from './Dialog.svelte';
	import ErrorBanner from './ErrorBanner.svelte';
	import Spinner from './Spinner.svelte';

	// While `onconfirm` runs the dialog stays open and cannot be dismissed; it
	// closes when the promise resolves and shows the error when it rejects.
	let {
		open = $bindable(false),
		title,
		message,
		confirmLabel = 'Confirm',
		cancelLabel = 'Cancel',
		danger = false,
		onconfirm,
		oncancel,
		children
	}: {
		open?: boolean;
		title: string;
		message: string;
		confirmLabel?: string;
		cancelLabel?: string;
		danger?: boolean;
		onconfirm: () => void | Promise<void>;
		oncancel?: () => void;
		/** Extra content below the message. */
		children?: Snippet;
	} = $props();

	let busy = $state(false);
	let error = $state<unknown>(null);

	$effect(() => {
		if (!open) error = null;
	});

	async function confirm() {
		busy = true;
		error = null;
		try {
			await onconfirm();
			open = false;
		} catch (failure) {
			error = failure;
		} finally {
			busy = false;
		}
	}

	function cancel() {
		open = false;
		oncancel?.();
	}
</script>

<Dialog bind:open {title} size="sm" dismissible={!busy} onclose={() => oncancel?.()}>
	<p>{message}</p>
	{@render children?.()}
	{#if error}
		<ErrorBanner {error} />
	{/if}
	{#snippet footer()}
		<button type="button" class="button" disabled={busy} onclick={cancel} data-autofocus>
			{cancelLabel}
		</button>
		<button
			type="button"
			class="button {danger ? 'danger' : 'primary'}"
			disabled={busy}
			onclick={confirm}
		>
			{#if busy}<Spinner size="sm" label="" />{/if}
			{confirmLabel}
		</button>
	{/snippet}
</Dialog>
