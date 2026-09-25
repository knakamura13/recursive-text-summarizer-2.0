<script lang="ts">
	import { api, errorMessage, isAbortError } from '$lib/api/client';
	import type { ExportFormat } from '$lib/api/types';
	import Dialog from '$lib/components/common/Dialog.svelte';
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import Tabs, { tabId, tabPanelId } from '$lib/components/common/Tabs.svelte';
	import { toasts } from '$lib/stores/toasts.svelte';

	let { open = $bindable(false), runId }: { open?: boolean; runId: string } = $props();

	const FORMATS: { id: ExportFormat; label: string; hint: string }[] = [
		{ id: 'txt', label: 'Text', hint: 'The summary as plain text.' },
		{ id: 'md', label: 'Markdown', hint: 'The summary with page-cited footnotes for each sentence.' },
		{ id: 'json', label: 'Audit JSON', hint: 'Every step, claim, and verdict behind the summary.' }
	];

	let format = $state<ExportFormat>('md');
	let content = $state<string | null>(null);
	let error = $state.raw<unknown>(null);
	let reload = $state(0);

	const current = $derived(FORMATS.find((item) => item.id === format) ?? FORMATS[0]);

	$effect(() => {
		if (!open) return;
		void reload;
		const controller = new AbortController();
		content = null;
		error = null;
		api.getExport(runId, format, controller.signal).then(
			(text) => (content = format === 'json' ? prettyJson(text) : text),
			(failure: unknown) => {
				if (!isAbortError(failure)) error = failure;
			}
		);
		return () => controller.abort();
	});

	function prettyJson(text: string): string {
		try {
			return JSON.stringify(JSON.parse(text), null, 2);
		} catch {
			return text;
		}
	}

	async function copy() {
		if (content === null) return;
		try {
			await navigator.clipboard.writeText(content);
			toasts.success(`${current.label} copied.`);
		} catch (failure) {
			toasts.error(`Copying is not available here: ${errorMessage(failure)}`);
		}
	}
</script>

<Dialog bind:open title="Export the summary" size="lg">
	<Tabs
		items={FORMATS.map(({ id, label }) => ({ id, label }))}
		bind:active={() => format, (id) => (format = id as ExportFormat)}
		label="Export format"
		idPrefix="export"
	/>
	<div class="panel" role="tabpanel" id={tabPanelId('export', format)} aria-labelledby={tabId('export', format)}>
		<p class="hint">{current.hint}</p>
		{#if error}
			<ErrorBanner {error} title="Could not load the export" onretry={() => (reload += 1)} />
		{:else if content === null}
			<p class="loading"><Spinner size="sm" label="" /> Loading…</p>
		{:else}
			<pre class="preview" class:mono={format === 'json'} data-testid="export-preview">{content}</pre>
		{/if}
	</div>

	{#snippet footer()}
		<button type="button" class="button" disabled={content === null} onclick={copy}>Copy</button>
		<a class="button primary" href={api.exportUrl(runId, format)} download>Download {current.label}</a>
	{/snippet}
</Dialog>

<style>
	.panel {
		display: flex;
		flex-direction: column;
		gap: var(--space-3);
		padding-top: var(--space-3);
	}

	.hint {
		font-size: 0.875rem;
		color: var(--color-text-muted);
	}

	.loading {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		color: var(--color-text-muted);
	}

	.preview {
		margin: 0;
		max-height: min(55vh, 32rem);
		overflow: auto;
		padding: var(--space-4);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		background: var(--color-bg);
		font-family: var(--font-serif);
		font-size: 0.95rem;
		line-height: 1.6;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}

	.preview.mono {
		font-family: var(--font-mono);
		font-size: 0.8rem;
		line-height: 1.5;
		white-space: pre;
		overflow-wrap: normal;
	}
</style>
