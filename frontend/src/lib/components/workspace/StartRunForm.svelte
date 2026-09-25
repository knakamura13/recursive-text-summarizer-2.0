<script lang="ts">
	import { onMount } from 'svelte';
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import RunConfigFields from '$lib/components/common/RunConfigFields.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import { api, errorMessage, isAbortError } from '$lib/api/client';
	import type { DocumentDetail, Preflight, Run, RunConfig } from '$lib/api/types';
	import { formatCount, formatEta, stageLabel } from '$lib/format';
	import { fieldErrorsFrom, validateRunConfig, type RunConfigErrors } from '$lib/runConfig';
	import { activity } from '$lib/stores/activity.svelte';
	import { canSummarize, models } from '$lib/stores/models.svelte';
	import { settings } from '$lib/stores/settings.svelte';

	interface Props {
		doc: DocumentDetail;
		onstarted: (run: Run) => void;
		/** Shown when the Document has Runs, to go back to the latest one. */
		oncancel?: () => void;
	}

	let { doc, onstarted, oncancel }: Props = $props();

	const PREFLIGHT_DELAY_MS = 350;
	const WINDOW_SOURCE: Record<NonNullable<Preflight['context_window_source']>, string> = {
		configured: 'as set in Advanced',
		model: 'reported by the model',
		assumed: 'assumed; the model did not report its window'
	};

	let config = $state<RunConfig | null>(settings.value ? { ...settings.value.defaults } : null);
	let settingsError = $state<unknown>(null);
	let preflight = $state.raw<Preflight | null>(null);
	let preflightError = $state<unknown>(null);
	/** The config (as JSON) that `preflight`/`preflightError` answer. */
	let checkedKey = $state<string | null>(null);
	let submitting = $state(false);
	let submitError = $state<unknown>(null);
	let serverFieldErrors = $state<Record<string, string>>({});

	function loadSettings() {
		settingsError = null;
		settings.load().then(
			(value) => {
				config ??= { ...value.defaults };
			},
			(error: unknown) => (settingsError = error)
		);
	}

	onMount(() => {
		if (config === null) loadSettings();
		if (models.list.length === 0) void models.refresh();
	});

	// Preselect the only sensible choice when the settings name no model.
	$effect(() => {
		if (config && !config.model) {
			const first = models.list.find(canSummarize);
			if (first) config.model = first.name;
		}
	});

	const localErrors = $derived(config ? validateRunConfig(config, { requireModel: true }) : {});
	const fieldErrors = $derived<RunConfigErrors>({ ...serverFieldErrors, ...localErrors });
	const valid = $derived(Object.keys(localErrors).length === 0);
	const configKey = $derived(config ? JSON.stringify(config) : null);
	const ready = $derived(doc.import_state === 'ready');
	const checked = $derived(configKey !== null && checkedKey === configKey);
	const activeRun = $derived(activity.activeRun);

	$effect(() => {
		const key = configKey;
		if (key === null || !ready || !valid) return;
		const snapshot = JSON.parse(key) as RunConfig;
		const controller = new AbortController();
		const timer = setTimeout(() => {
			api.preflight(doc.document_id, snapshot, controller.signal).then(
				(result) => {
					preflight = result;
					preflightError = null;
					checkedKey = key;
				},
				(error: unknown) => {
					if (isAbortError(error)) return;
					preflight = null;
					preflightError = error;
					checkedKey = key;
				}
			);
		}, PREFLIGHT_DELAY_MS);
		return () => {
			clearTimeout(timer);
			controller.abort();
		};
	});

	$effect(() => {
		// Server-side field errors describe the submitted values only.
		void configKey;
		serverFieldErrors = {};
	});

	type Reason = { text: string; pending?: boolean } | { run: Run };

	const reasons = $derived.by((): Reason[] => {
		const list: Reason[] = [];
		if (doc.import_state === 'importing') list.push({ text: 'The Document is still importing.' });
		if (doc.import_state === 'failed') list.push({ text: 'The Import failed, so there is no text to summarize.' });
		if (config === null) return [...list, { text: 'Loading the default settings…', pending: true }];
		if (!config.model) {
			list.push({
				text:
					models.list.length > 0
						? 'Choose a model.'
						: models.error
							? `No model list: ${models.error}`
							: models.loading
								? 'Loading installed models…'
								: 'No models are installed in Ollama. Install one with “ollama pull <model>”.',
				pending: models.loading
			});
		}
		if (activeRun) list.push({ run: activeRun });
		if (Object.keys(localErrors).some((field) => field !== 'model')) list.push({ text: 'Fix the settings marked below.' });
		if (ready && valid && !activeRun) {
			if (!checked) list.push({ text: 'Checking the configuration…', pending: true });
			else if (preflightError) {
				list.push({ text: 'The configuration check failed: ' + errorMessage(preflightError) });
			}
			else if (preflight && !preflight.ok) {
				for (const error of preflight.errors) list.push({ text: error.message });
				if (preflight.errors.length === 0) list.push({ text: 'The configuration check did not pass.' });
			}
		}
		if (submitting) list.push({ text: 'Starting…', pending: true });
		return list;
	});

	function newIdempotencyKey(): string {
		if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
		return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, '0')).join('');
	}

	async function start(event: SubmitEvent) {
		event.preventDefault();
		if (reasons.length > 0 || config === null) return;
		submitting = true;
		submitError = null;
		try {
			const run = await api.createRun(doc.document_id, $state.snapshot(config), newIdempotencyKey());
			onstarted(run);
		} catch (error) {
			submitError = error;
			serverFieldErrors = fieldErrorsFrom(error);
		} finally {
			submitting = false;
		}
	}
</script>

<form class="start-form" onsubmit={start} aria-labelledby="start-run-title" novalidate>
	<div class="head">
		<h2 id="start-run-title">Start a Run</h2>
		{#if oncancel}
			<button type="button" class="button ghost small" onclick={oncancel}>Back to the latest Run</button>
		{/if}
	</div>

	{#if settingsError}
		<ErrorBanner error={settingsError} title="Could not load the default settings" onretry={loadSettings} />
	{/if}
	{#if models.error}
		<ErrorBanner error={models.error} title="Could not list the installed models" onretry={() => models.refresh()} />
	{/if}

	{#if config}
		<RunConfigFields bind:config models={models.list} errors={fieldErrors} disabled={submitting} idPrefix="start" />
	{/if}

	<section class="preflight" aria-live="polite" aria-label="Configuration check">
		{#if !ready}
			<!-- The reasons list explains why nothing is checked. -->
		{:else if !valid}
			<p class="muted">The configuration is checked once every setting is valid.</p>
		{:else if !checked}
			<p class="muted"><Spinner size="sm" label="" /> Checking the configuration…</p>
		{:else if preflightError}
			<ErrorBanner error={preflightError} title="The configuration check failed" />
		{:else if preflight}
			<dl class="facts">
				{#if preflight.selected_strategy}
					<div><dt>Strategy</dt><dd>{preflight.selected_strategy === 'direct' ? 'Direct (one call)' : 'Hierarchical'}</dd></div>
				{/if}
				{#if preflight.context_window_tokens !== null}
					<div>
						<dt>Context window</dt>
						<dd>
							{formatCount(preflight.context_window_tokens)} tokens{#if preflight.context_window_source},
								{WINDOW_SOURCE[preflight.context_window_source]}{/if}
						</dd>
					</div>
				{/if}
				{#if preflight.document_tokens !== null}
					<div><dt>Document</dt><dd>about {formatCount(preflight.document_tokens)} tokens</dd></div>
				{/if}
				{#if preflight.estimated_leaf_count !== null}
					<div><dt>Segments</dt><dd>about {formatCount(preflight.estimated_leaf_count)}</dd></div>
				{/if}
				{#if preflight.estimated_model_calls !== null}
					<div><dt>Model calls</dt><dd>about {formatCount(preflight.estimated_model_calls)}</dd></div>
				{/if}
			</dl>
			{#if preflight.warnings.length > 0}
				<ul class="notices">
					{#each preflight.warnings as warning (warning.code + warning.message)}
						<li class="warning">{warning.message}</li>
					{/each}
				</ul>
			{/if}
		{/if}
	</section>

	{#if submitError}
		<ErrorBanner error={submitError} title="The Run did not start" ondismiss={() => (submitError = null)} />
	{/if}

	<div class="submit">
		<button type="submit" class="button primary" disabled={reasons.length > 0} aria-describedby="start-reasons">
			{submitting ? 'Starting…' : 'Start Run'}
		</button>
		<ul class="reasons" id="start-reasons" aria-live="polite">
			{#each reasons as reason, i (i)}
				{#if 'run' in reason}
					<li class="blocked">
						Another Run is active:
						<a href="/documents/{encodeURIComponent(reason.run.document_id)}?run={encodeURIComponent(reason.run.run_id)}"
							>{reason.run.document_title}</a
						>{#if reason.run.progress?.stage}{` · ${stageLabel(reason.run.progress.stage)}`}{/if}{#if reason.run.progress?.eta_seconds != null}{` · ${formatEta(reason.run.progress.eta_seconds)}`}{/if}.
						Only one Run can be active at a time.
					</li>
				{:else}
					<li class:pending={reason.pending}>{reason.text}</li>
				{/if}
			{/each}
		</ul>
	</div>
</form>

<style>
	.start-form {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
	}

	.head {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-2);
	}

	h2 {
		margin: 0;
		font-size: 1.1rem;
	}

	.preflight:empty {
		display: none;
	}

	.muted {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin: 0;
		color: var(--color-text-muted);
		font-size: 0.9rem;
	}

	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 11rem), 1fr));
		gap: var(--space-2) var(--space-4);
		margin: 0;
	}

	.facts dt {
		font-size: 0.75rem;
		color: var(--color-text-muted);
	}

	.facts dd {
		margin: 0;
	}

	.notices {
		margin: var(--space-2) 0 0;
		padding: 0;
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
	}

	.warning {
		padding: var(--space-2);
		border-radius: var(--radius-sm);
		background: var(--color-amber-soft);
		font-size: 0.9rem;
	}

	.submit {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--space-2);
	}

	.reasons {
		margin: 0;
		padding-left: 1.1rem;
		font-size: 0.9rem;
		color: var(--color-text-muted);
	}

	.reasons:empty {
		display: none;
	}

	.reasons .pending {
		font-style: italic;
	}

	.blocked {
		color: var(--color-text);
	}
</style>
