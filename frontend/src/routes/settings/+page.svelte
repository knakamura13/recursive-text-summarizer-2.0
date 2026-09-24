<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { api } from '$lib/api/client';
	import type { OllamaHealth, RunConfig } from '$lib/api/types';
	import EmptyState from '$lib/components/common/EmptyState.svelte';
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import RunConfigFields from '$lib/components/common/RunConfigFields.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import { formatBytes, formatRelativeTime } from '$lib/format';
	import { fieldErrorsFrom, runConfigPatch, validateRunConfig } from '$lib/runConfig';
	import { models } from '$lib/stores/models.svelte';
	import { settings } from '$lib/stores/settings.svelte';
	import { toasts } from '$lib/stores/toasts.svelte';

	let loadError = $state.raw<unknown>(null);

	let host = $state('');
	let hostError = $state<string | null>(null);
	let testing = $state(false);
	let savingHost = $state(false);
	let health = $state.raw<OllamaHealth | null>(null);
	let healthError = $state.raw<unknown>(null);

	let draft = $state<RunConfig | null>(null);
	let saving = $state(false);
	let saveError = $state.raw<unknown>(null);
	let serverErrors = $state.raw<Record<string, string>>({});
	let savedOnce = $state(false);
	let defaultsForm: HTMLFormElement | undefined = $state();

	const saved = $derived(settings.value);
	const hostChanged = $derived(saved !== null && host.trim() !== saved.ollama_host);
	const clientErrors = $derived(draft ? validateRunConfig(draft) : {});
	const errors = $derived({ ...serverErrors, ...clientErrors });
	const patch = $derived(saved && draft ? runConfigPatch(saved.defaults, draft) : null);
	const invalid = $derived(Object.keys(clientErrors).length > 0);

	async function load() {
		loadError = null;
		try {
			const value = await settings.load();
			host = value.ollama_host;
			draft = { ...value.defaults };
		} catch (error) {
			loadError = error;
			return;
		}
		void checkHealth();
		void models.refresh();
	}

	onMount(() => {
		void load();
	});

	async function checkHealth(hostToTest?: string) {
		healthError = null;
		try {
			health = await api.ollamaHealth(hostToTest);
		} catch (error) {
			health = null;
			healthError = error;
		}
	}

	function hostProblem(value: string): string | null {
		try {
			const url = new URL(value.trim());
			return url.protocol === 'http:' || url.protocol === 'https:'
				? null
				: 'Use an http:// or https:// address.';
		} catch {
			return 'Enter a URL such as http://localhost:11434.';
		}
	}

	async function testConnection(event: SubmitEvent) {
		event.preventDefault();
		hostError = hostProblem(host);
		if (hostError) return;
		testing = true;
		health = null;
		healthError = null;
		await checkHealth(host.trim());
		testing = false;
	}

	async function saveHost() {
		hostError = hostProblem(host);
		if (hostError || !hostChanged) return;
		savingHost = true;
		healthError = null;
		try {
			const value = await settings.save({ ollama_host: host.trim() });
			host = value.ollama_host;
			toasts.success('Ollama address saved.');
			void models.refresh();
		} catch (error) {
			hostError = fieldErrorsFrom(error).ollama_host ?? null;
			if (!hostError) healthError = error;
		} finally {
			savingHost = false;
		}
	}

	async function saveDefaults(event: SubmitEvent) {
		event.preventDefault();
		if (!draft || !saved) return;
		if (invalid) {
			await tick();
			defaultsForm?.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus();
			return;
		}
		const update = runConfigPatch(saved.defaults, draft);
		if (!update) return;
		saving = true;
		saveError = null;
		serverErrors = {};
		try {
			const value = await settings.save({ defaults: update });
			draft = { ...value.defaults };
			savedOnce = true;
			toasts.success('Run defaults saved.');
		} catch (error) {
			saveError = error;
			serverErrors = fieldErrorsFrom(error);
		} finally {
			saving = false;
		}
	}
</script>

<svelte:head>
	<title>Settings · Recursive Summarizer</title>
</svelte:head>

<div class="settings">
	<h1>Settings</h1>

	{#if loadError}
		<ErrorBanner title="Settings could not be loaded" error={loadError} onretry={load} />
	{:else if !draft}
		<div class="loading"><Spinner size="lg" label="Loading settings" /></div>
	{:else}
		<section aria-labelledby="ollama-heading">
			<h2 id="ollama-heading">Ollama</h2>
			<form class="host-form" novalidate onsubmit={testConnection}>
				<div class="field">
					<label for="ollama-host">Ollama address</label>
					<input
						id="ollama-host"
						type="url"
						inputmode="url"
						autocomplete="off"
						spellcheck="false"
						placeholder="http://localhost:11434"
						bind:value={host}
						disabled={testing || savingHost}
						aria-invalid={hostError ? 'true' : undefined}
						aria-describedby={hostError ? 'ollama-host-error' : 'ollama-host-hint'}
					/>
					{#if hostError}
						<p class="field-error" id="ollama-host-error">{hostError}</p>
					{:else}
						<p class="field-hint" id="ollama-host-hint">Where the app sends model calls.</p>
					{/if}
				</div>
				<div class="host-actions">
					<button type="submit" class="button" disabled={testing || savingHost}>
						{#if testing}<Spinner size="sm" label="" />{/if}
						Test connection
					</button>
					{#if hostChanged}
						<button type="button" class="button primary" disabled={testing || savingHost} onclick={saveHost}>
							{#if savingHost}<Spinner size="sm" label="" />{/if}
							Save address
						</button>
					{/if}
				</div>
			</form>
			<div class="health" aria-live="polite">
				{#if testing}
					<p class="muted">Checking the connection…</p>
				{:else if healthError}
					<ErrorBanner error={healthError} />
				{:else if health}
					<p class={health.connected ? 'ok' : 'bad'}>
						{health.connected
							? `Connected to Ollama${health.version ? ` ${health.version}` : ''} at ${health.host}.`
							: health.message}
					</p>
				{/if}
			</div>
		</section>

		<section aria-labelledby="models-heading">
			<div class="section-head">
				<h2 id="models-heading">Installed models</h2>
				<button
					type="button"
					class="button small"
					disabled={models.loading}
					onclick={() => models.refresh()}
				>
					{#if models.loading}<Spinner size="sm" label="" />{/if}
					Refresh
				</button>
			</div>
			{#if models.error}
				<ErrorBanner error={models.error} onretry={() => models.refresh()} />
			{:else if models.list.length === 0 && !models.loading}
				<EmptyState
					headingLevel={3}
					title="No models installed"
					message="Install one with “ollama pull <model>”, then refresh."
				/>
			{/if}
			{#if models.list.length > 0}
				<ul class="models">
					{#each models.list as model (model.name)}
						<li>
							<div class="model-text">
								<span class="model-name">{model.name}</span>
								<span class="model-meta">
									{[
										model.parameter_size,
										model.family,
										model.quantization,
										model.size_bytes === null ? null : formatBytes(model.size_bytes),
										model.modified_at ? `updated ${formatRelativeTime(model.modified_at)}` : null
									]
										.filter(Boolean)
										.join(' · ')}
								</span>
							</div>
							{#if draft.model === model.name}
								<span class="default-tag">Default</span>
							{:else}
								<button
									type="button"
									class="button small"
									aria-label="Use {model.name} as the default model"
									onclick={() => draft && (draft.model = model.name)}>Use as default</button
								>
							{/if}
						</li>
					{/each}
				</ul>
			{/if}
		</section>

		<section aria-labelledby="defaults-heading">
			<h2 id="defaults-heading">Run defaults</h2>
			<p class="muted">New Runs start with these settings; each Run can change them.</p>
			<form bind:this={defaultsForm} class="defaults" novalidate onsubmit={saveDefaults}>
				<RunConfigFields
					bind:config={draft}
					models={models.list}
					{errors}
					disabled={saving}
					idPrefix="defaults"
				/>
				{#if saveError}
					<ErrorBanner title="The defaults were not saved" error={saveError} />
				{/if}
				<div class="form-actions">
					<p class="save-state" aria-live="polite">
						{#if patch}
							Unsaved changes
						{:else if savedOnce}
							All changes saved
						{/if}
					</p>
					<button
						type="button"
						class="button"
						disabled={saving || !patch}
						onclick={() => {
							if (saved) draft = { ...saved.defaults };
							serverErrors = {};
							saveError = null;
						}}>Discard changes</button
					>
					<button type="submit" class="button primary" disabled={saving || !patch}>
						{#if saving}<Spinner size="sm" label="" />{/if}
						Save defaults
					</button>
				</div>
			</form>
		</section>
	{/if}
</div>

<style>
	.settings {
		display: flex;
		flex-direction: column;
		gap: var(--space-6);
		max-width: 48rem;
		margin: 0 auto;
		padding: var(--space-6) var(--space-4) var(--space-8);
	}

	section {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
		padding: var(--space-5);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-lg);
		background: var(--color-bg);
	}

	.loading {
		display: flex;
		justify-content: center;
		padding: var(--space-8);
		color: var(--color-text-muted);
	}

	.host-form {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		gap: var(--space-3);
	}

	.host-form .field {
		flex: 1 1 18rem;
	}

	.host-form .button {
		margin-top: 1.625rem;
	}

	.host-actions {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-2);
	}

	.ok {
		color: var(--color-sage);
		font-weight: 500;
	}

	.bad {
		color: var(--color-danger);
		font-weight: 500;
	}

	.muted {
		color: var(--color-text-muted);
	}

	.section-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-3);
	}

	.models {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
	}

	.models li {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-2) var(--space-3);
		padding: var(--space-3) 0;
	}

	.models li + li {
		border-top: 1px solid var(--color-border);
	}

	.model-text {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}

	.model-name {
		font-weight: 600;
		font-family: var(--font-mono);
		font-size: 0.9375rem;
	}

	.model-meta {
		font-size: 0.875rem;
		color: var(--color-text-muted);
	}

	.default-tag {
		padding: 0.125rem 0.625rem;
		border-radius: 999px;
		background: var(--color-sage-soft);
		color: var(--color-sage);
		font-size: 0.8125rem;
		font-weight: 600;
	}

	.defaults {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
	}

	.form-actions {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: flex-end;
		gap: var(--space-3);
	}

	.save-state {
		margin-right: auto;
		color: var(--color-text-muted);
	}

	@media (max-width: 639px) {
		.settings {
			padding: var(--space-4) var(--space-3) var(--space-8);
		}

		section {
			padding: var(--space-4);
		}

		.host-form .button {
			margin-top: 0;
		}

		.host-actions {
			width: 100%;
		}

		.host-actions .button {
			flex: 1 1 auto;
		}
	}
</style>
