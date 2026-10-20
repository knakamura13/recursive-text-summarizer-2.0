<script lang="ts">
	import type { OllamaModel, ProviderName, RunConfig } from '$lib/api/types';
	import type { NumericRunConfigField, RunConfigErrors } from '$lib/runConfig';
	import { canSummarize } from '$lib/stores/models.svelte';

	// Edits a RunConfig in place: provider, model, target words, strategy,
	// verification, and an Advanced section. Empty "automatic" fields hold null.
	let {
		config = $bindable(),
		models = [],
		errors = {},
		disabled = false,
		openrouterKeySet = null,
		idPrefix
	}: {
		config: RunConfig;
		/** Installed models for the select; a configured model that is missing stays selectable. */
		models?: OllamaModel[];
		errors?: RunConfigErrors;
		disabled?: boolean;
		/** Whether the server has an OpenRouter key; null while unknown, so nothing is warned early. */
		openrouterKeySet?: boolean | null;
		idPrefix: string;
	} = $props();

	const openrouter = $derived(config.provider === 'openrouter');

	// A model name belongs to one provider, and an upstream pin only to
	// OpenRouter, so changing the provider swaps both for what was last entered
	// for the other provider (empty at first) instead of carrying them across.
	const remembered: Record<ProviderName, { model: string; upstream: string }> = {
		ollama: { model: '', upstream: '' },
		openrouter: { model: '', upstream: '' }
	};

	function setProvider(provider: ProviderName) {
		if (config.provider === provider) return;
		remembered[config.provider] = { model: config.model, upstream: config.openrouter_upstream };
		config.provider = provider;
		config.model = remembered[provider].model;
		config.openrouter_upstream = provider === 'openrouter' ? remembered[provider].upstream : '';
	}

	interface AdvancedField {
		field: NumericRunConfigField;
		label: string;
		hint: string;
		step?: string;
		placeholder?: string;
	}

	const ADVANCED: AdvancedField[] = [
		{
			field: 'context_window',
			label: 'Context window (tokens)',
			hint: 'Empty: use the model’s own window.',
			placeholder: 'Automatic'
		},
		{
			field: 'max_output_tokens',
			label: 'Max output tokens',
			hint: 'Longest answer per model call.'
		},
		{
			field: 'max_concurrency',
			label: 'Concurrent model calls',
			hint: 'Calls sent to Ollama at the same time.'
		},
		{
			field: 'timeout_seconds',
			label: 'Timeout (seconds)',
			hint: 'How long one model call may take.',
			step: 'any'
		},
		{
			field: 'max_retries',
			label: 'Retries per call',
			hint: 'For connection errors and timeouts.'
		},
		{
			field: 'max_repair_passes',
			label: 'Repair passes',
			hint: 'Rewrites of summary sentences that fail verification.'
		},
		{
			field: 'chunk_tokens',
			label: 'Segment size (tokens)',
			hint: 'Empty: fit segments to the context window.',
			placeholder: 'Automatic'
		},
		{
			field: 'max_merge_children',
			label: 'Max summaries per merge',
			hint: 'Empty: as many as fit the context window.',
			placeholder: 'Automatic'
		}
	];

	function hintFor(item: AdvancedField): string {
		if (!openrouter) return item.hint;
		if (item.field === 'context_window') {
			return 'Empty: use the window OpenRouter lists for the model.';
		}
		if (item.field === 'max_concurrency') return 'Calls sent to OpenRouter at the same time.';
		return item.hint;
	}

	const installed = $derived(models.some((model) => model.name === config.model));
	const advancedInvalid = $derived(ADVANCED.some(({ field }) => errors[field]));

	const keyMissing = $derived(openrouter && openrouterKeySet === false);

	function describedBy(field: keyof RunConfigErrors, hasHint: boolean): string | undefined {
		const ids: string[] = [];
		if (hasHint) ids.push(`${idPrefix}-${field}-hint`);
		if (errors[field]) ids.push(`${idPrefix}-${field}-error`);
		if (field === 'model' && keyMissing) ids.push(`${idPrefix}-openrouter-key`);
		return ids.length > 0 ? ids.join(' ') : undefined;
	}
</script>

{#snippet error(field: keyof RunConfigErrors)}
	{#if errors[field]}
		<p class="field-error" id="{idPrefix}-{field}-error">{errors[field]}</p>
	{/if}
{/snippet}

<div class="run-config">
	<div class="basic">
		<div class="field">
			<label for="{idPrefix}-provider">Provider</label>
			<select
				id="{idPrefix}-provider"
				value={config.provider}
				onchange={(event) => setProvider(event.currentTarget.value as ProviderName)}
				{disabled}
				aria-describedby="{idPrefix}-provider-hint"
			>
				<option value="ollama">Ollama</option>
				<option value="openrouter">OpenRouter (hosted)</option>
			</select>
			<p class="field-hint" id="{idPrefix}-provider-hint">
				{openrouter
					? 'The Document’s text is sent to OpenRouter, which bills each request.'
					: 'Runs on the Ollama host set in Settings.'}
			</p>
			{#if keyMissing}
				<p class="field-error" id="{idPrefix}-openrouter-key" role="alert">
					The server has no OpenRouter API key. Set OPENROUTER_API_KEY in the environment it runs
					in, then restart it.
				</p>
			{/if}
		</div>

		<div class="field" class:span={openrouter}>
			<label for="{idPrefix}-model">Model</label>
			{#if openrouter}
				<input
					id="{idPrefix}-model"
					type="text"
					autocomplete="off"
					spellcheck="false"
					placeholder="vendor/model"
					bind:value={config.model}
					{disabled}
					aria-invalid={errors.model ? 'true' : undefined}
					aria-describedby={describedBy('model', true)}
				/>
				<p class="field-hint" id="{idPrefix}-model-hint">
					An OpenRouter model ID, for example meta-llama/llama-3.3-70b-instruct.
				</p>
			{:else}
				<select
					id="{idPrefix}-model"
					bind:value={config.model}
					{disabled}
					aria-invalid={errors.model ? 'true' : undefined}
					aria-describedby={describedBy('model', false)}
				>
					<option value="">Choose a model</option>
					{#if config.model && !installed}
						<option value={config.model}>{config.model} (not installed)</option>
					{/if}
					{#each models.filter(canSummarize) as model (model.name)}
						<option value={model.name}>
							{model.name}{model.parameter_size ? ` · ${model.parameter_size}` : ''}
						</option>
					{/each}
				</select>
			{/if}
			{@render error('model')}
		</div>

		{#if openrouter}
			<div class="field">
				<label for="{idPrefix}-openrouter_upstream">Upstream provider (optional)</label>
				<input
					id="{idPrefix}-openrouter_upstream"
					type="text"
					autocomplete="off"
					spellcheck="false"
					placeholder="Any"
					bind:value={config.openrouter_upstream}
					{disabled}
					aria-invalid={errors.openrouter_upstream ? 'true' : undefined}
					aria-describedby={describedBy('openrouter_upstream', true)}
				/>
				<p class="field-hint" id="{idPrefix}-openrouter_upstream-hint">
					Send every request only to this provider, for example deepinfra, and never fall back.
				</p>
				{@render error('openrouter_upstream')}
			</div>
		{/if}

		<div class="field">
			<label for="{idPrefix}-target_words">Target length (words)</label>
			<input
				id="{idPrefix}-target_words"
				type="number"
				inputmode="numeric"
				min="25"
				max="5000"
				step="1"
				bind:value={config.target_words}
				{disabled}
				aria-invalid={errors.target_words ? 'true' : undefined}
				aria-describedby={describedBy('target_words', false)}
			/>
			{@render error('target_words')}
		</div>

		<div class="field">
			<label for="{idPrefix}-strategy">Strategy</label>
			<select
				id="{idPrefix}-strategy"
				bind:value={config.strategy}
				{disabled}
				aria-describedby="{idPrefix}-strategy-hint"
			>
				<option value="auto">Automatic</option>
				<option value="direct">Direct (whole Document in one call)</option>
				<option value="hierarchical">Hierarchical (segments, then merges)</option>
			</select>
			<p class="field-hint" id="{idPrefix}-strategy-hint">
				Automatic picks Direct when the Document fits the context window.
			</p>
		</div>

		<label class="checkbox-field span">
			<input type="checkbox" bind:checked={config.verify} {disabled} />
			<span>Verify every summary sentence against the source</span>
		</label>
		<label class="checkbox-field span">
			<input type="checkbox" bind:checked={config.strict_numbers} {disabled} />
			<span>Keep numbers exactly as written. Off, a summary may round or omit them.</span>
		</label>
		<label class="checkbox-field span">
			<input type="checkbox" bind:checked={config.strict_names} {disabled} />
			<span>Keep multi-word names exactly as written. Off, a summary may shorten them.</span>
		</label>
		<label class="checkbox-field span">
			<input type="checkbox" bind:checked={config.preserve_sections} {disabled} />
			<span>
				Write verified prose for each section of a document with headings, beside the final
				summary. This adds requests; a document with no headings runs as usual.
			</span>
		</label>
	</div>

	<details class="advanced" open={advancedInvalid || undefined}>
		<summary>Advanced</summary>
		<div class="grid">
			{#each ADVANCED as item (item.field)}
				<div class="field">
					<label for="{idPrefix}-{item.field}">{item.label}</label>
					<input
						id="{idPrefix}-{item.field}"
						type="number"
						inputmode={item.step ? 'decimal' : 'numeric'}
						step={item.step ?? '1'}
						placeholder={item.placeholder}
						bind:value={config[item.field]}
						{disabled}
						aria-invalid={errors[item.field] ? 'true' : undefined}
						aria-describedby={describedBy(item.field, true)}
					/>
					<p class="field-hint" id="{idPrefix}-{item.field}-hint">{hintFor(item)}</p>
					{@render error(item.field)}
				</div>
			{/each}
		</div>
	</details>
</div>

<style>
	.run-config {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
	}

	.basic {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 14rem), 1fr));
		gap: var(--space-4);
		align-items: start;
	}

	.checkbox-field.span,
	.field.span {
		grid-column: 1 / -1;
	}

	.advanced {
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		padding: 0 var(--space-4);
	}

	summary {
		display: flex;
		align-items: center;
		min-height: var(--touch-target);
		font-weight: 600;
		cursor: pointer;
	}

	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 14rem), 1fr));
		gap: var(--space-4);
		padding-bottom: var(--space-4);
	}
</style>
