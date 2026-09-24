<script lang="ts">
	import type { OllamaModel, RunConfig } from '$lib/api/types';
	import type { NumericRunConfigField, RunConfigErrors } from '$lib/runConfig';

	// Edits a RunConfig in place: model, target words, strategy, verification,
	// and an Advanced section. Empty "automatic" fields hold null.
	let {
		config = $bindable(),
		models = [],
		errors = {},
		disabled = false,
		idPrefix
	}: {
		config: RunConfig;
		/** Installed models for the select; a configured model that is missing stays selectable. */
		models?: OllamaModel[];
		errors?: RunConfigErrors;
		disabled?: boolean;
		idPrefix: string;
	} = $props();

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

	const installed = $derived(models.some((model) => model.name === config.model));
	const advancedInvalid = $derived(ADVANCED.some(({ field }) => errors[field]));

	function describedBy(field: keyof RunConfigErrors, hasHint: boolean): string | undefined {
		const ids: string[] = [];
		if (hasHint) ids.push(`${idPrefix}-${field}-hint`);
		if (errors[field]) ids.push(`${idPrefix}-${field}-error`);
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
			<label for="{idPrefix}-model">Model</label>
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
				{#each models as model (model.name)}
					<option value={model.name}>
						{model.name}{model.parameter_size ? ` · ${model.parameter_size}` : ''}
					</option>
				{/each}
			</select>
			{@render error('model')}
		</div>

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

		<label class="checkbox-field">
			<input type="checkbox" bind:checked={config.verify} {disabled} />
			<span>Verify every summary sentence against the source</span>
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
					<p class="field-hint" id="{idPrefix}-{item.field}-hint">{item.hint}</p>
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
