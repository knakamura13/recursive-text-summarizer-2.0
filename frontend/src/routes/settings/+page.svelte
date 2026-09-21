<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$lib/api/client';

	let ollamaHost = $state('http://localhost:11434');
	let model = $state('');
	let healthMessage = $state('');
	let models = $state<string[]>([]);
	let error = $state<string | null>(null);

	onMount(async () => {
		try {
			const settings = await api.getSettings();
			ollamaHost = settings.ollama_host;
			model = settings.defaults.model;
		} catch (loadError) {
			error = loadError instanceof Error ? loadError.message : 'Failed to load settings';
		}
	});

	async function save() {
		await api.updateSettings({
			ollama_host: ollamaHost,
			defaults: { model, target_words: 300, strategy: 'auto', verify: true }
		});
	}

	async function testConnection() {
		const health = await api.ollamaHealth();
		healthMessage = health.message;
		const response = await api.ollamaModels();
		models = response.models.map((item) => item.name);
	}
</script>

<section class="settings-page">
	<h2>Settings</h2>
	{#if error}
		<p class="error">{error}</p>
	{/if}
	<label>
		Ollama host
		<input bind:value={ollamaHost} />
	</label>
	<label>
		Default model
		<input bind:value={model} placeholder="Required before first run" />
	</label>
	<div class="actions">
		<button type="button" onclick={testConnection}>Test connection</button>
		<button type="button" class="primary" onclick={save}>Save defaults</button>
	</div>
	{#if healthMessage}
		<p aria-live="polite">{healthMessage}</p>
	{/if}
	{#if models.length}
		<ul>
			{#each models as installed}
				<li>{installed}</li>
			{/each}
		</ul>
	{/if}
</section>

<style>
	.settings-page {
		max-width: 40rem;
		padding: var(--space-8);
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
	}

	label {
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}

	input {
		padding: var(--space-2) var(--space-3);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
	}

	.actions {
		display: flex;
		gap: var(--space-3);
	}

	.primary {
		background: var(--color-sage);
		color: white;
		border: none;
		border-radius: var(--radius-sm);
		padding: var(--space-2) var(--space-4);
	}

	.error {
		color: var(--color-danger);
	}
</style>
