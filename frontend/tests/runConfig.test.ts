import { describe, expect, it } from 'vitest';
import type { RunConfig } from '$lib/api/types';
import { runConfigPatch, validateRunConfig } from '$lib/runConfig';

function config(overrides: Partial<RunConfig> = {}): RunConfig {
	return {
		provider: 'ollama',
		model: 'llama3.1:8b',
		openrouter_upstream: '',
		target_words: 300,
		strategy: 'auto',
		verify: true,
		max_repair_passes: 1,
		citations: true,
		context_window: null,
		max_output_tokens: 1024,
		safety_margin_tokens: 256,
		safety_margin_fraction: 0.02,
		chunk_tokens: null,
		overlap_tokens: 0,
		max_merge_children: null,
		max_concurrency: 1,
		timeout_seconds: 600,
		max_retries: 5,
		strict_numbers: false,
		strict_names: false,
		preserve_sections: false,
		...overrides
	};
}

describe('validateRunConfig for OpenRouter', () => {
	it('accepts a vendor-qualified model with and without an upstream', () => {
		for (const upstream of ['', 'deepinfra', 'deepinfra/turbo']) {
			const errors = validateRunConfig(
				config({ provider: 'openrouter', model: 'meta-llama/llama-3.3-70b-instruct', openrouter_upstream: upstream }),
				{ requireModel: true }
			);
			expect(errors).toEqual({});
		}
	});

	it('asks for a typed model name rather than a choice', () => {
		const errors = validateRunConfig(config({ provider: 'openrouter', model: ' ' }), {
			requireModel: true
		});
		expect(errors.model).toBe('Enter a model name such as vendor/model.');
		expect(validateRunConfig(config({ model: '' }), { requireModel: true }).model).toBe(
			'Choose a model.'
		);
	});

	it.each(['two words', 'a/b/c', '-leading', 'x'.repeat(65)])('rejects the upstream %j', (upstream) => {
		const errors = validateRunConfig(
			config({ provider: 'openrouter', model: 'v/m', openrouter_upstream: upstream })
		);
		expect(errors.openrouter_upstream).toMatch(/provider name/);
	});

	it('rejects an upstream on an Ollama Run', () => {
		const errors = validateRunConfig(config({ openrouter_upstream: 'deepinfra' }));
		expect(errors.openrouter_upstream).toMatch(/Only OpenRouter/);
	});
});

describe('runConfigPatch for the provider fields', () => {
	it('sends the provider, model and cleared upstream together', () => {
		const saved = config({ provider: 'openrouter', model: 'v/m', openrouter_upstream: 'deepinfra' });
		const edited = config({ provider: 'ollama', model: 'llama3.1:8b', openrouter_upstream: '' });
		expect(runConfigPatch(saved, edited)).toEqual({
			provider: 'ollama',
			model: 'llama3.1:8b',
			openrouter_upstream: ''
		});
	});

	it('is null when nothing changed', () => {
		expect(runConfigPatch(config(), config())).toBeNull();
	});
});
