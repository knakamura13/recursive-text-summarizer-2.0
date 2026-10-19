import { expect, test } from '@playwright/test';
import { makeDocument, MockApi } from './support/mockApi';
import { SMALL_SOURCE } from './support/workspaceFixtures';

const DOC = 'doc-openrouter';
const MODEL = 'meta-llama/llama-3.3-70b-instruct';

function seed() {
	return {
		documents: [makeDocument({ document_id: DOC, title: 'Minutes', char_count: SMALL_SOURCE.length })],
		sources: { [DOC]: SMALL_SOURCE }
	};
}

test('choosing OpenRouter swaps the model list for a typed name and an upstream field', async ({ page }) => {
	const mock = await MockApi.install(page, seed());
	mock.on('POST', '/preflight', () => ({
		json: {
			ok: true,
			selected_strategy: 'direct',
			context_window_tokens: 131_072,
			context_window_source: 'model',
			provider: 'openrouter',
			token_counts_estimated: true,
			usable_input_capacity: 120_000,
			document_tokens: 600,
			estimated_leaf_count: 1,
			estimated_model_calls: 2,
			model_installed: null,
			errors: [],
			warnings: [
				{ code: 'token_counts_estimated', message: 'Token counts are estimates.', severity: 'warning' }
			]
		}
	}));

	await page.goto(`/documents/${DOC}`);
	await page.getByLabel('Provider', { exact: true }).selectOption('openrouter');

	const model = page.getByLabel('Model', { exact: true });
	await expect(model).toHaveJSProperty('tagName', 'INPUT');
	await expect(model).toHaveValue('');
	await expect(page.getByRole('button', { name: 'Start Run' })).toBeDisabled();
	await expect(page.locator('#start-reasons')).toContainText('vendor/model');

	await model.fill(MODEL);
	await page.getByLabel('Upstream provider (optional)').fill('deepinfra');

	await expect(page.getByText('listed by OpenRouter')).toBeVisible();
	await expect(page.getByText('(estimated)')).toBeVisible();
	await expect(page.getByText('Token counts are estimates.')).toBeVisible();
	await expect(page.getByRole('button', { name: 'Start Run' })).toBeEnabled();

	await page.getByRole('button', { name: 'Start Run' }).click();
	const [created] = mock.calls('POST', '/runs');
	expect(created.body).toMatchObject({
		config: { provider: 'openrouter', model: MODEL, openrouter_upstream: 'deepinfra' }
	});
	expect(JSON.stringify(created.body)).not.toMatch(/sk-or-/);
});

test('a server without a key says so and an invalid upstream blocks Start', async ({ page }) => {
	await MockApi.install(page, {
		...seed(),
		openrouter: { key_set: false, message: 'OpenRouter needs an API key.' }
	});

	await page.goto(`/documents/${DOC}`);
	await page.getByLabel('Provider', { exact: true }).selectOption('openrouter');

	await expect(page.getByRole('status').filter({ hasText: 'OPENROUTER_API_KEY' })).toBeVisible();

	await page.getByLabel('Model', { exact: true }).fill(MODEL);
	await page.getByLabel('Upstream provider (optional)').fill('two words');
	await expect(page.getByText('Use a provider name such as deepinfra')).toBeVisible();
	await expect(page.getByRole('button', { name: 'Start Run' })).toBeDisabled();
});

test('switching back to Ollama clears the model and the upstream', async ({ page }) => {
	await MockApi.install(page, seed());

	await page.goto(`/documents/${DOC}`);
	await page.getByLabel('Provider', { exact: true }).selectOption('openrouter');
	await page.getByLabel('Model', { exact: true }).fill(MODEL);
	await page.getByLabel('Upstream provider (optional)').fill('deepinfra');
	await page.getByLabel('Provider', { exact: true }).selectOption('ollama');

	await expect(page.getByLabel('Model', { exact: true })).toHaveJSProperty('tagName', 'SELECT');
	await expect(page.getByLabel('Upstream provider (optional)')).toHaveCount(0);
});

test('Settings reports whether the server has a key and never asks for it', async ({ page }) => {
	await MockApi.install(page, {
		openrouter: { key_set: false, message: 'OpenRouter needs an API key.' }
	});

	await page.goto('/settings');

	await expect(page.getByRole('heading', { name: 'OpenRouter' })).toBeVisible();
	await expect(page.getByText('OpenRouter needs an API key.')).toBeVisible();
	await expect(page.getByLabel(/api key/i)).toHaveCount(0);
});
