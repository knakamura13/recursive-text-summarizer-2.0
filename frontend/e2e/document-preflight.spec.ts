import { expect, test } from '@playwright/test';
import { apiError, makeDocument, MockApi } from './support/mockApi';
import { SMALL_SOURCE } from './support/workspaceFixtures';

const DOC = 'doc-preflight';

test('a failed preflight shows the server message and keeps Start disabled', async ({ page }) => {
	const mock = await MockApi.install(page, {
		documents: [makeDocument({ document_id: DOC, title: 'Minutes', char_count: SMALL_SOURCE.length })],
		sources: { [DOC]: SMALL_SOURCE }
	});
	mock.on('POST', '/preflight', () =>
		apiError(503, 'ollama_unreachable', 'Ollama is not reachable at this host.')
	);

	await page.goto(`/documents/${DOC}`);
	await expect(page.getByRole('alert').filter({ hasText: 'Ollama is not reachable at this host.' })).toBeVisible();
	await expect(page.getByRole('button', { name: 'Start Run' })).toBeDisabled();
	await expect(page.locator('#start-reasons')).toContainText('The configuration check failed');
	expect(mock.calls('POST', '/runs')).toHaveLength(0);
});
