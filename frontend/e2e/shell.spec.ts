import { expect, test } from '@playwright/test';
import { apiError, makeDocument, MockApi } from './support/mockApi';

const UNSAVED_HOST = 'https://ollama.example:11434';

test('library searches metadata and imports a file while keeping the 320px layout usable', async ({ page }) => {
	await page.setViewportSize({ width: 320, height: 780 });
	const harbour = makeDocument({
		document_id: 'doc-harbour',
		title: 'Harbour minutes',
		filename: 'harbour-minutes.pdf',
		format: 'pdf',
		size_bytes: 12_345,
		page_count: 17
	});
	const budget = makeDocument({ document_id: 'doc-budget', title: 'Budget report' });
	const mock = await MockApi.install(page, { documents: [harbour, budget] });

	await page.goto('/');
	await mock.waitForStream();
	const library = page.getByRole('list', { name: 'Documents' });
	const harbourRow = library.getByRole('listitem').filter({ hasText: 'Harbour minutes' });
	await expect(harbourRow).toContainText('PDF');
	await expect(harbourRow).toContainText('12 KB');
	await expect(harbourRow).toContainText('17 pages');

	const search = page.getByRole('searchbox', { name: 'Search Documents' });
	await search.fill('budget');
	await expect(page.getByRole('link', { name: 'Budget report' })).toBeVisible();
	await expect(page.getByRole('link', { name: 'Harbour minutes' })).toHaveCount(0);
	await search.fill('');
	await expect(page.getByRole('link', { name: 'Harbour minutes' })).toBeVisible();
	const budgetRow = library.getByRole('listitem').filter({ hasText: 'Budget report' });
	await budgetRow.getByRole('button', { name: 'Delete “Budget report”' }).click();
	const confirmation = page.getByRole('dialog', { name: 'Delete “Budget report”?' });
	await expect(confirmation).toBeVisible();
	await confirmation.getByRole('button', { name: 'Delete', exact: true }).click();
	await expect(confirmation).toBeHidden();
	await expect(page.locator('#main')).toBeFocused();
	await expect(page.getByRole('link', { name: 'Budget report' })).toHaveCount(0);


	const importButton = page.getByRole('button', { name: 'Import files' });
	await importButton.click();
	const dialog = page.getByRole('dialog', { name: 'Import documents' });
	await expect(dialog).toBeVisible();
	await page.keyboard.press('Escape');
	await expect(dialog).toBeHidden();
	await expect
		.poll(() =>
			importButton.evaluate(
				(element) =>
					element === document.activeElement || document.activeElement?.getAttribute('id') === 'main'
			)
		)
		.toBe(true);

	await importButton.click();
	await expect(dialog).toBeVisible();
	const picker = dialog.locator('input[type="file"]');
	await expect(picker).toHaveAttribute('accept', /\.epub/);
	await expect(picker).toHaveAttribute('accept', /\.jpeg/);
	await expect(picker).toHaveAttribute('accept', /\.tiff/);
	await expect(dialog).toContainText('(.png, .jpg, .jpeg, .tif, .tiff) for OCR');
	await picker.setInputFiles({
		name: 'meeting-notes.txt',
		mimeType: 'text/plain',
		buffer: Buffer.from('A short set of meeting notes.')
	});
	await mock.waitForCall('POST', '/documents');
	await expect(dialog.getByText(/Waiting to import/)).toBeVisible();
	await dialog.getByRole('button', { name: 'Done' }).click();
	await expect
		.poll(() =>
			importButton.evaluate(
				(element) =>
					element === document.activeElement || document.activeElement?.getAttribute('id') === 'main'
			)
		)
		.toBe(true);
	await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
});


test('import queue shows the server reason for a rejected file', async ({ page }) => {
	const mock = await MockApi.install(page);
	mock.on('POST', '/documents', () =>
		apiError(415, 'unsupported_format', 'This file format is not supported by the Document importer.')
	);

	await page.goto('/');
	await mock.waitForStream();
	await page.getByRole('button', { name: 'Import files' }).first().click();
	const dialog = page.getByRole('dialog', { name: 'Import documents' });
	await dialog.locator('input[type="file"]').setInputFiles({
		name: 'notes.unknown',
		mimeType: 'application/octet-stream',
		buffer: Buffer.from('not a supported Document')
	});
	await mock.waitForCall('POST', '/documents');
	const item = dialog.getByRole('listitem').filter({ hasText: 'notes.unknown' });
	await expect(item.getByRole('alert')).toHaveText(
		'This file format is not supported by the Document importer.'
	);
	await expect(item.getByRole('button', { name: 'Try again' })).toHaveCount(0);
});

test('Settings tests an unsaved Ollama address without persisting it and keeps errors visible', async ({ page }) => {
	const mock = await MockApi.install(page);
	const testedHosts: string[] = [];

	let rejectedDraftOnce = false;
	mock.on('GET', '/ollama/health', ({ query }, api) => {
		const host = query.get('host') ?? api.settings.ollama_host;
		testedHosts.push(host);
		if (host === UNSAVED_HOST && !rejectedDraftOnce) {
			rejectedDraftOnce = true;
			return apiError(503, 'ollama_unreachable', 'The unsaved Ollama address cannot be reached.', null, true);
		}
		return {
			json: { ...api.ollama, host, message: 'Connected to the selected address.' }
		};
	});

	await page.goto('/settings');
	await mock.waitForStream();
	const input = page.getByRole('textbox', { name: 'Ollama address' });
	await expect(input).toHaveValue('http://localhost:11434');
	await input.fill(UNSAVED_HOST);
	await page.getByRole('button', { name: 'Test connection' }).click();
	await expect(page.getByRole('alert')).toContainText('The unsaved Ollama address cannot be reached.');
	await expect.poll(() => mock.calls('GET', '/ollama/health').length).toBeGreaterThanOrEqual(2);
	await expect(page.getByRole('button', { name: 'Save address' })).toBeVisible();
	await expect.poll(() => mock.calls('PATCH', '/settings').length).toBe(0);
	await expect.poll(() => mock.settings.ollama_host).toBe('http://localhost:11434');

	await page.getByRole('button', { name: 'Test connection' }).click();
	await expect(page.getByText(`Connected to Ollama ${mock.ollama.version} at ${UNSAVED_HOST}.`)).toBeVisible();
	await expect.poll(() => testedHosts.at(-1)).toBe(UNSAVED_HOST);
	await expect.poll(() => mock.calls('PATCH', '/settings').length).toBe(0);
	await expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
		await page.evaluate(() => window.innerWidth)
	);

	await page.getByRole('button', { name: 'Save address' }).click();
	await expect.poll(() => mock.settings.ollama_host).toBe(UNSAVED_HOST);
	await expect.poll(() => mock.calls('PATCH', '/settings').length).toBe(1);
	await expect(page.getByRole('button', { name: 'Save address' })).toHaveCount(0);
});

test('library surfaces failures while refreshing an import that just completed', async ({ page }) => {
	const document = makeDocument({
		document_id: 'doc-imported',
		title: 'Scanned pages',
		import_state: 'importing',
		import_progress: { phase: 'extracting', done: 1, total: 4, unit: 'pages', message: null }
	});
	const mock = await MockApi.install(page, { documents: [document] });
	mock.on('GET', '/documents/:id', () =>
		apiError(503, 'document_unavailable', 'The imported Document could not be refreshed.')
	);

	await page.goto('/');
	await mock.waitForStream();
	await expect(page.getByText('Extracting text · 1 of 4 pages')).toBeVisible();
	mock.documents[0] = { ...document, import_state: 'ready', import_progress: null };
	await mock.pushActivity();
	await expect(page.getByRole('alert')).toContainText('The imported Document could not be refreshed.');
});
