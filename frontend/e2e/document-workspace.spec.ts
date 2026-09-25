import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { apiError, makeDocument, makeRun, MockApi } from './support/mockApi';
import {
	LARGE_LENGTH,
	SMALL_SOURCE,
	largePages,
	largeSlice,
	largeTree
} from './support/workspaceFixtures';

const DOC = 'doc-workspace';
const RUN = 'run-workspace';

function layoutOf(page: Page): 'phone' | 'tablet' | 'desktop' {
	const width = page.viewportSize()?.width ?? 1280;
	return width < 640 ? 'phone' : width < 1024 ? 'tablet' : 'desktop';
}

test('windows a 2,000-node tree and restores its deep-linked selection', async ({ page }) => {
	const run = makeRun({ run_id: RUN, document_id: DOC, document_title: 'Field notes', state: 'completed' });
	await MockApi.install(page, {
		documents: [makeDocument({ document_id: DOC, title: 'Field notes', char_count: SMALL_SOURCE.length })],
		sources: { [DOC]: SMALL_SOURCE },
		runs: [run],
		trees: { [RUN]: largeTree() }
	});

	await page.goto(`/documents/${DOC}?run=${RUN}&tab=tree`);
	const workspace = page.locator('div.workspace[data-layout]');
	await expect(workspace).toHaveAttribute('data-layout', layoutOf(page));
	const tree = page.getByRole('tree', { name: 'Summary tree' });
	await expect(tree).toBeVisible();
	await expect.poll(() => tree.getByRole('treeitem').count()).toBeLessThan(100);

	// The phone breakpoint must remain a single, non-scrolling column even at 320 px.
	await page.setViewportSize({ width: 320, height: 780 });
	await expect(workspace).toHaveAttribute('data-layout', 'phone');
	await expect
		.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
		.toBe(true);

	await tree.press('End');
	const lastNode = tree.getByRole('treeitem', { name: /Segment 1959/ });
	await expect(lastNode).toBeVisible();
	await tree.press('Enter');
	await expect(page).toHaveURL(new RegExp(`[?&]node=L0N1959(?:&|$)`));

	await page.reload();
	await expect(page).toHaveURL(new RegExp(`[?&]run=${RUN}.*[?&]tab=tree.*[?&]node=L0N1959`));
	const restored = page.getByRole('tree', { name: 'Summary tree' }).getByRole('treeitem', { name: /Segment 1959/ });
	await expect(restored).toBeVisible();
	await expect(restored).toHaveAttribute('aria-selected', 'true');
});

test('windows a 2.1 M-character Source and jumps to a page without loading the Document at once', async ({ page }) => {
	const run = makeRun({ run_id: RUN, document_id: DOC, document_title: 'Long textbook', state: 'completed' });
	const pages = largePages();
	const mock = await MockApi.install(page, {
		documents: [
			makeDocument({
				document_id: DOC,
				title: 'Long textbook',
				char_count: LARGE_LENGTH,
				page_count: pages.length
			})
		],
		runs: [run],
		pages: { [DOC]: pages }
	});
	mock.on('GET', '/documents/:id/source', ({ query }) => {
		const offset = Number(query.get('offset') ?? 0);
		const limit = Number(query.get('limit') ?? 16_384);
		return {
			json: {
				text: largeSlice(offset, limit),
				offset,
				total_length: LARGE_LENGTH,
				has_more: offset + limit < LARGE_LENGTH
			}
		};
	});

	await page.setViewportSize({ width: 320, height: 780 });
	await page.goto(`/documents/${DOC}?run=${RUN}&tab=source`);
	const source = page.getByRole('region', { name: 'Document text' });
	await expect(source).toBeVisible();
	await expect(source.locator('[data-chunk][data-loaded]')).toHaveCount(1);
	await expect.poll(() => source.locator('[data-chunk]').count()).toBeLessThanOrEqual(3);

	await page.getByLabel('Go to page').fill('500');
	await page.getByRole('button', { name: 'Go' }).click();
	await expect(source.locator('[data-page="500"]')).toBeVisible();
	await expect.poll(() => source.locator('[data-chunk]').count()).toBeLessThanOrEqual(3);

	const slices = mock.calls('GET', '/documents/:id/source');
	expect(slices.length).toBeGreaterThan(0);
	expect(slices.every((request) => Number(request.query.get('limit')) <= 16_384)).toBe(true);
});

test('shows a failed Run history request and retries it instead of offering a new Run', async ({ page }) => {
	const mock = await MockApi.install(page, {
		documents: [makeDocument({ document_id: DOC, title: 'Field notes', char_count: SMALL_SOURCE.length })],
		sources: { [DOC]: SMALL_SOURCE }
	});
	let attempts = 0;
	mock.on('GET', '/documents/:id/runs', () => {
		attempts += 1;
		return attempts === 1
			? apiError(503, 'history_unavailable', 'Run history is temporarily unavailable.')
			: undefined;
	});

	await page.goto('/documents/' + DOC);
	const error = page.getByRole('alert').filter({ hasText: 'Run history is temporarily unavailable.' });
	await expect(error).toBeVisible();
	await expect(page.getByRole('heading', { name: 'Start a Run' })).toHaveCount(0);
	await error.getByRole('button', { name: 'Try again' }).click();
	await expect(page.getByRole('heading', { name: 'Start a Run' })).toBeVisible();
	expect(mock.calls('GET', '/documents/:id/runs')).toHaveLength(2);
});
