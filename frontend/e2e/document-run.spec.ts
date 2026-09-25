import { setTimeout as sleep } from 'node:timers/promises';
import { expect, test, type Page } from '@playwright/test';
import { apiError, makeDocument, makeProgress, makeRun, MockApi, type MockSeed } from './support/mockApi';
import { openTab, SMALL_SOURCE, treeNode } from './support/workspaceFixtures';

const DOC = 'doc-run';

function setup(page: Page, seed: MockSeed = {}) {
	return MockApi.install(page, {
		documents: [makeDocument({ document_id: DOC, title: 'Harbour minutes', char_count: SMALL_SOURCE.length })],
		sources: { [DOC]: SMALL_SOURCE },
		...seed
	});
}

test('start a Run, follow its progress, Stop it, and Resume it', async ({ page }) => {
	const mock = await setup(page);
	await page.goto(`/documents/${DOC}`);

	await expect(page.getByRole('heading', { name: 'Start a Run' })).toBeVisible();
	const start = page.getByRole('button', { name: 'Start Run' });
	// Enabled once the debounced preflight for the current settings has answered.
	await expect(start).toBeEnabled();
	await expect(page.getByText('about 9', { exact: true })).toBeVisible();
	await start.click();

	await expect(page).toHaveURL(/[?&]run=run-/);
	const [create] = mock.calls('POST', '/runs');
	expect(create.headers['idempotency-key']).toBeTruthy();
	const run = mock.runs[0];
	const strip = page.getByTestId('run-strip');
	await expect(strip.getByText('Queued')).toBeVisible();

	// The worker picks it up: a state change, then progress on the activity stream.
	run.state = 'running';
	run.updated_at = new Date().toISOString();
	run.progress = makeProgress({ cursor: 3, eta_seconds: 840 });
	await mock.pushRun(run);
	await mock.pushActivity();
	await expect(strip.getByText('Summarizing')).toBeVisible();
	await expect(strip.getByText('12 of 40 segments')).toBeVisible();
	await expect(strip.getByText(/about 14 min left/)).toBeVisible();

	run.progress = makeProgress({
		cursor: 6,
		eta_seconds: 420,
		leaves: { done: 20, total: 40, reused: 0, failed: 0 },
		stages: makeProgress().stages.map((stage) =>
			stage.stage === 'summarizing' ? { ...stage, completed: 20 } : stage
		)
	});
	await mock.pushActivity();
	await expect(strip.getByText('20 of 40 segments')).toBeVisible();
	await expect(strip.getByText(/about 7 min left/)).toBeVisible();

	// Nodes appear in the Tree as the Run plans and finishes them.
	await mock.pushNodes(run.run_id, [
		treeNode({ node_id: 'L0N0001', label: 'Segment 1', state: 'completed' }),
		treeNode({ node_id: 'L0N0002', order: 1, label: 'Segment 2', state: 'active' })
	]);
	await openTab(page, 'Tree');
	const tree = page.getByRole('tree', { name: 'Summary tree' });
	await expect(tree.getByRole('treeitem', { name: /Segment 2/ })).toBeVisible();
	await mock.pushNodes(run.run_id, [treeNode({ node_id: 'L0N0002', order: 1, label: 'Segment 2', state: 'completed' })]);
	await expect(tree.getByRole('treeitem', { name: /Segment 2/ })).toContainText('4 s');

	// Stop answers slowly here; the button must say "Stopping…" at once.
	mock.on('POST', '/runs/:id/stop', async () => {
		await sleep(1500);
		return undefined;
	});
	await strip.getByRole('button', { name: 'Stop' }).click();
	await expect(strip.getByRole('button', { name: 'Stopping…' })).toBeDisabled({ timeout: 1000 });
	await expect(strip.locator('[data-state="stopping"]')).toBeVisible();

	run.state = 'stopped';
	run.can_resume = true;
	run.updated_at = new Date(Date.now() + 1000).toISOString();
	await mock.pushRun(run);
	await expect(strip.locator('[data-state="stopped"]')).toBeVisible();

	await strip.getByRole('button', { name: 'Resume' }).click();
	await expect(strip.locator('[data-state="queued"]')).toBeVisible();
	await expect(strip.getByRole('button', { name: 'Stop' })).toBeEnabled();
	expect(mock.calls('POST', '/runs/:id/resume')).toHaveLength(1);
});

test('a failed Run shows its message, item, hint, and error details', async ({ page }) => {
	const failed = makeRun({
		run_id: 'run-failed',
		document_id: DOC,
		document_title: 'Harbour minutes',
		state: 'failed',
		failure: {
			code: 'item_invalid_output',
			message: 'Segment 37 · pp. 112–115 produced invalid output after 3 tries: missing "units"',
			stage: 'summarizing',
			item: 'Segment 37 · pp. 112–115',
			detail: 'LeafSummaryError: response is missing the "units" field',
			hint: 'Resume to retry this item, or try another model.'
		}
	});
	await setup(page, { runs: [failed] });
	await page.goto(`/documents/${DOC}?run=run-failed&tab=status`);

	const panel = page.getByRole('alert').filter({ hasText: 'The Run failed' });
	await expect(panel).toContainText('produced invalid output after 3 tries');
	await expect(panel).toContainText('Stage: Summarizing');
	await expect(panel).toContainText('Item: Segment 37 · pp. 112–115');
	await expect(panel).toContainText('Resume to retry this item, or try another model.');
	await expect(panel.getByText('LeafSummaryError')).toBeHidden();
	await panel.getByText('Error details').click();
	await expect(panel.getByText('LeafSummaryError: response is missing the "units" field')).toBeVisible();
	await expect(page.getByTestId('run-strip').getByRole('button', { name: 'Resume' })).toBeEnabled();
});

test('Start stays disabled while another Run is active and links to it', async ({ page }) => {
	const other = makeRun({
		run_id: 'run-elsewhere',
		document_id: 'doc-other',
		document_title: 'Budget report',
		state: 'running'
	});
	const mock = await setup(page, {
		documents: [
			makeDocument({ document_id: DOC, title: 'Harbour minutes', char_count: SMALL_SOURCE.length }),
			makeDocument({ document_id: 'doc-other', title: 'Budget report' })
		],
		runs: [other]
	});
	await page.goto(`/documents/${DOC}`);

	const reasons = page.locator('#start-reasons');
	await expect(reasons).toContainText('Another Run is active');
	await expect(reasons.getByRole('link', { name: 'Budget report' })).toHaveAttribute(
		'href',
		'/documents/doc-other?run=run-elsewhere'
	);
	const start = page.getByRole('button', { name: 'Start Run' });
	await expect(start).toBeDisabled();
	// Native disabled state prevents submission.
	expect(mock.calls('POST', '/runs')).toHaveLength(0);
});

test('a rejected Start shows the server message', async ({ page }) => {
	const mock = await setup(page);
	mock.on('POST', '/runs', () => apiError(400, 'model_required', 'Choose a model before starting a Run.'));
	await page.goto(`/documents/${DOC}`);
	const start = page.getByRole('button', { name: 'Start Run' });
	await expect(start).toBeEnabled();
	await start.click();
	await expect(page.getByRole('alert').filter({ hasText: 'Choose a model before starting a Run.' })).toBeVisible();
	await expect(page).not.toHaveURL(/[?&]run=/);
});
