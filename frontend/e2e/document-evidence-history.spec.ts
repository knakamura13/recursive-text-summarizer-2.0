import { expect, test } from '@playwright/test';
import { makeDocument, makeRun, MockApi } from './support/mockApi';
import { availableSummary, SMALL_SOURCE } from './support/workspaceFixtures';

const DOC = 'doc-summary';
const RUN = 'run-summary';
const QUOTE = 'dredging budget';
const START = SMALL_SOURCE.indexOf(QUOTE);

const citation = {
	segment_id: 'D000001',
	quote: QUOTE,
	quote_found: true,
	start: START,
	end: START + QUOTE.length,
	page_start: 1,
	page_end: 1
};

test('shows verified summary evidence in the Source and exposes all exports', async ({ page }) => {
	const run = makeRun({ run_id: RUN, document_id: DOC, document_title: 'Harbour minutes', state: 'completed' });
	const summary = availableSummary({
		text: 'The commission reviewed the dredging budget.',
		sentences: [
			{
				index: 0,
				paragraph: 0,
				text: 'The commission reviewed the dredging budget.',
				verdict: 'supported',
				evidence: [citation]
			}
		],
		removed_sentences: [{ text: 'An unsupported claim.', verdict: 'insufficient_support', reason: 'No source passage supports it.' }],
		citations: [
			{
				citation_id: '1',
				segment_id: citation.segment_id,
				start: citation.start,
				end: citation.end,
				page_start: citation.page_start,
				page_end: citation.page_end
			}
		],
		word_count: 7,
		target_words: 100,
		short_of_target: true,
		verification_state: 'completed',
		publication: 'verified_subset',
		notices: [
			{
				code: 'verified_sentence_subset',
				message:
					'Some sentences of the written draft could not be verified and were removed; the remaining sentences passed verification.',
				severity: 'warning'
			}
		]
	});
	await MockApi.install(page, {
		documents: [makeDocument({ document_id: DOC, title: 'Harbour minutes', char_count: SMALL_SOURCE.length, page_count: 1 })],
		sources: { [DOC]: SMALL_SOURCE },
		pages: { [DOC]: [{ page: 1, start: 0, end: SMALL_SOURCE.length, ocr: false, blank: false }] },
		runs: [run],
		summaries: { [RUN]: summary }
	});

	await page.goto(`/documents/${DOC}?run=${RUN}&tab=summary`);
	const article = page.getByRole('article', { name: 'Summary' });
	await expect(article).toContainText('could not be verified and were removed; the remaining sentences passed verification.');
	await expect(article.getByRole('link', { name: 'Text' })).toHaveAttribute('href', /\/export\/txt$/);
	await expect(article.getByRole('link', { name: 'Markdown' })).toHaveAttribute('href', /\/export\/md$/);
	await expect(article.getByRole('link', { name: 'Audit JSON' })).toHaveAttribute('href', /\/export\/json$/);

	const removed = article.getByText('1 sentence was removed by verification');
	await removed.click();
	await expect(article).toContainText('No source passage supports it.');

	await article.getByRole('button', { name: /The commission reviewed the dredging budget/ }).click();
	const evidence = article.getByRole('button', { name: new RegExp(`${QUOTE}.*Whole document`) });
	await expect(evidence).toBeVisible();
	await evidence.click();
	const source = page.getByRole('region', { name: 'Document text' });
	await expect(page.getByTestId('source-focus')).toContainText('Evidence · Whole document · p. 1');
	await expect(source.locator('mark')).toContainText(QUOTE);
});

test('an unverified editorial summary never claims verification passed', async ({ page }) => {
	const run = makeRun({
		run_id: RUN,
		document_id: DOC,
		document_title: 'Harbour minutes',
		state: 'completed'
	});
	run.config.verify = false;
	await MockApi.install(page, {
		documents: [makeDocument({ document_id: DOC, title: 'Harbour minutes', char_count: SMALL_SOURCE.length })],
		sources: { [DOC]: SMALL_SOURCE },
		runs: [run],
		summaries: {
			[RUN]: availableSummary({
				publication: 'editorial',
				verification_state: 'not_run',
				notices: [{ code: 'verification_off', message: 'Verification was off for this Run.', severity: 'info' }],
				sentences: [{ index: 0, paragraph: 0, text: 'The commission reviewed the dredging budget.', verdict: 'unchecked', evidence: [] }]
			})
		}
	});
	await page.goto('/documents/' + DOC + '?run=' + RUN + '&tab=summary');
	const article = page.getByRole('article', { name: 'Summary' });
	await expect(article).toContainText('Verification was off for this Run.');
	await expect(article).not.toContainText('passed verification');
});
test('opens a historical Run and deletes it only after confirmation', async ({ page }) => {
	const latest = makeRun({
		run_id: 'run-latest',
		document_id: 'doc-history',
		document_title: 'History',
		state: 'completed',
		created_at: '2026-09-23T10:00:00Z'
	});
	const older = makeRun({
		run_id: 'run-older',
		document_id: 'doc-history',
		document_title: 'History',
		state: 'failed',
		created_at: '2026-09-22T10:00:00Z',
		failure: {
			code: 'internal_error',
			message: 'Earlier Attempt failed.',
			stage: 'summarizing',
			item: null,
			detail: null,
			hint: 'Resume to try again.'
		}
	});
	await MockApi.install(page, {
		documents: [makeDocument({ document_id: 'doc-history', title: 'History', char_count: SMALL_SOURCE.length })],
		sources: { 'doc-history': SMALL_SOURCE },
		runs: [latest, older]
	});

	await page.goto('/documents/doc-history?run=run-latest&tab=runs');
	const oldRow = page.locator('[data-run-id="run-older"]');
	await expect(oldRow).toContainText('failed');
	await expect(oldRow).toContainText('Earlier Attempt failed.');
	await expect(oldRow).toContainText('llama3.1:8b');
	await oldRow.getByRole('button', { name: 'Open' }).click();
	await expect(page).toHaveURL(/run=run-older/);
	await expect(page.getByRole('alert').filter({ hasText: 'Earlier Attempt failed.' })).toBeVisible();

	await page.getByRole('tablist', { name: 'Document views' }).getByRole('tab', { name: /^Runs/ }).click();
	const selectedOldRow = page.locator('[data-run-id="run-older"]');
	await selectedOldRow.getByRole('button', { name: 'Delete' }).click();
	const dialog = page.getByRole('dialog', { name: 'Delete this Run?' });
	await expect(dialog).toBeVisible();
	await expect(dialog.getByText('Its summary, tree, and progress are deleted. The Document stays.')).toBeVisible();
	await dialog.getByRole('button', { name: 'Delete Run' }).click();
	await expect(page.locator('[data-run-id="run-older"]')).toHaveCount(0);
	await expect(page.getByText('Run deleted.')).toBeVisible();
});
