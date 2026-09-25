import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Activity, DocumentSummary } from '../src/lib/api/types';
import { FakeEventSource } from './support/fakeEventSource';

function doc(overrides: Partial<DocumentSummary>): DocumentSummary {
	return {
		document_id: 'doc-1',
		title: 'Notes',
		filename: 'notes.txt',
		format: 'txt',
		origin: 'upload',
		size_bytes: 10,
		page_count: null,
		char_count: null,
		import_state: 'ready',
		import_progress: null,
		import_error: null,
		latest_run: null,
		created_at: '2026-09-20T10:00:00Z',
		updated_at: '2026-09-20T10:00:00Z',
		...overrides
	} as DocumentSummary;
}

const snapshot = (overrides: Partial<Activity>): Activity => ({ active_run: null, importing: [], cursor: 1, ...overrides });

let stop: () => void;

beforeEach(() => {
	FakeEventSource.reset();
	vi.stubGlobal('EventSource', FakeEventSource);
	vi.resetModules();
});

afterEach(() => {
	stop?.();
	vi.unstubAllGlobals();
});

describe('documents store', () => {
	it('refreshes an upload whose import finished before any snapshot listed it as importing', async () => {
		const importing = doc({ document_id: 'doc-fast', import_state: 'importing' });
		const ready = doc({ document_id: 'doc-fast', import_state: 'ready', char_count: 42 });
		const fetch = vi.fn(async (input: RequestInfo | URL) => {
			const url = String(input);
			const body = url.endsWith('/documents/doc-fast') ? ready : { documents: [] };
			return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
		});
		vi.stubGlobal('fetch', fetch);
		const { activity } = await import('../src/lib/stores/activity.svelte');
		const { documents } = await import('../src/lib/stores/documents.svelte');
		stop = activity.start();
		await documents.refresh();
		documents.upsert(importing);

		FakeEventSource.latest.open();
		FakeEventSource.latest.emit('activity', snapshot({}), '1');

		await vi.waitFor(() => expect(documents.get('doc-fast')?.import_state).toBe('ready'));
		expect(documents.get('doc-fast')?.char_count).toBe(42);
	});
});
