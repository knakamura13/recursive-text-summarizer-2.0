import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Activity, NodesEvent, Run } from '../src/lib/api/types';
import type * as ActivityModule from '../src/lib/stores/activity.svelte';
import { FakeEventSource } from './support/fakeEventSource';

let store: (typeof ActivityModule)['activity'];
let stop: () => void;
let visibility: DocumentVisibilityState = 'visible';

function setVisibility(state: DocumentVisibilityState) {
	visibility = state;
	document.dispatchEvent(new Event('visibilitychange'));
}

function run(overrides: Partial<Run>): Run {
	return {
		run_id: 'run-1',
		document_id: 'doc-1',
		document_title: 'Annual report',
		state: 'running',
		progress: null,
		created_at: '2026-09-20T10:00:00Z',
		updated_at: '2026-09-20T10:05:00Z',
		...overrides
	} as Run;
}

const snapshot = (overrides: Partial<Activity>): Activity => ({
	active_run: null,
	importing: [],
	cursor: 0,
	...overrides
});

beforeEach(async () => {
	vi.useFakeTimers();
	FakeEventSource.reset();
	vi.stubGlobal('EventSource', FakeEventSource);
	visibility = 'visible';
	vi.spyOn(document, 'visibilityState', 'get').mockImplementation(() => visibility);
	// A fresh store per test: it is a module-level singleton.
	vi.resetModules();
	store = (await import('../src/lib/stores/activity.svelte')).activity;
});

afterEach(() => {
	stop?.();
	vi.useRealTimers();
});

describe('activity store', () => {
	it('closes the stream while the tab is hidden and resumes after the last cursor', () => {
		stop = store.start();
		const first = FakeEventSource.latest;
		expect(first.url).toBe('/api/v1/activity/stream?after=0');
		first.open();
		first.emit('activity', snapshot({ cursor: 5 }), '5');
		first.emit('nodes', { run_id: 'run-1', nodes: [], cursor: 9 }, '9');
		expect(store.cursor).toBe(9);

		setVisibility('hidden');
		expect(first.closed).toBe(true);
		expect(store.status).toBe('closed');
		vi.advanceTimersByTime(60_000);
		expect(FakeEventSource.instances).toHaveLength(1);

		setVisibility('visible');
		expect(FakeEventSource.instances).toHaveLength(2);
		expect(FakeEventSource.latest.url).toBe('/api/v1/activity/stream?after=9');
	});

	it('reconnects after the transport id when a node frame precedes a terminal Run frame', () => {
		const terminal = vi.fn();
		store.onRun(terminal);
		stop = store.start();
		const first = FakeEventSource.latest;
		first.open();
		first.emit('activity', snapshot({ cursor: 5, active_run: run({}) }), '5');
		first.emit('nodes', { run_id: 'run-1', nodes: [], cursor: 9 }, '5');
		expect(store.cursor).toBe(5);
		setVisibility('hidden');
		setVisibility('visible');
		const next = FakeEventSource.latest;
		expect(next.url).toBe('/api/v1/activity/stream?after=5');
		next.open();
		next.emit('run', run({ state: 'completed' }), '5');
		next.emit('activity', snapshot({ cursor: 9 }), '9');
		expect(terminal).toHaveBeenCalledWith(expect.objectContaining({ state: 'completed' }));
		expect(store.cursor).toBe(9);
	});

	it('asks subscribers to refetch after a reconnect, never after the first connection', () => {
		const onReconnect = vi.fn();
		store.onReconnect(onReconnect);
		stop = store.start();
		FakeEventSource.latest.open();
		expect(onReconnect).not.toHaveBeenCalled();

		FakeEventSource.latest.fail();
		vi.advanceTimersByTime(1_000);
		expect(onReconnect).not.toHaveBeenCalled();
		FakeEventSource.latest.open();
		expect(onReconnect).toHaveBeenCalledTimes(1);

		setVisibility('hidden');
		setVisibility('visible');
		FakeEventSource.latest.open();
		expect(onReconnect).toHaveBeenCalledTimes(2);
	});

	it('does not connect while the tab starts hidden', () => {
		visibility = 'hidden';
		stop = store.start();
		expect(FakeEventSource.instances).toHaveLength(0);
		setVisibility('visible');
		expect(FakeEventSource.instances).toHaveLength(1);
	});

	it('delivers nodes and run events in order and keeps the active Run in step', () => {
		const seen: string[] = [];
		store.onNodes((event: NodesEvent) => seen.push(`nodes ${event.cursor}`));
		store.onRun((changed) => seen.push(`run ${changed.state}`));
		stop = store.start();
		const source = FakeEventSource.latest;
		source.open();
		const progress = { stage: 'summarizing' } as Run['progress'];
		source.emit('activity', snapshot({ cursor: 3, active_run: run({ progress }) }), '3');

		source.emit('nodes', { run_id: 'run-1', nodes: [], cursor: 4 }, '4');
		source.emit('run', run({ state: 'stopping' }), '5');
		expect(store.activeRun?.state).toBe('stopping');
		expect(store.activeRun?.progress).toEqual(progress);

		source.emit('nodes', { run_id: 'run-1', nodes: [], cursor: 6 }, '6');
		source.emit('run', run({ state: 'stopped' }), '7');
		expect(store.activeRun).toBeNull();
		expect(seen).toEqual(['nodes 4', 'run stopping', 'nodes 6', 'run stopped']);
		expect(store.cursor).toBe(7);
	});

	it('adopts the cursor of a snapshot even when it is lower (the server database was reset)', () => {
		stop = store.start();
		const source = FakeEventSource.latest;
		source.open();
		source.emit('activity', snapshot({ cursor: 50 }), '50');
		source.emit('activity', snapshot({ cursor: 2 }), '2');
		expect(store.cursor).toBe(2);
	});

	it('stops notifying a handler after it unsubscribes', () => {
		const handler = vi.fn();
		const unsubscribe = store.onRun(handler);
		stop = store.start();
		FakeEventSource.latest.open();
		FakeEventSource.latest.emit('run', run({ state: 'queued' }));
		unsubscribe();
		FakeEventSource.latest.emit('run', run({ state: 'running' }));
		expect(handler).toHaveBeenCalledTimes(1);
	});
});
