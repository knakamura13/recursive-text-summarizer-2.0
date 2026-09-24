import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { NodesEvent, Run, RunProgress, RunState, Tree, TreeNode } from '../../src/lib/api/types';

const mocks = vi.hoisted(() => ({
	api: { getRun: vi.fn(), getTree: vi.fn(), getSummary: vi.fn() },
	toastError: vi.fn()
}));

vi.mock('$lib/api/client', () => ({
	api: mocks.api,
	errorMessage: (error: unknown) => (error instanceof Error ? error.message : String(error)),
	isAbortError: (error: unknown) => (error as { name?: string } | null)?.name === 'AbortError'
}));
vi.mock('$lib/stores/toasts.svelte', () => ({ toasts: { error: mocks.toastError } }));
vi.mock('$lib/stores/activity.svelte', () => ({ activity: null }));

import { applyRunEvents, RunStream, snapshotFromRest, type ActivityFeed } from '../../src/lib/stores/runStream.svelte';

function makeRun(state: RunState, overrides: Partial<Run> = {}): Run {
	return {
		run_id: 'run-1',
		document_id: 'doc-1',
		document_title: 'Report',
		state,
		requested_strategy: 'auto',
		selected_strategy: 'hierarchical',
		config: {} as Run['config'],
		created_at: '2026-09-23T10:00:00Z',
		updated_at: '2026-09-23T10:00:00Z',
		attempt: null,
		attempt_count: 1,
		failure: null,
		can_stop: state === 'queued' || state === 'running',
		can_resume: state === 'stopped' || state === 'failed' || state === 'interrupted',
		progress: null,
		...overrides
	};
}

function makeProgress(cursor: number, overrides: Partial<RunProgress> = {}): RunProgress {
	return {
		cursor,
		attempt_number: 1,
		started_at: '2026-09-23T10:00:00Z',
		elapsed_seconds: 10,
		stage: 'summarizing',
		stages: [],
		leaves: { done: 1, total: 4, reused: 0, failed: 0 },
		merges: { done: 0, total: null, reused: 0, failed: 0 },
		merge_levels: [],
		claims: { done: 0, total: null, reused: 0, failed: 0 },
		current_items: [],
		eta_seconds: 60,
		eta_basis: null,
		...overrides
	};
}

function leaf(id: string, order: number, overrides: Partial<TreeNode> = {}): TreeNode {
	return {
		node_id: id,
		parent_id: null,
		level: 0,
		order,
		kind: 'leaf',
		label: `Segment ${order + 1}`,
		state: 'pending',
		child_count: 0,
		page_start: null,
		page_end: null,
		duration_seconds: null,
		...overrides
	};
}

const tree = (nodes: TreeNode[], cursor: number): Tree => ({ nodes, cursor });

describe('applyRunEvents', () => {
	const base = snapshotFromRest(makeRun('running'), tree([leaf('L1', 0), leaf('L2', 1)], 10), 0);

	it('updates changed nodes only and keeps the shape index when only states change', () => {
		const next = applyRunEvents(
			base,
			[{ type: 'nodes', nodes: [leaf('L1', 0), leaf('L2', 1, { state: 'active' })], cursor: 12 }],
			5
		);
		expect(next.nodes.get('L1')).toBe(base.nodes.get('L1'));
		expect(next.nodes.get('L2')?.state).toBe('active');
		expect(next.children).toBe(base.children);
		expect(next.nodesCursor).toBe(12);
	});

	it('rebuilds the tree shape when a merge level adopts the leaves', () => {
		const merge: TreeNode = { ...leaf('M1', 0), level: 1, kind: 'merge', label: 'Level 1 · Group 1' };
		const next = applyRunEvents(
			base,
			[{ type: 'nodes', nodes: [merge, leaf('L1', 0, { parent_id: 'M1' }), leaf('L2', 1, { parent_id: 'M1' })], cursor: 20 }],
			5
		);
		expect(next.children.get(null)).toEqual(['M1']);
		expect(next.children.get('M1')).toEqual(['L1', 'L2']);
	});

	it('skips node data the tree already reflects', () => {
		const stale = applyRunEvents(base, [{ type: 'nodes', nodes: [leaf('L1', 0, { state: 'active' })], cursor: 10 }], 5);
		expect(stale).toBe(base);
		expect(stale.nodes.get('L1')?.state).toBe('pending');
	});

	it('ignores progress from an older Attempt or cursor and stamps accepted progress', () => {
		const current = applyRunEvents(
			base,
			[{ type: 'run', run: makeRun('running', { progress: makeProgress(40, { attempt_number: 2 }) }) }],
			7
		);
		expect(current.progress?.attempt_number).toBe(2);
		expect(current.progressAt).toBe(7);
		const stale = applyRunEvents(
			current,
			[
				{ type: 'run', run: makeRun('running', { progress: makeProgress(50, { attempt_number: 1 }) }) },
				{ type: 'run', run: makeRun('running', { progress: makeProgress(35, { attempt_number: 2 }) }) }
			],
			9
		);
		expect(stale.progress).toBe(current.progress);
		expect(stale.progressAt).toBe(7);
	});

	it('keeps the newest Run snapshot and ignores other Runs', () => {
		const stopping = applyRunEvents(base, [{ type: 'run', run: makeRun('stopping', { updated_at: '2026-09-23T10:05:00Z' }) }], 1);
		const late = applyRunEvents(
			stopping,
			[
				{ type: 'run', run: makeRun('running', { updated_at: '2026-09-23T10:04:00Z' }) },
				{ type: 'run', run: makeRun('completed', { run_id: 'other', updated_at: '2026-09-23T11:00:00Z' }) }
			],
			2
		);
		expect(late.run?.state).toBe('stopping');
	});

	it('replaces the nodes with a newer refetched tree and drops nodes it no longer has', () => {
		const refreshed = applyRunEvents(
			base,
			[{ type: 'tree', tree: tree([leaf('L1', 0), leaf('L3', 2, { state: 'completed' })], 30) }],
			3
		);
		expect([...refreshed.nodes.keys()]).toEqual(['L1', 'L3']);
		expect(refreshed.nodes.get('L1')).toBe(base.nodes.get('L1'));
		expect(refreshed.children.get(null)).toEqual(['L1', 'L3']);
		expect(applyRunEvents(refreshed, [{ type: 'tree', tree: tree([], 25) }], 4)).toBe(refreshed);
	});
});

/** An activity store stand-in whose handlers the test fires like SSE events. */
function fakeFeed() {
	const handlers = { nodes: new Set<(e: NodesEvent) => void>(), run: new Set<(r: Run) => void>(), reconnect: new Set<() => void>() };
	const state = $state<{ activeRun: Run | null }>({ activeRun: null });
	const subscribe = <T>(set: Set<T>, handler: T) => {
		set.add(handler);
		return () => set.delete(handler);
	};
	const feed: ActivityFeed = {
		get activeRun() {
			return state.activeRun;
		},
		onNodes: (handler) => subscribe(handlers.nodes, handler),
		onRun: (handler) => subscribe(handlers.run, handler),
		onReconnect: (handler) => subscribe(handlers.reconnect, handler)
	};
	return {
		feed,
		handlers,
		setActiveRun(run: Run | null) {
			state.activeRun = run;
			flushSync();
		},
		nodes: (event: NodesEvent) => handlers.nodes.forEach((handler) => handler(event)),
		run: (run: Run) => handlers.run.forEach((handler) => handler(run)),
		reconnect: () => handlers.reconnect.forEach((handler) => handler())
	};
}

describe('RunStream', () => {
	let flushes: (() => void)[];
	const schedule = (flush: () => void) => {
		flushes.push(flush);
		return () => {
			flushes = flushes.filter((item) => item !== flush);
		};
	};
	const frame = () => {
		const pending = flushes;
		flushes = [];
		for (const flush of pending) flush();
	};

	beforeEach(() => {
		flushes = [];
		mocks.api.getRun.mockReset();
		mocks.api.getTree.mockReset();
		mocks.api.getSummary.mockReset();
		mocks.toastError.mockReset();
	});

	async function open(run: Run, nodes: TreeNode[] = [leaf('L1', 0)], cursor = 10) {
		const activity = fakeFeed();
		mocks.api.getRun.mockResolvedValue(run);
		mocks.api.getTree.mockResolvedValue(tree(nodes, cursor));
		const onStateChange = vi.fn();
		const stream = new RunStream('run-1', { schedule, now: () => 1000, onStateChange, feed: activity.feed });
		await vi.waitFor(() => expect(stream.loading).toBe(false));
		return { stream, onStateChange, activity };
	}

	it('applies node events of its own Run once per frame and ignores other Runs', async () => {
		const { stream, activity } = await open(makeRun('running'));
		activity.nodes({ run_id: 'run-1', nodes: [leaf('L1', 0, { state: 'active' })], cursor: 11 });
		activity.nodes({ run_id: 'other', nodes: [leaf('L1', 0, { state: 'failed' })], cursor: 12 });
		expect(stream.nodes.get('L1')?.state).toBe('pending');
		expect(flushes).toHaveLength(1);
		frame();
		expect(stream.nodes.get('L1')?.state).toBe('active');
	});

	it('takes live progress from the active Run of the activity stream', async () => {
		const { stream, activity } = await open(makeRun('running', { progress: makeProgress(10) }));
		activity.setActiveRun(makeRun('running', { progress: makeProgress(15, { eta_seconds: 42 }) }));
		frame();
		expect(stream.progress?.eta_seconds).toBe(42);
		activity.setActiveRun(makeRun('running', { run_id: 'other', progress: makeProgress(99, { eta_seconds: 1 }) }));
		frame();
		expect(stream.progress?.eta_seconds).toBe(42);
	});

	it('keeps events that arrive during the initial load and drops those the tree already has', async () => {
		const activity = fakeFeed();
		let resolveTree: (tree: Tree) => void = () => {};
		mocks.api.getRun.mockResolvedValue(makeRun('running'));
		// Executor form: Node 20 (the project's runtime) has no Promise.withResolvers.
		mocks.api.getTree.mockReturnValue(new Promise<Tree>((resolve) => (resolveTree = resolve)));
		const stream = new RunStream('run-1', { schedule, feed: activity.feed });
		activity.nodes({ run_id: 'run-1', nodes: [leaf('L1', 0, { state: 'active' })], cursor: 9 });
		activity.nodes({ run_id: 'run-1', nodes: [leaf('L2', 1, { state: 'completed' })], cursor: 12 });
		frame();
		resolveTree(tree([leaf('L1', 0, { state: 'completed' }), leaf('L2', 1)], 10));
		await vi.waitFor(() => expect(stream.loading).toBe(false));
		expect(stream.nodes.get('L1')?.state).toBe('completed');
		expect(stream.nodes.get('L2')?.state).toBe('completed');
	});

	it('refetches the tree after the activity stream reconnects', async () => {
		const { stream, activity } = await open(makeRun('running'), [leaf('L1', 0)], 10);
		mocks.api.getTree.mockResolvedValue(tree([leaf('L1', 0, { state: 'completed' })], 20));
		activity.reconnect();
		await vi.waitFor(() => expect(flushes).toHaveLength(1));
		frame();
		expect(stream.nodes.get('L1')?.state).toBe('completed');
	});

	it('does not refetch anything for a Run that already ended', async () => {
		const { activity } = await open(makeRun('failed'));
		activity.reconnect();
		await Promise.resolve();
		expect(mocks.api.getTree).toHaveBeenCalledTimes(1);
		expect(mocks.api.getSummary).not.toHaveBeenCalled();
	});

	it('reports the end of the Run, fetches its final state once, and loads the summary', async () => {
		mocks.api.getSummary.mockResolvedValue({ available: true, text: 'Done.' });
		const { stream, onStateChange, activity } = await open(makeRun('running'));
		mocks.api.getRun.mockResolvedValue(
			makeRun('completed', { updated_at: '2026-09-23T10:09:00Z', progress: makeProgress(80, { stage: 'publishing' }) })
		);
		activity.run(makeRun('completed', { updated_at: '2026-09-23T10:09:00Z' }));
		frame();
		expect(stream.run?.state).toBe('completed');
		expect(onStateChange).toHaveBeenLastCalledWith(expect.objectContaining({ state: 'completed' }), 'running');
		await vi.waitFor(() => expect(stream.summary).toEqual({ available: true, text: 'Done.' }));
		await vi.waitFor(() => expect(stream.progress?.stage).toBe('publishing'));
		expect(mocks.api.getRun).toHaveBeenCalledTimes(2);
	});

	it('follows a resumed Run: Stop applies at once, Resume brings live events again', async () => {
		const { stream, activity } = await open(makeRun('running'), [leaf('L1', 0, { state: 'completed' })], 44);
		stream.applyRun(makeRun('stopping', { updated_at: '2026-09-23T10:01:00Z' }));
		expect(stream.run?.state).toBe('stopping');
		activity.run(makeRun('stopped', { updated_at: '2026-09-23T10:02:00Z' }));
		frame();
		expect(stream.run?.state).toBe('stopped');
		stream.applyRun(makeRun('queued', { updated_at: '2026-09-23T10:10:00Z', attempt_count: 2 }));
		activity.nodes({ run_id: 'run-1', nodes: [leaf('L2', 1, { state: 'active' })], cursor: 45 });
		frame();
		expect(stream.run?.state).toBe('queued');
		expect(stream.nodes.get('L2')?.state).toBe('active');
	});

	it('stops listening after destroy', async () => {
		const { stream, activity } = await open(makeRun('running'));
		stream.destroy();
		expect(activity.handlers.nodes.size).toBe(0);
		expect(activity.handlers.run.size).toBe(0);
		activity.setActiveRun(makeRun('running', { progress: makeProgress(50) }));
		expect(flushes).toHaveLength(0);
	});

	it('exposes a load error and recovers on reload', async () => {
		const activity = fakeFeed();
		mocks.api.getRun.mockRejectedValueOnce(new Error('Run not found'));
		mocks.api.getTree.mockResolvedValue(tree([], 0));
		const stream = new RunStream('run-1', { schedule, feed: activity.feed });
		await vi.waitFor(() => expect(stream.error).toBeInstanceOf(Error));
		mocks.api.getRun.mockResolvedValue(makeRun('completed'));
		mocks.api.getSummary.mockResolvedValue({ available: false });
		stream.reload();
		await vi.waitFor(() => expect(stream.run?.state).toBe('completed'));
		expect(stream.error).toBeNull();
	});
});
