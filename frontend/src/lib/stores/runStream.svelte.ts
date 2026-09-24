// Live state of one Run: the Run, its progress snapshot, and its node tree. Loaded over REST;
// while the Run is active it is kept current by the app's single activity stream (contract 7:
// no per-Run EventSource). Events are applied in batches, one per animation frame, by the pure
// `applyRunEvents` reducer. Runs that are not active never change, so they are fetched once.
import { untrack } from 'svelte';
import { api, errorMessage, isAbortError } from '$lib/api/client';
import type { FinalSummary, NodesEvent, Run, RunProgress, RunState, Tree, TreeNode } from '$lib/api/types';
import { activity } from '$lib/stores/activity.svelte';
import { toasts } from '$lib/stores/toasts.svelte';
import { affectsStructure, buildChildrenIndex, type ChildrenIndex } from '$lib/tree/treeModel';

const ACTIVE_STATES: Record<RunState, boolean> = {
	queued: true,
	running: true,
	stopping: true,
	stopped: false,
	failed: false,
	interrupted: false,
	completed: false
};

/** Queued, running, or stopping: the Run changes over time and blocks other Runs. */
export function isActiveRunState(state: RunState): boolean {
	return ACTIVE_STATES[state];
}

export interface RunSnapshot {
	run: Run | null;
	progress: RunProgress | null;
	/** Clock reading (ms, same clock as the reducer's `now`) when `progress` arrived. */
	progressAt: number;
	nodes: ReadonlyMap<string, TreeNode>;
	children: ChildrenIndex;
	/** The node data is complete through this run_events id (tree cursor, then nodes events). */
	nodesCursor: number;
}

export type RunEvent =
	/** A RunResponse from the activity stream, the active Run's snapshot, or a REST call. */
	| { type: 'run'; run: Run }
	/** Changed nodes of the active Run. */
	| { type: 'nodes'; nodes: readonly TreeNode[]; cursor: number }
	/** The whole tree, refetched after the activity stream reconnected. */
	| { type: 'tree'; tree: Tree };

export function emptySnapshot(): RunSnapshot {
	const nodes = new Map<string, TreeNode>();
	return { run: null, progress: null, progressAt: 0, nodes, children: buildChildrenIndex(nodes), nodesCursor: 0 };
}

/** Initial snapshot from `GET /runs/{id}` and `GET /runs/{id}/tree`. */
export function snapshotFromRest(run: Run, tree: Tree, now: number): RunSnapshot {
	const nodes = new Map<string, TreeNode>();
	for (const node of tree.nodes) nodes.set(node.node_id, node);
	return { run, progress: run.progress, progressAt: now, nodes, children: buildChildrenIndex(nodes), nodesCursor: tree.cursor };
}

function newerRun(current: Run | null, next: Run): boolean {
	if (current === null) return true;
	const currentTime = Date.parse(current.updated_at);
	const nextTime = Date.parse(next.updated_at);
	if (Number.isNaN(currentTime) || Number.isNaN(nextTime)) return true;
	return nextTime >= currentTime || next.attempt_count > current.attempt_count;
}

function newerProgress(current: RunProgress | null, next: RunProgress): boolean {
	if (current === null) return true;
	const currentAttempt = current.attempt_number ?? 0;
	const nextAttempt = next.attempt_number ?? 0;
	if (nextAttempt !== currentAttempt) return nextAttempt > currentAttempt;
	return next.cursor >= current.cursor;
}

function sameNode(a: TreeNode, b: TreeNode): boolean {
	for (const key of Object.keys(b) as (keyof TreeNode)[]) {
		if (a[key] !== b[key]) return false;
	}
	return true;
}

/**
 * Apply a batch of events in order. Returns the same snapshot when nothing changed. Unchanged
 * nodes keep their object identity so only changed tree rows re-render; the children index is
 * rebuilt only when the tree's shape changed. Node data older than `nodesCursor` is skipped.
 */
export function applyRunEvents(snapshot: RunSnapshot, events: readonly RunEvent[], now: number): RunSnapshot {
	let { run, progress, progressAt, nodesCursor } = snapshot;
	let nodes: Map<string, TreeNode> | null = null;
	let reshaped = false;
	for (const event of events) {
		switch (event.type) {
			case 'run':
				if (run !== null && event.run.run_id !== run.run_id) break;
				if (newerRun(run, event.run)) run = event.run;
				if (event.run.progress !== null && newerProgress(progress, event.run.progress)) {
					progress = event.run.progress;
					progressAt = now;
				}
				break;
			case 'nodes':
				if (event.cursor <= nodesCursor) break;
				nodesCursor = event.cursor;
				for (const node of event.nodes) {
					const previous = (nodes ?? snapshot.nodes).get(node.node_id);
					if (previous !== undefined && sameNode(previous, node)) continue;
					nodes ??= new Map(snapshot.nodes);
					nodes.set(node.node_id, node);
					if (affectsStructure(previous, node)) reshaped = true;
				}
				break;
			case 'tree': {
				if (event.tree.cursor <= nodesCursor) break;
				nodesCursor = event.tree.cursor;
				const current = nodes ?? snapshot.nodes;
				const next = new Map<string, TreeNode>();
				let changed = event.tree.nodes.length !== current.size;
				for (const node of event.tree.nodes) {
					const previous = current.get(node.node_id);
					if (previous !== undefined && sameNode(previous, node)) {
						next.set(node.node_id, previous);
						continue;
					}
					next.set(node.node_id, node);
					changed = true;
					if (affectsStructure(previous, node)) reshaped = true;
				}
				if (next.size !== current.size) reshaped = true;
				if (changed) nodes = next;
				break;
			}
		}
	}
	if (run === snapshot.run && progress === snapshot.progress && nodesCursor === snapshot.nodesCursor && nodes === null) {
		return snapshot;
	}
	const nextNodes = nodes ?? snapshot.nodes;
	return {
		run,
		progress,
		progressAt,
		nodes: nextNodes,
		children: reshaped ? buildChildrenIndex(nextNodes) : snapshot.children,
		nodesCursor
	};
}

/** Schedules `flush` once; returns a canceller. Test seam for frame batching. */
export type FlushScheduler = (flush: () => void) => () => void;

/** Next animation frame, or 250 ms when the tab is hidden and frames are paused. */
export const frameScheduler: FlushScheduler = (flush) => {
	let done = false;
	const run = () => {
		if (done) return;
		done = true;
		cancelAnimationFrame(frame);
		clearTimeout(timer);
		flush();
	};
	const frame = requestAnimationFrame(run);
	const timer = setTimeout(run, 250);
	return () => {
		done = true;
		cancelAnimationFrame(frame);
		clearTimeout(timer);
	};
};

/** The parts of the activity store a RunStream listens to (contract 8b). */
export interface ActivityFeed {
	readonly activeRun: Run | null;
	onNodes(handler: (event: NodesEvent) => void): () => void;
	onRun(handler: (run: Run) => void): () => void;
	onReconnect(handler: () => void): () => void;
}

export interface RunStreamOptions {
	/** Called after the Run's state changes, and once after the first load (previous = null). */
	onStateChange?: (run: Run, previous: RunState | null) => void;
	feed?: ActivityFeed;
	now?: () => number;
	schedule?: FlushScheduler;
}

export class RunStream {
	readonly runId: string;
	#snapshot = $state.raw<RunSnapshot>(emptySnapshot());
	#loading = $state(true);
	#error = $state.raw<unknown>(null);
	#summary = $state.raw<FinalSummary | null>(null);
	#summaryLoading = $state(false);
	#summaryError = $state.raw<unknown>(null);

	readonly #options: RunStreamOptions;
	readonly #now: () => number;
	readonly #schedule: FlushScheduler;
	readonly #controller = new AbortController();
	readonly #stop: (() => void)[];
	#queue: RunEvent[] = [];
	#cancelFlush: (() => void) | null = null;
	/** The REST snapshot is applied; events that arrive earlier wait in the queue. */
	#ready = false;
	#destroyed = false;

	constructor(runId: string, options: RunStreamOptions = {}) {
		this.runId = runId;
		this.#options = options;
		this.#now = options.now ?? (() => performance.now());
		this.#schedule = options.schedule ?? frameScheduler;
		const feed = options.feed ?? activity;
		// Listen before loading so nothing that happens during the load is missed; the reducer
		// drops node data the tree already reflects.
		this.#stop = [
			feed.onNodes((event) => {
				if (event.run_id === runId) this.#enqueue({ type: 'nodes', nodes: event.nodes, cursor: event.cursor });
			}),
			feed.onRun((run) => {
				if (run.run_id === runId) this.#enqueue({ type: 'run', run });
			}),
			feed.onReconnect(() => {
				// Node events missed while the stream was down are not replayed; refetch once.
				const run = this.#snapshot.run;
				if (this.#ready && run && isActiveRunState(run.state)) void this.#refetchTree();
			}),
			// The active Run's snapshot, with its progress, arrives with every activity event.
			$effect.root(() => {
				$effect(() => {
					const active = feed.activeRun;
					if (active?.run_id === runId) untrack(() => this.#enqueue({ type: 'run', run: active }));
				});
			})
		];
		void this.#load();
	}

	get run(): Run | null {
		return this.#snapshot.run;
	}
	get progress(): RunProgress | null {
		return this.#snapshot.progress;
	}
	get progressAt(): number {
		return this.#snapshot.progressAt;
	}
	get nodes(): ReadonlyMap<string, TreeNode> {
		return this.#snapshot.nodes;
	}
	get children(): ChildrenIndex {
		return this.#snapshot.children;
	}
	get loading(): boolean {
		return this.#loading;
	}
	get error(): unknown {
		return this.#error;
	}
	get summary(): FinalSummary | null {
		return this.#summary;
	}
	get summaryLoading(): boolean {
		return this.#summaryLoading;
	}
	get summaryError(): unknown {
		return this.#summaryError;
	}

	/** Retry the initial load after an error. */
	reload(): void {
		if (this.#destroyed) return;
		this.#error = null;
		this.#loading = true;
		void this.#load();
	}

	/** Apply a Run returned by a REST call (Stop, Resume) right away. */
	applyRun(run: Run): void {
		this.#flush();
		this.#commit([{ type: 'run', run }]);
	}

	async loadSummary(): Promise<void> {
		if (this.#destroyed) return;
		this.#summaryLoading = true;
		this.#summaryError = null;
		try {
			const summary = await api.getSummary(this.runId, this.#controller.signal);
			if (!this.#destroyed) this.#summary = summary;
		} catch (error) {
			if (!isAbortError(error) && !this.#destroyed) this.#summaryError = error;
		} finally {
			if (!this.#destroyed) this.#summaryLoading = false;
		}
	}

	destroy(): void {
		this.#destroyed = true;
		this.#cancelFlush?.();
		this.#cancelFlush = null;
		this.#queue = [];
		for (const stop of this.#stop) stop();
		this.#controller.abort();
	}

	async #load(): Promise<void> {
		try {
			const signal = this.#controller.signal;
			const [run, tree] = await Promise.all([api.getRun(this.runId, signal), api.getTree(this.runId, signal)]);
			if (this.#destroyed) return;
			this.#snapshot = snapshotFromRest(run, tree, this.#now());
			this.#ready = true;
			this.#loading = false;
			this.#afterRunChange(null);
			this.#flush();
		} catch (error) {
			if (isAbortError(error) || this.#destroyed) return;
			this.#error = error;
			this.#loading = false;
			this.#queue = [];
		}
	}

	async #refetchTree(): Promise<void> {
		try {
			const tree = await api.getTree(this.runId, this.#controller.signal);
			this.#enqueue({ type: 'tree', tree });
		} catch (error) {
			if (!isAbortError(error) && !this.#destroyed) toasts.error(`Could not refresh the tree: ${errorMessage(error)}`);
		}
	}

	/** Fetch the final RunResponse once when the Run ends: the terminal event carries no progress. */
	async #refreshRun(): Promise<void> {
		try {
			const run = await api.getRun(this.runId, this.#controller.signal);
			this.#commit([{ type: 'run', run }]);
		} catch (error) {
			if (!isAbortError(error) && !this.#destroyed) toasts.error(`Could not refresh the Run: ${errorMessage(error)}`);
		}
	}

	#enqueue(event: RunEvent): void {
		// After a failed load the next reload fetches fresh state; queued events would be stale.
		if (this.#destroyed || (!this.#ready && this.#error !== null)) return;
		this.#queue.push(event);
		this.#cancelFlush ??= this.#schedule(() => {
			this.#cancelFlush = null;
			this.#flush();
		});
	}

	#flush(): void {
		if (!this.#ready) return;
		this.#cancelFlush?.();
		this.#cancelFlush = null;
		if (this.#queue.length === 0) return;
		const events = this.#queue;
		this.#queue = [];
		this.#commit(events);
	}

	#commit(events: readonly RunEvent[]): void {
		if (this.#destroyed || !this.#ready) return;
		const before = this.#snapshot;
		const after = applyRunEvents(before, events, this.#now());
		if (after === before) return;
		this.#snapshot = after;
		if (after.run !== before.run && after.run?.state !== before.run?.state) this.#afterRunChange(before.run?.state ?? null);
	}

	#afterRunChange(previous: RunState | null): void {
		const run = this.#snapshot.run;
		if (run === null) return;
		if (previous !== null && isActiveRunState(previous) && !isActiveRunState(run.state)) void this.#refreshRun();
		if (run.state === 'completed' && this.#summary === null && !this.#summaryLoading) void this.loadSummary();
		this.#options.onStateChange?.(run, previous);
	}
}
