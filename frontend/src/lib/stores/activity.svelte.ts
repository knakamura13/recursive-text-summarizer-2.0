// The tab's only EventSource. /activity/stream carries `activity` snapshots
// (the active Run with progress, importing Documents, cursor), `nodes`
// (changed nodes of the active Run), and `run` (any Run state change). The
// layout starts it once. It closes while the tab is hidden and reconnects
// with the last cursor when visible; onReconnect subscribers then refetch
// whatever may have gone stale.
import { api } from '$lib/api/client';
import { openEventStream, type EventStream, type StreamStatus } from '$lib/api/sse';
import type { Activity, DocumentSummary, NodesEvent, Run, RunState } from '$lib/api/types';

export type ActivityListener = (next: Activity, previous: Activity) => void;

const ACTIVE_STATES: readonly RunState[] = ['queued', 'running', 'stopping'];

class ActivityStore {
	#activeRun = $state.raw<Run | null>(null);
	#importing = $state.raw<DocumentSummary[]>([]);
	#status = $state<StreamStatus>('closed');
	#cursor = $state(0);
	#activityListeners = new Set<ActivityListener>();
	#nodesListeners = new Set<(event: NodesEvent) => void>();
	#runListeners = new Set<(run: Run) => void>();
	#reconnectListeners = new Set<() => void>();
	#stream: EventStream | null = null;
	#started = false;
	#openedBefore = false;

	get activeRun(): Run | null {
		return this.#activeRun;
	}

	get importing(): DocumentSummary[] {
		return this.#importing;
	}

	get status(): StreamStatus {
		return this.#status;
	}

	/** The latest event cursor; reconnects resume after it. */
	get cursor(): number {
		return this.#cursor;
	}

	/** Every `activity` snapshot, with the one it replaced. Returns the unsubscribe function. */
	onActivity(handler: ActivityListener): () => void {
		return this.#subscribe(this.#activityListeners, handler);
	}

	/** Changed nodes of the active Run. Returns the unsubscribe function. */
	onNodes(handler: (event: NodesEvent) => void): () => void {
		return this.#subscribe(this.#nodesListeners, handler);
	}

	/** Any Run state change, including the active Run ending. Returns the unsubscribe function. */
	onRun(handler: (run: Run) => void): () => void {
		return this.#subscribe(this.#runListeners, handler);
	}

	/** After the stream reopens (network loss, hidden tab): refetch what may be stale. */
	onReconnect(handler: () => void): () => void {
		return this.#subscribe(this.#reconnectListeners, handler);
	}

	/** Opens the stream (idempotent). Returns the function that closes it again. */
	start(): () => void {
		if (!this.#started) {
			this.#started = true;
			document.addEventListener('visibilitychange', this.#onVisibility);
			if (document.visibilityState !== 'hidden') this.#open();
		}
		return () => this.stop();
	}

	stop(): void {
		if (!this.#started) return;
		this.#started = false;
		document.removeEventListener('visibilitychange', this.#onVisibility);
		this.#close();
	}

	#subscribe<T>(listeners: Set<T>, handler: T): () => void {
		listeners.add(handler);
		return () => listeners.delete(handler);
	}

	#open(): void {
		if (this.#stream) return;
		this.#stream = openEventStream(
			() => api.activityStreamUrl(this.#cursor),
			{
				activity: (data, lastEventId) => {
					const snapshot = data as Activity;
					// The snapshot's cursor is the server's current position: adopt it
					// even when lower (a reset database restarts the sequence).
					this.#cursor = Number.isFinite(snapshot.cursor)
						? snapshot.cursor
						: Math.max(this.#cursor, Number.parseInt(lastEventId, 10) || 0);
					this.#applyActivity(snapshot);
				},
				nodes: (data, lastEventId) => {
					const event = data as NodesEvent;
					// Node payloads describe the batch's latest projection, but the
					// transport id advances only after its final frame is delivered.
					this.#advance(lastEventId);
					for (const listener of this.#nodesListeners) listener(event);
				},
				run: (data, lastEventId) => {
					const run = data as Run;
					this.#advance(lastEventId);
					this.#applyRun(run);
					for (const listener of this.#runListeners) listener(run);
				}
			},
			{ onStatus: (status) => this.#onStatus(status) }
		);
	}

	#close(): void {
		const stream = this.#stream;
		this.#stream = null;
		stream?.close();
	}

	#onVisibility = (): void => {
		if (document.visibilityState === 'hidden') this.#close();
		else this.#open();
	};

	#onStatus(status: StreamStatus): void {
		this.#status = status;
		if (status !== 'open') return;
		const reopened = this.#openedBefore;
		this.#openedBefore = true;
		if (reopened) for (const listener of this.#reconnectListeners) listener();
	}

	#advance(lastEventId: string): void {
		const next = Math.max(this.#cursor, Number.parseInt(lastEventId, 10) || 0);
		if (next !== this.#cursor) this.#cursor = next;
	}

	#applyActivity(next: Activity): void {
		const previous: Activity = {
			active_run: this.#activeRun,
			importing: this.#importing,
			cursor: this.#cursor
		};
		this.#activeRun = next.active_run ?? null;
		this.#importing = next.importing ?? [];
		const applied: Activity = {
			active_run: this.#activeRun,
			importing: this.#importing,
			cursor: this.#cursor
		};
		for (const listener of this.#activityListeners) listener(applied, previous);
	}

	/** A state change reaches the header before the next snapshot does. */
	#applyRun(run: Run): void {
		const current = this.#activeRun;
		const active = ACTIVE_STATES.includes(run.state);
		if (current?.run_id === run.run_id) {
			this.#activeRun = active ? { ...run, progress: current.progress } : null;
		} else if (active && !current) {
			this.#activeRun = run;
		}
	}
}

export const activity = new ActivityStore();
