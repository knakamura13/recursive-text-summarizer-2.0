// EventSource wrapper that owns reconnection: the browser's built-in retry
// would reuse the original URL, but the stream resumes from the latest cursor,
// so every reconnect re-evaluates `url()`. Backoff doubles from 1 s to 30 s and
// resets once a connection opens; regaining connectivity skips the remaining
// wait.

export type StreamStatus = 'connecting' | 'open' | 'reconnecting' | 'closed';

export type StreamHandlers = Record<string, (data: unknown, lastEventId: string) => void>;

export interface StreamOptions {
	onStatus?: (status: StreamStatus) => void;
	/** Event names that end the stream after their handler runs. */
	closeOn?: string[];
}

export interface EventStream {
	close(): void;
}

export const RECONNECT_INITIAL_MS = 1_000;
export const RECONNECT_MAX_MS = 30_000;

export function openEventStream(
	url: () => string,
	handlers: StreamHandlers,
	options: StreamOptions = {}
): EventStream {
	const closeOn = options.closeOn ?? [];
	let source: EventSource | null = null;
	let retryTimer: ReturnType<typeof setTimeout> | undefined;
	let failures = 0;
	let closed = false;
	let status: StreamStatus | null = null;

	function setStatus(next: StreamStatus): void {
		if (next === status) return;
		status = next;
		options.onStatus?.(next);
	}

	function connect(): void {
		retryTimer = undefined;
		setStatus(failures === 0 ? 'connecting' : 'reconnecting');
		const current = new EventSource(url());
		source = current;
		current.onopen = () => {
			if (source !== current) return;
			failures = 0;
			setStatus('open');
		};
		current.onerror = () => {
			if (source !== current) return;
			current.close();
			source = null;
			const delay = Math.min(RECONNECT_MAX_MS, RECONNECT_INITIAL_MS * 2 ** failures);
			failures += 1;
			setStatus('reconnecting');
			retryTimer = setTimeout(connect, delay);
		};
		for (const [name, handler] of Object.entries(handlers)) {
			current.addEventListener(name, (event) => {
				if (source !== current) return;
				const message = event as MessageEvent<string>;
				let data: unknown = message.data;
				try {
					data = JSON.parse(message.data);
				} catch {
					// Not JSON: hand the raw text to the handler.
				}
				try {
					handler(data, message.lastEventId);
				} finally {
					if (closeOn.includes(name)) close();
				}
			});
		}
	}

	function reconnectNow(): void {
		if (closed || retryTimer === undefined) return;
		clearTimeout(retryTimer);
		connect();
	}

	function close(): void {
		if (closed) return;
		closed = true;
		clearTimeout(retryTimer);
		retryTimer = undefined;
		source?.close();
		source = null;
		window.removeEventListener('online', reconnectNow);
		setStatus('closed');
	}

	window.addEventListener('online', reconnectNow);
	connect();
	return { close };
}
