// Controllable stand-in for EventSource (jsdom has none). Install with
// vi.stubGlobal('EventSource', FakeEventSource) and drive connections by hand.
export class FakeEventSource {
	static instances: FakeEventSource[] = [];

	static get latest(): FakeEventSource {
		const latest = FakeEventSource.instances.at(-1);
		if (!latest) throw new Error('No EventSource was created');
		return latest;
	}

	static reset(): void {
		FakeEventSource.instances = [];
	}

	readonly url: string;
	closed = false;
	onopen: ((event: Event) => void) | null = null;
	onerror: ((event: Event) => void) | null = null;
	#listeners = new Map<string, ((event: MessageEvent) => void)[]>();

	constructor(url: string) {
		this.url = url;
		FakeEventSource.instances.push(this);
	}

	addEventListener(type: string, listener: (event: MessageEvent) => void): void {
		this.#listeners.set(type, [...(this.#listeners.get(type) ?? []), listener]);
	}

	close(): void {
		this.closed = true;
	}

	open(): void {
		this.onopen?.(new Event('open'));
	}

	fail(): void {
		this.onerror?.(new Event('error'));
	}

	/** Delivers a named event; `data` objects are JSON-encoded like the server's. */
	emit(type: string, data: unknown, lastEventId = ''): void {
		const text = typeof data === 'string' ? data : JSON.stringify(data);
		for (const listener of this.#listeners.get(type) ?? []) {
			listener(new MessageEvent(type, { data: text, lastEventId }));
		}
	}
}
