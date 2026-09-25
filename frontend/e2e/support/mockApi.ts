// In-memory stand-in for the /api/v1 backend, installed with page.route so
// e2e specs run without the Python server.
//
//   const mock = await MockApi.install(page, { documents: [makeDocument({ title: 'Notes' })] });
//   mock.on('POST', '/documents', () => apiError(415, 'unsupported_format', 'Not supported.'));
//   await page.goto('/');
//
// State lives in public fields (documents, runs, trees, ...) that specs may
// mutate at any time; every response reads the current state. Handlers added
// with `on()` run before the built-in ones (newest first); returning
// undefined falls through.
//
// SSE: page.route cannot hold a response open, so a real EventSource would
// reconnect after every response. install() therefore replaces EventSource
// with a shim that fetches the stream URL through page.route, delivers the
// events of that response, and then stays open like the real server's
// stream. Later events come from push()/pushActivity()/pushRun()/
// pushNodes(); dropStreams() simulates a lost connection (the app reconnects
// with ?after=<cursor>).
import { setTimeout as sleep } from 'node:timers/promises';
import type { Page, Route } from '@playwright/test';
import type {
	Activity,
	DocumentDetail,
	FinalSummary,
	NodeDetail,
	OllamaHealth,
	OllamaModel,
	Preflight,
	Run,
	RunConfig,
	RunProgress,
	RunState,
	SegmentRef,
	Settings,
	SourcePage,
	TreeNode
} from '../../src/lib/api/types';

export const CSRF_TOKEN = 'e2e-csrf-token';

export interface MockRequest {
	method: string;
	/** Path below /api/v1, e.g. "/documents/doc-1". */
	path: string;
	params: Record<string, string>;
	query: URLSearchParams;
	headers: Record<string, string>;
	/** Parsed JSON body; undefined when absent or not JSON. */
	body: unknown;
	/** Raw request body (multipart uploads). */
	raw: Buffer | null;
}

export interface MockResponse {
	/** Defaults to 200. */
	status?: number;
	/** Serialized as the JSON body. */
	json?: unknown;
	/** Raw body, used when `json` is absent. */
	body?: string | Buffer;
	contentType?: string;
	headers?: Record<string, string>;
	/** Holds the response back, e.g. to observe a pending state. */
	delayMs?: number;
}

export type MockHandler = (
	request: MockRequest,
	mock: MockApi
) => MockResponse | undefined | Promise<MockResponse | undefined>;

export interface SseEvent {
	event: string;
	data: unknown;
	id?: number | string;
}

export interface MockSeed {
	documents?: DocumentDetail[];
	/** Canonical text per document_id, served by /documents/{id}/source. */
	sources?: Record<string, string>;
	pages?: Record<string, SourcePage[]>;
	settings?: Settings;
	models?: OllamaModel[];
	ollama?: OllamaHealth;
	runs?: Run[];
	trees?: Record<string, TreeNode[]>;
	/** Keyed `${runId}/${nodeId}`. */
	nodes?: Record<string, NodeDetail>;
	segments?: Record<string, SegmentRef[]>;
	summaries?: Record<string, FinalSummary>;
	preflight?: Preflight;
}

// --- Factories ---------------------------------------------------------------

let sequence = 0;
const nextId = (prefix: string) => `${prefix}-${++sequence}`;
const iso = (minutesAgo = 0) => new Date(Date.now() - minutesAgo * 60_000).toISOString();

export function defaultRunConfig(overrides: Partial<RunConfig> = {}): RunConfig {
	return {
		model: 'llama3.1:8b',
		target_words: 300,
		strategy: 'auto',
		verify: true,
		max_repair_passes: 1,
		citations: true,
		context_window: null,
		max_output_tokens: 1024,
		safety_margin_tokens: 256,
		safety_margin_fraction: 0.02,
		chunk_tokens: null,
		overlap_tokens: 0,
		max_merge_children: null,
		max_concurrency: 1,
		timeout_seconds: 180,
		max_retries: 5,
		...overrides
	};
}

export function makeSettings(overrides: Partial<Settings> = {}): Settings {
	return { ollama_host: 'http://localhost:11434', defaults: defaultRunConfig(), ...overrides };
}

export function makeModel(name: string, overrides: Partial<OllamaModel> = {}): OllamaModel {
	return {
		name,
		size_bytes: 4_920_000_000,
		parameter_size: '8.0B',
		family: 'llama',
		quantization: 'Q4_K_M',
		modified_at: iso(60 * 24 * 3),
		...overrides
	};
}

export function makeDocument(overrides: Partial<DocumentDetail> = {}): DocumentDetail {
	const id = overrides.document_id ?? nextId('doc');
	const title = overrides.title ?? `Document ${id}`;
	return {
		document_id: id,
		title,
		filename: `${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}.txt`,
		format: 'txt',
		origin: 'upload',
		size_bytes: 12_345,
		import_state: 'ready',
		import_progress: null,
		import_error: null,
		char_count: 12_000,
		page_count: null,
		created_at: iso(120),
		updated_at: iso(60),
		latest_run: null,
		source_sha256: null,
		import_report: null,
		...overrides
	};
}

export function makeProgress(overrides: Partial<RunProgress> = {}): RunProgress {
	return {
		cursor: 1,
		attempt_number: 1,
		started_at: iso(5),
		elapsed_seconds: 300,
		stage: 'summarizing',
		stages: [
			{ stage: 'preparing', state: 'completed', completed: null, total: null, detail: 'hierarchical' },
			{ stage: 'segmenting', state: 'completed', completed: 40, total: 40, detail: null },
			{ stage: 'summarizing', state: 'active', completed: 12, total: 40, detail: null },
			{ stage: 'merging', state: 'pending', completed: null, total: null, detail: null },
			{ stage: 'writing', state: 'pending', completed: null, total: null, detail: null },
			{ stage: 'verifying', state: 'pending', completed: null, total: null, detail: null },
			{ stage: 'publishing', state: 'pending', completed: null, total: null, detail: null }
		],
		leaves: { done: 12, total: 40, reused: 0, failed: 0 },
		merges: { done: 0, total: null, reused: 0, failed: 0 },
		merge_levels: [],
		claims: { done: 0, total: null, reused: 0, failed: 0 },
		current_items: [],
		eta_seconds: 840,
		eta_basis: 'Based on 12 model calls in this Attempt.',
		...overrides
	};
}

const ACTIVE_STATES: RunState[] = ['queued', 'running', 'stopping'];

export function makeRun(overrides: Partial<Run> = {}): Run {
	const state = overrides.state ?? 'completed';
	const active = ACTIVE_STATES.includes(state);
	return {
		run_id: nextId('run'),
		document_id: 'doc-1',
		document_title: 'Document',
		state,
		requested_strategy: 'auto',
		selected_strategy: 'hierarchical',
		config: defaultRunConfig(),
		created_at: iso(30),
		updated_at: iso(1),
		attempt: {
			attempt_id: nextId('attempt'),
			attempt_number: 1,
			state,
			started_at: iso(30),
			ended_at: active ? null : iso(1),
			failure: null
		},
		attempt_count: 1,
		failure: null,
		can_stop: state === 'queued' || state === 'running',
		can_resume: state === 'stopped' || state === 'failed' || state === 'interrupted',
		progress: active ? makeProgress() : null,
		...overrides
	};
}

// --- Responses ----------------------------------------------------------------

export function apiError(
	status: number,
	code: string,
	message: string,
	details: Record<string, unknown> | null = null,
	retryable = false
): MockResponse {
	return { status, json: { code, message, details, retryable } };
}

/** A text/event-stream body; the response ends after the events. */
export function sse(events: SseEvent[]): MockResponse {
	const body = events
		.map(({ event, data, id }) => {
			const lines = [`event: ${event}`];
			if (id !== undefined) lines.push(`id: ${id}`);
			lines.push(`data: ${JSON.stringify(data)}`);
			return `${lines.join('\n')}\n\n`;
		})
		.join('');
	return {
		body,
		contentType: 'text/event-stream',
		headers: { 'Cache-Control': 'no-cache' }
	};
}

function summaryOf(document: DocumentDetail) {
	const { source_sha256: _sha, import_report: _report, ...summary } = document;
	return summary;
}

function withoutProgress(run: Run): Run {
	return { ...run, progress: null };
}

/** The file part of a multipart/form-data body. */
function parseUpload(raw: Buffer | null, contentType: string): { filename: string; content: Buffer } {
	const boundary = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType);
	if (!raw || !boundary) return { filename: 'upload', content: Buffer.alloc(0) };
	const delimiter = Buffer.from(`--${boundary[1] ?? boundary[2]}`);
	let start = raw.indexOf(delimiter);
	while (start !== -1) {
		const next = raw.indexOf(delimiter, start + delimiter.length);
		if (next === -1) break;
		const part = raw.subarray(start + delimiter.length + 2, next - 2);
		const headerEnd = part.indexOf('\r\n\r\n');
		const headers = part.subarray(0, headerEnd).toString('utf8');
		const filename = /filename="([^"]*)"/i.exec(headers);
		if (filename) return { filename: filename[1], content: part.subarray(headerEnd + 4) };
		start = next;
	}
	return { filename: 'upload', content: Buffer.alloc(0) };
}

// --- Mock ---------------------------------------------------------------------

interface CompiledRoute {
	method: string;
	pattern: RegExp;
	names: string[];
	handler: MockHandler;
}

function compile(method: string, path: string, handler: MockHandler): CompiledRoute {
	const names: string[] = [];
	const source = path
		.split('/')
		.map((segment) => {
			if (!segment.startsWith(':')) return segment.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
			names.push(segment.slice(1));
			return '([^/]+)';
		})
		.join('/');
	return { method: method.toUpperCase(), pattern: new RegExp(`^${source}$`), names, handler };
}

interface MockSseWindow {
	__mockSse: {
		push(type: string, data: string, id?: string): void;
		drop(): void;
		openCount(): number;
	};
}

/**
 * Browser-side EventSource replacement (see the file comment). Playwright
 * serializes this function into the page, so it must stay self-contained.
 */
function installEventSourceShim(): void {
	const CLOSED = 2;
	const sources = new Set<ShimEventSource>();

	class ShimEventSource extends EventTarget {
		static readonly CONNECTING = 0;
		static readonly OPEN = 1;
		static readonly CLOSED = 2;
		readonly CONNECTING = 0;
		readonly OPEN = 1;
		readonly CLOSED = 2;
		readonly url: string;
		readonly withCredentials = false;
		readyState = 0;
		lastEventId = '';
		onopen: ((event: Event) => void) | null = null;
		onmessage: ((event: MessageEvent) => void) | null = null;
		onerror: ((event: Event) => void) | null = null;

		constructor(url: string | URL) {
			super();
			this.url = new URL(String(url), location.href).href;
			sources.add(this);
			void this.connect();
		}

		async connect(): Promise<void> {
			let text: string;
			try {
				const response = await fetch(this.url, { headers: { Accept: 'text/event-stream' } });
				const type = response.headers.get('content-type') ?? '';
				if (!response.ok || !type.includes('text/event-stream')) throw new Error(`HTTP ${response.status}`);
				text = await response.text();
			} catch {
				this.fail();
				return;
			}
			if (this.readyState === CLOSED) return;
			this.readyState = 1;
			this.fire(new Event('open'));
			for (const block of text.split(/\r?\n\r?\n/)) {
				let type = 'message';
				let id: string | undefined;
				const data: string[] = [];
				for (const line of block.split(/\r?\n/)) {
					const colon = line.indexOf(':');
					if (colon <= 0) continue;
					const field = line.slice(0, colon);
					const value = line.slice(colon + 1).replace(/^ /, '');
					if (field === 'event') type = value;
					else if (field === 'data') data.push(value);
					else if (field === 'id') id = value;
				}
				if (data.length > 0) this.emit(type, data.join('\n'), id);
			}
		}

		emit(type: string, data: string, id?: string): void {
			if (this.readyState !== 1) return;
			if (id !== undefined) this.lastEventId = id;
			this.fire(new MessageEvent(type, { data, lastEventId: this.lastEventId }));
		}

		fail(): void {
			if (this.readyState === CLOSED) return;
			this.readyState = CLOSED;
			sources.delete(this);
			this.fire(new Event('error'));
		}

		close(): void {
			this.readyState = CLOSED;
			sources.delete(this);
		}

		fire(event: Event): void {
			this.dispatchEvent(event);
			if (event.type === 'open') this.onopen?.call(this, event);
			else if (event.type === 'error') this.onerror?.call(this, event);
			else if (event.type === 'message') this.onmessage?.call(this, event as MessageEvent);
		}
	}

	const control: MockSseWindow['__mockSse'] = {
		push(type, data, id) {
			for (const source of sources) source.emit(type, data, id);
		},
		drop() {
			for (const source of [...sources]) source.fail();
		},
		openCount() {
			return [...sources].filter((source) => source.readyState === 1).length;
		}
	};
	Object.defineProperty(window, 'EventSource', { value: ShimEventSource, configurable: true, writable: true });
	Object.defineProperty(window, '__mockSse', { value: control, configurable: true });
}

export class MockApi {
	readonly page: Page;
	documents: DocumentDetail[];
	sources: Record<string, string>;
	pages: Record<string, SourcePage[]>;
	settings: Settings;
	models: OllamaModel[];
	ollama: OllamaHealth;
	runs: Run[];
	trees: Record<string, TreeNode[]>;
	nodes: Record<string, NodeDetail>;
	segments: Record<string, SegmentRef[]>;
	summaries: Record<string, FinalSummary>;
	preflight: Preflight;
	/** Every /api/v1 request, in order. */
	readonly requests: MockRequest[] = [];
	/** Requests no handler answered (they got a 404). */
	readonly unhandled: MockRequest[] = [];

	#overrides: CompiledRoute[] = [];
	#routes: CompiledRoute[] = [];
	#idempotency = new Map<string, string>();
	/** Event cursor: stamped on stream events, tree responses, and Activity. */
	cursor = 1;

	private constructor(page: Page, seed: MockSeed) {
		this.page = page;
		this.documents = seed.documents ?? [];
		this.sources = seed.sources ?? {};
		this.pages = seed.pages ?? {};
		this.settings = seed.settings ?? makeSettings();
		this.models = seed.models ?? [makeModel('llama3.1:8b'), makeModel('qwen2.5:14b', { parameter_size: '14.8B', family: 'qwen2' })];
		this.ollama = seed.ollama ?? {
			connected: true,
			message: 'Connected',
			host: this.settings.ollama_host,
			version: '0.6.2'
		};
		this.runs = seed.runs ?? [];
		this.trees = seed.trees ?? {};
		this.nodes = seed.nodes ?? {};
		this.segments = seed.segments ?? {};
		this.summaries = seed.summaries ?? {};
		this.preflight = seed.preflight ?? {
			ok: true,
			selected_strategy: 'hierarchical',
			context_window_tokens: 32_768,
			context_window_source: 'model',
			usable_input_capacity: 28_000,
			document_tokens: 90_000,
			estimated_leaf_count: 4,
			estimated_model_calls: 9,
			model_installed: true,
			errors: [],
			warnings: []
		};
		this.#routes = this.#builtInRoutes();
	}

	static async install(page: Page, seed: MockSeed = {}): Promise<MockApi> {
		const mock = new MockApi(page, seed);
		await page.addInitScript(installEventSourceShim);
		await page.route(
			(url) => url.pathname.startsWith('/api/v1/'),
			(route) => mock.#handle(route)
		);
		return mock;
	}

	/** Adds a handler that runs before the built-in routes. `path` is below /api/v1, with `:name` params. */
	on(method: string, path: string, handler: MockHandler): void {
		this.#overrides.unshift(compile(method, path, handler));
	}

	/** The current Activity: the active Run (with progress), importing Documents, and the cursor. */
	activity(): Activity {
		const active = this.runs.find((run) => ACTIVE_STATES.includes(run.state)) ?? null;
		return {
			active_run: active ? { ...active, progress: active.progress ?? makeProgress() } : null,
			importing: this.documents.filter((document) => document.import_state === 'importing').map(summaryOf),
			cursor: this.cursor
		};
	}

	/** Recorded requests for `method` + `path` pattern (same syntax as `on`). */
	calls(method: string, path: string): MockRequest[] {
		const { pattern } = compile(method, path, () => undefined);
		return this.requests.filter(
			(request) => request.method === method.toUpperCase() && pattern.test(request.path)
		);
	}

	/** Resolves with the first matching request, waiting for it if needed. */
	async waitForCall(method: string, path: string, timeoutMs = 10_000): Promise<MockRequest> {
		const deadline = Date.now() + timeoutMs;
		for (;;) {
			const [found] = this.calls(method, path);
			if (found) return found;
			if (Date.now() > deadline) throw new Error(`No ${method} ${path} request within ${timeoutMs} ms`);
			await sleep(50);
		}
	}

	document(id: string): DocumentDetail | undefined {
		return this.documents.find((document) => document.document_id === id);
	}

	/** Resolves once the app has an open activity stream. */
	async waitForStream(timeoutMs = 10_000): Promise<void> {
		const deadline = Date.now() + timeoutMs;
		while ((await this.openStreams()) === 0) {
			if (Date.now() > deadline) throw new Error(`No open activity stream within ${timeoutMs} ms`);
			await sleep(50);
		}
	}

	/** Number of open streams in the page (the app keeps exactly one while visible). */
	openStreams(): Promise<number> {
		return this.page.evaluate(() => (window as unknown as MockSseWindow).__mockSse.openCount());
	}

	/** Sends one event on the open stream, JSON-encoded like the server. Waits for the stream first. */
	async push(event: string, data: unknown, id: number = this.cursor): Promise<void> {
		await this.waitForStream();
		await this.page.evaluate(
			([type, payload, eventId]) => (window as unknown as MockSseWindow).__mockSse.push(type, payload, eventId),
			[event, JSON.stringify(data), String(id)] as const
		);
	}

	/** Sends the current activity() snapshot. */
	async pushActivity(): Promise<void> {
		await this.push('activity', this.activity(), this.cursor);
	}

	/** Sends a `run` state change (progress null) with the next cursor. */
	async pushRun(run: Run): Promise<void> {
		this.cursor += 1;
		await this.push('run', withoutProgress(run), this.cursor);
	}

	/** Sends changed nodes of a Run with the next cursor. */
	async pushNodes(runId: string, nodes: TreeNode[]): Promise<void> {
		this.cursor += 1;
		await this.push('nodes', { run_id: runId, nodes, cursor: this.cursor }, this.cursor);
	}

	/** Fails every open stream, as a lost connection would; the app reconnects with its cursor. */
	async dropStreams(): Promise<void> {
		await this.page.evaluate(() => (window as unknown as MockSseWindow).__mockSse.drop());
	}

	async #handle(route: Route): Promise<void> {
		const request = route.request();
		const url = new URL(request.url());
		const path = url.pathname.slice('/api/v1'.length);
		const headers = await request.allHeaders();
		const raw = request.postDataBuffer();
		let body: unknown;
		if (raw && (headers['content-type'] ?? '').includes('application/json')) {
			try {
				body = JSON.parse(raw.toString('utf8'));
			} catch {
				body = undefined;
			}
		}
		const base: MockRequest = {
			method: request.method(),
			path,
			params: {},
			query: url.searchParams,
			headers,
			body,
			raw
		};
		this.requests.push(base);

		let response: MockResponse | undefined;
		if (base.method !== 'GET' && headers['x-csrf-token'] !== CSRF_TOKEN) {
			response = apiError(403, 'csrf_invalid', 'The session token is missing or stale. Retry the request.', null, true);
		}
		for (const candidate of [...this.#overrides, ...this.#routes]) {
			if (response) break;
			if (candidate.method !== base.method) continue;
			const match = candidate.pattern.exec(path);
			if (!match) continue;
			const params: Record<string, string> = {};
			candidate.names.forEach((name, index) => (params[name] = decodeURIComponent(match[index + 1])));
			response = await candidate.handler({ ...base, params }, this);
		}
		if (!response) {
			this.unhandled.push(base);
			response = apiError(404, 'not_found', `Unmocked API route: ${base.method} ${path}`);
		}
		if (response.delayMs) await sleep(response.delayMs);
		const status = response.status ?? 200;
		const isJson = response.json !== undefined;
		const contentType = isJson ? 'application/json' : (response.contentType ?? 'text/plain; charset=utf-8');
		await route.fulfill({
			status,
			headers: {
				'X-CSRF-Token': CSRF_TOKEN,
				...(status === 204 ? {} : { 'Content-Type': contentType }),
				...response.headers
			},
			body: isJson ? JSON.stringify(response.json) : (response.body ?? '')
		});
	}

	#runOr404(id: string): Run | MockResponse {
		return this.runs.find((run) => run.run_id === id) ?? apiError(404, 'run_not_found', 'Run not found.');
	}

	#builtInRoutes(): CompiledRoute[] {
		const notFoundDocument = () => apiError(404, 'document_not_found', 'Document not found.');
		const routes: [string, string, MockHandler][] = [
			['GET', '/health', () => ({ json: { status: 'ok' } })],

			['GET', '/settings', () => ({ json: this.settings })],
			[
				'PATCH',
				'/settings',
				({ body }) => {
					const update = (body ?? {}) as { ollama_host?: string; defaults?: Record<string, unknown> };
					if (update.ollama_host !== undefined) this.settings.ollama_host = update.ollama_host;
					if (update.defaults) {
						const { clear = [], ...fields } = update.defaults as { clear?: string[] };
						const defaults: Record<string, unknown> = { ...this.settings.defaults };
						for (const [key, value] of Object.entries(fields)) if (value !== null) defaults[key] = value;
						for (const key of clear) defaults[key] = null;
						this.settings.defaults = defaults as unknown as RunConfig;
					}
					return { json: this.settings };
				}
			],

			['GET', '/ollama/health', () => ({ json: { ...this.ollama, host: this.settings.ollama_host } })],
			[
				'GET',
				'/ollama/models',
				() =>
					this.ollama.connected
						? { json: { models: this.models } }
						: apiError(503, 'ollama_unreachable', this.ollama.message, null, true)
			],

			[
				'GET',
				'/documents',
				({ query }) => {
					const search = (query.get('search') ?? '').toLowerCase();
					const documents = this.documents
						.filter(
							(document) =>
								!search ||
								document.title.toLowerCase().includes(search) ||
								document.filename.toLowerCase().includes(search)
						)
						.sort((a, b) => b.updated_at.localeCompare(a.updated_at))
						.map(summaryOf);
					return { json: { documents } };
				}
			],
			[
				'POST',
				'/documents',
				({ raw, headers }) => {
					const { filename, content } = parseUpload(raw, headers['content-type'] ?? '');
					const existing = this.documents.find(
						(document) =>
							document.filename === filename &&
							document.size_bytes === content.length &&
							document.import_state !== 'failed'
					);
					if (existing) {
						return { json: { document: summaryOf(existing), already_imported: true } };
					}
					const document = makeDocument({
						title: filename.replace(/\.[^.]+$/, ''),
						filename,
						size_bytes: content.length,
						import_state: 'importing',
						import_progress: { phase: 'queued', done: 0, total: null, unit: null, message: null },
						char_count: null,
						created_at: iso(),
						updated_at: iso()
					});
					this.documents.unshift(document);
					this.sources[document.document_id] = content.toString('utf8');
					return { status: 201, json: { document: summaryOf(document), already_imported: false } };
				}
			],
			[
				'POST',
				'/documents/text',
				({ body }) => {
					const { title, text } = body as { title?: string; text: string };
					const firstLine = text.trim().split('\n')[0].slice(0, 80);
					const document = makeDocument({
						title: title?.trim() || firstLine,
						filename: 'pasted-text.txt',
						origin: 'paste',
						size_bytes: Buffer.byteLength(text),
						char_count: Array.from(text).length,
						created_at: iso(),
						updated_at: iso()
					});
					this.documents.unshift(document);
					this.sources[document.document_id] = text;
					return { status: 201, json: { document: summaryOf(document), already_imported: false } };
				}
			],
			['GET', '/documents/:id', ({ params }) => {
				const document = this.document(params.id);
				return document ? { json: document } : notFoundDocument();
			}],
			[
				'PATCH',
				'/documents/:id',
				({ params, body }) => {
					const document = this.document(params.id);
					if (!document) return notFoundDocument();
					document.title = (body as { title: string }).title;
					document.updated_at = iso();
					return { json: document };
				}
			],
			[
				'DELETE',
				'/documents/:id',
				({ params }) => {
					const document = this.document(params.id);
					if (!document) return notFoundDocument();
					const active = this.runs.find(
						(run) => run.document_id === params.id && ACTIVE_STATES.includes(run.state)
					);
					if (active) {
						return apiError(409, 'run_active', 'Stop the active Run before deleting this Document.', {
							run_id: active.run_id,
							document_id: active.document_id,
							document_title: active.document_title,
							state: active.state
						});
					}
					this.documents = this.documents.filter((item) => item.document_id !== params.id);
					this.runs = this.runs.filter((run) => run.document_id !== params.id);
					return { status: 204 };
				}
			],
			[
				'GET',
				'/documents/:id/source',
				({ params, query }) => {
					const text = this.sources[params.id];
					if (text === undefined) return notFoundDocument();
					const points = Array.from(text);
					const offset = Number(query.get('offset') ?? 0);
					const limit = Number(query.get('limit') ?? 16_384);
					return {
						json: {
							text: points.slice(offset, offset + limit).join(''),
							offset,
							total_length: points.length,
							has_more: offset + limit < points.length
						}
					};
				}
			],
			['GET', '/documents/:id/pages', ({ params }) => ({ json: { pages: this.pages[params.id] ?? [] } })],
			[
				'GET',
				'/documents/:id/original',
				({ params }) => ({ body: this.sources[params.id] ?? '', contentType: 'text/plain; charset=utf-8' })
			],
			[
				'GET',
				'/documents/:id/runs',
				({ params }) => ({
					json: {
						runs: this.runs
							.filter((run) => run.document_id === params.id)
							.sort((a, b) => b.created_at.localeCompare(a.created_at))
							.map(withoutProgress)
					}
				})
			],

			['POST', '/preflight', () => ({ json: this.preflight })],
			[
				'POST',
				'/runs',
				({ body, headers }) => {
					const key = headers['idempotency-key'];
					const repeated = key ? this.runs.find((run) => run.run_id === this.#idempotency.get(key)) : undefined;
					if (repeated) return { json: repeated };
					const { document_id, config } = body as { document_id: string; config: RunConfig };
					const document = this.document(document_id);
					if (!document) return notFoundDocument();
					const active = this.runs.find((run) => ACTIVE_STATES.includes(run.state));
					if (active) {
						return apiError(409, 'run_active', 'Another Run is active.', {
							run_id: active.run_id,
							document_id: active.document_id,
							document_title: active.document_title,
							state: active.state
						});
					}
					const run = makeRun({
						document_id,
						document_title: document.title,
						state: 'queued',
						config,
						created_at: iso(),
						updated_at: iso()
					});
					this.runs.unshift(run);
					if (key) this.#idempotency.set(key, run.run_id);
					return { status: 201, json: run };
				}
			],
			[
				'GET',
				'/runs/active',
				() => ({ json: { run: this.activity().active_run } })
			],
			['GET', '/runs/:id', ({ params }) => {
				const run = this.#runOr404(params.id);
				return 'run_id' in run ? { json: run } : run;
			}],
			[
				'POST',
				'/runs/:id/stop',
				({ params }) => {
					const run = this.#runOr404(params.id);
					if (!('run_id' in run)) return run;
					if (!run.can_stop) return apiError(409, 'run_not_stoppable', 'This Run is not active.');
					run.state = run.state === 'queued' ? 'stopped' : 'stopping';
					run.can_stop = false;
					run.updated_at = iso();
					return { status: 202, json: run };
				}
			],
			[
				'POST',
				'/runs/:id/resume',
				({ params }) => {
					const run = this.#runOr404(params.id);
					if (!('run_id' in run)) return run;
					if (!run.can_resume) return apiError(409, 'run_not_resumable', 'This Run cannot be resumed.');
					run.state = 'queued';
					run.can_resume = false;
					run.can_stop = true;
					run.attempt_count += 1;
					run.updated_at = iso();
					return { status: 202, json: run };
				}
			],
			[
				'DELETE',
				'/runs/:id',
				({ params }) => {
					const run = this.#runOr404(params.id);
					if (!('run_id' in run)) return run;
					if (ACTIVE_STATES.includes(run.state)) {
						return apiError(409, 'run_active', 'Stop the Run before deleting it.', {
							run_id: run.run_id,
							document_id: run.document_id,
							document_title: run.document_title,
							state: run.state
						});
					}
					this.runs = this.runs.filter((item) => item.run_id !== params.id);
					return { status: 204 };
				}
			],
			[
				'GET',
				'/runs/:id/tree',
				({ params }) => ({ json: { nodes: this.trees[params.id] ?? [], cursor: this.cursor } })
			],
			[
				'GET',
				'/runs/:id/nodes/:nodeId',
				({ params }) =>
					this.nodes[`${params.id}/${params.nodeId}`]
						? { json: this.nodes[`${params.id}/${params.nodeId}`] }
						: apiError(404, 'node_not_found', 'Node not found.')
			],
			['GET', '/runs/:id/segments', ({ params }) => ({ json: { segments: this.segments[params.id] ?? [] } })],
			[
				'GET',
				'/runs/:id/summary',
				({ params }) => ({
					json: this.summaries[params.id] ?? {
						available: false,
						text: null,
						sentences: [],
						removed_sentences: [],
						citations: [],
						word_count: null,
						target_words: null,
						short_of_target: false,
						notices: [],
						verification_state: 'not_run',
						publication: null
					}
				})
			],
			[
				'GET',
				'/runs/:id/export/:format',
				({ params }) => ({
					body: this.summaries[params.id]?.text ?? '',
					headers: { 'Content-Disposition': `attachment; filename="summary.${params.format}"` }
				})
			],

			['GET', '/activity', () => ({ json: this.activity() })],
			[
				'GET',
				'/activity/stream',
				({ query }) => {
					const events: SseEvent[] = [{ event: 'activity', data: this.activity(), id: this.cursor }];
					const active = this.activity().active_run;
					if (Number(query.get('after') ?? 0) > 0 && active) {
						events.push({
							event: 'nodes',
							data: { run_id: active.run_id, nodes: this.trees[active.run_id] ?? [], cursor: this.cursor },
							id: this.cursor
						});
					}
					return sse(events);
				}
			]
		];
		return routes.map(([method, path, handler]) => compile(method, path, handler));
	}
}
