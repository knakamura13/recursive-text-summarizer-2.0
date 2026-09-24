// Typed client for the /api/v1 HTTP API.
//
// Errors: every non-2xx JSON response becomes an ApiError carrying the server's
// {code, message, details, retryable}; a failed fetch, a timeout, or a proxy
// answering for an unreachable backend becomes a NetworkError. Aborting a
// caller's signal rejects with a DOMException named "AbortError" (see
// isAbortError) so callers can ignore superseded requests.
//
// CSRF: the server sends the session's token in X-CSRF-Token on every /api
// response. The client remembers the latest one, fetches it lazily through
// /health before the first mutating request, and retries a mutating request
// once when the server rejects it as csrf_invalid.
import type {
	Activity,
	DocumentCreated,
	DocumentDetail,
	DocumentSummary,
	ErrorBody,
	ExportFormat,
	FinalSummary,
	NodeDetail,
	OllamaHealth,
	OllamaModel,
	Preflight,
	Run,
	RunConfig,
	SegmentRef,
	Settings,
	SettingsUpdate,
	SourcePage,
	SourceSlice,
	Tree
} from './types';

export const API_BASE = '/api/v1';
export const DEFAULT_TIMEOUT_MS = 30_000;
/** Pasted text may be up to 20 MiB; give its upload more time than a normal request. */
const PASTE_TIMEOUT_MS = 120_000;
const CSRF_HEADER = 'X-CSRF-Token';

export class ApiError extends Error {
	readonly status: number;
	readonly code: string;
	readonly details: Record<string, unknown> | null;
	readonly retryable: boolean;

	constructor(status: number, body: ErrorBody) {
		super(body.message);
		this.name = 'ApiError';
		this.status = status;
		this.code = body.code;
		this.details = body.details;
		this.retryable = body.retryable;
	}
}

/** The server could not be reached, did not answer in time, or a proxy answered for it. */
export class NetworkError extends Error {
	constructor(message: string, options?: { cause?: unknown }) {
		super(message, options);
		this.name = 'NetworkError';
	}
}

export function isAbortError(error: unknown): boolean {
	return (
		typeof error === 'object' && error !== null && (error as { name?: unknown }).name === 'AbortError'
	);
}

/** The server message for an error of any kind, for inline errors and toasts. */
export function errorMessage(error: unknown): string {
	if (error instanceof Error && error.message) return error.message;
	if (typeof error === 'string' && error) return error;
	return 'Something went wrong.';
}

function abortError(): DOMException {
	return new DOMException('The request was aborted.', 'AbortError');
}

// --- Transport -------------------------------------------------------------

type Method = 'GET' | 'POST' | 'PATCH' | 'DELETE';

interface RequestOptions {
	method?: Method;
	/** Serialized as JSON. */
	body?: unknown;
	headers?: Record<string, string>;
	signal?: AbortSignal;
	/** Milliseconds before the request fails with a NetworkError; null disables the limit. */
	timeoutMs?: number | null;
}

interface RawResponse {
	status: number;
	statusText: string;
	text: string;
}

let csrfToken: string | null = null;
let csrfBootstrap: Promise<void> | null = null;

/** Fetches a session token through /health unless one is already known. */
function ensureCsrfToken(): Promise<void> {
	if (csrfToken) return Promise.resolve();
	csrfBootstrap ??= send('/health', { method: 'GET' })
		.then((raw) => {
			decode<unknown>(raw);
		})
		.finally(() => {
			csrfBootstrap = null;
		});
	return csrfBootstrap;
}

async function send(path: string, options: RequestOptions): Promise<RawResponse> {
	const method = options.method ?? 'GET';
	const external = options.signal;
	if (external?.aborted) throw abortError();

	const controller = new AbortController();
	const timeoutMs = options.timeoutMs === undefined ? DEFAULT_TIMEOUT_MS : options.timeoutMs;
	let timedOut = false;
	const timer =
		timeoutMs === null
			? undefined
			: setTimeout(() => {
					timedOut = true;
					controller.abort();
				}, timeoutMs);
	const forwardAbort = () => controller.abort();
	external?.addEventListener('abort', forwardAbort, { once: true });

	const headers = new Headers(options.headers);
	headers.set('Accept', 'application/json');
	let body: string | undefined;
	if (options.body !== undefined) {
		headers.set('Content-Type', 'application/json');
		body = JSON.stringify(options.body);
	}
	if (method !== 'GET' && csrfToken) headers.set(CSRF_HEADER, csrfToken);

	try {
		const response = await fetch(`${API_BASE}${path}`, {
			method,
			headers,
			body,
			credentials: 'same-origin',
			signal: controller.signal
		});
		csrfToken = response.headers.get(CSRF_HEADER) ?? csrfToken;
		const text = response.status === 204 ? '' : await response.text();
		return { status: response.status, statusText: response.statusText, text };
	} catch (error) {
		if (external?.aborted) throw abortError();
		if (timedOut) {
			throw new NetworkError(
				`The server did not respond within ${Math.round((timeoutMs ?? 0) / 1000)} s.`,
				{ cause: error }
			);
		}
		throw new NetworkError('Cannot reach the server. Check that the app is running.', {
			cause: error
		});
	} finally {
		clearTimeout(timer);
		external?.removeEventListener('abort', forwardAbort);
	}
}

function parseErrorBody(text: string): ErrorBody | null {
	let parsed: unknown;
	try {
		parsed = JSON.parse(text);
	} catch {
		return null;
	}
	if (typeof parsed !== 'object' || parsed === null) return null;
	const body = parsed as Record<string, unknown>;
	if (typeof body.code === 'string' && typeof body.message === 'string') {
		const details = body.details;
		return {
			code: body.code,
			message: body.message,
			details:
				typeof details === 'object' && details !== null && !Array.isArray(details)
					? (details as Record<string, unknown>)
					: null,
			retryable: body.retryable === true
		};
	}
	return null;
}

function decode<T>(raw: RawResponse): T {
	if (raw.status >= 200 && raw.status < 300) {
		if (raw.text === '') return undefined as T;
		try {
			return JSON.parse(raw.text) as T;
		} catch {
			throw new ApiError(raw.status, {
				code: 'invalid_response',
				message: 'The server returned a response the app cannot read.',
				details: null,
				retryable: false
			});
		}
	}
	const body = parseErrorBody(raw.text);
	if (body) throw new ApiError(raw.status, body);
	// The API always answers errors with JSON; anything else comes from a proxy
	// or gateway standing in for a backend that is down.
	if (raw.status >= 500) {
		throw new NetworkError(`The server is not responding (HTTP ${raw.status}).`);
	}
	throw new ApiError(raw.status, {
		code: `http_${raw.status}`,
		message: `The request failed (HTTP ${raw.status}${raw.statusText ? ` ${raw.statusText}` : ''}).`,
		details: null,
		retryable: false
	});
}

/**
 * Sends a mutating request with the session token, refreshing the token and
 * retrying once when the server rejects it as stale.
 */
async function sendWithCsrf(attempt: () => Promise<RawResponse>): Promise<RawResponse> {
	await ensureCsrfToken();
	const sentToken = csrfToken;
	const raw = await attempt();
	if (raw.status !== 403 || parseErrorBody(raw.text)?.code !== 'csrf_invalid') return raw;
	// The rejection normally carries the current token; fetch one if it did not.
	if (csrfToken === sentToken) {
		csrfToken = null;
		await ensureCsrfToken();
	}
	return attempt();
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
	const method = options.method ?? 'GET';
	const raw =
		method === 'GET' ? await send(path, options) : await sendWithCsrf(() => send(path, options));
	return decode<T>(raw);
}

function uploadOnce(
	file: File,
	onProgress: ((loaded: number, total: number) => void) | undefined,
	signal: AbortSignal | undefined
): Promise<RawResponse> {
	// Executor form: Promise.withResolvers is missing in Node 20 (vitest) and Safari < 17.4.
	return new Promise((resolve, reject) => {
		if (signal?.aborted) {
			reject(abortError());
			return;
		}
		const xhr = new XMLHttpRequest();
		const abort = () => xhr.abort();
		signal?.addEventListener('abort', abort, { once: true });
		xhr.open('POST', `${API_BASE}/documents`);
		xhr.setRequestHeader('Accept', 'application/json');
		if (csrfToken) xhr.setRequestHeader(CSRF_HEADER, csrfToken);
		xhr.upload.onprogress = (event) => {
			onProgress?.(event.loaded, event.lengthComputable ? event.total : file.size);
		};
		xhr.onload = () => {
			csrfToken = xhr.getResponseHeader(CSRF_HEADER) ?? csrfToken;
			resolve({ status: xhr.status, statusText: xhr.statusText, text: xhr.responseText });
		};
		xhr.onerror = () =>
			reject(new NetworkError('The file could not be sent. Check that the app is running.'));
		xhr.onabort = () => reject(abortError());
		xhr.onloadend = () => signal?.removeEventListener('abort', abort);
		const form = new FormData();
		form.append('file', file, file.name);
		xhr.send(form);
	});
}

function route(...segments: string[]): string {
	return segments.map((segment) => `/${encodeURIComponent(segment)}`).join('');
}

// --- Endpoints ---------------------------------------------------------------

export const api = {
	async health(): Promise<void> {
		await request<unknown>('/health');
	},

	getSettings: (): Promise<Settings> => request<Settings>('/settings'),
	updateSettings: (update: SettingsUpdate): Promise<Settings> =>
		request<Settings>('/settings', { method: 'PATCH', body: update }),

	ollamaHealth: (host?: string): Promise<OllamaHealth> =>
		request<OllamaHealth>(
			host === undefined ? '/ollama/health' : `/ollama/health?host=${encodeURIComponent(host)}`
		),
	async ollamaModels(): Promise<OllamaModel[]> {
		return (await request<{ models: OllamaModel[] }>('/ollama/models')).models;
	},

	async listDocuments(search?: string, signal?: AbortSignal): Promise<DocumentSummary[]> {
		const query = search?.trim() ? `?search=${encodeURIComponent(search.trim())}` : '';
		return (await request<{ documents: DocumentSummary[] }>(`/documents${query}`, { signal }))
			.documents;
	},
	getDocument: (id: string, signal?: AbortSignal): Promise<DocumentDetail> =>
		request<DocumentDetail>(route('documents', id), { signal }),
	async uploadDocument(
		file: File,
		options: { onProgress?: (loaded: number, total: number) => void; signal?: AbortSignal } = {}
	): Promise<DocumentCreated> {
		const raw = await sendWithCsrf(() => uploadOnce(file, options.onProgress, options.signal));
		return decode<DocumentCreated>(raw);
	},
	pasteText: (body: { title?: string; text: string }): Promise<DocumentCreated> =>
		request<DocumentCreated>('/documents/text', {
			method: 'POST',
			body,
			timeoutMs: PASTE_TIMEOUT_MS
		}),
	renameDocument: (id: string, title: string): Promise<DocumentDetail> =>
		request<DocumentDetail>(route('documents', id), { method: 'PATCH', body: { title } }),
	async deleteDocument(id: string): Promise<void> {
		await request<void>(route('documents', id), { method: 'DELETE' });
	},
	getSource: (id: string, offset: number, limit: number, signal?: AbortSignal): Promise<SourceSlice> =>
		request<SourceSlice>(`${route('documents', id, 'source')}?offset=${offset}&limit=${limit}`, {
			signal
		}),
	async getPages(id: string, signal?: AbortSignal): Promise<SourcePage[]> {
		return (await request<{ pages: SourcePage[] }>(route('documents', id, 'pages'), { signal }))
			.pages;
	},
	originalUrl: (id: string): string => `${API_BASE}${route('documents', id, 'original')}`,

	preflight: (documentId: string, config: RunConfig, signal?: AbortSignal): Promise<Preflight> =>
		request<Preflight>('/preflight', {
			method: 'POST',
			body: { document_id: documentId, config },
			signal
		}),
	createRun: (documentId: string, config: RunConfig, idempotencyKey: string): Promise<Run> =>
		request<Run>('/runs', {
			method: 'POST',
			body: { document_id: documentId, config },
			headers: { 'Idempotency-Key': idempotencyKey }
		}),
	getRun: (runId: string, signal?: AbortSignal): Promise<Run> =>
		request<Run>(route('runs', runId), { signal }),
	async listRuns(documentId: string, signal?: AbortSignal): Promise<Run[]> {
		return (await request<{ runs: Run[] }>(route('documents', documentId, 'runs'), { signal })).runs;
	},
	stopRun: (runId: string): Promise<Run> =>
		request<Run>(route('runs', runId, 'stop'), { method: 'POST' }),
	resumeRun: (runId: string): Promise<Run> =>
		request<Run>(route('runs', runId, 'resume'), { method: 'POST' }),
	async deleteRun(runId: string): Promise<void> {
		await request<void>(route('runs', runId), { method: 'DELETE' });
	},

	getActivity: (signal?: AbortSignal): Promise<Activity> => request<Activity>('/activity', { signal }),

	getTree: (runId: string, signal?: AbortSignal): Promise<Tree> =>
		request<Tree>(route('runs', runId, 'tree'), { signal }),
	getNode: (runId: string, nodeId: string, signal?: AbortSignal): Promise<NodeDetail> =>
		request<NodeDetail>(route('runs', runId, 'nodes', nodeId), { signal }),
	async getSegments(runId: string, signal?: AbortSignal): Promise<SegmentRef[]> {
		return (await request<{ segments: SegmentRef[] }>(route('runs', runId, 'segments'), { signal }))
			.segments;
	},
	getSummary: (runId: string, signal?: AbortSignal): Promise<FinalSummary> =>
		request<FinalSummary>(route('runs', runId, 'summary'), { signal }),
	exportUrl: (runId: string, format: ExportFormat): string =>
		`${API_BASE}${route('runs', runId, 'export', format)}`,

	/** The tab's only SSE endpoint; `after` is the last cursor seen (0 on the first connect). */
	activityStreamUrl: (after: number): string => `${API_BASE}/activity/stream?after=${after}`
};
