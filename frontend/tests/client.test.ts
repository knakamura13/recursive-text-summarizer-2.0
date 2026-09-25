import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type * as ClientModule from '../src/lib/api/client';

type Client = typeof ClientModule;

interface Reply {
	status?: number;
	json?: unknown;
	text?: string;
	headers?: Record<string, string>;
}

interface Call {
	method: string;
	url: string;
	headers: Headers;
	body: unknown;
}

function reply({ status = 200, json, text, headers = {} }: Reply): Response {
	const body = status === 204 ? null : json !== undefined ? JSON.stringify(json) : (text ?? '');
	return new Response(body, {
		status,
		headers: { ...(json !== undefined ? { 'Content-Type': 'application/json' } : {}), ...headers }
	});
}

const errorBody = (code: string, message: string, extra: Record<string, unknown> = {}) => ({
	code,
	message,
	details: null,
	retryable: false,
	...extra
});

let client: Client;
let calls: Call[];

/** Answers fetch calls with `handler`; records every call. */
function serve(handler: (call: Call, index: number) => Response | Promise<Response>) {
	vi.stubGlobal(
		'fetch',
		vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
			const call: Call = {
				method: init.method ?? 'GET',
				url: String(input),
				headers: new Headers(init.headers),
				body: init.body ? JSON.parse(String(init.body)) : undefined
			};
			calls.push(call);
			return handler(call, calls.length - 1);
		})
	);
}

beforeEach(async () => {
	// A fresh module per test: the CSRF token lives in module state.
	vi.resetModules();
	client = await import('../src/lib/api/client');
	calls = [];
});

afterEach(() => {
	vi.useRealTimers();
});

describe('error mapping', () => {
	it('turns a JSON error body into an ApiError carrying the server fields', async () => {
		serve(() =>
			reply({
				status: 409,
				json: errorBody('run_active', 'Another Run is active.', {
					details: { run_id: 'r1', document_id: 'd1' }
				})
			})
		);
		const failure = await client.api.getRun('r2').catch((error: unknown) => error);
		expect(failure).toBeInstanceOf(client.ApiError);
		expect(failure).toMatchObject({
			status: 409,
			code: 'run_active',
			message: 'Another Run is active.',
			details: { run_id: 'r1', document_id: 'd1' },
			retryable: false
		});
	});

	it('treats a non-JSON 5xx from a proxy as a NetworkError', async () => {
		serve(() => reply({ status: 502, text: 'Bad Gateway', headers: { 'Content-Type': 'text/plain' } }));
		await expect(client.api.getSettings()).rejects.toBeInstanceOf(client.NetworkError);
	});

	it('reports a non-JSON 4xx as an ApiError named after the status', async () => {
		serve(() => reply({ status: 404, text: '<html>Not Found</html>' }));
		await expect(client.api.getSettings()).rejects.toMatchObject({ code: 'http_404', status: 404 });
	});

	it('wraps a failed fetch in a NetworkError', async () => {
		serve(() => Promise.reject(new TypeError('Failed to fetch')));
		await expect(client.api.listDocuments()).rejects.toBeInstanceOf(client.NetworkError);
	});

	it('fails with a NetworkError after 30 s without an answer', async () => {
		vi.useFakeTimers();
		vi.stubGlobal(
			'fetch',
			vi.fn(
				(_input: RequestInfo | URL, init: RequestInit = {}) =>
					new Promise<Response>((_resolve, reject) => {
						init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
					})
			)
		);
		let failure: unknown;
		const pending = client.api.getSettings().catch((error: unknown) => (failure = error));
		await vi.advanceTimersByTimeAsync(29_999);
		expect(failure).toBeUndefined();
		await vi.advanceTimersByTimeAsync(1);
		await pending;
		expect(failure).toBeInstanceOf(client.NetworkError);
		expect((failure as Error).message).toMatch(/30 s/);
	});

	it('rejects with an AbortError, not a NetworkError, when the caller aborts', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn(
				(_input: RequestInfo | URL, init: RequestInit = {}) =>
					new Promise<Response>((_resolve, reject) => {
						init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
					})
			)
		);
		const controller = new AbortController();
		const pending = client.api.getDocument('d1', controller.signal);
		controller.abort();
		const failure = await pending.catch((error: unknown) => error);
		expect(client.isAbortError(failure)).toBe(true);
		expect(failure).not.toBeInstanceOf(client.NetworkError);
	});
});

describe('Ollama health checks', () => {
	it('targets an explicitly supplied unsaved host as a safely encoded query value', async () => {
		const host = 'http://ollama.local:11434/api?profile=office&name=fast model';
		serve(() => reply({ json: { connected: true, message: 'Connected', host, version: null } }));

		await expect(client.api.ollamaHealth(host)).resolves.toMatchObject({ connected: true, host });
		const request = new URL(calls[0].url, 'http://app.local');
		expect(request.pathname).toBe('/api/v1/ollama/health');
		expect(request.searchParams.get('host')).toBe(host);
	});
});

describe('CSRF token', () => {
	const settings = { ollama_host: 'http://localhost:11434', defaults: {} };

	it('fetches a token through /health before the first mutating request and reuses it', async () => {
		serve((call) =>
			call.url.endsWith('/health')
				? reply({ json: { status: 'ok' }, headers: { 'X-CSRF-Token': 'token-1' } })
				: reply({ json: settings })
		);
		await client.api.updateSettings({ ollama_host: 'http://localhost:11434' });
		await client.api.updateSettings({ ollama_host: 'http://127.0.0.1:11434' });

		expect(calls.map((call) => `${call.method} ${call.url}`)).toEqual([
			'GET /api/v1/health',
			'PATCH /api/v1/settings',
			'PATCH /api/v1/settings'
		]);
		expect(calls[1].headers.get('X-CSRF-Token')).toBe('token-1');
		expect(calls[2].headers.get('X-CSRF-Token')).toBe('token-1');
	});

	it('adopts the token of any response', async () => {
		serve((call) =>
			call.method === 'GET'
				? reply({ json: settings, headers: { 'X-CSRF-Token': 'from-get' } })
				: reply({ json: settings })
		);
		await client.api.getSettings();
		await client.api.updateSettings({});
		expect(calls.map((call) => call.url)).toEqual(['/api/v1/settings', '/api/v1/settings']);
		expect(calls[1].headers.get('X-CSRF-Token')).toBe('from-get');
	});

	it('retries once with the token sent along with a csrf_invalid rejection', async () => {
		const run = { run_id: 'r1', state: 'queued' };
		serve((call, index) => {
			if (call.url.endsWith('/health')) {
				return reply({ json: { status: 'ok' }, headers: { 'X-CSRF-Token': 'stale' } });
			}
			return index === 1
				? reply({
						status: 403,
						json: errorBody('csrf_invalid', 'The session token is missing or stale.', { retryable: true }),
						headers: { 'X-CSRF-Token': 'fresh' }
					})
				: reply({ status: 201, json: run });
		});
		const created = await client.api.createRun('d1', { model: 'm' } as never, 'key-1');

		expect(created).toEqual(run);
		const posts = calls.filter((call) => call.method === 'POST');
		expect(posts.map((call) => call.headers.get('X-CSRF-Token'))).toEqual(['stale', 'fresh']);
		expect(posts.map((call) => call.headers.get('Idempotency-Key'))).toEqual(['key-1', 'key-1']);
	});

	it('refreshes the token through /health when the rejection carries none', async () => {
		let healthCalls = 0;
		serve((call) => {
			if (call.url.endsWith('/health')) {
				healthCalls += 1;
				return reply({ json: { status: 'ok' }, headers: { 'X-CSRF-Token': `token-${healthCalls}` } });
			}
			return call.headers.get('X-CSRF-Token') === 'token-2'
				? reply({ status: 202, json: { run_id: 'r1', state: 'stopping' } })
				: reply({ status: 403, json: errorBody('csrf_invalid', 'Stale token.', { retryable: true }) });
		});
		await expect(client.api.stopRun('r1')).resolves.toMatchObject({ state: 'stopping' });
		expect(healthCalls).toBe(2);
	});

	it('gives up after one retry', async () => {
		serve((call) =>
			call.url.endsWith('/health')
				? reply({ json: { status: 'ok' }, headers: { 'X-CSRF-Token': 't' } })
				: reply({
						status: 403,
						json: errorBody('csrf_invalid', 'Stale token.', { retryable: true }),
						headers: { 'X-CSRF-Token': 't' }
					})
		);
		await expect(client.api.deleteDocument('d1')).rejects.toMatchObject({ code: 'csrf_invalid' });
		expect(calls.filter((call) => call.method === 'DELETE')).toHaveLength(2);
	});
});

describe('uploadDocument', () => {
	class FakeXhr {
		static instances: FakeXhr[] = [];
		method = '';
		url = '';
		headers: Record<string, string> = {};
		body: unknown = null;
		aborted = false;
		status = 0;
		statusText = '';
		responseText = '';
		#responseHeaders: Record<string, string> = {};
		upload: { onprogress: ((event: ProgressEvent) => void) | null } = { onprogress: null };
		onload: (() => void) | null = null;
		onerror: (() => void) | null = null;
		onabort: (() => void) | null = null;
		onloadend: (() => void) | null = null;

		constructor() {
			FakeXhr.instances.push(this);
		}
		open(method: string, url: string) {
			this.method = method;
			this.url = url;
		}
		setRequestHeader(name: string, value: string) {
			this.headers[name] = value;
		}
		getResponseHeader(name: string): string | null {
			return this.#responseHeaders[name.toLowerCase()] ?? null;
		}
		send(body: unknown) {
			this.body = body;
		}
		abort() {
			this.aborted = true;
			this.onabort?.();
			this.onloadend?.();
		}
		progress(loaded: number, total: number) {
			this.upload.onprogress?.({ loaded, total, lengthComputable: true } as ProgressEvent);
		}
		respond(status: number, json: unknown, headers: Record<string, string> = {}) {
			this.status = status;
			this.responseText = JSON.stringify(json);
			this.#responseHeaders = Object.fromEntries(
				Object.entries(headers).map(([key, value]) => [key.toLowerCase(), value])
			);
			this.onload?.();
			this.onloadend?.();
		}
	}

	const created = {
		document: { document_id: 'd1', title: 'notes', import_state: 'importing' },
		already_imported: false
	};
	const file = () => new File(['hello world'], 'notes.txt', { type: 'text/plain' });

	async function nextXhr(count: number): Promise<FakeXhr> {
		await vi.waitFor(() => expect(FakeXhr.instances).toHaveLength(count));
		return FakeXhr.instances[count - 1];
	}

	beforeEach(() => {
		FakeXhr.instances = [];
		vi.stubGlobal('XMLHttpRequest', FakeXhr);
		serve(() => reply({ json: { status: 'ok' }, headers: { 'X-CSRF-Token': 'upload-token' } }));
	});

	it('sends the file with the session token, reports progress, and returns the created Document', async () => {
		const progress: [number, number][] = [];
		const pending = client.api.uploadDocument(file(), {
			onProgress: (loaded, total) => progress.push([loaded, total])
		});
		const xhr = await nextXhr(1);
		expect(xhr.method).toBe('POST');
		expect(xhr.url).toBe('/api/v1/documents');
		expect(xhr.headers['X-CSRF-Token']).toBe('upload-token');
		expect((xhr.body as FormData).get('file')).toBeInstanceOf(File);

		xhr.progress(40, 100);
		xhr.progress(100, 100);
		xhr.respond(201, created);
		await expect(pending).resolves.toEqual(created);
		expect(progress).toEqual([
			[40, 100],
			[100, 100]
		]);
	});

	it('rejects with the server error for a refused file', async () => {
		const pending = client.api.uploadDocument(file());
		const xhr = await nextXhr(1);
		xhr.respond(415, errorBody('unsupported_format', 'This file type is not supported.', {
			details: { supported: ['pdf', 'txt'] }
		}));
		await expect(pending).rejects.toMatchObject({
			status: 415,
			code: 'unsupported_format',
			message: 'This file type is not supported.',
			details: { supported: ['pdf', 'txt'] }
		});
	});

	it('sends the file again with the fresh token after a csrf_invalid rejection', async () => {
		const pending = client.api.uploadDocument(file());
		const first = await nextXhr(1);
		first.respond(403, errorBody('csrf_invalid', 'Stale token.', { retryable: true }), {
			'X-CSRF-Token': 'rotated'
		});
		const second = await nextXhr(2);
		expect(second.headers['X-CSRF-Token']).toBe('rotated');
		second.respond(201, created);
		await expect(pending).resolves.toEqual(created);
	});

	it('aborts the request when the signal fires', async () => {
		const controller = new AbortController();
		const pending = client.api.uploadDocument(file(), { signal: controller.signal });
		const xhr = await nextXhr(1);
		controller.abort();
		expect(xhr.aborted).toBe(true);
		const failure = await pending.catch((error: unknown) => error);
		expect(client.isAbortError(failure)).toBe(true);
	});

	it('maps a connection failure to a NetworkError', async () => {
		const pending = client.api.uploadDocument(file());
		const xhr = await nextXhr(1);
		xhr.onerror?.();
		await expect(pending).rejects.toBeInstanceOf(client.NetworkError);
	});
});
