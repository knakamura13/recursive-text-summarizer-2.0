export type DocumentSummary = {
	document_id: string;
	title: string;
	filename: string;
	format: 'txt' | 'md' | 'pdf';
	size_bytes: number;
	import_state: 'ready' | 'pending_confirmation' | 'failed';
	latest_run_state?: string | null;
};

export type RunConfig = {
	model: string;
	target_words?: number;
	strategy?: 'auto' | 'direct' | 'hierarchical';
	verify?: boolean;
	max_repair_passes?: number;
	citations?: boolean;
	context_window?: number | null;
	max_output_tokens?: number;
	safety_margin_tokens?: number;
	safety_margin_fraction?: number;
	chunk_tokens?: number | null;
	overlap_tokens?: number;
	max_merge_children?: number | null;
	max_concurrency?: number;
	timeout_seconds?: number;
	max_retries?: number;
};

let csrfToken: string | null = null;

async function ensureSession(): Promise<void> {
	const response = await fetch('/api/v1/health', { credentials: 'include' });
	if (!response.ok) {
		throw new Error('Failed to establish session');
	}
	const token = response.headers.get('X-CSRF-Token');
	if (token) {
		csrfToken = token;
	}
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
	if (!csrfToken && init.method && init.method !== 'GET') {
		await ensureSession();
	}
	const headers = new Headers(init.headers);
	if (init.method && init.method !== 'GET' && csrfToken) {
		headers.set('X-CSRF-Token', csrfToken);
	}
	const response = await fetch(path, {
		...init,
		headers,
		credentials: 'include'
	});
	if (!response.ok) {
		const body = await response.text();
		throw new Error(body || response.statusText);
	}
	if (response.status === 204) {
		return undefined as T;
	}
	return (await response.json()) as T;
}

export const api = {
	health: () => apiFetch<{ status: string }>('/api/v1/health'),
	listDocuments: (search?: string) =>
		apiFetch<{ documents: DocumentSummary[]; already_imported?: boolean }>(
			`/api/v1/documents${search ? `?search=${encodeURIComponent(search)}` : ''}`
		),
	getDocument: (id: string) =>
		apiFetch<DocumentSummary & { latest_run_id?: string | null }>(`/api/v1/documents/${id}`),
	uploadDocument: async (file: File) => {
		const form = new FormData();
		form.append('file', file);
		return apiFetch<{ documents: DocumentSummary[]; already_imported?: boolean }>(
			'/api/v1/documents',
			{ method: 'POST', body: form }
		);
	},
	renameDocument: (id: string, title: string) =>
		apiFetch(`/api/v1/documents/${id}`, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ title })
		}),
	deleteDocument: (id: string) =>
		apiFetch(`/api/v1/documents/${id}`, { method: 'DELETE' }),
	getSourceSlice: (id: string, offset = 0, limit = 4000) =>
		apiFetch<{ text: string; offset: number; total_length: number; has_more: boolean }>(
			`/api/v1/documents/${id}/source?offset=${offset}&limit=${limit}`
		),
	getSettings: () =>
		apiFetch<{ ollama_host: string; defaults: RunConfig }>('/api/v1/settings'),
	updateSettings: (payload: unknown) =>
		apiFetch('/api/v1/settings', {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(payload)
		}),
	ollamaHealth: () => apiFetch<{ connected: boolean; message: string }>('/api/v1/ollama/health'),
	ollamaModels: () => apiFetch<{ models: { name: string }[] }>('/api/v1/ollama/models'),
	preflight: (documentId: string, config: RunConfig) =>
		apiFetch<{ fits: boolean; warnings?: string[] }>('/api/v1/preflight', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ document_id: documentId, config })
		}),
	createRun: (documentId: string, config: RunConfig, idempotencyKey: string) =>
		apiFetch<{ run_id: string }>('/api/v1/runs', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Idempotency-Key': idempotencyKey
			},
			body: JSON.stringify({ document_id: documentId, config })
		}),
	getRun: (runId: string) => apiFetch<{ state: string }>(`/api/v1/runs/${runId}`),
	cancelRun: (runId: string) =>
		apiFetch(`/api/v1/runs/${runId}/cancel`, { method: 'POST' }),
	resumeRun: (runId: string) =>
		apiFetch(`/api/v1/runs/${runId}/resume`, { method: 'POST' }),
	getRunTree: (runId: string) =>
		apiFetch<{ nodes: import('$lib/stores/workspace').HierarchyNode[] }>(
			`/api/v1/runs/${runId}/tree`
		),
	getRunSummary: (runId: string) =>
		apiFetch<{ available: boolean; text?: string | null }>(`/api/v1/runs/${runId}/summary`),
	confirmExtraction: (documentId: string) =>
		apiFetch(`/api/v1/documents/${documentId}/confirm-extraction`, { method: 'POST' })
};
