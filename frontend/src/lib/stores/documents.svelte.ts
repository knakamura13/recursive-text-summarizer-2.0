// The library list. Setting `search` refetches after a short pause. The
// activity stream keeps it live: import progress of listed Documents,
// Documents that finish importing, the latest Run state of each Document, and
// a refetch after the stream reconnects (events may have been missed).
import { api, ApiError, isAbortError } from '$lib/api/client';
import type { Activity, DocumentSummary, Run } from '$lib/api/types';
import { activity } from './activity.svelte';

const SEARCH_DEBOUNCE_MS = 250;

class DocumentsStore {
	#list = $state.raw<DocumentSummary[]>([]);
	#loading = $state(false);
	#loaded = $state(false);
	#error = $state.raw<unknown>(null);
	#search = $state('');
	#request: AbortController | null = null;
	#debounce: ReturnType<typeof setTimeout> | undefined;

	constructor() {
		activity.onActivity((next, previous) => this.#applyActivity(next, previous));
		activity.onRun((run) => this.#applyRun(run));
		activity.onReconnect(() => {
			if (this.#loaded) void this.refresh();
		});
	}

	get list(): DocumentSummary[] {
		return this.#list;
	}

	get loading(): boolean {
		return this.#loading;
	}

	/** True once a list request has succeeded. */
	get loaded(): boolean {
		return this.#loaded;
	}

	/** The error of the last list request, null after a success. */
	get error(): unknown {
		return this.#error;
	}

	get search(): string {
		return this.#search;
	}

	set search(value: string) {
		if (value === this.#search) return;
		this.#search = value;
		clearTimeout(this.#debounce);
		this.#debounce = setTimeout(() => void this.refresh(), SEARCH_DEBOUNCE_MS);
	}

	get(id: string): DocumentSummary | undefined {
		return this.#list.find((document) => document.document_id === id);
	}

	async refresh(): Promise<void> {
		clearTimeout(this.#debounce);
		this.#request?.abort();
		const request = new AbortController();
		this.#request = request;
		this.#loading = true;
		try {
			this.#list = await api.listDocuments(this.#search, request.signal);
			this.#error = null;
			this.#loaded = true;
		} catch (error) {
			if (!isAbortError(error)) this.#error = error;
		} finally {
			if (this.#request === request) {
				this.#request = null;
				this.#loading = false;
			}
		}
	}

	/** Replaces the Document in place, or adds it at the top. */
	upsert(document: DocumentSummary): void {
		const index = this.#list.findIndex((item) => item.document_id === document.document_id);
		this.#list =
			index === -1 ? [document, ...this.#list] : this.#list.with(index, document);
	}

	remove(id: string): void {
		this.#list = this.#list.filter((document) => document.document_id !== id);
	}

	async #reload(id: string): Promise<void> {
		try {
			const document = await api.getDocument(id);
			if (this.get(id)) this.upsert(document);
		} catch (error) {
			if (error instanceof ApiError && error.status === 404) this.remove(id);
			else this.#error = error;
		}
	}

	#applyActivity(next: Activity, previous: Activity): void {
		if (!this.#loaded) return;
		const filtered = this.#search.trim() !== '';
		for (const document of next.importing) {
			if (this.get(document.document_id) || !filtered) this.upsert(document);
		}
		const stillImporting = new Set(next.importing.map((document) => document.document_id));
		const finished = new Set<string>();
		for (const document of previous.importing) finished.add(document.document_id);
		// An import can finish between two snapshots, so it never appears in either.
		for (const document of this.#list) {
			if (document.import_state === 'importing') finished.add(document.document_id);
		}
		for (const id of finished) {
			if (!stillImporting.has(id)) void this.#reload(id);
		}
	}

	#applyRun(run: Run): void {
		const listed = this.get(run.document_id);
		const latest = listed?.latest_run;
		if (!listed) return;
		// An older Run changing state (e.g. resumed) does not replace a newer latest Run.
		if (latest && latest.run_id !== run.run_id && latest.created_at > run.created_at) return;
		if (latest?.run_id === run.run_id && latest.state === run.state) return;
		this.upsert({
			...listed,
			latest_run: {
				run_id: run.run_id,
				state: run.state,
				created_at: run.created_at,
				updated_at: run.updated_at
			}
		});
	}
}

export const documents = new DocumentsStore();
