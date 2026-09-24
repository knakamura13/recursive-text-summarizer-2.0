// Files the user handed to the Import dialog in this session. Each file is
// uploaded with progress (two at a time); once the server has it, the
// Document's own import_state and import_progress (documents store) take over.
import { api, isAbortError } from '$lib/api/client';
import { formatBytes } from '$lib/format';
import { documents } from './documents.svelte';

/** Upload limit enforced by the server (413 file_too_large beyond it). */
export const MAX_IMPORT_BYTES = 500 * 1024 * 1024;

/** File picker filter: every format the server can import (extensions plus MIME types for mobile pickers). */
export const IMPORT_ACCEPT = [
	'.txt',
	'.md',
	'.markdown',
	'.srt',
	'.vtt',
	'.pdf',
	'.docx',
	'.odt',
	'.rtf',
	'.html',
	'.htm',
	'.epub',
	'.png',
	'.jpg',
	'.jpeg',
	'.tif',
	'.tiff',
	'text/plain',
	'text/markdown',
	'text/html',
	'text/vtt',
	'application/x-subrip',
	'application/pdf',
	'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
	'application/vnd.oasis.opendocument.text',
	'application/rtf',
	'text/rtf',
	'application/epub+zip',
	'image/png',
	'image/jpeg',
	'image/tiff'
].join(',');

export const SUPPORTED_FORMATS =
'Text and Markdown (.txt, .md, .markdown), subtitles (.srt, .vtt), PDF, Word (.docx), OpenDocument (.odt), RTF, HTML, EPUB, and images (.png, .jpg, .jpeg, .tif, .tiff) for OCR';

export type ImportItemPhase = 'waiting' | 'uploading' | 'sent' | 'failed';

export interface ImportItem {
	id: number;
	name: string;
	size: number;
	phase: ImportItemPhase;
	/** Bytes sent so far and the request size. */
	loaded: number;
	total: number;
	/** The Document created for the file, or the existing one it duplicates. */
	documentId: string | null;
	/** The server already had these exact bytes. */
	duplicate: boolean;
	error: unknown;
}

const CONCURRENT_UPLOADS = 2;

class ImportsStore {
	#items = $state.raw<ImportItem[]>([]);
	#files = new Map<number, File>();
	#controllers = new Map<number, AbortController>();
	#settle = new Map<number, (item: ImportItem | null) => void>();
	#nextId = 1;

	get items(): readonly ImportItem[] {
		return this.#items;
	}

	/** Files not yet received by the server. */
	get uploadingCount(): number {
		return this.#items.filter((item) => item.phase === 'waiting' || item.phase === 'uploading')
			.length;
	}

	/**
	 * Queues the files. Each promise resolves with the item once its upload
	 * ends (phase `sent` or `failed`), or with null when the user removes it first.
	 */
	add(files: Iterable<File>): Promise<ImportItem | null>[] {
		const added: ImportItem[] = [];
		const settled: Promise<ImportItem | null>[] = [];
		for (const file of files) {
			const tooLarge = file.size > MAX_IMPORT_BYTES;
			const item: ImportItem = {
				id: this.#nextId++,
				name: file.name,
				size: file.size,
				phase: tooLarge ? 'failed' : 'waiting',
				loaded: 0,
				total: file.size,
				documentId: null,
				duplicate: false,
				error: tooLarge
					? new Error(
							`“${file.name}” is ${formatBytes(file.size)}; the import limit is ${formatBytes(MAX_IMPORT_BYTES)}.`
						)
					: null
			};
			added.push(item);
			if (tooLarge) {
				settled.push(Promise.resolve(item));
			} else {
				this.#files.set(item.id, file);
				settled.push(new Promise((resolve) => this.#settle.set(item.id, resolve)));
			}
		}
		this.#items = [...this.#items, ...added];
		this.#pump();
		return settled;
	}

	/** Uploads a failed file again. */
	retry(id: number): Promise<ImportItem | null> {
		const item = this.#items.find((candidate) => candidate.id === id);
		if (!item || item.phase !== 'failed' || !this.#files.has(id)) return Promise.resolve(null);
		const settled = new Promise<ImportItem | null>((resolve) => this.#settle.set(id, resolve));
		this.#update(id, { phase: 'waiting', error: null, loaded: 0 });
		this.#pump();
		return settled;
	}

	/** Drops the item; an upload in flight is aborted. */
	remove(id: number): void {
		this.#controllers.get(id)?.abort();
		this.#files.delete(id);
		this.#items = this.#items.filter((item) => item.id !== id);
		this.#settle.get(id)?.(null);
		this.#settle.delete(id);
		this.#pump();
	}

	clearFinished(): void {
		for (const item of this.#items) {
			if (item.phase === 'sent' || item.phase === 'failed') this.#files.delete(item.id);
		}
		this.#items = this.#items.filter((item) => item.phase === 'waiting' || item.phase === 'uploading');
	}

	#update(id: number, changes: Partial<ImportItem>): ImportItem | null {
		let updated: ImportItem | null = null;
		this.#items = this.#items.map((item) => {
			if (item.id !== id) return item;
			updated = { ...item, ...changes };
			return updated;
		});
		return updated;
	}

	#finish(id: number, changes: Partial<ImportItem>): void {
		const item = this.#update(id, changes);
		this.#settle.get(id)?.(item);
		this.#settle.delete(id);
	}

	#pump(): void {
		let active = this.#items.filter((item) => item.phase === 'uploading').length;
		for (const item of this.#items) {
			if (active >= CONCURRENT_UPLOADS) return;
			if (item.phase !== 'waiting') continue;
			active += 1;
			void this.#upload(item.id);
		}
	}

	async #upload(id: number): Promise<void> {
		const file = this.#files.get(id);
		if (!file) return;
		const controller = new AbortController();
		this.#controllers.set(id, controller);
		this.#update(id, { phase: 'uploading', loaded: 0, total: file.size });
		let shownPercent = -1;
		try {
			const created = await api.uploadDocument(file, {
				signal: controller.signal,
				onProgress: (loaded, total) => {
					// Progress events arrive every few milliseconds; re-render per whole percent.
					const percent = total > 0 ? Math.floor((loaded / total) * 100) : 0;
					if (percent === shownPercent) return;
					shownPercent = percent;
					this.#update(id, { loaded, total });
				}
			});
			documents.upsert(created.document);
			this.#files.delete(id);
			this.#finish(id, {
				phase: 'sent',
				documentId: created.document.document_id,
				duplicate: created.already_imported,
				error: null
			});
		} catch (error) {
			// An abort means the user removed the item; remove() already settled it.
			if (!isAbortError(error)) this.#finish(id, { phase: 'failed', error });
		} finally {
			this.#controllers.delete(id);
			this.#pump();
		}
	}
}

export const imports = new ImportsStore();
