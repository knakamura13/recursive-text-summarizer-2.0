// Models installed in Ollama. `error` holds the server message (for example
// ollama_unreachable) and clears on the next successful refresh.
import { api, errorMessage } from '$lib/api/client';
import type { OllamaModel } from '$lib/api/types';

/** Embedding models (nomic-embed-text, bge, ...) cannot write summaries. */
export function canSummarize(model: OllamaModel): boolean {
	return !/embed/i.test(model.name) && !/bert$/i.test(model.family ?? '');
}

class ModelsStore {
	#list = $state.raw<OllamaModel[]>([]);
	#error = $state<string | null>(null);
	#loading = $state(false);
	#pending: Promise<void> | null = null;

	get list(): OllamaModel[] {
		return this.#list;
	}

	get error(): string | null {
		return this.#error;
	}

	get loading(): boolean {
		return this.#loading;
	}

	/** Refetches the list; concurrent calls share one request. */
	refresh(): Promise<void> {
		this.#pending ??= this.#load().finally(() => {
			this.#pending = null;
		});
		return this.#pending;
	}

	async #load(): Promise<void> {
		this.#loading = true;
		try {
			this.#list = await api.ollamaModels();
			this.#error = null;
		} catch (error) {
			this.#error = errorMessage(error);
		} finally {
			this.#loading = false;
		}
	}
}

export const models = new ModelsStore();
