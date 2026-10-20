// Whether the server has an OpenRouter API key. Only the yes/no and a message
// are known here: the key lives in the server's environment and is never sent.
import { api, errorMessage } from '$lib/api/client';
import type { OpenRouterStatus } from '$lib/api/types';

class OpenRouterStore {
	#status = $state.raw<OpenRouterStatus | null>(null);
	#error = $state<string | null>(null);
	#pending: Promise<void> | null = null;

	get status(): OpenRouterStatus | null {
		return this.#status;
	}

	/** True or false once known; null while unknown, so a form does not warn before it knows. */
	get keySet(): boolean | null {
		return this.#status ? this.#status.key_set : null;
	}

	get error(): string | null {
		return this.#error;
	}

	/** Refetches the status; concurrent calls share one request. */
	refresh(): Promise<void> {
		this.#pending ??= this.#load().finally(() => {
			this.#pending = null;
		});
		return this.#pending;
	}

	async #load(): Promise<void> {
		try {
			this.#status = await api.openrouterStatus();
			this.#error = null;
		} catch (error) {
			this.#error = errorMessage(error);
		}
	}
}

export const openrouter = new OpenRouterStore();
