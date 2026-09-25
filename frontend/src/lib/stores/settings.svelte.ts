// Stored settings: the Ollama host and the default Run configuration.
import { api } from '$lib/api/client';
import type { Settings, SettingsUpdate } from '$lib/api/types';

class SettingsStore {
	#value = $state.raw<Settings | null>(null);
	#pending: Promise<Settings> | null = null;

	get value(): Settings | null {
		return this.#value;
	}

	/** Fetches the current settings; concurrent calls share one request. Rejects with the API error. */
	load(): Promise<Settings> {
		this.#pending ??= api
			.getSettings()
			.then((value) => (this.#value = value))
			.finally(() => {
				this.#pending = null;
			});
		return this.#pending;
	}

	/** Applies a merge patch and keeps the server's result. Rejects with the API error. */
	async save(update: SettingsUpdate): Promise<Settings> {
		const value = await api.updateSettings(update);
		this.#value = value;
		return value;
	}
}

export const settings = new SettingsStore();
