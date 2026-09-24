// Short-lived notifications shown by ToastHost. Repeating a message that is
// still visible restarts its timer instead of stacking a copy.

export type ToastKind = 'error' | 'info' | 'success';

export interface Toast {
	id: number;
	kind: ToastKind;
	message: string;
}

const DURATION_MS: Record<ToastKind, number> = { error: 10_000, info: 6_000, success: 4_000 };
const MAX_VISIBLE = 4;

class Toasts {
	#items = $state.raw<Toast[]>([]);
	#timers = new Map<number, ReturnType<typeof setTimeout>>();
	#nextId = 1;

	get items(): readonly Toast[] {
		return this.#items;
	}

	error(message: string): void {
		this.#show('error', message);
	}

	info(message: string): void {
		this.#show('info', message);
	}

	success(message: string): void {
		this.#show('success', message);
	}

	dismiss(id: number): void {
		clearTimeout(this.#timers.get(id));
		this.#timers.delete(id);
		this.#items = this.#items.filter((toast) => toast.id !== id);
	}

	#show(kind: ToastKind, message: string): void {
		const existing = this.#items.find((toast) => toast.kind === kind && toast.message === message);
		const toast = existing ?? { id: this.#nextId++, kind, message };
		if (!existing) {
			const visible = [...this.#items, toast];
			for (const dropped of visible.slice(0, -MAX_VISIBLE)) this.dismiss(dropped.id);
			this.#items = visible.slice(-MAX_VISIBLE);
		}
		clearTimeout(this.#timers.get(toast.id));
		this.#timers.set(
			toast.id,
			setTimeout(() => this.dismiss(toast.id), DURATION_MS[kind])
		);
	}
}

export const toasts = new Toasts();
