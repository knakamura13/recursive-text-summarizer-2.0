import { createSubscriber } from 'svelte/reactivity';

/**
 * Monotonic clock (performance.now, the clock RunStream stamps progress with) that ticks once a
 * second, but only while some effect or template reads it; no reader means no interval.
 */
class Ticker {
	#now = performance.now();
	#subscribe = createSubscriber((update) => {
		this.#now = performance.now();
		const id = setInterval(() => {
			this.#now = performance.now();
			update();
		}, 1000);
		return () => clearInterval(id);
	});

	get now(): number {
		if (!$effect.tracking()) return performance.now();
		this.#subscribe();
		return this.#now;
	}
}

export const ticker = new Ticker();
