import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { openEventStream, type StreamStatus } from '../src/lib/api/sse';
import { FakeEventSource } from './support/fakeEventSource';

beforeEach(() => {
	vi.useFakeTimers();
	FakeEventSource.reset();
	vi.stubGlobal('EventSource', FakeEventSource);
});

afterEach(() => {
	vi.useRealTimers();
});

describe('openEventStream', () => {
	it('reconnects with the URL as it is at reconnect time', () => {
		let cursor = 0;
		const stream = openEventStream(() => `/api/v1/activity/stream?after=${cursor}`, {});
		const first = FakeEventSource.latest;
		expect(first.url).toBe('/api/v1/activity/stream?after=0');

		first.open();
		cursor = 42;
		first.fail();
		expect(first.closed).toBe(true);
		expect(FakeEventSource.instances).toHaveLength(1);

		vi.advanceTimersByTime(1_000);
		expect(FakeEventSource.instances).toHaveLength(2);
		expect(FakeEventSource.latest.url).toBe('/api/v1/activity/stream?after=42');
		stream.close();
	});

	it('doubles the wait after each failed attempt up to 30 s and starts over after a connection opens', () => {
		const stream = openEventStream(() => '/stream', {});
		const waits: number[] = [];
		for (let attempt = 0; attempt < 7; attempt += 1) {
			const count = FakeEventSource.instances.length;
			FakeEventSource.latest.fail();
			let waited = 0;
			while (FakeEventSource.instances.length === count) {
				vi.advanceTimersByTime(250);
				waited += 250;
			}
			waits.push(waited);
		}
		expect(waits).toEqual([1_000, 2_000, 4_000, 8_000, 16_000, 30_000, 30_000]);

		FakeEventSource.latest.open();
		FakeEventSource.latest.fail();
		vi.advanceTimersByTime(999);
		const count = FakeEventSource.instances.length;
		vi.advanceTimersByTime(1);
		expect(FakeEventSource.instances).toHaveLength(count + 1);
		stream.close();
	});

	it('hands handlers the parsed data and the event id, and raw text when the data is not JSON', () => {
		const received: [unknown, string][] = [];
		const stream = openEventStream(() => '/stream', {
			activity: (data, id) => received.push([data, id])
		});
		FakeEventSource.latest.open();
		FakeEventSource.latest.emit('activity', { cursor: 7 }, '7');
		FakeEventSource.latest.emit('activity', 'not json', '8');
		expect(received).toEqual([
			[{ cursor: 7 }, '7'],
			['not json', '8']
		]);
		stream.close();
	});

	it('ends the stream on a closeOn event after its handler ran, and never reconnects', () => {
		const statuses: StreamStatus[] = [];
		const terminal = vi.fn();
		openEventStream(
			() => '/stream',
			{ terminal },
			{ closeOn: ['terminal'], onStatus: (status) => statuses.push(status) }
		);
		const source = FakeEventSource.latest;
		source.open();
		source.emit('terminal', { state: 'completed' });

		expect(terminal).toHaveBeenCalledWith({ state: 'completed' }, '');
		expect(source.closed).toBe(true);
		source.fail();
		vi.advanceTimersByTime(60_000);
		expect(FakeEventSource.instances).toHaveLength(1);
		expect(statuses).toEqual(['connecting', 'open', 'closed']);
	});

	it('still closes on a closeOn event whose handler throws', () => {
		openEventStream(
			() => '/stream',
			{
				terminal: () => {
					throw new Error('handler failed');
				}
			},
			{ closeOn: ['terminal'] }
		);
		const source = FakeEventSource.latest;
		expect(() => source.emit('terminal', {})).toThrow('handler failed');
		expect(source.closed).toBe(true);
	});

	it('reports reconnecting until a connection opens again', () => {
		const statuses: StreamStatus[] = [];
		const stream = openEventStream(() => '/stream', {}, { onStatus: (status) => statuses.push(status) });
		FakeEventSource.latest.open();
		FakeEventSource.latest.fail();
		vi.advanceTimersByTime(1_000);
		FakeEventSource.latest.fail();
		vi.advanceTimersByTime(2_000);
		FakeEventSource.latest.open();
		stream.close();
		expect(statuses).toEqual(['connecting', 'open', 'reconnecting', 'open', 'closed']);
	});

	it('cancels a pending reconnect when closed during the wait', () => {
		const stream = openEventStream(() => '/stream', {});
		FakeEventSource.latest.fail();
		stream.close();
		vi.advanceTimersByTime(60_000);
		expect(FakeEventSource.instances).toHaveLength(1);
	});

	it('skips the remaining wait when the browser comes back online', () => {
		const stream = openEventStream(() => '/stream', {});
		FakeEventSource.latest.fail();
		vi.advanceTimersByTime(1_000);
		FakeEventSource.latest.fail();
		expect(FakeEventSource.instances).toHaveLength(2);
		window.dispatchEvent(new Event('online'));
		expect(FakeEventSource.instances).toHaveLength(3);
		stream.close();
	});
});
