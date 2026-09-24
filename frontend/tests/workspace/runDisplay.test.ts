import { describe, expect, it } from 'vitest';
import type { RunProgress } from '../../src/lib/api/types';
import {
	activeStageBar,
	liveElapsed,
	liveEta,
	reaskNote,
	stageRows
} from '../../src/lib/components/workspace/runDisplay';

function progress(overrides: Partial<RunProgress> = {}): RunProgress {
	return {
		cursor: 1,
		attempt_number: 1,
		started_at: null,
		elapsed_seconds: 30,
		stage: 'summarizing',
		stages: [
			{ stage: 'preparing', state: 'completed', completed: null, total: null, detail: 'hierarchical' },
			{ stage: 'segmenting', state: 'completed', completed: null, total: null, detail: null },
			{ stage: 'summarizing', state: 'active', completed: 12, total: 40, detail: null }
		],
		leaves: { done: 12, total: 40, reused: 3, failed: 1 },
		merges: { done: 0, total: null, reused: 0, failed: 0 },
		merge_levels: [],
		claims: { done: 0, total: null, reused: 0, failed: 0 },
		current_items: [],
		eta_seconds: 90,
		eta_basis: null,
		...overrides
	};
}

describe('live elapsed and ETA', () => {
	it('ticks elapsed forward from the snapshot while the Run is live', () => {
		expect(liveElapsed(30, 1_000, 6_500, true)).toBe(35.5);
		expect(liveElapsed(30, 1_000, 6_500, false)).toBe(30);
		expect(liveElapsed(null, 1_000, 6_500, true)).toBeNull();
	});

	it('counts the ETA down and never below zero', () => {
		expect(liveEta(90, 0, 30_000, true)).toBe(60);
		expect(liveEta(10, 0, 30_000, true)).toBe(0);
		expect(liveEta(90, 0, 30_000, false)).toBe(90);
		expect(liveEta(null, 0, 30_000, true)).toBeNull();
	});

	it('ignores a clock reading taken before the snapshot arrived', () => {
		expect(liveElapsed(30, 5_000, 4_000, true)).toBe(30);
		expect(liveEta(90, 5_000, 4_000, true)).toBe(90);
	});
});

describe('stageRows', () => {
	it('lists every stage in pipeline order with counts and details', () => {
		const rows = stageRows(progress());
		expect(rows.map((row) => row.stage)).toEqual([
			'preparing',
			'segmenting',
			'summarizing',
			'merging',
			'writing',
			'verifying',
			'publishing'
		]);
		expect(rows[0].notes[0]).toMatch(/hierarchical/);
		expect(rows[2]).toMatchObject({ state: 'active', counts: '12 of 40 segments' });
		expect(rows[2].notes).toEqual(['3 reused from earlier work', '1 failed']);
		expect(rows[3].state).toBe('pending');
	});

	it('shows merge progress per level and verification progress in claims', () => {
		const rows = stageRows(
			progress({
				stage: 'verifying',
				merges: { done: 11, total: 11, reused: 0, failed: 0 },
				merge_levels: [
					{ level: 1, done: 9, total: 9 },
					{ level: 2, done: 2, total: 2 }
				],
				claims: { done: 5, total: 18, reused: 0, failed: 0 },
				stages: [{ stage: 'verifying', state: 'active', completed: 5, total: 18, detail: 'Repair pass 1' }]
			})
		);
		expect(rows[3].counts).toBe('11 of 11 merges');
		expect(rows[3].notes).toEqual(['Level 1: 9 of 9', 'Level 2: 2 of 2']);
		expect(rows[5]).toMatchObject({ counts: '5 of 18 claims checked', notes: ['Repair pass 1'] });
	});

	it('shows all stages pending before the first snapshot', () => {
		expect(stageRows(null).every((row) => row.state === 'pending' && row.counts === null)).toBe(true);
	});
});

describe('activeStageBar', () => {
	it('fills for counted stages and is indeterminate otherwise', () => {
		expect(activeStageBar(progress())).toEqual({ label: 'Summarizing', value: 12, max: 40 });
		expect(
			activeStageBar(
				progress({ stage: 'writing', stages: [{ stage: 'writing', state: 'active', completed: null, total: null, detail: null }] })
			)
		).toEqual({ label: 'Writing', value: 0, max: null });
		expect(activeStageBar(progress({ stage: null }))).toBeNull();
	});
});

describe('reaskNote', () => {
	it('notes re-asks after invalid output but not first tries', () => {
		const item = { kind: 'leaf', work_id: 'L0N0003', label: 'Segment 3', started_at: null };
		expect(reaskNote({ ...item, attempt: 2 })).toBe('Re-asked after invalid output (try 2 of 3)');
		expect(reaskNote({ ...item, attempt: 1 })).toBeNull();
		expect(reaskNote({ ...item, attempt: null })).toBeNull();
	});
});
