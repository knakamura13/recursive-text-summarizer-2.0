// Pure helpers that turn a Run's progress snapshot into what the status panel shows.
import { formatCount, formatPages, stageLabel } from '$lib/format';
import type { CurrentItem, RunConfig, RunProgress, StageName, StageProgress } from '$lib/api/types';

/** Stage order of the pipeline, used when the server has not sent stage rows yet. */
export const STAGE_ORDER: readonly StageName[] = [
	'preparing',
	'segmenting',
	'summarizing',
	'merging',
	'writing',
	'verifying',
	'publishing'
];

/** One model call plus up to two re-asks with the validation error (D12). */
export const MAX_ITEM_TRIES = 3;

/**
 * Seconds shown for a server value measured at `receivedAt`, counted forward on the local clock
 * while the Run is live so the timer ticks without refetching.
 */
export function liveElapsed(base: number | null, receivedAt: number, now: number, live: boolean): number | null {
	if (base === null) return null;
	return live ? base + Math.max(0, now - receivedAt) / 1000 : base;
}

/** Remaining seconds counted down from the server's estimate, never below zero. */
export function liveEta(eta: number | null, receivedAt: number, now: number, live: boolean): number | null {
	if (eta === null) return null;
	return live ? Math.max(0, eta - Math.max(0, now - receivedAt) / 1000) : eta;
}

export interface StageRow {
	stage: StageName;
	label: string;
	state: StageProgress['state'];
	/** "12 of 40 segments" style count, when the stage has one. */
	counts: string | null;
	/** Extra lines: stage detail, reused/failed counts, per-level merge progress. */
	notes: string[];
}

function countText(done: number, total: number | null, noun: string): string {
	return total === null ? `${formatCount(done)} ${noun}` : `${formatCount(done)} of ${formatCount(total)} ${noun}`;
}

/** Stepper rows with counts and details for every stage. */
export function stageRows(progress: RunProgress | null): StageRow[] {
	const byStage = new Map((progress?.stages ?? []).map((row) => [row.stage, row]));
	return STAGE_ORDER.map((stage) => {
		const row = byStage.get(stage);
		const state = row?.state ?? 'pending';
		const notes: string[] = [];
		let counts: string | null = null;
		const stageCount = (noun: string) =>
			row && row.completed !== null ? countText(row.completed, row.total, noun) : null;
		switch (stage) {
			case 'preparing':
				if (row?.detail === 'direct') notes.push('Strategy: direct (one pass over the whole Document)');
				else if (row?.detail === 'hierarchical') notes.push('Strategy: hierarchical (segments, then merges)');
				else if (row?.detail) notes.push(row.detail);
				break;
			case 'summarizing': {
				const leaves = progress?.leaves;
				counts = leaves ? countText(leaves.done, leaves.total, 'segments') : stageCount('segments');
				if (leaves && leaves.reused > 0) notes.push(`${formatCount(leaves.reused)} reused from earlier work`);
				if (leaves && leaves.failed > 0) notes.push(`${formatCount(leaves.failed)} failed`);
				if (row?.detail) notes.push(row.detail);
				break;
			}
			case 'merging': {
				const merges = progress?.merges;
				counts =
					merges && (merges.done > 0 || merges.total !== null)
						? countText(merges.done, merges.total, 'merges')
						: stageCount('merges');
				for (const level of progress?.merge_levels ?? []) {
					notes.push(`Level ${level.level}: ${formatCount(level.done)} of ${formatCount(level.total)}`);
				}
				if (merges && merges.reused > 0) notes.push(`${formatCount(merges.reused)} reused from earlier work`);
				if (row?.detail) notes.push(row.detail);
				break;
			}
			case 'verifying': {
				const claims = progress?.claims;
				counts =
					claims && (claims.done > 0 || claims.total !== null)
						? countText(claims.done, claims.total, 'claims checked')
						: stageCount('claims checked');
				if (row?.detail) notes.push(row.detail);
				break;
			}
			default:
				counts = stageCount('items');
				if (row?.detail) notes.push(row.detail);
		}
		return { stage, label: stageLabel(stage), state, counts, notes };
	});
}

export interface StageBar {
	label: string;
	value: number;
	/** Null draws an indeterminate bar. */
	max: number | null;
}

/** Progress bar for the active stage: counted stages fill, the others are indeterminate. */
export function activeStageBar(progress: RunProgress | null): StageBar | null {
	if (progress === null || progress.stage === null) return null;
	const stage = progress.stage;
	const row = progress.stages.find((item) => item.stage === stage);
	const counter =
		stage === 'summarizing' ? progress.leaves : stage === 'merging' ? progress.merges : stage === 'verifying' ? progress.claims : null;
	const total = row?.total ?? counter?.total ?? null;
	const done = row?.completed ?? counter?.done ?? 0;
	const label = stageLabel(stage);
	if (total === null || total <= 0) return { label, value: 0, max: null };
	return { label, value: Math.min(done, total), max: total };
}

/** A note for items being re-asked after invalid output, or null for a first try. */
export function reaskNote(item: CurrentItem): string | null {
	if (item.attempt === null || item.attempt < 2) return null;
	return `Re-asked after invalid output (try ${item.attempt} of ${MAX_ITEM_TRIES})`;
}

/** One-line description of a Run configuration for the Runs list. */
export function describeConfig(config: RunConfig): string {
	const parts = [
		config.model || 'no model',
		`${formatCount(config.target_words)} words`,
		config.strategy === 'auto' ? 'auto strategy' : config.strategy,
		config.verify ? 'verified' : 'not verified'
	];
	return parts.join(' · ');
}

/** "Segment 3 · pp. 4–5" style label for an evidence reference. */
export function evidenceLabel(segmentId: string, pageStart: number | null, pageEnd: number | null): string {
	const pages = formatPages(pageStart, pageEnd);
	return pages ? `${segmentId} · ${pages}` : segmentId;
}
