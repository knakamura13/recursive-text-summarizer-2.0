// Client-side checks and merge patches for RunConfig, mirroring the limits of
// summarizer_web/models/api.py so forms can explain a bad value before saving.
import { ApiError } from '$lib/api/client';
import type { ClearableConfigField, RunConfig, RunConfigPatch } from '$lib/api/types';
import { formatCount } from '$lib/format';

export type RunConfigField = keyof RunConfig;
export type RunConfigErrors = Partial<Record<RunConfigField, string>>;

export interface NumberLimit {
	min: number;
	max?: number;
	integer: boolean;
	/** Empty means "automatic" (null). */
	nullable: boolean;
}

export const RUN_CONFIG_LIMITS = {
	target_words: { min: 25, max: 5000, integer: true, nullable: false },
	max_repair_passes: { min: 0, max: 5, integer: true, nullable: false },
	context_window: { min: 1024, max: 4_194_304, integer: true, nullable: true },
	max_output_tokens: { min: 128, max: 131_072, integer: true, nullable: false },
	safety_margin_tokens: { min: 0, max: 131_072, integer: true, nullable: false },
	safety_margin_fraction: { min: 0, max: 0.5, integer: false, nullable: false },
	chunk_tokens: { min: 128, integer: true, nullable: true },
	overlap_tokens: { min: 0, integer: true, nullable: false },
	max_merge_children: { min: 2, max: 64, integer: true, nullable: true },
	max_concurrency: { min: 1, max: 16, integer: true, nullable: false },
	timeout_seconds: { min: 10, max: 3600, integer: false, nullable: false },
	max_retries: { min: 1, max: 20, integer: true, nullable: false }
} satisfies Partial<Record<RunConfigField, NumberLimit>>;

export type NumericRunConfigField = keyof typeof RUN_CONFIG_LIMITS;

const CLEARABLE: readonly RunConfigField[] = [
	'context_window',
	'chunk_tokens',
	'max_merge_children'
] satisfies ClearableConfigField[];

/** Messages keyed by field for every value the server would reject. */
export function validateRunConfig(
	config: RunConfig,
	options: { requireModel?: boolean } = {}
): RunConfigErrors {
	const errors: RunConfigErrors = {};
	if (options.requireModel && !config.model.trim()) errors.model = 'Choose a model.';
	for (const [field, limit] of Object.entries(RUN_CONFIG_LIMITS) as [
		NumericRunConfigField,
		NumberLimit
	][]) {
		const value = config[field] as number | null | undefined;
		if (value === null || value === undefined || Number.isNaN(value)) {
			if (!limit.nullable) errors[field] = 'Enter a number.';
		} else if (limit.integer && !Number.isInteger(value)) {
			errors[field] = 'Enter a whole number.';
		} else if (value < limit.min || (limit.max !== undefined && value > limit.max)) {
			errors[field] =
				limit.max === undefined
					? `Enter at least ${formatCount(limit.min)}.`
					: `Enter a value from ${formatCount(limit.min)} to ${formatCount(limit.max)}.`;
		}
	}
	if (
		!errors.overlap_tokens &&
		config.chunk_tokens !== null &&
		config.overlap_tokens >= config.chunk_tokens
	) {
		errors.overlap_tokens = 'Must be smaller than the chunk size.';
	}
	return errors;
}

/**
 * The merge patch that turns `saved` into `edited`, or null when nothing
 * changed. Nullable fields emptied by the user go to `clear` because null
 * means "unchanged" in a patch.
 */
export function runConfigPatch(saved: RunConfig, edited: RunConfig): RunConfigPatch | null {
	const patch: Record<string, unknown> = {};
	const clear: ClearableConfigField[] = [];
	for (const field of Object.keys(edited) as RunConfigField[]) {
		if (edited[field] === saved[field]) continue;
		if (edited[field] === null && CLEARABLE.includes(field)) {
			clear.push(field as ClearableConfigField);
		} else {
			patch[field] = edited[field];
		}
	}
	if (clear.length > 0) patch.clear = clear;
	return Object.keys(patch).length > 0 ? (patch as RunConfigPatch) : null;
}

/**
 * Field messages from a 422 invalid_request, keyed by the last named segment
 * of each validation error location (e.g. `target_words`, `ollama_host`).
 */
export function fieldErrorsFrom(error: unknown): Record<string, string> {
	if (!(error instanceof ApiError) || error.code !== 'invalid_request') return {};
	const items = error.details?.errors;
	if (!Array.isArray(items)) return {};
	const result: Record<string, string> = {};
	for (const item of items as { loc?: unknown; msg?: unknown }[]) {
		if (!Array.isArray(item?.loc) || typeof item.msg !== 'string') continue;
		const field = item.loc.findLast((part) => typeof part === 'string' && part !== 'body');
		if (typeof field === 'string' && !(field in result)) result[field] = item.msg;
	}
	return result;
}
