import { describe, expect, it } from 'vitest';
import {
	formatBytes,
	formatDuration,
	formatEta,
	formatImportProgress,
	formatPages,
	formatRelativeTime,
	importPhaseLabel,
	runStateLabel,
	stageLabel
} from '../src/lib/format';

describe('display formatting', () => {
	it('keeps file sizes readable at byte-unit boundaries', () => {
		expect(formatBytes(0)).toBe('0 B');
		expect(formatBytes(1023)).toBe('1023 B');
		expect(formatBytes(1024)).toBe('1 KB');
		expect(formatBytes(1024 ** 2)).toBe('1 MB');
	});

	it('formats elapsed time and does not invent an ETA before there is an estimate', () => {
		expect(formatDuration(65)).toBe('1 min 5 s');
		expect(formatDuration(7200)).toBe('2 h');
		expect(formatEta(null)).toBe('estimating time left');
		expect(formatEta(59)).toBe('less than a minute left');
	});

	it('labels pages, Run states, stages, and import progress for users', () => {
		expect(formatPages(4, 4)).toBe('p. 4');
		expect(formatPages(4, 9)).toBe('pp. 4–9');
		expect(formatPages(null, null)).toBe('');
		expect(runStateLabel('stopped')).toBe('Stopped');
		expect(stageLabel('verifying')).toBe('Verifying');
		expect(importPhaseLabel('ocr')).toBe('Recognizing text (OCR)');
		expect(
			formatImportProgress({ phase: 'ocr', done: 5, total: 10, unit: 'pages', message: null })
		).toBe('Recognizing text (OCR) · 5 of 10 pages');
	});

	it('uses a stable fallback for malformed relative timestamps', () => {
		expect(formatRelativeTime('not-a-timestamp')).toBe('');
	});
});
