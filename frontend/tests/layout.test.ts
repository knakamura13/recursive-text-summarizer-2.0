import { describe, expect, it } from 'vitest';
import { pythonOffsetToUtf16Index, sliceByPythonOffsets } from '../src/lib/unicodeOffsets';
import fixture from './fixtures/workspace-shell.json';

describe('workspace shell fixture', () => {
	it('includes required layout regions', () => {
		expect(fixture.document.title).toBeTruthy();
		expect(fixture.hierarchy.length).toBeGreaterThan(0);
		expect(fixture.stages.length).toBeGreaterThan(0);
		expect(fixture.sourcePreview).toBeTruthy();
		expect(fixture.finalSummary).toBeTruthy();
	});

	it('uses real hierarchy labels rather than fabricated sections', () => {
		const labels = fixture.hierarchy.map((node) => node.label);
		expect(labels).toContain('Full document');
		expect(labels.some((label) => label.includes('Self-Attention'))).toBe(true);
	});
});

describe('unicodeOffsets', () => {
	it('converts python code-point offsets for astral characters', () => {
		const text = 'a😊b';
		expect(pythonOffsetToUtf16Index(text, 2)).toBe(3);
		expect(sliceByPythonOffsets(text, 1, 2)).toBe('😊');
	});
});
