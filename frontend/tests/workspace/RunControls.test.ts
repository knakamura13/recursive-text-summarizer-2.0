import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import type { Run, RunState } from '../../src/lib/api/types';
import RunControls from '../../src/lib/components/workspace/RunControls.svelte';

function makeRun(state: RunState, overrides: Partial<Run> = {}): Run {
	return {
		run_id: 'run-1',
		document_id: 'doc-1',
		document_title: 'Report',
		state,
		requested_strategy: 'auto',
		selected_strategy: null,
		config: {} as Run['config'],
		created_at: '2026-09-23T10:00:00Z',
		updated_at: '2026-09-23T10:00:00Z',
		attempt: null,
		attempt_count: 1,
		failure: null,
		can_stop: state === 'queued' || state === 'running',
		can_resume: state === 'stopped' || state === 'failed' || state === 'interrupted',
		progress: null,
		...overrides
	};
}

function setup(run: Run, props: { blockedBy?: Run | null; stopPending?: boolean; resumePending?: boolean } = {}) {
	const onstop = vi.fn();
	const onresume = vi.fn();
	const onnew = vi.fn();
	render(RunControls, {
		props: {
			run,
			blockedBy: props.blockedBy ?? null,
			stopPending: props.stopPending ?? false,
			resumePending: props.resumePending ?? false,
			onstop,
			onresume,
			onnew
		}
	});
	return { onstop, onresume, onnew };
}

describe('RunControls', () => {
	it.each(['queued', 'running'] as const)('offers Stop for a %s Run and nothing else', async (state) => {
		const { onstop } = setup(makeRun(state));
		const stop = screen.getByRole('button', { name: 'Stop' });
		expect(stop).toBeEnabled();
		expect(screen.queryByRole('button', { name: 'Resume' })).toBeNull();
		expect(screen.queryByRole('button', { name: 'Start new Run' })).toBeNull();
		await fireEvent.click(stop);
		expect(onstop).toHaveBeenCalledOnce();
	});

	it('shows "Stopping…" right after Stop is clicked, before the server answers', () => {
		setup(makeRun('running'), { stopPending: true });
		expect(screen.getByRole('button', { name: 'Stopping…' })).toBeDisabled();
	});

	it('shows "Stopping…" while the Run is stopping', () => {
		setup(makeRun('stopping'));
		expect(screen.getByRole('button', { name: 'Stopping…' })).toBeDisabled();
	});

	it.each(['stopped', 'failed', 'interrupted'] as const)('offers Resume and Start new Run for a %s Run', async (state) => {
		const { onresume, onnew } = setup(makeRun(state));
		await fireEvent.click(screen.getByRole('button', { name: 'Resume' }));
		await fireEvent.click(screen.getByRole('button', { name: 'Start new Run' }));
		expect(onresume).toHaveBeenCalledOnce();
		expect(onnew).toHaveBeenCalledOnce();
		expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull();
	});

	it('blocks Resume while another Run is active and links to it', () => {
		const other = makeRun('running', { run_id: 'run-2', document_id: 'doc-2', document_title: 'Annual report' });
		setup(makeRun('failed'), { blockedBy: other });
		expect(screen.getByRole('button', { name: 'Resume' })).toBeDisabled();
		expect(screen.getByRole('link', { name: 'Annual report' })).toHaveAttribute('href', '/documents/doc-2?run=run-2');
	});

	it('does not treat the Run itself as the blocker', () => {
		setup(makeRun('stopped'), { blockedBy: makeRun('stopped') });
		expect(screen.getByRole('button', { name: 'Resume' })).toBeEnabled();
	});

	it('shows "Resuming…" while Resume is in flight', () => {
		setup(makeRun('stopped'), { resumePending: true });
		expect(screen.getByRole('button', { name: 'Resuming…' })).toBeDisabled();
	});

	it('offers only Start new Run for a completed Run', () => {
		setup(makeRun('completed'));
		expect(screen.getByRole('button', { name: 'Start new Run' })).toBeEnabled();
		expect(screen.queryByRole('button', { name: 'Resume' })).toBeNull();
		expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull();
	});
});
