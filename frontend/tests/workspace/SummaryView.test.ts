import { fireEvent, render, screen, within } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import type { FinalSummary, Run, SummaryHeading, SummarySentence } from '../../src/lib/api/types';
import SummaryView from '../../src/lib/components/workspace/SummaryView.svelte';

const RUN: Run = {
	run_id: 'run-1',
	document_id: 'doc-1',
	document_title: 'Harbor report',
	state: 'completed',
	requested_strategy: 'auto',
	selected_strategy: null,
	config: {} as Run['config'],
	created_at: '2026-09-30T10:00:00Z',
	updated_at: '2026-09-30T10:00:00Z',
	attempt: null,
	attempt_count: 1,
	failure: null,
	can_stop: false,
	can_resume: false,
	progress: null
};

function sentence(index: number, paragraph: number, text: string, overrides: Partial<SummarySentence> = {}): SummarySentence {
	return { index, paragraph, text, verdict: 'unchecked', evidence: [], ...overrides };
}

function heading(sectionId: string, level: number, text: string, beforeSentence: number): SummaryHeading {
	return { section_id: sectionId, level, text, before_sentence: beforeSentence };
}

function makeSummary(overrides: Partial<FinalSummary>): FinalSummary {
	return {
		available: true,
		text: null,
		sentences: [],
		headings: [],
		removed_sentences: [],
		citations: [],
		word_count: null,
		target_words: null,
		short_of_target: false,
		notices: [],
		verification_state: 'not_run',
		publication: 'editorial',
		...overrides
	};
}

function setup(summary: FinalSummary) {
	const onshow = vi.fn();
	render(SummaryView, {
		props: { run: RUN, summary, loading: false, error: null, onretry: vi.fn(), onshow, showOnSelect: true }
	});
	return { onshow };
}

/** The summary text's blocks in order: "h3 Title", "p: first | second" (its sentences), "evidence". */
function blocks(): string[] {
	const first = screen.getAllByRole('button').find((element) => element.classList.contains('sentence'));
	const container = first?.closest('p')?.parentElement;
	if (!container) throw new Error('No summary sentences rendered');
	return Array.from(container.children, (block) => {
		if (/^H[1-6]$/.test(block.tagName)) return `${block.tagName.toLowerCase()} ${block.textContent}`;
		if (block.tagName === 'P') {
			return `p: ${Array.from(block.querySelectorAll('[role="button"]'), (item) => item.textContent).join(' | ')}`;
		}
		return block.classList.contains('evidence-panel') ? 'evidence' : block.tagName;
	});
}

// "The report opens.\n\n# Water Quality\n\nOxygen was low. Nitrogen peaked.\n\n# Habitat\n\n## Shorebirds\n\n
// Plovers nested.\n\n# Funding & <em>grants</em>": an untitled opening, a heading-only parent before its
// first child, and an empty section last.
const SECTIONED = makeSummary({
	sentences: [
		sentence(0, 0, 'The report opens.'),
		sentence(1, 2, 'Oxygen was low.'),
		sentence(2, 2, 'Nitrogen peaked.', {
			verdict: 'supported',
			evidence: [
				{ segment_id: 'S000002', quote: 'nitrogen reached 2.3', quote_found: true, start: 120, end: 140, page_start: 1, page_end: 1 }
			]
		}),
		sentence(3, 5, 'Plovers nested.')
	],
	headings: [
		heading('s2', 1, 'Water Quality', 1),
		heading('s3', 1, 'Habitat', 3),
		heading('s4', 2, 'Shorebirds', 3),
		heading('s5', 1, 'Funding & <em>grants</em>', 4)
	],
	verification_state: 'completed'
});

describe('SummaryView', () => {
	it('shows each section heading by level between its sentence groups', () => {
		setup(SECTIONED);

		expect(blocks()).toEqual([
			'p: The report opens.',
			'h3 Water Quality',
			'p: Oxygen was low. | Nitrogen peaked.',
			'h3 Habitat',
			'h4 Shorebirds',
			'p: Plovers nested.',
			'h3 Funding & <em>grants</em>'
		]);
		const funding = screen.getByRole('heading', { level: 3, name: 'Funding & <em>grants</em>' });
		expect(funding.querySelector('em')).toBeNull();
	});

	it('nests an outline that starts below level 1 under the view and stops at h6', () => {
		setup(
			makeSummary({
				sentences: [sentence(0, 1, 'Deck spans 200 feet.'), sentence(1, 6, 'Piers cracked.')],
				headings: [
					heading('s1', 2, 'Bridge', 0),
					heading('s2', 3, 'Deck', 1),
					heading('s3', 4, 'Joints', 1),
					heading('s4', 6, 'Piers', 1)
				]
			})
		);

		expect(blocks()).toEqual(['h3 Bridge', 'p: Deck spans 200 feet.', 'h4 Deck', 'h5 Joints', 'h6 Piers', 'p: Piers cracked.']);
	});

	it('selects a sentence under a heading and shows its evidence before the next heading', async () => {
		const { onshow } = setup(SECTIONED);

		const nitrogen = screen.getByRole('button', { name: 'Nitrogen peaked.' });
		await fireEvent.click(nitrogen);

		expect(nitrogen).toHaveAttribute('aria-pressed', 'true');
		expect(onshow).toHaveBeenCalledWith({ start: 120, end: 140 }, expect.stringContaining('Evidence'));
		expect(screen.getByText('nitrogen reached 2.3')).toBeInTheDocument();
		expect(blocks().slice(1, 4)).toEqual(['h3 Water Quality', 'p: Oxygen was low. | Nitrogen peaked.', 'evidence']);
		expect(blocks()[4]).toBe('h3 Habitat');
	});

	it('renders a summary written whole as paragraphs only', () => {
		setup(
			makeSummary({
				text: 'The deck spans 200 feet. It was inspected in 1998.\n\nThe pier cracked.',
				sentences: [
					sentence(0, 0, 'The deck spans 200 feet.'),
					sentence(1, 0, 'It was inspected in 1998.'),
					sentence(2, 1, 'The pier cracked.')
				]
			})
		);

		expect(blocks()).toEqual(['p: The deck spans 200 feet. | It was inspected in 1998.', 'p: The pier cracked.']);
		expect(within(screen.getByRole('article', { name: 'Summary' })).queryAllByRole('heading')).toEqual([]);
	});
});
