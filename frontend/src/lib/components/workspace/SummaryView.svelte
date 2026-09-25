<script lang="ts">
	import EmptyState from '$lib/components/common/EmptyState.svelte';
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import { api } from '$lib/api/client';
	import type { FinalSummary, Run, SummarySentence } from '$lib/api/types';
	import { formatCount } from '$lib/format';
	import type { TextRange } from '$lib/source/sourceModel';
	import { toasts } from '$lib/stores/toasts.svelte';
	import EvidenceList from './EvidenceList.svelte';
	import { evidenceLabel } from './runDisplay';

	interface Props {
		run: Run;
		summary: FinalSummary | null;
		loading: boolean;
		error: unknown;
		onretry: () => void;
		/** Highlight a cited passage in the Source. */
		onshow: (range: TextRange, label: string) => void;
		/** Also highlight a sentence's first evidence when it is selected (Source visible beside). */
		showOnSelect: boolean;
	}

	let { run, summary, loading, error, onretry, onshow, showOnSelect }: Props = $props();

	const VERDICT: Record<SummarySentence['verdict'], string> = {
		supported: 'Supported by the cited source passages',
		unchecked: 'Not checked (verification was off)',
		not_meaningfully_verifiable: 'Not checkable against the source'
	};
	// Other publication kinds arrive as server notices (verified_sentence_subset, verified_content_unit_fallback).
	const passedAsWritten = $derived(summary?.publication === 'editorial' && summary.verification_state === 'completed');

	let selected = $state<number | null>(null);

	const paragraphs = $derived.by(() => {
		const groups: SummarySentence[][] = [];
		let current = -1;
		for (const sentence of summary?.sentences ?? []) {
			if (sentence.paragraph !== current || groups.length === 0) {
				groups.push([]);
				current = sentence.paragraph;
			}
			groups[groups.length - 1].push(sentence);
		}
		return groups;
	});
	const selectedSentence = $derived(summary?.sentences.find((sentence) => sentence.index === selected) ?? null);
	const plainText = $derived(summary?.text ?? summary?.sentences.map((sentence) => sentence.text).join(' ') ?? '');

	function select(sentence: SummarySentence) {
		selected = selected === sentence.index ? null : sentence.index;
		const first = sentence.evidence.find((ref) => ref.start !== null && ref.end !== null);
		if (selected !== null && showOnSelect && first && first.start !== null && first.end !== null) {
			onshow({ start: first.start, end: first.end }, `Evidence · ${evidenceLabel(first.segment_id, first.page_start, first.page_end)}`);
		}
	}

	function onSentenceKey(event: KeyboardEvent, sentence: SummarySentence) {
		if (event.key !== 'Enter' && event.key !== ' ') return;
		event.preventDefault();
		select(sentence);
	}

	async function copy() {
		try {
			await navigator.clipboard.writeText(plainText);
			toasts.success('Summary copied.');
		} catch {
			toasts.error('Copying is not available here; select the text and copy it instead.');
		}
	}
</script>

{#if error}
	<ErrorBanner {error} title="Could not load the summary" {onretry} />
{:else if run.state !== 'completed'}
	<EmptyState
		title="No summary yet"
		message={run.state === 'failed' || run.state === 'stopped' || run.state === 'interrupted'
			? 'This Run ended before the summary was written. Resume it to finish the summary.'
			: 'The summary appears here when the Run completes. Watch the Tree tab to see it being built.'}
		headingLevel={3}
	/>
{:else if loading && !summary}
	<p class="muted"><Spinner size="sm" label="" /> Loading the summary…</p>
{:else if summary && !summary.available}
	<EmptyState title="The summary is not available" message="This Run completed without a published summary." headingLevel={3} />
{:else if summary}
	<article class="summary" aria-label="Summary">
		<div class="toolbar">
			<p class="words" class:short={summary.short_of_target}>
				{#if summary.word_count !== null}{formatCount(summary.word_count)} words{/if}{#if summary.target_words !== null}{` of ${formatCount(summary.target_words)} targeted`}{/if}
			</p>
			<div class="actions">
				<button type="button" class="button small" onclick={copy}>Copy</button>
				<a class="button small ghost" href={api.exportUrl(run.run_id, 'txt')} download>Text</a>
				<a class="button small ghost" href={api.exportUrl(run.run_id, 'md')} download>Markdown</a>
				<a class="button small ghost" href={api.exportUrl(run.run_id, 'json')} download>Audit JSON</a>
			</div>
		</div>

		{#if passedAsWritten || summary.notices.length > 0}
			<ul class="notices">
				{#if passedAsWritten}
					<li class="notice severity-info">The written summary passed verification.</li>
				{/if}
				{#each summary.notices as notice (notice.code + notice.message)}
					<li class="notice severity-{notice.severity}">{notice.message}</li>
				{/each}
			</ul>
		{/if}

		{#if summary.verification_state === 'completed'}
			<p class="legend">
				<span class="sample verdict-supported">Underlined</span> sentences are supported by the source. Select a sentence
				to see its evidence.
			</p>
		{/if}

		<div class="text">
			{#each paragraphs as paragraph, p (p)}
				<p>
					{#each paragraph as sentence (sentence.index)}
						<span
							class="sentence verdict-{sentence.verdict}"
							class:selected={selected === sentence.index}
							role="button"
							tabindex="0"
							aria-pressed={selected === sentence.index}
							title={VERDICT[sentence.verdict]}
							onclick={() => select(sentence)}
							onkeydown={(event) => onSentenceKey(event, sentence)}>{sentence.text}</span
						>{' '}
					{/each}
				</p>
				{#if selectedSentence && paragraph.includes(selectedSentence)}
					<div class="evidence-panel" aria-live="polite">
						<p class="verdict">{VERDICT[selectedSentence.verdict]}</p>
						{#if selectedSentence.evidence.length > 0}
							<EvidenceList evidence={selectedSentence.evidence} {onshow} />
						{:else}
							<p class="muted">No evidence is recorded for this sentence.</p>
						{/if}
					</div>
				{/if}
			{/each}
		</div>

		{#if summary.citations.length > 0}
			<section class="citations">
				<h3>Sources</h3>
				<ul>
					{#each summary.citations as citation (citation.citation_id)}
						{@const label = `[${citation.citation_id}] ${evidenceLabel(citation.segment_id, citation.page_start, citation.page_end)}`}
						<li>
							{#if citation.start !== null && citation.end !== null}
								{@const range = { start: citation.start, end: citation.end }}
								<button type="button" class="citation" onclick={() => onshow(range, label)}>{label}</button>
							{:else}
								<span class="citation static">{label}</span>
							{/if}
						</li>
					{/each}
				</ul>
			</section>
		{/if}

		{#if summary.removed_sentences.length > 0}
			<details class="removed">
				<summary>
					{summary.removed_sentences.length === 1
						? '1 sentence was removed by verification'
						: `${summary.removed_sentences.length} sentences were removed by verification`}
				</summary>
				<ul>
					{#each summary.removed_sentences as removed, i (i)}
						<li>
							<p class="removed-text">{removed.text}</p>
							<p class="removed-why">{removed.reason ?? `Verdict: ${removed.verdict}`}</p>
						</li>
					{/each}
				</ul>
			</details>
		{/if}
	</article>
{/if}

<style>
	.summary {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
		max-width: 44rem;
	}

	.toolbar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-2);
		padding-bottom: var(--space-3);
		border-bottom: 1px solid var(--color-border);
	}

	.words {
		margin: 0;
		color: var(--color-text-muted);
		font-size: 0.875rem;
		font-variant-numeric: tabular-nums;
	}

	.words.short {
		color: var(--color-amber-strong);
	}

	.actions {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-1);
	}

	.notices {
		margin: 0;
		padding: 0;
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}

	.notice {
		padding: 0.1rem 0 0.1rem var(--space-3);
		border-left: 3px solid var(--color-border-strong);
		color: var(--color-text-muted);
		font-size: 0.875rem;
	}

	.severity-info {
		border-left-color: var(--color-success);
	}

	.severity-warning {
		border-left-color: var(--color-amber);
		color: var(--color-text);
	}

	.severity-error {
		border-left-color: var(--color-danger);
		color: var(--color-danger);
	}

	.legend {
		margin: 0;
		font-size: 0.8125rem;
		color: var(--color-text-subtle);
	}

	.text p {
		margin: 0 0 var(--space-4);
		font-family: var(--font-serif);
		font-size: 1.1875rem;
		line-height: 1.7;
		color: var(--color-text);
	}

	.sentence {
		cursor: pointer;
		border-radius: 2px;
		transition: background-color 120ms ease;
	}

	.sentence:hover {
		background: var(--color-amber-soft);
	}

	.sentence.selected {
		background: var(--color-highlight);
	}

	.verdict-supported {
		text-decoration: underline;
		text-decoration-color: var(--color-success-border);
		text-decoration-thickness: 2px;
		text-underline-offset: 0.22em;
	}

	.verdict-supported:hover,
	.verdict-supported.selected {
		text-decoration-color: var(--color-success);
	}

	.verdict-not_meaningfully_verifiable {
		text-decoration: underline dotted;
		text-decoration-color: var(--color-text-subtle);
		text-underline-offset: 0.22em;
	}

	.evidence-panel {
		margin: calc(-1 * var(--space-2)) 0 var(--space-4);
		padding: var(--space-3) var(--space-4);
		border-left: 3px solid var(--color-highlight);
		background: var(--color-surface-elevated);
		border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}

	.verdict {
		margin: 0;
		font-size: 0.72rem;
		font-weight: 600;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--color-success);
	}

	.muted {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin: 0;
		color: var(--color-text-muted);
		font-size: 0.9rem;
	}

	.citations {
		padding-top: var(--space-3);
		border-top: 1px solid var(--color-border);
	}

	.citations h3 {
		margin: 0 0 var(--space-2);
		font-size: 0.72rem;
		font-weight: 600;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--color-text-subtle);
	}

	.citations ul {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-1) var(--space-2);
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.citation {
		min-height: 2.25rem;
		padding: 0 var(--space-3);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
		background: var(--color-surface-elevated);
		color: inherit;
		font: inherit;
		font-size: 0.8125rem;
		cursor: pointer;
	}

	.citation:hover {
		border-color: var(--color-border-strong);
	}

	.citation.static {
		display: inline-flex;
		align-items: center;
		cursor: default;
	}

	.removed {
		padding-top: var(--space-3);
		border-top: 1px solid var(--color-border);
	}

	.removed summary {
		min-height: 44px;
		display: flex;
		align-items: center;
		cursor: pointer;
		font-size: 0.875rem;
		color: var(--color-text-muted);
	}

	.removed ul {
		margin: 0;
		padding-left: 1.2rem;
		display: flex;
		flex-direction: column;
		gap: var(--space-3);
	}

	.removed-text {
		margin: 0;
		font-family: var(--font-serif);
		text-decoration: line-through;
		text-decoration-color: var(--color-accent);
		color: var(--color-text-muted);
	}

	.removed-why {
		margin: 0.15rem 0 0;
		font-size: 0.8125rem;
		color: var(--color-text-subtle);
	}

	@media (max-width: 639px) {
		.text p {
			font-size: 1.0625rem;
		}
	}
</style>
