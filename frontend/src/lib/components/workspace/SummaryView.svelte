<script lang="ts">
	import EmptyState from '$lib/components/common/EmptyState.svelte';
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import { api } from '$lib/api/client';
	import type { FinalSummary, Run, SummarySentence } from '$lib/api/types';
	import { formatCount, formatPages } from '$lib/format';
	import type { TextRange } from '$lib/source/sourceModel';
	import { toasts } from '$lib/stores/toasts.svelte';
	import EvidenceList from './EvidenceList.svelte';

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
	const PUBLICATION: Record<NonNullable<FinalSummary['publication']>, string> = {
		editorial: 'The written summary passed verification.',
		verified_subset: 'Sentences that failed verification were removed; the rest passed.',
		content_unit_fallback: 'The written summary did not pass verification, so this summary is built from verified content units.'
	};

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
			const pages = formatPages(first.page_start, first.page_end);
			onshow({ start: first.start, end: first.end }, `Evidence · ${first.segment_id}${pages ? ` · ${pages}` : ''}`);
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
				{#if summary.word_count !== null}{formatCount(summary.word_count)} words{/if}{#if summary.target_words !== null}
					of {formatCount(summary.target_words)} targeted{/if}
			</p>
			<div class="actions">
				<button type="button" class="button small" onclick={copy}>Copy</button>
				<a class="button small ghost" href={api.exportUrl(run.run_id, 'txt')} download>Text</a>
				<a class="button small ghost" href={api.exportUrl(run.run_id, 'md')} download>Markdown</a>
				<a class="button small ghost" href={api.exportUrl(run.run_id, 'json')} download>Audit JSON</a>
			</div>
		</div>

		{#if (summary.publication && summary.verification_state === 'completed') || summary.notices.length > 0}
			<ul class="notices">
				{#if summary.publication && summary.verification_state === 'completed'}
					<li class="notice severity-{summary.publication === 'editorial' ? 'info' : 'warning'}">
						{PUBLICATION[summary.publication]}
					</li>
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
						{@const pages = formatPages(citation.page_start, citation.page_end)}
						{@const label = `[${citation.citation_id}] ${citation.segment_id}${pages ? ` · ${pages}` : ''}`}
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
	}

	.toolbar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-2);
	}

	.words {
		margin: 0;
		color: var(--color-text-muted);
		font-size: 0.9rem;
	}

	.words.short {
		color: var(--color-amber-strong, var(--color-amber));
	}

	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-1);
	}

	.notices {
		margin: 0;
		padding: 0;
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
	}

	.notice {
		padding: var(--space-2) var(--space-3);
		border-radius: var(--radius-sm);
		background: var(--color-surface);
		font-size: 0.9rem;
	}

	.severity-warning {
		background: var(--color-amber-soft);
	}

	.severity-error {
		background: var(--color-danger-soft);
	}

	.legend {
		margin: 0;
		font-size: 0.85rem;
		color: var(--color-text-muted);
	}

	.text p {
		margin: 0 0 var(--space-3);
		line-height: 1.7;
	}

	.sentence {
		cursor: pointer;
		border-radius: 2px;
	}

	.sentence:hover {
		background: var(--color-surface);
	}

	.sentence.selected {
		background: var(--color-amber-soft);
	}

	.verdict-supported {
		text-decoration: underline;
		text-decoration-color: var(--color-sage);
		text-decoration-thickness: 2px;
		text-underline-offset: 3px;
	}

	.verdict-not_meaningfully_verifiable {
		text-decoration: underline dotted;
		text-decoration-color: var(--color-text-subtle);
		text-underline-offset: 3px;
	}

	.evidence-panel {
		margin: 0 0 var(--space-3);
		padding: var(--space-3);
		border-left: 3px solid var(--color-amber);
		background: var(--color-surface);
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}

	.verdict {
		margin: 0;
		font-size: 0.85rem;
		font-weight: 600;
	}

	.muted {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin: 0;
		color: var(--color-text-muted);
		font-size: 0.9rem;
	}

	.citations h3 {
		margin: 0 0 var(--space-2);
		font-size: 0.9rem;
	}

	.citations ul {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-1);
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.citation {
		min-height: 44px;
		padding: 0 var(--space-3);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
		background: var(--color-bg);
		color: inherit;
		font: inherit;
		font-size: 0.85rem;
		cursor: pointer;
	}

	.citation.static {
		display: inline-flex;
		align-items: center;
		cursor: default;
	}

	.removed summary {
		min-height: 44px;
		display: flex;
		align-items: center;
		cursor: pointer;
	}

	.removed ul {
		margin: 0;
		padding-left: 1.2rem;
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}

	.removed-text {
		margin: 0;
		text-decoration: line-through;
		color: var(--color-text-muted);
	}

	.removed-why {
		margin: 0;
		font-size: 0.85rem;
	}
</style>
