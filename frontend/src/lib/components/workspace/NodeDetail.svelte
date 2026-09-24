<script lang="ts">
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import type { NodeDetail, SegmentRef, TreeNode } from '$lib/api/types';
	import { formatDuration, formatPages } from '$lib/format';
	import type { TextRange } from '$lib/source/sourceModel';
	import EvidenceList from './EvidenceList.svelte';

	interface Props {
		/** Live tree row (state and label update while the Run progresses). */
		node: TreeNode | undefined;
		detail: NodeDetail | null;
		loading: boolean;
		error: unknown;
		onretry: () => void;
		onshow: (range: TextRange, label: string, tone: 'span' | 'quote') => void;
		onselectnode: (nodeId: string) => void;
	}

	let { node, detail, loading, error, onretry, onshow, onselectnode }: Props = $props();

	const KIND_LABEL = { leaf: 'Segment summary', merge: 'Merge', passthrough: 'Passthrough' } as const;
	const STATE_LABEL = { pending: 'Pending', active: 'Working', completed: 'Done', failed: 'Failed' } as const;

	const state = $derived(node?.state ?? detail?.state ?? 'pending');
	const label = $derived(node?.label ?? detail?.label ?? '');
	const hasDetailData = $derived(
		detail !== null &&
			(detail.content_units.length > 0 ||
				detail.annotations.length > 0 ||
				detail.entities.length > 0 ||
				detail.quotations.length > 0)
	);

	function showSegment(segment: SegmentRef) {
		const pages = formatPages(segment.page_start, segment.page_end);
		onshow(
			{ start: segment.core_start, end: segment.core_end },
			`Segment ${segment.order + 1}${pages ? ` · ${pages}` : ''}`,
			'span'
		);
	}
</script>

<article class="node-detail" aria-labelledby="node-detail-title">
	<header class="head">
		<h2 id="node-detail-title">{label}</h2>
		<p class="meta">
			<span class="state state-{state}">{STATE_LABEL[state]}</span>
			{#if detail}· {KIND_LABEL[detail.kind]} · level {detail.level}{/if}
			{#if detail?.duration_seconds != null}· took {formatDuration(detail.duration_seconds)}{/if}
		</p>
	</header>

	{#if error}
		<ErrorBanner {error} title="Could not load this node" {onretry} />
	{:else if loading && !detail}
		<p class="muted"><Spinner size="sm" label="" /> Loading…</p>
	{:else if detail}
		{#if detail.error}
			<div class="error" role="alert">
				<h3>Error</h3>
				<p>{detail.error}</p>
			</div>
		{/if}

		{#if detail.summary_text}
			<section>
				<h3>Summary</h3>
				<p class="summary">{detail.summary_text}</p>
			</section>
		{:else if state === 'active'}
			<p class="muted">Being summarized now…</p>
		{:else if state === 'pending'}
			<p class="muted">Not summarized yet.</p>
		{/if}

		{#if detail.content_units.length > 0}
			<section>
				<h3>Content units ({detail.content_units.length})</h3>
				<ol class="units">
					{#each detail.content_units as unit, i (i)}
						<li>
							<p class="unit-text">
								{unit.text}
								<span class="tags">
									<span class="tag">{unit.kind}</span>
									{#if unit.uncertain}<span class="tag uncertain">uncertain</span>{/if}
								</span>
							</p>
							{#if unit.qualification}<p class="qualification">Qualification: {unit.qualification}</p>{/if}
							<EvidenceList evidence={unit.evidence} onshow={(range, text) => onshow(range, text, 'quote')} />
						</li>
					{/each}
				</ol>
			</section>
		{/if}

		{#if detail.annotations.length > 0}
			<section>
				<h3>Annotations</h3>
				<ul class="units">
					{#each detail.annotations as annotation, i (i)}
						<li>
							<p class="unit-text">
								<span class="tag">{annotation.kind}</span>
								{annotation.text}
							</p>
							<EvidenceList evidence={annotation.evidence} onshow={(range, text) => onshow(range, text, 'quote')} />
						</li>
					{/each}
				</ul>
			</section>
		{/if}

		{#if detail.entities.length > 0}
			<section>
				<h3>Entities</h3>
				<ul class="chips">
					{#each detail.entities as entity (entity)}<li>{entity}</li>{/each}
				</ul>
			</section>
		{/if}

		{#if detail.quotations.length > 0}
			<section>
				<h3>Quotations</h3>
				<EvidenceList evidence={detail.quotations} onshow={(range, text) => onshow(range, text, 'quote')} />
			</section>
		{/if}

		{#if !hasDetailData && state === 'completed'}
			<p class="muted">This Run has no content units for this node (it was made before they were recorded).</p>
		{/if}

		{#if detail.covered_segments.length > 0}
			<section>
				<h3>Covers {detail.covered_segments.length === 1 ? '1 segment' : `${detail.covered_segments.length} segments`}</h3>
				<ul class="segments">
					{#each detail.covered_segments as segment (segment.segment_id)}
						{@const pages = formatPages(segment.page_start, segment.page_end)}
						<li>
							<button type="button" class="segment" onclick={() => showSegment(segment)}>
								Segment {segment.order + 1}{pages ? ` · ${pages}` : ''}
							</button>
						</li>
					{/each}
				</ul>
			</section>
		{/if}

		{#if detail.parent_id || detail.child_ids.length > 0}
			<section class="relations">
				{#if detail.parent_id}
					<button type="button" class="button small ghost" onclick={() => onselectnode(detail.parent_id!)}>
						Open parent
					</button>
				{/if}
				{#if detail.child_ids.length > 0}
					<span class="muted">Merges {detail.child_ids.length} summaries</span>
				{/if}
			</section>
		{/if}

		{#if detail.started_at || detail.completed_at}
			<p class="timing">
				{#if detail.started_at}Started {new Date(detail.started_at).toLocaleString()}{/if}{#if detail.completed_at}
					· finished {new Date(detail.completed_at).toLocaleString()}{/if}
			</p>
		{/if}
	{/if}
</article>

<style>
	.node-detail {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
	}

	.head h2 {
		margin: 0;
		font-size: 1.05rem;
		overflow-wrap: anywhere;
	}

	.meta {
		margin: var(--space-1) 0 0;
		font-size: 0.85rem;
		color: var(--color-text-muted);
	}

	.state {
		font-weight: 600;
	}

	.state-active {
		color: var(--color-amber-strong, var(--color-amber));
	}

	.state-completed {
		color: var(--color-sage);
	}

	.state-failed {
		color: var(--color-danger);
	}

	section {
		display: flex;
		flex-direction: column;
		gap: var(--space-2);
	}

	h3 {
		margin: 0;
		font-size: 0.9rem;
	}

	.summary {
		margin: 0;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}

	.muted {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin: 0;
		color: var(--color-text-muted);
		font-size: 0.9rem;
	}

	.error {
		padding: var(--space-3);
		border-radius: var(--radius-md);
		background: var(--color-danger-soft);
	}

	.error p {
		margin: var(--space-1) 0 0;
		overflow-wrap: anywhere;
	}

	.units {
		margin: 0;
		padding-left: 1.2rem;
		display: flex;
		flex-direction: column;
		gap: var(--space-3);
	}

	.unit-text {
		margin: 0 0 var(--space-1);
		overflow-wrap: anywhere;
	}

	.tags {
		display: inline-flex;
		gap: var(--space-1);
		margin-left: var(--space-1);
	}

	.tag {
		display: inline-block;
		padding: 0 0.35rem;
		border-radius: var(--radius-sm);
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		font-size: 0.75rem;
		color: var(--color-text-muted);
	}

	.tag.uncertain {
		background: var(--color-amber-soft);
		border-color: var(--color-amber-border);
	}

	.qualification {
		margin: 0 0 var(--space-1);
		font-size: 0.85rem;
		color: var(--color-text-muted);
	}

	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-1);
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.chips li {
		padding: 0.15rem 0.5rem;
		border-radius: 999px;
		background: var(--color-surface);
		border: 1px solid var(--color-border);
		font-size: 0.85rem;
	}

	.segments {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-1);
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.segment {
		min-height: 44px;
		padding: 0 var(--space-3);
		border: 1px solid var(--color-sage-border);
		border-radius: var(--radius-sm);
		background: var(--color-sage-soft);
		color: inherit;
		font: inherit;
		font-size: 0.85rem;
		cursor: pointer;
	}

	.relations {
		flex-direction: row;
		align-items: center;
		flex-wrap: wrap;
	}

	.timing {
		margin: 0;
		font-size: 0.8rem;
		color: var(--color-text-muted);
	}
</style>
