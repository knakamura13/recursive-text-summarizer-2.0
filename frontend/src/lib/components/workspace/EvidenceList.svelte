<script lang="ts">
	import type { EvidenceRef } from '$lib/api/types';
	import type { TextRange } from '$lib/source/sourceModel';
	import { evidenceLabel } from './runDisplay';

	interface Props {
		evidence: readonly EvidenceRef[];
		/** Highlight a cited passage in the Source. */
		onshow: (range: TextRange, label: string) => void;
	}

	let { evidence, onshow }: Props = $props();
</script>

{#if evidence.length > 0}
	<ul class="evidence">
		{#each evidence as ref, i (i)}
			{@const where = evidenceLabel(ref.segment_id, ref.page_start, ref.page_end)}
			<li>
				{#if ref.start !== null && ref.end !== null}
					{@const range = { start: ref.start, end: ref.end }}
					<button type="button" class="evidence-item" onclick={() => onshow(range, `Evidence · ${where}`)}>
						{#if ref.quote}<q>{ref.quote}</q>{:else}<span>Cited passage</span>{/if}
						<span class="where">
							{where}{#if ref.quote && !ref.quote_found} · quote not found word for word; showing its segment{/if}
						</span>
					</button>
				{:else}
					<div class="evidence-item static">
						{#if ref.quote}<q>{ref.quote}</q>{/if}
						<span class="where">{where}</span>
					</div>
				{/if}
			</li>
		{/each}
	</ul>
{/if}

<style>
	.evidence {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
	}

	.evidence-item {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		width: 100%;
		min-height: 44px;
		padding: var(--space-2) var(--space-3);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-sm);
		background: var(--color-bg);
		color: inherit;
		font: inherit;
		font-size: 0.9rem;
		text-align: left;
	}

	button.evidence-item {
		cursor: pointer;
	}

	button.evidence-item:hover {
		border-color: var(--color-amber);
		background: var(--color-amber-soft);
	}

	.evidence-item.static {
		background: var(--color-surface);
		border-color: var(--color-border);
	}

	q {
		font-family: var(--font-serif);
		font-size: 1rem;
		line-height: 1.5;
		overflow-wrap: anywhere;
	}

	.where {
		font-size: 0.8rem;
		color: var(--color-text-muted);
	}
</style>
