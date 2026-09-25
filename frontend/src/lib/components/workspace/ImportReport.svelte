<script lang="ts">
	import type { ImportReport } from '$lib/api/types';
	import { formatCount, formatDuration, formatName } from '$lib/format';

	interface Props {
		report: ImportReport;
	}

	let { report }: Props = $props();

	/** 1–3, 7, 9–12 */
	function pageList(pages: readonly number[]): string {
		const sorted = [...pages].sort((a, b) => a - b);
		const parts: string[] = [];
		for (let i = 0; i < sorted.length; i++) {
			const start = sorted[i];
			while (i + 1 < sorted.length && sorted[i + 1] === sorted[i] + 1) i++;
			parts.push(start === sorted[i] ? `${start}` : `${start}–${sorted[i]}`);
		}
		return parts.join(', ');
	}

	const headline = $derived(
		[
			formatName(report.detected_format),
			report.page_count !== null ? `${formatCount(report.page_count)} pages` : null,
			report.ocr_pages.length > 0 ? `${formatCount(report.ocr_pages.length)} OCR'd` : null,
			report.blank_pages.length > 0 ? `${formatCount(report.blank_pages.length)} blank` : null,
			`${formatCount(report.word_count)} words`,
			report.notices.length > 0 ? `${report.notices.length} ${report.notices.length === 1 ? 'notice' : 'notices'}` : null
		]
			.filter(Boolean)
			.join(' · ')
	);
	const warnings = $derived(report.notices.filter((notice) => notice.severity !== 'info').length);
</script>

<details class="import-report" open={warnings > 0}>
	<summary>
		<span class="title">Import report</span>
		<span class="headline">{headline}</span>
	</summary>
	<div class="body">
		{#if report.notices.length > 0}
			<ul class="notices">
				{#each report.notices as notice (notice.code + notice.message)}
					<li class="notice severity-{notice.severity}">{notice.message}</li>
				{/each}
			</ul>
		{/if}
		<dl class="facts">
			<div><dt>Format</dt><dd>{formatName(report.detected_format)}</dd></div>
			{#if report.encoding}<div><dt>Encoding</dt><dd>{report.encoding}</dd></div>{/if}
			{#if report.page_count !== null}<div><dt>Pages</dt><dd>{formatCount(report.page_count)}</dd></div>{/if}
			{#if report.text_layer_pages !== null}
				<div><dt>Pages with a text layer</dt><dd>{formatCount(report.text_layer_pages)}</dd></div>
			{/if}
			{#if report.ocr_pages.length > 0}
				<div><dt>OCR'd pages</dt><dd>{pageList(report.ocr_pages)}</dd></div>
			{/if}
			{#if report.blank_pages.length > 0}
				<div><dt>Blank pages</dt><dd>{pageList(report.blank_pages)}</dd></div>
			{/if}
			<div><dt>Characters</dt><dd>{formatCount(report.char_count)}</dd></div>
			<div><dt>Words</dt><dd>{formatCount(report.word_count)}</dd></div>
			{#if report.duration_seconds !== null}
				<div><dt>Import time</dt><dd>{formatDuration(report.duration_seconds)}</dd></div>
			{/if}
		</dl>
		{#if report.preview}
			<h3 class="preview-title">Text preview</h3>
			<p class="preview">{report.preview}</p>
		{/if}
	</div>
</details>

<style>
	.import-report {
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
	}

	summary {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 0 var(--space-2);
		min-height: 44px;
		padding: var(--space-2) var(--space-3);
		cursor: pointer;
	}

	.title {
		font-weight: 600;
	}

	.headline {
		font-size: 0.85rem;
		color: var(--color-text-muted);
	}

	.body {
		display: flex;
		flex-direction: column;
		gap: var(--space-3);
		padding: 0 var(--space-3) var(--space-3);
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
		padding: var(--space-2);
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

	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(10rem, 1fr));
		gap: var(--space-2) var(--space-4);
		margin: 0;
	}

	.facts dt {
		font-size: 0.75rem;
		color: var(--color-text-muted);
	}

	.facts dd {
		margin: 0;
		overflow-wrap: anywhere;
	}

	.preview-title {
		margin: 0;
		font-size: 0.9rem;
	}

	.preview {
		margin: 0;
		max-height: 12rem;
		overflow: auto;
		padding: var(--space-2);
		background: var(--color-surface);
		border-radius: var(--radius-sm);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		font-size: 0.85rem;
	}
</style>
