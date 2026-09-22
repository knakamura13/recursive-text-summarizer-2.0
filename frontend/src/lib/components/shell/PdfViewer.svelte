<script lang="ts">
	import { onMount } from 'svelte';

	let {
		url,
		page = 1
	}: {
		url: string;
		page?: number;
	} = $props();

	let canvas: HTMLCanvasElement | undefined = $state();
	let currentPage = $state(page);
	let totalPages = $state(0);
	let scale = $state(1.2);

	onMount(async () => {
		const pdfjs = await import('pdfjs-dist');
		pdfjs.GlobalWorkerOptions.workerSrc = '/pdfjs/pdf.worker.min.mjs';
		const loadingTask = pdfjs.getDocument(url);
		const pdf = await loadingTask.promise;
		totalPages = pdf.numPages;
		await renderPage(pdf, currentPage);
	});

	async function renderPage(pdf: import('pdfjs-dist').PDFDocumentProxy, pageNumber: number) {
		if (!canvas) return;
		const pdfPage = await pdf.getPage(pageNumber);
		const viewport = pdfPage.getViewport({ scale });
		const context = canvas.getContext('2d');
		if (!context) return;
		canvas.height = viewport.height;
		canvas.width = viewport.width;
		await pdfPage.render({ canvasContext: context, viewport, canvas }).promise;
	}
</script>

<div class="pdf-viewer" aria-label="PDF viewer">
	<div class="toolbar">
		<button
			type="button"
			aria-label="Previous page"
			disabled={currentPage <= 1}
			onclick={() => (currentPage -= 1)}
		>
			Prev
		</button>
		<span>Page {currentPage} of {totalPages || '…'}</span>
		<button
			type="button"
			aria-label="Next page"
			disabled={totalPages > 0 && currentPage >= totalPages}
			onclick={() => (currentPage += 1)}
		>
			Next
		</button>
		<label>
			Zoom
			<input type="range" min="0.8" max="2" step="0.1" bind:value={scale} />
		</label>
	</div>
	<canvas bind:this={canvas}></canvas>
</div>

<style>
	.pdf-viewer {
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		padding: var(--space-3);
		overflow: auto;
	}

	.toolbar {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		margin-bottom: var(--space-3);
	}
</style>
