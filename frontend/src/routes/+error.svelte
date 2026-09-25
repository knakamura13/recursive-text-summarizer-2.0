<script lang="ts">
	import { page } from '$app/state';

	const notFound = $derived(page.status === 404);
</script>

<svelte:head>
	<title>{notFound ? 'Page not found' : 'Error'} · Recursive Summarizer</title>
</svelte:head>

<section class="error-page">
	<h1>{notFound ? 'Page not found' : 'Something went wrong'}</h1>
	<p>
		{notFound
			? 'There is nothing at this address.'
			: (page.error?.message ?? 'The page could not be shown.')}
	</p>
	<p class="status">Error {page.status}</p>
	<a class="button primary" href="/">Go to the Library</a>
</section>

<style>
	.error-page {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--space-3);
		max-width: 40rem;
		margin: 0 auto;
		padding: var(--space-8) var(--space-4);
	}

	.status {
		color: var(--color-text-muted);
		font-size: 0.875rem;
	}
</style>
