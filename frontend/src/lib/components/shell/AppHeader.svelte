<script lang="ts">
	import { page } from '$app/stores';

	const tabs = [
		{ href: '/', label: 'Summary', match: (path: string) => path.startsWith('/documents') || path === '/' },
		{ href: '/sources', label: 'Sources', match: (path: string) => path.includes('/sources') },
		{ href: '/settings', label: 'Settings', match: (path: string) => path.startsWith('/settings') }
	];
</script>

<header class="app-header">
	<div class="brand">
		<h1>Recursive Summarizer</h1>
	</div>
	<nav aria-label="Primary">
		<ul>
			{#each tabs as tab}
				<li>
					<a
						href={tab.href}
						aria-current={tab.match($page.url.pathname) ? 'page' : undefined}
						class:selected={tab.match($page.url.pathname)}
					>
						{tab.label}
					</a>
				</li>
			{/each}
		</ul>
	</nav>
</header>

<style>
	.app-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: var(--space-4) var(--space-6);
		border-bottom: 1px solid var(--color-border);
		background: var(--color-bg);
	}

	.brand h1 {
		margin: 0;
		font-size: 1.125rem;
		font-weight: 600;
		color: var(--color-text);
	}

	nav ul {
		display: flex;
		gap: var(--space-2);
		list-style: none;
		margin: 0;
		padding: 0;
	}

	nav a {
		display: inline-block;
		padding: var(--space-2) var(--space-4);
		border-radius: var(--radius-sm);
		color: var(--color-text-muted);
		text-decoration: none;
		font-weight: 500;
	}

	nav a.selected,
	nav a[aria-current='page'] {
		background: var(--color-sage-soft);
		color: var(--color-sage);
	}
</style>
