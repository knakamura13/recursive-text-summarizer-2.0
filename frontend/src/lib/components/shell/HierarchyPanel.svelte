<script lang="ts">
	import type { HierarchyNode } from '$lib/stores/workspace';

	let {
		nodes = [],
		selectedId = null,
		onSelect
	}: {
		nodes?: HierarchyNode[];
		selectedId?: string | null;
		onSelect?: (nodeId: string) => void;
	} = $props();

	const roots = $derived(nodes.filter((node) => node.parent_id === null));
	const childrenByParent = $derived(
		nodes.reduce<Record<string, HierarchyNode[]>>((acc, node) => {
			if (!node.parent_id) return acc;
			(acc[node.parent_id] ||= []).push(node);
			return acc;
		}, {})
	);

	function statusSymbol(state: HierarchyNode['state']): string {
		if (state === 'completed') return '✓';
		if (state === 'active') return '●';
		return '○';
	}
</script>

<aside class="hierarchy-panel" aria-label="Document hierarchy">
	<div class="panel-header">
		<h2>Document Hierarchy</h2>
	</div>
	<ul class="tree" role="tree">
		{#each roots as root}
			<li role="treeitem" aria-expanded="true">
				<button
					type="button"
					class:selected={selectedId === root.node_id}
					onclick={() => onSelect?.(root.node_id)}
				>
					<span class="status">{statusSymbol(root.state)}</span>
					{root.label}
				</button>
				{#if childrenByParent[root.node_id]}
					<ul role="group">
						{#each childrenByParent[root.node_id] as child}
							<li role="treeitem">
								<button
									type="button"
									class:selected={selectedId === child.node_id}
									onclick={() => onSelect?.(child.node_id)}
								>
									<span class="status">{statusSymbol(child.state)}</span>
									{child.label}
								</button>
								{#if childrenByParent[child.node_id]}
									<ul role="group">
										{#each childrenByParent[child.node_id] as grandchild}
											<li role="treeitem">
												<button
													type="button"
													class:selected={selectedId === grandchild.node_id}
													onclick={() => onSelect?.(grandchild.node_id)}
												>
													<span class="status">{statusSymbol(grandchild.state)}</span>
													{grandchild.label}
												</button>
											</li>
										{/each}
									</ul>
								{/if}
							</li>
						{/each}
					</ul>
				{/if}
			</li>
		{/each}
	</ul>
</aside>

<style>
	.hierarchy-panel {
		border-left: 1px solid var(--color-border);
		background: var(--color-surface);
		min-width: var(--sidebar-right-width);
		padding: var(--space-5);
		overflow: auto;
	}

	.panel-header h2 {
		margin: 0 0 var(--space-4);
		font-size: 0.95rem;
	}

	.tree,
	.tree ul {
		list-style: none;
		margin: 0;
		padding: 0;
	}

	.tree ul {
		margin-left: var(--space-4);
	}

	.tree button {
		width: 100%;
		text-align: left;
		border: none;
		background: transparent;
		padding: var(--space-2) var(--space-3);
		border-radius: var(--radius-sm);
		cursor: pointer;
		display: flex;
		gap: var(--space-2);
		align-items: center;
	}

	.tree button.selected {
		background: var(--color-sage-soft);
		color: var(--color-sage);
	}

	.status {
		width: 1rem;
		text-align: center;
		font-size: 0.8rem;
	}
</style>
