<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { MediaQuery } from 'svelte/reactivity';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import EmptyState from '$lib/components/common/EmptyState.svelte';
	import ErrorBanner from '$lib/components/common/ErrorBanner.svelte';
	import Sheet from '$lib/components/common/Sheet.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import Tabs, { tabId, tabPanelId, type TabItem } from '$lib/components/common/Tabs.svelte';
	import { api, errorMessage, isAbortError } from '$lib/api/client';
	import type { DocumentDetail, NodeDetail as NodeDetailData, Run, RunState, SourcePage } from '$lib/api/types';
	import { formatBytes, formatCount, formatName } from '$lib/format';
	import type { SourceFocus, TextRange } from '$lib/source/sourceModel';
	import { activity } from '$lib/stores/activity.svelte';
	import { documents } from '$lib/stores/documents.svelte';
	import { RunStream } from '$lib/stores/runStream.svelte';
	import { toasts } from '$lib/stores/toasts.svelte';
	import ImportPanel from './ImportPanel.svelte';
	import ImportReport from './ImportReport.svelte';
	import NodeDetail from './NodeDetail.svelte';
	import RunHistory from './RunHistory.svelte';
	import RunStatus from './RunStatus.svelte';
	import RunStrip from './RunStrip.svelte';
	import SourceViewer from './SourceViewer.svelte';
	import StartRunForm from './StartRunForm.svelte';
	import SummaryView from './SummaryView.svelte';
	import TreeView from './TreeView.svelte';

	let { documentId }: { documentId: string } = $props();

	type Tab = 'summary' | 'status' | 'tree' | 'source' | 'runs';
	const TAB_LABELS: Record<Tab, string> = {
		summary: 'Summary',
		status: 'Status',
		tree: 'Tree',
		source: 'Source',
		runs: 'Runs'
	};

	// Layout: phone < 640 px (tab bar + node sheet), tablet two panes, desktop >= 1024 px.
	const wide = new MediaQuery('min-width: 640px');
	const desktop = new MediaQuery('min-width: 1024px');
	const phone = $derived(!wide.current);

	// --- URL state: ?run=<id|new>&tab=<tab>&node=<id> --------------------------------------
	const query = $derived(page.url.searchParams);
	const runParam = $derived(query.get('run'));
	const nodeParam = $derived(query.get('node'));
	const tabParam = $derived.by((): Tab | null => {
		const tab = query.get('tab');
		return tab !== null && Object.hasOwn(TAB_LABELS, tab) ? (tab as Tab) : null;
	});

	function setQuery(patch: Partial<Record<'run' | 'tab' | 'node', string | null>>, push = false) {
		const url = new URL(page.url);
		for (const [key, value] of Object.entries(patch)) {
			if (value === null || value === undefined) url.searchParams.delete(key);
			else url.searchParams.set(key, value);
		}
		if (url.search === page.url.search) return;
		void goto(url, { replaceState: !push, keepFocus: true, noScroll: true });
	}

	// --- Document --------------------------------------------------------------------------
	let doc = $state.raw<DocumentDetail | null>(null);
	let docError = $state<unknown>(null);
	let pages = $state.raw<SourcePage[]>([]);

	async function loadDocument() {
		try {
			const next = await api.getDocument(documentId);
			if (doc && doc.import_state !== next.import_state) documents.upsert(next);
			doc = next;
			docError = null;
		} catch (error) {
			if (!isAbortError(error)) docError = error;
		}
	}

	// Import progress: poll while importing; the Document opens when its text is ready.
	$effect(() => {
		if (doc?.import_state !== 'importing') return;
		const timer = setInterval(() => void loadDocument(), 1000);
		return () => clearInterval(timer);
	});

	$effect(() => {
		if (doc?.import_state !== 'ready') return;
		const controller = new AbortController();
		api.getPages(documentId, controller.signal).then(
			(result) => (pages = result),
			(error: unknown) => {
				if (!isAbortError(error)) toasts.error(`Page positions are unavailable: ${errorMessage(error)}`);
			}
		);
		return () => controller.abort();
	});

	// --- Runs --------------------------------------------------------------------------------
	let runs = $state.raw<Run[] | null>(null);
	let runsError = $state<unknown>(null);

	async function loadRuns() {
		try {
			runs = await api.listRuns(documentId);
			runsError = null;
		} catch (error) {
			runsError = error;
		}
	}

	const startMode = $derived(runParam === 'new' || (runParam === null && runs !== null && runs.length === 0));
	const selectedRunId = $derived(runParam === 'new' ? null : (runParam ?? runs?.[0]?.run_id ?? null));

	let stream = $state.raw<RunStream | null>(null);
	let stopPending = $state(false);
	let resumingRunId = $state<string | null>(null);

	function onRunState(_run: Run, previous: RunState | null) {
		// "Stopping…" now comes from the Run's own state (or the Stop ended it at once).
		if (previous !== null) stopPending = false;
	}

	$effect(() => {
		const id = selectedRunId;
		if (id === null) {
			stream = null;
			return;
		}
		const created = new RunStream(id, { onStateChange: onRunState });
		stream = created;
		stopPending = false;
		return () => created.destroy();
	});

	/** The Runs list with the selected Run's live state. */
	const runList = $derived(
		runs?.map((run) => (stream?.run && run.run_id === stream.run.run_id ? stream.run : run)) ?? null
	);
	const blockedBy = $derived(activity.activeRun);

	onMount(() => {
		void loadDocument();
		void loadRuns();
		// Runs of this Document that start, stop, or end (also from other tabs) refresh the list.
		return activity.onRun((run) => {
			if (run.document_id === documentId) void loadRuns();
		});
	});

	function onStarted(run: Run) {
		runs = [run, ...(runs ?? []).filter((item) => item.run_id !== run.run_id)];
		setQuery({ run: run.run_id, tab: 'status', node: null }, true);
	}

	async function stopRun() {
		const current = stream;
		const run = current?.run;
		if (!current || !run) return;
		stopPending = true;
		try {
			current.applyRun(await api.stopRun(run.run_id));
		} catch (error) {
			stopPending = false;
			toasts.error(`Could not stop the Run: ${errorMessage(error)}`);
		}
	}

	async function resumeRun(target: Run) {
		resumingRunId = target.run_id;
		try {
			const resumed = await api.resumeRun(target.run_id);
			if (stream?.runId === resumed.run_id) stream.applyRun(resumed);
			else setQuery({ run: resumed.run_id, tab: 'status', node: null }, true);
			void loadRuns();
		} catch (error) {
			toasts.error(`Could not resume the Run: ${errorMessage(error)}`);
		} finally {
			resumingRunId = null;
		}
	}

	async function deleteRun(target: Run) {
		await api.deleteRun(target.run_id);
		runs = runs?.filter((run) => run.run_id !== target.run_id) ?? null;
		if (runParam === target.run_id) setQuery({ run: null, node: null });
		toasts.success('Run deleted.');
		void loadRuns();
		void documents.refresh();
	}

	// --- Tabs ----------------------------------------------------------------------------------
	const defaultTab = $derived<Tab>(stream?.run?.state === 'completed' ? 'summary' : 'status');
	const currentTab = $derived.by((): Tab => {
		const tab = tabParam ?? defaultTab;
		// Wider layouts show the Source in the side pane, not as a tab.
		return !phone && tab === 'source' ? defaultTab : tab;
	});
	const failedNodes = $derived.by(() => {
		let count = 0;
		for (const node of stream?.nodes.values() ?? []) if (node.state === 'failed') count += 1;
		return count;
	});
	const tabItems = $derived.by((): TabItem[] => {
		const order: Tab[] = phone ? ['summary', 'status', 'tree', 'source', 'runs'] : ['summary', 'status', 'tree', 'runs'];
		return order.map((id) => ({
			id,
			label: TAB_LABELS[id],
			badge:
				id === 'tree' && failedNodes > 0
					? `${failedNodes} failed`
					: id === 'runs' && runs && runs.length > 0
						? runs.length
						: null
		}));
	});

	// --- Node selection and detail -----------------------------------------------------------
	const selectedNode = $derived(nodeParam !== null ? stream?.nodes.get(nodeParam) : undefined);
	// Refetch the detail when the live row changes (e.g. the node completes while open).
	const nodeVersion = $derived(
		selectedNode ? `${selectedNode.state}|${selectedNode.label}|${selectedNode.duration_seconds}|${selectedNode.child_count}` : ''
	);
	let nodeDetail = $state.raw<NodeDetailData | null>(null);
	let nodeError = $state<unknown>(null);
	let nodeLoading = $state(false);
	let nodeReload = $state(0);
	let spanShownFor: string | null = null;

	$effect(() => {
		const runId = stream?.runId;
		const nodeId = nodeParam;
		void nodeVersion;
		void nodeReload;
		if (!runId || nodeId === null) {
			nodeDetail = null;
			nodeError = null;
			nodeLoading = false;
			spanShownFor = null;
			return;
		}
		if (untrack(() => nodeDetail?.node_id) !== nodeId) nodeDetail = null;
		nodeLoading = true;
		nodeError = null;
		const controller = new AbortController();
		api.getNode(runId, nodeId, controller.signal).then(
			(detail) => {
				nodeDetail = detail;
				nodeLoading = false;
				showCoveredSpan(detail);
			},
			(error: unknown) => {
				if (isAbortError(error)) return;
				nodeError = error;
				nodeLoading = false;
			}
		);
		return () => controller.abort();
	});

	function selectNode(nodeId: string) {
		setQuery(phone ? { node: nodeId, tab: 'tree' } : { node: nodeId });
	}

	// --- Source focus --------------------------------------------------------------------------
	let sourceFocus = $state.raw<SourceFocus | null>(null);
	let focusKey = 0;
	/** Where the Source was scrolled, restored when the viewer remounts (phone tab switches). */
	const reading = { offset: 0 };

	function showInSource(range: TextRange, label: string, tone: SourceFocus['tone'], openSource = true) {
		focusKey += 1;
		sourceFocus = { range, label, tone, key: focusKey };
		if (phone && openSource) setQuery({ tab: 'source' });
	}

	/** Selecting a node tints the text it covers (once per node, without leaving the tree). */
	function showCoveredSpan(detail: NodeDetailData) {
		if (spanShownFor === detail.node_id || detail.covered_segments.length === 0) return;
		spanShownFor = detail.node_id;
		let start = Infinity;
		let end = -Infinity;
		for (const segment of detail.covered_segments) {
			start = Math.min(start, segment.core_start);
			end = Math.max(end, segment.core_end);
		}
		showInSource({ start, end }, `Covered by ${detail.label}`, 'span', false);
	}

	const docMeta = $derived(
		doc
			? [
					formatName(doc.format),
					formatBytes(doc.size_bytes),
					doc.page_count !== null ? `${formatCount(doc.page_count)} pages` : null,
					doc.char_count !== null ? `${formatCount(doc.char_count)} characters` : null
				]
					.filter(Boolean)
					.join(' · ')
			: ''
	);
</script>

<svelte:head>
	<title>{doc ? `${doc.title} · Summarizer` : 'Document · Summarizer'}</title>
</svelte:head>

{#snippet emptyRun(title: string, message: string)}
	<EmptyState {title} {message} headingLevel={3} />
{/snippet}

{#snippet sourceViewer()}
	{#if doc}
		<SourceViewer
			documentId={doc.document_id}
			totalLength={doc.char_count}
			{pages}
			focus={sourceFocus}
			onclearfocus={() => (sourceFocus = null)}
			initialOffset={reading.offset}
			onposition={(offset) => (reading.offset = offset)}
		/>
	{/if}
{/snippet}

{#snippet nodeDetailView()}
	<NodeDetail
		node={selectedNode}
		detail={nodeDetail}
		loading={nodeLoading}
		error={nodeError}
		onretry={() => (nodeReload += 1)}
		onshow={(range, label, tone) => showInSource(range, label, tone)}
		onselectnode={selectNode}
	/>
{/snippet}

<div class="workspace" class:phone class:desktop={desktop.current} data-layout={phone ? 'phone' : desktop.current ? 'desktop' : 'tablet'}>
	{#if !doc}
		<div class="center">
			{#if docError}
				<ErrorBanner error={docError} title="Could not open this Document" onretry={() => void loadDocument()} />
				<a class="button" href="/">Back to the library</a>
			{:else}
				<Spinner label="Loading the Document" />
			{/if}
		</div>
	{:else}
		<header class="doc-header">
			<div class="title-block">
				<h1 class="doc-title">{doc.title}</h1>
				<p class="doc-meta">
					{docMeta} · <a href={api.originalUrl(doc.document_id)} download>Original file</a>
				</p>
			</div>
			{#if doc.import_state === 'ready' && !startMode && stream?.run}
				<RunStrip
					run={stream.run}
					progress={stream.progress}
					progressAt={stream.progressAt}
					{blockedBy}
					{stopPending}
					resumePending={resumingRunId === stream.run.run_id}
					onstop={stopRun}
					onresume={() => stream?.run && resumeRun(stream.run)}
					onnew={() => setQuery({ run: 'new', tab: 'status', node: null }, true)}
				/>
			{/if}
		</header>

		{#if doc.import_state !== 'ready'}
			<div class="import-area">
				<ImportPanel {doc} ondeleted={() => void goto('/')} />
			</div>
		{:else}
			<div class="panes">
				<div class="main-pane">
					<div class="tabbar">
						<Tabs
							items={tabItems}
							bind:active={() => currentTab, (id) => setQuery({ tab: id })}
							label="Document views"
							idPrefix="doc"
							fill={phone}
						/>
					</div>
					<div
						class="panel"
						class:fill={currentTab === 'tree' || currentTab === 'source'}
						role="tabpanel"
						id={tabPanelId('doc', currentTab)}
						aria-labelledby={tabId('doc', currentTab)}
					>
						{#if currentTab === 'status'}
							<div class="stack">
								{#if startMode}
									<StartRunForm
										{doc}
										onstarted={onStarted}
										oncancel={runs && runs.length > 0 ? () => setQuery({ run: null }) : undefined}
									/>
								{:else if stream?.error}
									<ErrorBanner error={stream.error} title="Could not load this Run" onretry={() => stream?.reload()} />
									<button type="button" class="button" onclick={() => setQuery({ run: null, node: null })}>
										Show the latest Run
									</button>
				{:else if stream?.run}
					<RunStatus run={stream.run} progress={stream.progress} progressAt={stream.progressAt} />
				{:else if runsError && selectedRunId === null}
					<ErrorBanner error={runsError} title="Could not load the Run history" onretry={() => void loadRuns()} />
				{:else}
					<Spinner label="Loading the Run" />
								{/if}
								{#if doc.import_report}
									<ImportReport report={doc.import_report} />
								{/if}
							</div>
						{:else if currentTab === 'summary'}
							<div class="stack">
								{#if stream?.run}
									<SummaryView
										run={stream.run}
										summary={stream.summary}
										loading={stream.summaryLoading}
										error={stream.summaryError}
										onretry={() => stream?.loadSummary()}
										onshow={(range, label) => showInSource(range, label, 'quote')}
										showOnSelect={!phone}
									/>
								{:else if startMode}
									{@render emptyRun('No summary yet', 'Start a Run from the Status tab to summarize this Document.')}
								{:else}
									<Spinner label="Loading the Run" />
								{/if}
							</div>
						{:else if currentTab === 'tree'}
							{#if stream && stream.nodes.size > 0}
								<TreeView nodes={stream.nodes} children={stream.children} selected={nodeParam} onselect={selectNode} />
							{:else}
								<div class="stack">
									{#if startMode}
										{@render emptyRun('No tree yet', 'Start a Run to watch its summary tree being built.')}
									{:else if stream?.loading}
										<Spinner label="Loading the tree" />
									{:else}
										{@render emptyRun(
											'No nodes yet',
											'Segments appear here as soon as the Run has split the Document.'
										)}
									{/if}
								</div>
							{/if}
						{:else if currentTab === 'source'}
							{@render sourceViewer()}
						{:else}
							<div class="stack">
								<RunHistory
									runs={runList}
									error={runsError}
									selectedRunId={startMode ? null : selectedRunId}
									activeRun={activity.activeRun}
									{resumingRunId}
									onretry={() => void loadRuns()}
									onopen={(runId) => setQuery({ run: runId, node: null, tab: null }, true)}
									onresume={resumeRun}
									ondelete={deleteRun}
									onnew={() => setQuery({ run: 'new', tab: 'status', node: null }, true)}
								/>
							</div>
						{/if}
					</div>
				</div>

				{#if !phone}
					<aside class="side-pane" aria-label="Details and source">
						{#if nodeParam !== null && stream}
							<div class="node-pane">
								<div class="node-pane-bar">
									<span class="node-pane-title">Node detail</span>
									<button type="button" class="button small ghost" onclick={() => setQuery({ node: null })}>
										Close
									</button>
								</div>
								{@render nodeDetailView()}
							</div>
						{/if}
						<div class="source-pane">
							{@render sourceViewer()}
						</div>
					</aside>
				{/if}
			</div>

			{#if phone && stream}
				<Sheet
					open={nodeParam !== null && currentTab === 'tree'}
					title={selectedNode?.label ?? 'Node detail'}
					onclose={() => setQuery({ node: null })}
				>
					{@render nodeDetailView()}
				</Sheet>
			{/if}
		{/if}
	{/if}
</div>

<style>
	.workspace {
		flex: 1 1 auto;
		min-height: 0;
		height: 100%;
		min-width: 0;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.center {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--space-3);
		padding: var(--space-6) var(--space-4);
	}

	.doc-header {
		flex: none;
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		justify-content: space-between;
		gap: var(--space-2) var(--space-4);
		padding: var(--space-3) var(--space-4);
		border-bottom: 1px solid var(--color-border);
		min-width: 0;
	}

	.title-block {
		min-width: 0;
		flex: 1 1 16rem;
	}

	.doc-title {
		margin: 0;
		font-size: 1.15rem;
		line-height: 1.3;
		overflow-wrap: anywhere;
	}

	.doc-meta {
		margin: var(--space-1) 0 0;
		font-size: 0.8rem;
		color: var(--color-text-muted);
		overflow-wrap: anywhere;
	}

	.doc-header :global([data-testid='run-strip']) {
		flex: 1 1 18rem;
	}

	.import-area {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
	}

	.panes {
		flex: 1;
		min-height: 0;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
	}

	.main-pane {
		display: flex;
		flex-direction: column;
		min-height: 0;
		min-width: 0;
	}

	.tabbar {
		flex: none;
		min-width: 0;
		border-bottom: 1px solid var(--color-border);
	}

	.panel {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		overscroll-behavior: contain;
	}

	.panel.fill {
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}

	.stack {
		display: flex;
		flex-direction: column;
		gap: var(--space-4);
		padding: var(--space-4);
		max-width: 60rem;
	}

	/* Phone: the tab bar sits at the bottom, within thumb reach. */
	.phone .tabbar {
		order: 2;
		border-bottom: none;
		border-top: 1px solid var(--color-border);
		padding-bottom: env(safe-area-inset-bottom, 0);
		background: var(--color-bg);
	}

	.phone .doc-header {
		padding: var(--space-2) var(--space-3);
	}

	.phone .doc-title {
		font-size: 1rem;
	}

	.phone .stack {
		padding: var(--space-3);
	}

	@media (min-width: 640px) {
		.panes {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		}
	}

	@media (min-width: 1024px) {
		.panes {
			grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
		}
	}

	.side-pane {
		display: flex;
		flex-direction: column;
		min-height: 0;
		min-width: 0;
		border-left: 1px solid var(--color-border);
	}

	.node-pane {
		flex: 0 1 auto;
		max-height: 55%;
		overflow-y: auto;
		overscroll-behavior: contain;
		padding: 0 var(--space-4) var(--space-4);
		border-bottom: 1px solid var(--color-border-strong);
	}

	.node-pane-bar {
		position: sticky;
		top: 0;
		z-index: 1;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-2);
		margin: 0 calc(-1 * var(--space-4)) var(--space-2);
		padding: var(--space-1) var(--space-4);
		background: var(--color-bg);
		border-bottom: 1px solid var(--color-border);
	}

	.node-pane-title {
		font-size: 0.8rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--color-text-muted);
	}

	.source-pane {
		flex: 1 1 0;
		min-height: 12rem;
		display: flex;
		flex-direction: column;
	}
</style>
