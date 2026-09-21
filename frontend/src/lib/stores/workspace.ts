import type { DocumentSummary } from '$lib/api/client';

export type WorkspaceTab = 'document' | 'sections' | 'full';

export type HierarchyNode = {
	node_id: string;
	parent_id: string | null;
	level: number;
	order: number;
	label: string;
	provisional: boolean;
	state: 'pending' | 'active' | 'completed' | 'skipped';
};

export type ProgressStage = {
	stage: string;
	state: 'active' | 'completed' | 'skipped' | 'pending';
	completed?: number | null;
	total?: number | null;
};

export type WorkspaceFixture = {
	document: DocumentSummary & { page_count?: number };
	stages: ProgressStage[];
	selectedNodeId: string;
	hierarchy: HierarchyNode[];
	sourcePreview: string;
	summaryPreview: string;
	finalSummary: string;
};
