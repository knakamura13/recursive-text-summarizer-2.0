// Mirrors summarizer_web/models/api.py. Offsets into document text are Python
// code-point offsets; convert with $lib/unicodeOffsets before slicing JS strings.
// Timestamps are ISO 8601 UTC strings.

export interface ErrorBody {
	code: string;
	message: string;
	retryable: boolean;
	details: Record<string, unknown> | null;
}

export interface Notice {
	code: string;
	message: string;
	severity: 'info' | 'warning' | 'error';
}

// --- Ollama ---------------------------------------------------------------

export interface OllamaHealth {
	connected: boolean;
	message: string;
	host: string;
	version: string | null;
}

export interface OllamaModel {
	name: string;
	size_bytes: number | null;
	parameter_size: string | null;
	family: string | null;
	quantization: string | null;
	modified_at: string | null;
}

// --- Run configuration and settings ---------------------------------------

export type StrategyName = 'auto' | 'direct' | 'hierarchical';
export type SelectedStrategy = 'direct' | 'hierarchical';

export interface RunConfig {
	model: string;
	target_words: number;
	strategy: StrategyName;
	verify: boolean;
	max_repair_passes: number;
	citations: boolean;
	context_window: number | null;
	max_output_tokens: number;
	safety_margin_tokens: number;
	safety_margin_fraction: number;
	chunk_tokens: number | null;
	overlap_tokens: number;
	max_merge_children: number | null;
	max_concurrency: number;
	timeout_seconds: number;
	max_retries: number;
	strict_numbers: boolean;
	strict_names: boolean;
}

export type ClearableConfigField = 'context_window' | 'chunk_tokens' | 'max_merge_children';

export type RunConfigPatch = Partial<RunConfig> & { clear?: ClearableConfigField[] };

export interface Settings {
	ollama_host: string;
	defaults: RunConfig;
}

export interface SettingsUpdate {
	ollama_host?: string;
	defaults?: RunConfigPatch;
}

// --- Documents -------------------------------------------------------------

export type DocumentFormat =
	| 'txt'
	| 'md'
	| 'pdf'
	| 'docx'
	| 'odt'
	| 'rtf'
	| 'html'
	| 'epub'
	| 'srt'
	| 'vtt'
	| 'png'
	| 'jpeg'
	| 'tiff';

export type ImportState = 'importing' | 'ready' | 'failed';
export type ImportPhase = 'uploading' | 'queued' | 'reading' | 'extracting' | 'ocr' | 'finalizing';

export type RunState =
	| 'queued'
	| 'running'
	| 'stopping'
	| 'stopped'
	| 'failed'
	| 'interrupted'
	| 'completed';

export interface ImportProgress {
	phase: ImportPhase;
	done: number;
	total: number | null;
	unit: 'pages' | 'bytes' | null;
	message: string | null;
}

export interface ImportReport {
	detected_format: DocumentFormat;
	encoding: string | null;
	page_count: number | null;
	text_layer_pages: number | null;
	ocr_pages: number[];
	blank_pages: number[];
	char_count: number;
	word_count: number;
	notices: Notice[];
	preview: string;
	extraction_version: string;
	duration_seconds: number | null;
}

export interface RunBrief {
	run_id: string;
	state: RunState;
	created_at: string;
	updated_at: string;
}

export interface DocumentSummary {
	document_id: string;
	title: string;
	filename: string;
	format: DocumentFormat;
	origin: 'upload' | 'paste';
	size_bytes: number;
	import_state: ImportState;
	import_progress: ImportProgress | null;
	import_error: string | null;
	char_count: number | null;
	page_count: number | null;
	created_at: string;
	updated_at: string;
	latest_run: RunBrief | null;
}

export interface DocumentDetail extends DocumentSummary {
	source_sha256: string | null;
	import_report: ImportReport | null;
}

export interface DocumentCreated {
	document: DocumentSummary;
	already_imported: boolean;
}

export interface SourceSlice {
	text: string;
	offset: number;
	total_length: number;
	has_more: boolean;
}

export interface SourcePage {
	page: number;
	start: number;
	end: number;
	ocr: boolean;
	blank: boolean;
}

// --- Preflight and runs ----------------------------------------------------

export interface Preflight {
	ok: boolean;
	selected_strategy: SelectedStrategy | null;
	context_window_tokens: number | null;
	context_window_source: 'configured' | 'model' | 'assumed' | null;
	usable_input_capacity: number | null;
	document_tokens: number | null;
	estimated_leaf_count: number | null;
	estimated_model_calls: number | null;
	model_installed: boolean | null;
	errors: Notice[];
	warnings: Notice[];
}

export interface RunFailure {
	code: string;
	message: string;
	stage: string | null;
	item: string | null;
	detail: string | null;
	hint: string | null;
}

export interface AttemptInfo {
	attempt_id: string;
	attempt_number: number;
	state: RunState;
	started_at: string | null;
	ended_at: string | null;
	failure: RunFailure | null;
}

export type StageName =
	| 'preparing'
	| 'segmenting'
	| 'summarizing'
	| 'merging'
	| 'writing'
	| 'verifying'
	| 'publishing';

export interface StageProgress {
	stage: StageName;
	state: 'pending' | 'active' | 'completed' | 'skipped';
	completed: number | null;
	total: number | null;
	detail: string | null;
}

export interface CountProgress {
	done: number;
	total: number | null;
	reused: number;
	failed: number;
}

export interface LevelProgress {
	level: number;
	done: number;
	total: number;
}

export interface CurrentItem {
	kind: string;
	work_id: string;
	label: string;
	started_at: string | null;
	attempt: number | null;
}

export interface RunProgress {
	cursor: number;
	attempt_number: number | null;
	started_at: string | null;
	elapsed_seconds: number | null;
	stage: StageName | null;
	stages: StageProgress[];
	leaves: CountProgress;
	merges: CountProgress;
	merge_levels: LevelProgress[];
	claims: CountProgress;
	current_items: CurrentItem[];
	eta_seconds: number | null;
	eta_basis: string | null;
}

export interface Run {
	run_id: string;
	document_id: string;
	document_title: string;
	state: RunState;
	requested_strategy: StrategyName;
	selected_strategy: SelectedStrategy | null;
	config: RunConfig;
	created_at: string;
	updated_at: string;
	attempt: AttemptInfo | null;
	attempt_count: number;
	failure: RunFailure | null;
	can_stop: boolean;
	can_resume: boolean;
	progress: RunProgress | null;
}

export interface Activity {
	active_run: Run | null;
	importing: DocumentSummary[];
	cursor: number;
}

// --- Tree and nodes --------------------------------------------------------

export type NodeKind = 'leaf' | 'merge' | 'passthrough';
export type NodeState = 'pending' | 'active' | 'completed' | 'failed';

export interface TreeNode {
	node_id: string;
	parent_id: string | null;
	level: number;
	order: number;
	kind: NodeKind;
	label: string;
	state: NodeState;
	child_count: number;
	page_start: number | null;
	page_end: number | null;
	duration_seconds: number | null;
}

export interface Tree {
	nodes: TreeNode[];
	cursor: number;
}

export interface EvidenceRef {
	segment_id: string;
	quote: string | null;
	quote_found: boolean;
	start: number | null;
	end: number | null;
	page_start: number | null;
	page_end: number | null;
}

export interface NodeContentUnit {
	text: string;
	kind: string;
	uncertain: boolean;
	qualification: string | null;
	evidence: EvidenceRef[];
}

export interface NodeAnnotation {
	kind: 'qualification' | 'contradiction';
	text: string;
	evidence: EvidenceRef[];
}

export interface SegmentRef {
	segment_id: string;
	order: number;
	start: number;
	end: number;
	core_start: number;
	core_end: number;
	page_start: number | null;
	page_end: number | null;
}

export interface NodeDetail {
	node_id: string;
	parent_id: string | null;
	level: number;
	order: number;
	kind: NodeKind;
	label: string;
	state: NodeState;
	summary_text: string | null;
	content_units: NodeContentUnit[];
	annotations: NodeAnnotation[];
	entities: string[];
	quotations: EvidenceRef[];
	covered_segments: SegmentRef[];
	child_ids: string[];
	started_at: string | null;
	completed_at: string | null;
	duration_seconds: number | null;
	error: string | null;
}

// --- Final summary -----------------------------------------------------------

export interface SummaryCitation {
	citation_id: string;
	segment_id: string;
	start: number | null;
	end: number | null;
	page_start: number | null;
	page_end: number | null;
}

export interface SummarySentence {
	index: number;
	paragraph: number;
	text: string;
	verdict: 'supported' | 'not_meaningfully_verifiable' | 'unchecked';
	evidence: EvidenceRef[];
}

export interface RemovedSentence {
	text: string;
	verdict: string;
	reason: string | null;
}

export interface FinalSummary {
	available: boolean;
	text: string | null;
	sentences: SummarySentence[];
	removed_sentences: RemovedSentence[];
	citations: SummaryCitation[];
	word_count: number | null;
	target_words: number | null;
	short_of_target: boolean;
	notices: Notice[];
	verification_state: 'not_run' | 'in_progress' | 'completed' | 'failed';
	publication: 'editorial' | 'verified_subset' | 'content_unit_fallback' | null;
}

// --- Server-sent events ------------------------------------------------------

// One EventSource per browser tab (HTTP/1.1 allows ~6 connections per host
// across all tabs): GET /api/v1/activity/stream?after=<cursor>
//   event "activity" data: Activity (active_run includes progress)
//   event "nodes"    data: NodesEvent, changed nodes of the active Run
//   event "run"      data: Run (progress: null) whenever a Run changes state,
//                    including the active Run reaching a terminal state
// Close it while document.hidden; reconnect with the last cursor. Runs that
// are not active never change, so fetch their tree and summary once.
export interface NodesEvent {
	run_id: string;
	nodes: TreeNode[];
	cursor: number;
}

export type ExportFormat = 'txt' | 'md' | 'json';
