// Display formatting shared by every page. English UI; output does not depend
// on the browser locale so layouts and tests stay stable.
import type {
	DocumentFormat,
	ImportPhase,
	ImportProgress,
	ImportState,
	RunState,
	StageName
} from '$lib/api/types';

const COUNT = new Intl.NumberFormat('en-US');
const BYTE_UNITS = ['KB', 'MB', 'GB', 'TB'];

/** 512 B, 1.5 KB, 12 KB, 500 MB (binary multiples). */
export function formatBytes(bytes: number): string {
	if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
	if (bytes < 1024) return `${Math.round(bytes)} B`;
	let value = bytes / 1024;
	let unit = 0;
	while (value >= 1024 && unit < BYTE_UNITS.length - 1) {
		value /= 1024;
		unit += 1;
	}
	const rounded = value < 10 ? Math.round(value * 10) / 10 : Math.round(value);
	// 1023.96 KB rounds to 1024 KB; show it as the next unit instead.
	if (rounded >= 1024 && unit < BYTE_UNITS.length - 1) return `1 ${BYTE_UNITS[unit + 1]}`;
	return `${rounded} ${BYTE_UNITS[unit]}`;
}

/** 2,100,000 */
export function formatCount(value: number): string {
	return COUNT.format(value);
}

/** 42 s, 3 min 5 s, 2 h 4 min. Negative or missing values read as 0 s. */
export function formatDuration(seconds: number): string {
	const total = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
	if (total < 60) return `${total} s`;
	const hours = Math.floor(total / 3600);
	const minutes = Math.floor((total % 3600) / 60);
	if (hours === 0) {
		const rest = total % 60;
		return rest === 0 ? `${minutes} min` : `${minutes} min ${rest} s`;
	}
	return minutes === 0 ? `${hours} h` : `${hours} h ${minutes} min`;
}

/** Time left as a phrase: "about 12 min left". Null while the server has no estimate. */
export function formatEta(seconds: number | null): string {
	if (seconds === null || !Number.isFinite(seconds)) return 'estimating time left';
	if (seconds < 60) return 'less than a minute left';
	const minutes = Math.round(seconds / 60);
	if (minutes < 60) return `about ${minutes} min left`;
	const hours = Math.floor(minutes / 60);
	const rest = minutes % 60;
	return rest === 0 ? `about ${hours} h left` : `about ${hours} h ${rest} min left`;
}

/** just now, 5 min ago, 3 h ago, yesterday, 4 days ago, then the date. */
export function formatRelativeTime(iso: string, now: number = Date.now()): string {
	const then = Date.parse(iso);
	if (Number.isNaN(then)) return '';
	const seconds = Math.max(0, (now - then) / 1000);
	if (seconds < 45) return 'just now';
	const minutes = Math.round(seconds / 60);
	if (minutes < 60) return `${minutes} min ago`;
	const hours = Math.round(seconds / 3600);
	if (hours < 24) return `${hours} h ago`;
	const days = Math.round(seconds / 86_400);
	if (days === 1) return 'yesterday';
	if (days < 7) return `${days} days ago`;
	const date = new Date(then);
	const sameYear = date.getFullYear() === new Date(now).getFullYear();
	return date.toLocaleDateString('en-US', {
		month: 'short',
		day: 'numeric',
		year: sameYear ? undefined : 'numeric'
	});
}

/** p. 5, pp. 5–9, or '' when the Document has no pages. */
export function formatPages(start: number | null, end: number | null): string {
	if (start === null && end === null) return '';
	const first = start ?? end;
	const last = end ?? start;
	return first === last ? `p. ${first}` : `pp. ${first}–${last}`;
}

const RUN_STATE_LABELS: Record<RunState, string> = {
	queued: 'Queued',
	running: 'Running',
	stopping: 'Stopping…',
	stopped: 'Stopped',
	failed: 'Failed',
	interrupted: 'Interrupted',
	completed: 'Completed'
};

export function runStateLabel(state: RunState): string {
	return RUN_STATE_LABELS[state] ?? state;
}

const IMPORT_STATE_LABELS: Record<ImportState, string> = {
	importing: 'Importing',
	ready: 'Ready',
	failed: 'Import failed'
};

export function importStateLabel(state: ImportState): string {
	return IMPORT_STATE_LABELS[state] ?? state;
}

const STAGE_LABELS: Record<StageName, string> = {
	preparing: 'Preparing',
	segmenting: 'Segmenting',
	summarizing: 'Summarizing',
	merging: 'Merging',
	writing: 'Writing',
	verifying: 'Verifying',
	publishing: 'Publishing'
};

export function stageLabel(stage: StageName): string {
	return STAGE_LABELS[stage] ?? stage;
}

const IMPORT_PHASE_LABELS: Record<ImportPhase, string> = {
	uploading: 'Uploading',
	queued: 'Waiting to import',
	reading: 'Reading the file',
	extracting: 'Extracting text',
	ocr: 'Recognizing text (OCR)',
	finalizing: 'Finishing'
};

export function importPhaseLabel(phase: ImportPhase): string {
	return IMPORT_PHASE_LABELS[phase] ?? phase;
}

/** "Extracting text · 45 of 312 pages"; "Importing" before the first progress report. */
export function formatImportProgress(progress: ImportProgress | null): string {
	if (!progress) return 'Importing';
	const phase = importPhaseLabel(progress.phase);
	if (progress.unit === 'pages') {
		return progress.total === null
			? `${phase} · ${formatCount(progress.done)} pages`
			: `${phase} · ${formatCount(progress.done)} of ${formatCount(progress.total)} pages`;
	}
	if (progress.unit === 'bytes') {
		return progress.total === null
			? `${phase} · ${formatBytes(progress.done)}`
			: `${phase} · ${formatBytes(progress.done)} of ${formatBytes(progress.total)}`;
	}
	return phase;
}

const FORMAT_NAMES: Record<DocumentFormat, string> = {
	txt: 'Text',
	md: 'Markdown',
	pdf: 'PDF',
	docx: 'Word',
	odt: 'OpenDocument',
	rtf: 'RTF',
	html: 'HTML',
	epub: 'EPUB',
	srt: 'SRT subtitles',
	vtt: 'WebVTT subtitles',
	png: 'PNG image',
	jpeg: 'JPEG image',
	tiff: 'TIFF image'
};

export function formatName(format: DocumentFormat): string {
	return FORMAT_NAMES[format] ?? format.toUpperCase();
}
