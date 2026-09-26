"""Versioned API request and response models.

Offsets into document text are Python code-point offsets. Timestamps are
ISO 8601 UTC strings.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Errors and health -------------------------------------------------------


class ErrorResponse(ApiModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] | None = None


class Notice(ApiModel):
    """A coded, human-readable message: preflight, import, or summary notes."""

    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "info"


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"


# --- Ollama ------------------------------------------------------------------


class OllamaHealthResponse(ApiModel):
    connected: bool
    message: str
    host: str
    version: str | None = None


class OllamaModel(ApiModel):
    name: str
    size_bytes: int | None = None
    parameter_size: str | None = None
    family: str | None = None
    quantization: str | None = None
    modified_at: str | None = None


class OllamaModelsResponse(ApiModel):
    models: list[OllamaModel]


# --- Run configuration and settings -----------------------------------------

StrategyName = Literal["auto", "direct", "hierarchical"]
SelectedStrategy = Literal["direct", "hierarchical"]


class RunConfig(ApiModel):
    """Run configuration. Validated on input; stored configs load with
    `RunConfig.model_construct` so older rows never fail to render."""

    model: str = ""
    target_words: int = Field(300, ge=25, le=5000)
    strategy: StrategyName = "auto"
    verify: bool = True
    max_repair_passes: int = Field(1, ge=0, le=5)
    citations: bool = True
    context_window: int | None = Field(None, ge=1024, le=4_194_304)
    max_output_tokens: int = Field(1024, ge=128, le=131_072)
    safety_margin_tokens: int = Field(256, ge=0, le=131_072)
    safety_margin_fraction: float = Field(0.02, ge=0.0, le=0.5)
    chunk_tokens: int | None = Field(None, ge=128)
    overlap_tokens: int = Field(0, ge=0)
    max_merge_children: int | None = Field(None, ge=2, le=64)
    max_concurrency: int = Field(1, ge=1, le=16)
    timeout_seconds: float = Field(600.0, ge=10.0, le=3600.0)
    max_retries: int = Field(5, ge=1, le=20)
    strict_numbers: bool = False
    strict_names: bool = False

    @model_validator(mode="after")
    def _overlap_below_chunk(self) -> RunConfig:
        if self.chunk_tokens is not None and self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        return self


class RunConfigPatch(ApiModel):
    """Partial update of the default run configuration; omitted fields keep
    their stored values."""

    model: str | None = None
    target_words: int | None = Field(None, ge=25, le=5000)
    strategy: StrategyName | None = None
    verify: bool | None = None
    max_repair_passes: int | None = Field(None, ge=0, le=5)
    citations: bool | None = None
    context_window: int | None = Field(None, ge=1024, le=4_194_304)
    max_output_tokens: int | None = Field(None, ge=128, le=131_072)
    safety_margin_tokens: int | None = Field(None, ge=0, le=131_072)
    safety_margin_fraction: float | None = Field(None, ge=0.0, le=0.5)
    chunk_tokens: int | None = Field(None, ge=128)
    overlap_tokens: int | None = Field(None, ge=0)
    max_merge_children: int | None = Field(None, ge=2, le=64)
    max_concurrency: int | None = Field(None, ge=1, le=16)
    timeout_seconds: float | None = Field(None, ge=10.0, le=3600.0)
    max_retries: int | None = Field(None, ge=1, le=20)
    strict_numbers: bool | None = None
    strict_names: bool | None = None
    clear: list[Literal["context_window", "chunk_tokens", "max_merge_children"]] = Field(
        default_factory=list,
        description="Nullable fields to reset to null, since null means 'unchanged' here.",
    )


class SettingsResponse(ApiModel):
    ollama_host: str
    defaults: RunConfig


class SettingsUpdate(ApiModel):
    ollama_host: str | None = None
    defaults: RunConfigPatch | None = None


# --- Documents ----------------------------------------------------------------

DocumentFormat = Literal[
    "txt", "md", "pdf", "docx", "odt", "rtf", "html", "epub", "srt", "vtt", "png", "jpeg", "tiff"
]
ImportState = Literal["importing", "ready", "failed"]
ImportPhase = Literal["uploading", "queued", "reading", "extracting", "ocr", "finalizing"]

RunState = Literal[
    "queued", "running", "stopping", "stopped", "failed", "interrupted", "completed"
]


class ImportProgress(ApiModel):
    phase: ImportPhase
    done: int = 0
    total: int | None = None
    unit: Literal["pages", "bytes"] | None = None
    message: str | None = None


class ImportReport(ApiModel):
    detected_format: DocumentFormat
    encoding: str | None = None
    page_count: int | None = None
    text_layer_pages: int | None = None
    ocr_pages: list[int] = Field(default_factory=list)
    blank_pages: list[int] = Field(default_factory=list)
    char_count: int
    word_count: int
    notices: list[Notice] = Field(default_factory=list)
    preview: str
    extraction_version: str
    duration_seconds: float | None = None


class RunBrief(ApiModel):
    run_id: str
    state: RunState
    created_at: str
    updated_at: str


class DocumentSummary(ApiModel):
    document_id: str
    title: str
    filename: str
    format: DocumentFormat
    origin: Literal["upload", "paste"] = "upload"
    size_bytes: int
    import_state: ImportState
    import_progress: ImportProgress | None = None
    import_error: str | None = None
    char_count: int | None = None
    page_count: int | None = None
    created_at: str
    updated_at: str
    latest_run: RunBrief | None = None


class DocumentListResponse(ApiModel):
    documents: list[DocumentSummary]


class DocumentCreatedResponse(ApiModel):
    document: DocumentSummary
    already_imported: bool = False


class PasteTextRequest(ApiModel):
    title: str | None = Field(None, max_length=300)
    text: str = Field(min_length=1)


class DocumentDetailResponse(DocumentSummary):
    source_sha256: str | None = None
    import_report: ImportReport | None = None


class DocumentRenameRequest(ApiModel):
    title: str = Field(min_length=1, max_length=300)


class SourceSliceResponse(ApiModel):
    text: str
    offset: int
    total_length: int
    has_more: bool


class SourcePage(ApiModel):
    page: int
    start: int
    end: int
    ocr: bool = False
    blank: bool = False


class SourcePagesResponse(ApiModel):
    pages: list[SourcePage]


# --- Preflight and runs -------------------------------------------------------


class PreflightRequest(ApiModel):
    document_id: str
    config: RunConfig


class PreflightResponse(ApiModel):
    ok: bool
    selected_strategy: SelectedStrategy | None = None
    context_window_tokens: int | None = None
    context_window_source: Literal["configured", "model", "assumed"] | None = None
    usable_input_capacity: int | None = None
    document_tokens: int | None = None
    estimated_leaf_count: int | None = None
    estimated_model_calls: int | None = None
    model_installed: bool | None = None
    errors: list[Notice] = Field(default_factory=list)
    warnings: list[Notice] = Field(default_factory=list)


class CreateRunRequest(ApiModel):
    document_id: str
    config: RunConfig


class RunFailure(ApiModel):
    code: str
    message: str
    stage: str | None = None
    item: str | None = None
    detail: str | None = None
    hint: str | None = None


class AttemptInfo(ApiModel):
    attempt_id: str
    attempt_number: int
    state: RunState
    started_at: str | None = None
    ended_at: str | None = None
    failure: RunFailure | None = None


StageNameValue = Literal[
    "preparing", "segmenting", "summarizing", "merging", "writing", "verifying", "publishing"
]


class StageProgress(ApiModel):
    stage: StageNameValue
    state: Literal["pending", "active", "completed", "skipped"]
    completed: int | None = None
    total: int | None = None
    detail: str | None = None


class CountProgress(ApiModel):
    done: int = 0
    total: int | None = None
    reused: int = 0
    failed: int = 0


class LevelProgress(ApiModel):
    level: int
    done: int
    total: int


class CurrentItem(ApiModel):
    kind: str
    work_id: str
    label: str
    started_at: str | None = None
    attempt: int | None = None


class RunProgress(ApiModel):
    cursor: int
    attempt_number: int | None = None
    started_at: str | None = None
    elapsed_seconds: float | None = None
    stage: StageNameValue | None = None
    stages: list[StageProgress]
    leaves: CountProgress = Field(default_factory=CountProgress)
    merges: CountProgress = Field(default_factory=CountProgress)
    merge_levels: list[LevelProgress] = Field(default_factory=list)
    claims: CountProgress = Field(default_factory=CountProgress)
    current_items: list[CurrentItem] = Field(default_factory=list)
    eta_seconds: float | None = None
    eta_basis: str | None = None


class RunResponse(ApiModel):
    run_id: str
    document_id: str
    document_title: str
    state: RunState
    requested_strategy: StrategyName
    selected_strategy: SelectedStrategy | None = None
    config: RunConfig
    created_at: str
    updated_at: str
    attempt: AttemptInfo | None = None
    attempt_count: int = 0
    failure: RunFailure | None = None
    can_stop: bool = False
    can_resume: bool = False
    progress: RunProgress | None = None


class RunListResponse(ApiModel):
    runs: list[RunResponse]


class ActiveRunResponse(ApiModel):
    run: RunResponse | None = None


class ActivityResponse(ApiModel):
    """Snapshot sent on the tab's single SSE stream; `cursor` is the max
    run_events.event_id it reflects."""

    active_run: RunResponse | None = None
    importing: list[DocumentSummary] = Field(default_factory=list)
    cursor: int = 0


# --- Tree and nodes -----------------------------------------------------------

NodeKind = Literal["leaf", "merge", "passthrough"]
NodeState = Literal["pending", "active", "completed", "failed"]


class NodeTreeItem(ApiModel):
    node_id: str
    parent_id: str | None = None
    level: int
    order: int
    kind: NodeKind
    label: str
    state: NodeState
    child_count: int = 0
    page_start: int | None = None
    page_end: int | None = None
    duration_seconds: float | None = None


class NodeTreeResponse(ApiModel):
    nodes: list[NodeTreeItem]
    cursor: int


class EvidenceRef(ApiModel):
    """Source support for an assertion. `start`/`end` locate the quote when
    it was found in the segment, otherwise the segment core."""

    segment_id: str
    quote: str | None = None
    quote_found: bool = False
    start: int | None = None
    end: int | None = None
    page_start: int | None = None
    page_end: int | None = None


class NodeContentUnit(ApiModel):
    text: str
    kind: str
    uncertain: bool = False
    qualification: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class NodeAnnotation(ApiModel):
    kind: Literal["qualification", "contradiction"]
    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class SegmentRef(ApiModel):
    segment_id: str
    order: int
    start: int
    end: int
    core_start: int
    core_end: int
    page_start: int | None = None
    page_end: int | None = None


class SegmentListResponse(ApiModel):
    segments: list[SegmentRef]


class NodeDetailResponse(ApiModel):
    node_id: str
    parent_id: str | None = None
    level: int
    order: int
    kind: NodeKind
    label: str
    state: NodeState
    summary_text: str | None = None
    content_units: list[NodeContentUnit] = Field(default_factory=list)
    annotations: list[NodeAnnotation] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    quotations: list[EvidenceRef] = Field(default_factory=list)
    covered_segments: list[SegmentRef] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


# --- Final summary ------------------------------------------------------------


class SummaryCitation(ApiModel):
    citation_id: str
    segment_id: str
    start: int | None = None
    end: int | None = None
    page_start: int | None = None
    page_end: int | None = None


class SummarySentence(ApiModel):
    index: int
    paragraph: int
    text: str
    verdict: Literal["supported", "not_meaningfully_verifiable", "unchecked"] = "unchecked"
    evidence: list[EvidenceRef] = Field(default_factory=list)


class RemovedSentence(ApiModel):
    text: str
    verdict: str
    reason: str | None = None


class FinalSummaryResponse(ApiModel):
    available: bool
    text: str | None = None
    sentences: list[SummarySentence] = Field(default_factory=list)
    removed_sentences: list[RemovedSentence] = Field(default_factory=list)
    citations: list[SummaryCitation] = Field(default_factory=list)
    word_count: int | None = None
    target_words: int | None = None
    short_of_target: bool = False
    notices: list[Notice] = Field(default_factory=list)
    verification_state: Literal["not_run", "in_progress", "completed", "failed"] = "not_run"
    publication: Literal["editorial", "verified_subset", "content_unit_fallback"] | None = None
