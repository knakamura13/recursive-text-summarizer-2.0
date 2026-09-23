"""Versioned API response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorResponse(ApiModel):
    code: str
    message: str
    retryable: bool = False


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"


class OllamaHealthResponse(ApiModel):
    connected: bool
    message: str


class OllamaModel(ApiModel):
    name: str


class OllamaModelsResponse(ApiModel):
    models: list[OllamaModel]


class RunConfig(ApiModel):
    model: str
    target_words: int = 300
    strategy: Literal["auto", "direct", "hierarchical"] = "auto"
    verify: bool = True
    max_repair_passes: int = 1
    citations: bool = True
    context_window: int | None = None
    max_output_tokens: int = 1024
    safety_margin_tokens: int = 256
    safety_margin_fraction: float = 0.02
    chunk_tokens: int | None = None
    overlap_tokens: int = 0
    max_merge_children: int | None = None
    max_concurrency: int = 1
    timeout_seconds: float = 180.0
    max_retries: int = 5


class SettingsResponse(ApiModel):
    ollama_host: str
    defaults: RunConfig


class SettingsUpdate(ApiModel):
    ollama_host: str | None = None
    defaults: RunConfig | None = None


class DocumentSummary(ApiModel):
    document_id: str
    title: str
    filename: str
    format: Literal["txt", "md", "pdf"]
    size_bytes: int
    import_state: Literal["ready", "pending_confirmation", "failed"]
    latest_run_state: str | None = None


class DocumentListResponse(ApiModel):
    documents: list[DocumentSummary]
    already_imported: bool = False


class DocumentDetailResponse(DocumentSummary):
    revision_id: str | None = None
    source_sha256: str | None = None
    page_count: int | None = None
    blank_pages: list[int] = Field(default_factory=list)
    latest_run_id: str | None = None
    latest_run_state: str | None = None


class DocumentRenameRequest(ApiModel):
    title: str


class SourceSliceResponse(ApiModel):
    text: str
    offset: int
    total_length: int
    has_more: bool


class SourceLocationsResponse(ApiModel):
    segments: list[dict[str, Any]]


class PreflightRequest(ApiModel):
    document_id: str
    config: RunConfig


class PreflightResponse(ApiModel):
    fits: bool
    selected_strategy: str
    context_window_tokens: int
    context_window_assumed: bool
    warnings: list[str] = Field(default_factory=list)
    usable_input_capacity: int


class CreateRunRequest(ApiModel):
    document_id: str
    config: RunConfig


class RunResponse(ApiModel):
    run_id: str
    document_id: str
    state: str
    strategy: str | None = None
    config: RunConfig
    attempt_id: str | None = None
    attempt_state: str | None = None
    failure_reason: str | None = None


class RunListResponse(ApiModel):
    runs: list[RunResponse]


class NodeTreeItem(ApiModel):
    node_id: str
    parent_id: str | None
    level: int
    order: int
    label: str
    provisional: bool
    state: Literal["pending", "active", "completed", "skipped"] = "pending"


class NodeTreeResponse(ApiModel):
    nodes: list[NodeTreeItem]


class NodeDetailResponse(ApiModel):
    node_id: str
    label: str
    summary_text: str | None
    provisional: bool
    source_passage: str | None
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    covered_segment_ids: list[str] = Field(default_factory=list)


class FinalSummaryResponse(ApiModel):
    available: bool
    text: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    word_count: int | None = None
    target_words: int | None = None
    short_of_target: bool = False
    audit_warnings: list[str] = Field(default_factory=list)
    verification_state: Literal[
        "not_run", "in_progress", "completed", "failed"
    ] = "not_run"
    verification: dict[str, Any] | None = None


class ConfirmExtractionResponse(ApiModel):
    document_id: str
    import_state: Literal["ready"]


class ProgressStage(ApiModel):
    stage: str
    state: Literal["active", "completed", "skipped"]
    completed: int | None = None
    total: int | None = None
