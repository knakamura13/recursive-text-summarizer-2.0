"""Provider-free run budget checks."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from summarizer.budget import BudgetError, select_strategy
from summarizer.config import AppConfig, StrategyConfig
from summarizer.ingestion import SourceDocument
from summarizer.tokenization import resolve_token_counter
from summarizer_web.models.api import PreflightRequest, PreflightResponse
from summarizer_web.services.documents_service import _latest_revision
from summarizer_web.services.settings_service import get_settings


def run_preflight(payload: PreflightRequest) -> PreflightResponse:
    if not payload.config.model.strip():
        raise HTTPException(status_code=400, detail="Model selection is required")
    revision = _latest_revision(payload.document_id)
    if revision is None:
        raise HTTPException(status_code=404, detail="Document source not found")
    text = Path(revision["canonical_path"]).read_text(encoding="utf-8")
    document = SourceDocument(text=text, source_id=revision["source_sha256"])
    settings = get_settings()
    strategy = StrategyConfig(
        strategy=payload.config.strategy,
        context_window=payload.config.context_window,
        max_output_tokens=payload.config.max_output_tokens,
        safety_margin_tokens=payload.config.safety_margin_tokens,
        safety_margin_fraction=payload.config.safety_margin_fraction,
    )
    counter = resolve_token_counter(provider="ollama", model=payload.config.model)
    try:
        report = select_strategy(
            document,
            counter,
            provider="ollama",
            model=payload.config.model,
            config=strategy,
        )
    except BudgetError as error:
        return PreflightResponse(
            fits=False,
            selected_strategy=payload.config.strategy,
            context_window_tokens=strategy.context_window or 0,
            context_window_assumed=True,
            warnings=[str(error)],
            usable_input_capacity=0,
        )
    warnings: list[str] = []
    if report.context_window_assumed:
        warnings.append("Context window is assumed for this model; set it explicitly if known.")
    if payload.config.strategy == "direct" and report.strategy != "direct":
        warnings.append("Direct strategy will not fit; hierarchical execution is required.")
    return PreflightResponse(
        fits=True,
        selected_strategy=report.strategy,
        context_window_tokens=report.context_window_tokens,
        context_window_assumed=report.context_window_assumed,
        warnings=warnings,
        usable_input_capacity=report.usable_input_capacity,
    )
