"""Run preflight: what a Run with a given configuration would do, without running it.

Preflight never raises for a bad configuration. Every problem is a coded
Notice; errors make ``ok`` false, warnings do not.

Errors: ``invalid_config`` (the configuration does not validate, or a
configured context window exceeds the model maximum), ``model_required``,
``document_not_found``, ``document_not_ready`` (importing or failed),
``model_not_installed``, ``context_window_unknown`` (no configured window and
no model metadata), ``budget`` (the Run would fail its budget checks).
Warnings: ``ollama_unreachable`` (the installed models could not be fetched),
``context_window_assumed`` (the context lookup failed), ``merge_capacity`` (a merge request may not hold two leaf summaries), and
``budget`` when the only window known is the assumed one.

The context window is resolved as the worker resolves it: a configured window
is used as is, otherwise min(model maximum, DEFAULT_OLLAMA_CONTEXT_WINDOW)
from ``/api/show``; an unknown model window blocks preflight unless configured.
Estimates come from the pipeline's own functions: ``select_strategy``
for the strategy and capacity, ``leaf_segmentation`` + ``segment_document``
for the leaf count, ``merge_input_budget`` + ``measure_merge_overhead`` for
the merge fanout, and the progress service's merge and verification call
estimators, which also drive the Run ETA.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from threading import Lock
from typing import Any

from pydantic import ValidationError

from summarizer.budget import (
    ASSUMED_CONTEXT_WINDOW,
    BudgetError,
    BudgetReport,
    resolve_context_window,
    select_strategy,
)
from summarizer.config import AppConfig, StrategyConfig
from summarizer.ingestion import SourceDocument
from summarizer.merge import child_fence_tokens, measure_merge_overhead
from summarizer.pipeline import leaf_segmentation, merge_input_budget
from summarizer.providers.ollama import DEFAULT_OLLAMA_CONTEXT_WINDOW
from summarizer.segmentation import SegmentationConfig, SegmentationError, segment_document
from summarizer.summaries import MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES
from summarizer.tokenization import TokenCounter, resolve_token_counter
from summarizer_web.errors import ApiError
from summarizer_web.models.api import Notice, PreflightResponse, RunConfig
from summarizer_web.services.documents_service import (
    canonical_text,
    get_document,
    latest_revision,
)
from summarizer_web.services.ollama_service import (
    OllamaContextLengthUnknownError,
    OllamaError,
    installed_models,
    is_installed,
    model_context_length,
)
from summarizer_web.services.progress_service import (
    estimate_merge_calls,
    estimate_verification_calls,
)
from summarizer_web.services.settings_service import get_settings
from summarizer_web.worker.runner import segmentation_config, strategy_config

# A leaf's JSON output is capped at max_output_tokens model tokens, while the
# byte counter the pipeline uses for Ollama charges one unit per UTF-8 byte;
# model output averages about four bytes per token.
_BYTES_PER_OUTPUT_TOKEN = 4
_LEAF_COUNT_CACHE_SIZE = 64
_leaf_counts: OrderedDict[tuple[str, str, int, int], int] = OrderedDict()
_leaf_counts_lock = Lock()

_FIELD_LABELS = {
    "model": "Model",
    "target_words": "Target words",
    "strategy": "Strategy",
    "verify": "Verify",
    "max_repair_passes": "Repair passes",
    "citations": "Citations",
    "context_window": "Context window",
    "max_output_tokens": "Max output tokens",
    "safety_margin_tokens": "Safety margin tokens",
    "safety_margin_fraction": "Safety margin fraction",
    "chunk_tokens": "Chunk tokens",
    "overlap_tokens": "Overlap tokens",
    "max_merge_children": "Max merge children",
    "max_concurrency": "Concurrency",
    "timeout_seconds": "Timeout",
    "max_retries": "Retries",
}


def run_preflight(payload: dict[str, Any]) -> PreflightResponse:
    """Check a PreflightRequest body whose ``config`` may be invalid."""
    document_id = payload.get("document_id")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ApiError(422, "invalid_request", "document_id is required.")
    errors: list[Notice] = []
    warnings: list[Notice] = []

    raw_config = payload.get("config")
    config = _validated_config(raw_config, errors)
    model = (config.model if config is not None else _raw_model(raw_config)).strip()
    if not model:
        errors.append(_error("model_required", "Choose a model for this Run."))
    document = _ready_document(document_id, errors)

    host = get_settings().ollama_host
    model_installed: bool | None = None
    if model:
        try:
            model_installed = is_installed(model, installed_models(host))
        except OllamaError as error:
            warnings.append(
                _warning(
                    "ollama_unreachable",
                    f"{error.message} The model and its context window could not be checked.",
                )
            )
        else:
            if not model_installed:
                errors.append(
                    _error(
                        "model_not_installed",
                        f"Model {model} is not installed. Run `ollama pull {model}` "
                        "or choose another model.",
                    )
                )
    if config is None or document is None or not model:
        return PreflightResponse(
            ok=False, model_installed=model_installed, errors=errors, warnings=warnings
        )

    window, source = _context_window(
        host, model, config, check_model=model_installed is True, errors=errors, warnings=warnings
    )
    if any(error.code == "context_window_unknown" for error in errors):
        return PreflightResponse(
            ok=False, model_installed=model_installed, errors=errors, warnings=warnings
        )
    strategy = strategy_config(config)
    if source == "model":
        strategy = replace(strategy, context_window=window)
    assumed = source == "assumed"
    counter = resolve_token_counter(provider="ollama", model=model)
    response = PreflightResponse(
        ok=False,
        context_window_tokens=resolve_context_window(
            provider="ollama", model=model, explicit=strategy.context_window
        ).tokens,
        context_window_source=source,
        model_installed=model_installed,
    )
    try:
        report = select_strategy(document, counter, provider="ollama", model=model, config=strategy)
    except BudgetError as error:
        (warnings if assumed else errors).append(_budget_notice(error, config, assumed=assumed))
        return _finish(response, errors, warnings)

    response = response.model_copy(
        update={
            "selected_strategy": report.strategy,
            "context_window_tokens": report.context_window_tokens,
            "usable_input_capacity": report.usable_input_capacity,
            "document_tokens": report.document_tokens,
        }
    )
    try:
        leaf_count, merge_calls = _tree_estimate(
            document, counter, report, config, strategy=strategy, model=model, host=host,
            warnings=warnings,
        )
    except (BudgetError, SegmentationError) as error:
        (warnings if assumed else errors).append(_budget_notice(error, config, assumed=assumed))
        return _finish(response, errors, warnings)
    calls = leaf_count + merge_calls + 1
    if config.verify:
        calls += estimate_verification_calls(config.target_words)
    response = response.model_copy(
        update={"estimated_leaf_count": leaf_count, "estimated_model_calls": calls}
    )
    return _finish(response, errors, warnings)


def _finish(
    response: PreflightResponse, errors: list[Notice], warnings: list[Notice]
) -> PreflightResponse:
    return response.model_copy(update={"ok": not errors, "errors": errors, "warnings": warnings})


def _validated_config(raw: object, errors: list[Notice]) -> RunConfig | None:
    if not isinstance(raw, dict):
        errors.append(_error("invalid_config", "The run configuration is missing."))
        return None
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as error:
        for item in error.errors():
            location = item.get("loc") or ()
            field = str(location[0]) if location else ""
            message = str(item.get("msg", "invalid value")).removeprefix("Value error, ")
            label = _FIELD_LABELS.get(field, field)
            errors.append(
                _error("invalid_config", f"{label}: {message}." if label else f"{message}.")
            )
        return None


def _raw_model(raw: object) -> str:
    model = raw.get("model") if isinstance(raw, dict) else None
    return model if isinstance(model, str) else ""


def _ready_document(document_id: str, errors: list[Notice]) -> SourceDocument | None:
    try:
        document = get_document(document_id)
    except ApiError as error:
        if error.code != "document_not_found":
            raise
        errors.append(_error("document_not_found", "This Document no longer exists."))
        return None
    if document.import_state == "importing":
        errors.append(
            _error(
                "document_not_ready",
                "The Document is still importing. Start the Run when the import finishes.",
            )
        )
        return None
    if document.import_state == "failed":
        reason = (document.import_error or "").strip().rstrip(".")
        errors.append(
            _error(
                "document_not_ready",
                f"The Document failed to import: {reason}." if reason
                else "The Document failed to import.",
            )
        )
        return None
    revision = latest_revision(document_id)
    if revision is None:
        errors.append(_error("document_not_ready", "The Document has no imported text yet."))
        return None
    return SourceDocument(
        text=canonical_text(revision["canonical_path"]), source_id=revision["source_sha256"]
    )


def _context_window(
    host: str,
    model: str,
    config: RunConfig,
    *,
    check_model: bool,
    errors: list[Notice],
    warnings: list[Notice],
) -> tuple[int | None, str]:
    """Return (window, source) the way the worker selects the Run's window."""
    maximum: int | None = None
    if check_model:
        try:
            maximum = model_context_length(host, model)
        except OllamaContextLengthUnknownError as error:
            if config.context_window is None:
                errors.append(
                    _error(
                        "context_window_unknown",
                        f"{error.message} Set Context window in Advanced.",
                    )
                )
        except OllamaError as error:
            if config.context_window is None:
                warnings.append(
                    _warning(
                        "context_window_assumed",
                        f"{error.message} The assumed context window of "
                        f"{ASSUMED_CONTEXT_WINDOW} tokens is used for these estimates.",
                    )
                )
    if config.context_window is not None:
        if maximum is not None and config.context_window > maximum:
            errors.append(
                _error(
                    "invalid_config",
                    f"Context window: {config.context_window} tokens exceeds the maximum "
                    f"of {maximum} tokens for {model}.",
                )
            )
        return config.context_window, "configured"
    if maximum is not None:
        return min(maximum, DEFAULT_OLLAMA_CONTEXT_WINDOW), "model"
    return None, "assumed"


def _tree_estimate(
    document: SourceDocument,
    counter: TokenCounter,
    report: BudgetReport,
    config: RunConfig,
    *,
    strategy: StrategyConfig,
    model: str,
    host: str,
    warnings: list[Notice],
) -> tuple[int, int]:
    """Return (leaf count, merge calls); raises BudgetError/SegmentationError."""
    if report.strategy == "direct":
        return 1, 0
    segmentation = leaf_segmentation(
        report,
        counter,
        app=AppConfig(model=model, provider="ollama", ollama_host=host),
        strategy=strategy,
        requested=segmentation_config(config),
    )
    leaf_count = _leaf_count(document, counter, segmentation)
    if leaf_count < 2:
        return leaf_count, 0
    budget = merge_input_budget(report)
    overhead = measure_merge_overhead(
        counter, level=1, provider_schema_reserve=MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES
    )
    capacity = budget - overhead
    if capacity <= 0:
        raise BudgetError(
            f"merge requests do not fit: the merge instructions need {overhead} of the "
            f"{budget} input tokens a context window of {report.context_window_tokens} "
            "tokens leaves"
        )
    per_token = 1 if counter.exact else _BYTES_PER_OUTPUT_TOKEN
    fanout = capacity // (config.max_output_tokens * per_token + child_fence_tokens(counter))
    if config.max_merge_children is not None:
        fanout = min(fanout, config.max_merge_children)
    if fanout < 2:
        warnings.append(
            _warning(
                "merge_capacity",
                "A merge request may not hold two leaf summaries at this context window, "
                "so the Run can fail while merging. Set a larger context window or lower "
                "max output tokens.",
            )
        )
        fanout = 2
    return leaf_count, estimate_merge_calls(leaf_count, fanout)


def _leaf_count(
    document: SourceDocument, counter: TokenCounter, segmentation: SegmentationConfig
) -> int:
    """Segment once per (source, counter, budget); large Documents take a while."""
    key = (
        document.source_id,
        counter.identity,
        segmentation.max_tokens,
        segmentation.overlap_tokens,
    )
    with _leaf_counts_lock:
        cached = _leaf_counts.get(key)
        if cached is not None:
            _leaf_counts.move_to_end(key)
            return cached
    count = len(segment_document(document, counter, segmentation))
    with _leaf_counts_lock:
        _leaf_counts[key] = count
        while len(_leaf_counts) > _LEAF_COUNT_CACHE_SIZE:
            _leaf_counts.popitem(last=False)
    return count


def _budget_notice(error: Exception, config: RunConfig, *, assumed: bool) -> Notice:
    text = str(error).strip().rstrip(".")
    message = f"{text[:1].upper()}{text[1:]}."
    if config.strategy == "direct":
        hint = "Choose the Auto or Hierarchical strategy, or set a larger context window."
    elif config.chunk_tokens is not None:
        hint = "Lower chunk tokens or max output tokens, or set a larger context window."
    else:
        hint = "Lower max output tokens or set a larger context window."
    if assumed:
        return _warning(
            "budget",
            f"{message} This uses the assumed context window of {ASSUMED_CONTEXT_WINDOW} "
            f"tokens. {hint}",
        )
    return _error("budget", f"{message} {hint}")


def _error(code: str, message: str) -> Notice:
    return Notice(code=code, message=message, severity="error")


def _warning(code: str, message: str) -> Notice:
    return Notice(code=code, message=message, severity="warning")
