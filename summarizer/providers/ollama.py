from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from threading import Lock
from typing import Any

import httpx
import ollama
from pydantic import ValidationError

from summarizer.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    normalize_output_text,
)
from summarizer.summaries import (
    MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES,
    MAX_QUOTATIONS_PER_NODE,
)


def _create_client(**kwargs: object) -> object:
    return ollama.Client(**kwargs)


_SUMMARY_SCHEMA_NAMES = frozenset(("leaf_summary", "merged_summary"))
DEFAULT_OLLAMA_CONTEXT_WINDOW = 32_768


def _local_response_schema(request: GenerationRequest) -> Mapping[str, object]:
    """Strengthen generated-summary constraints supported by Ollama's decoder.

    The shared schema stays within OpenAI's smaller supported subset. Ollama
    supports additional constraints that prevent locally generated records
    from reaching application validation with blank required values or empty
    evidence. Source-dependent provenance and quotation checks remain in the
    application validator.
    """
    schema = deepcopy(request.response_schema)
    if (
        request.schema_name not in _SUMMARY_SCHEMA_NAMES
        or not isinstance(schema, dict)
    ):
        return schema

    definitions = schema.get("$defs")
    root_properties = schema.get("properties")
    if not isinstance(definitions, dict) or not isinstance(root_properties, dict):
        return schema

    summary = root_properties.get("summary")
    provenance = root_properties.get("provenance")
    quotations = root_properties.get("quotations")
    level = root_properties.get("level")
    evidence_item = definitions.get("EvidenceItem")
    content_unit = definitions.get("ContentUnit")
    annotation = definitions.get("GroundedAnnotation")
    if not all(
        isinstance(value, dict)
        for value in (
            summary,
            provenance,
            quotations,
            level,
            evidence_item,
            content_unit,
            annotation,
        )
    ):
        return schema

    summary["minLength"] = 1
    quotations["maxItems"] = MAX_QUOTATIONS_PER_NODE
    for definition in (content_unit, annotation):
        properties = definition.get("properties")
        if not isinstance(properties, dict):
            return schema
        text = properties.get("text")
        evidence = properties.get("evidence")
        if not isinstance(text, dict) or not isinstance(evidence, dict):
            return schema
        text["minLength"] = 1
        evidence["minItems"] = 1

    evidence_properties = evidence_item.get("properties")
    if not isinstance(evidence_properties, dict):
        return schema
    segment_id = evidence_properties.get("segment_id")
    if not isinstance(segment_id, dict):
        return schema
    segment_id["minLength"] = 1
    quote = evidence_properties.get("quote")
    if not isinstance(quote, dict):
        return schema
    variants = quote.get("anyOf")
    if not isinstance(variants, list):
        return schema
    for variant in variants:
        if isinstance(variant, dict) and variant.get("type") == "string":
            variant["minLength"] = 1

    provenance["minItems"] = 1
    if request.schema_name == "leaf_summary" and request.operation_id:
        segment_id["enum"] = [request.operation_id]
        provenance["maxItems"] = 1
    if request.expected_summary_level is not None:
        level["enum"] = [request.expected_summary_level]

    candidate_sources: dict[str, list[str]] = {}
    for source_id, candidates in (request.quote_candidates_by_segment or {}).items():
        for candidate in candidates:
            candidate_sources.setdefault(candidate, []).append(source_id)
    allowed_sources = list(request.allowed_summary_segment_ids or ())
    if not allowed_sources and request.schema_name == "leaf_summary" and request.operation_id:
        allowed_sources = [request.operation_id]

    if candidate_sources or allowed_sources:
        base_bytes = len(
            json.dumps(request.response_schema, separators=(",", ":"), sort_keys=True)
            .encode("utf-8")
        )

        def evidence_variant(
            *, source_ids: list[str] | None, candidate: str | None
        ) -> dict[str, object]:
            source_schema: dict[str, object] = {"type": "string", "minLength": 1}
            if source_ids is not None:
                source_schema["enum"] = source_ids
            quote_schema: dict[str, object] = (
                {"type": "null"}
                if candidate is None
                else {"type": "string", "enum": [candidate]}
            )
            return {
                "type": "object",
                "properties": {
                    "segment_id": source_schema,
                    "quote": quote_schema,
                },
                "required": ["segment_id", "quote"],
                "additionalProperties": False,
            }

        accepted_sources: list[str] = []
        if allowed_sources:
            for source_id in allowed_sources:
                trial_sources = [*accepted_sources, source_id]
                evidence_item.clear()
                evidence_item["anyOf"] = [
                    evidence_variant(source_ids=trial_sources, candidate=None)
                ]
                provenance["items"] = {"type": "string", "enum": trial_sources}
                emitted_bytes = len(
                    json.dumps(schema, separators=(",", ":"), sort_keys=True).encode(
                        "utf-8"
                    )
                )
                if emitted_bytes - base_bytes > MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES:
                    break
                accepted_sources = trial_sources
            if not accepted_sources:
                raise ProviderRequestError(
                    "Ollama summary schema reserve cannot fit one source identifier"
                )
            provenance["items"] = {"type": "string", "enum": accepted_sources}

        pair_variants = [
            evidence_variant(
                source_ids=accepted_sources or None,
                candidate=None,
            )
        ]
        evidence_item.clear()
        evidence_item["anyOf"] = pair_variants
        capacity_exhausted = False
        for candidate, source_ids in candidate_sources.items():
            eligible_sources = [
                source_id
                for source_id in source_ids
                if not accepted_sources or source_id in accepted_sources
            ]
            represented_sources: list[str] = []
            for source_id in eligible_sources:
                represented_sources.append(source_id)
                candidate_variant = evidence_variant(
                    source_ids=represented_sources, candidate=candidate
                )
                evidence_item["anyOf"] = [*pair_variants, candidate_variant]
                emitted_bytes = len(
                    json.dumps(schema, separators=(",", ":"), sort_keys=True).encode(
                        "utf-8"
                    )
                )
                if emitted_bytes - base_bytes > MAX_PROVIDER_SUMMARY_SCHEMA_JSON_BYTES:
                    represented_sources.pop()
                    capacity_exhausted = True
                    break
            if represented_sources:
                pair_variants.append(
                    evidence_variant(
                        source_ids=represented_sources,
                        candidate=candidate,
                    )
                )
            evidence_item["anyOf"] = pair_variants
            if capacity_exhausted:
                break
    return schema


class OllamaProvider:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        client_factory: Callable[..., object] = _create_client,
    ) -> None:
        self._host = host
        self._client_factory = client_factory
        self._clients: dict[float, Any] = {}
        self._context_windows: dict[str, int] = {}
        self._clients_lock = Lock()

    def configure_context_window(
        self,
        model: str,
        requested: int | None,
        *,
        timeout_seconds: float,
    ) -> int:
        """Select and enforce a context no larger than the model architecture."""
        try:
            response = self._get_client(timeout_seconds).show(model)
        except httpx.TimeoutException as error:
            raise ProviderTimeoutError("Ollama model inspection timed out") from error
        except (ConnectionError, httpx.TransportError) as error:
            raise ProviderConnectionError(
                "Ollama connection failed; confirm the service is running"
            ) from error
        except ollama.ResponseError as error:
            if error.status_code == 404:
                raise ProviderRequestError(
                    "Ollama model was not found; pull it before retrying"
                ) from error
            raise ProviderRequestError("Ollama rejected model inspection") from error
        except ollama.RequestError as error:
            raise ProviderRequestError("Ollama model inspection was invalid") from error
        except (json.JSONDecodeError, ValidationError) as error:
            raise ProviderResponseError(
                "Ollama returned malformed model metadata"
            ) from error

        modelinfo = getattr(response, "modelinfo", None)
        architecture = (
            modelinfo.get("general.architecture")
            if isinstance(modelinfo, Mapping)
            else None
        )
        maximum = (
            modelinfo.get(f"{architecture}.context_length")
            if isinstance(architecture, str)
            else None
        )
        if not isinstance(maximum, int) or maximum <= 0:
            raise ProviderResponseError(
                "Ollama model metadata did not contain a valid context length"
            )
        selected = (
            min(maximum, DEFAULT_OLLAMA_CONTEXT_WINDOW)
            if requested is None
            else requested
        )
        if selected <= 0 or selected > maximum:
            raise ProviderRequestError(
                f"requested context window {selected} exceeds the model maximum {maximum}"
            )
        with self._clients_lock:
            self._context_windows[model] = selected
        return selected

    def generate(self, request: GenerationRequest) -> GenerationResult:
        arguments: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {
                    "role": "system",
                    "content": request.instructions,
                },
                {"role": "user", "content": request.input_text},
            ],
            "stream": False,
            "think": False,
        }
        with self._clients_lock:
            context_window = self._context_windows.get(request.model)
        options: dict[str, int] = {}
        if context_window is not None:
            options["num_ctx"] = context_window
        if request.max_output_tokens is not None:
            options["num_predict"] = request.max_output_tokens
        if options:
            arguments["options"] = options
        if request.response_schema is not None:
            # Ollama supports constraints beyond the subset accepted by the
            # hosted adapter. Callers still parse and validate defensively.
            arguments["format"] = _local_response_schema(request)

        try:
            response = self._get_client(request.timeout_seconds).chat(**arguments)
        except httpx.TimeoutException as error:
            raise ProviderTimeoutError("Ollama request timed out") from error
        except (ConnectionError, httpx.TransportError) as error:
            raise ProviderConnectionError(
                "Ollama connection failed; confirm the service is running"
            ) from error
        except ollama.ResponseError as error:
            status_code = error.status_code
            if status_code == 404:
                raise ProviderRequestError(
                    "Ollama model was not found; pull it before retrying"
                ) from error
            if status_code == 429:
                raise ProviderRateLimitError(
                    "Ollama request was rate limited"
                ) from error
            if status_code >= 500:
                raise ProviderServerError(
                    "Ollama server request failed"
                ) from error
            raise ProviderRequestError(
                "Ollama rejected the request"
            ) from error
        except ollama.RequestError as error:
            raise ProviderRequestError(
                "Ollama request was invalid"
            ) from error
        except (json.JSONDecodeError, ValidationError) as error:
            raise ProviderResponseError(
                "Ollama returned a malformed response"
            ) from error

        if getattr(response, "done", None) is not True:
            raise ProviderResponseError(
                "Ollama response did not complete"
            )
        if getattr(response, "done_reason", None) == "length":
            # A record cut off part-way is unparseable JSON, and calling it
            # malformed output hides the one fact that explains it.
            limit = (
                f"the configured output token limit of {request.max_output_tokens}"
                if request.max_output_tokens is not None
                else "the model's output limit"
            )
            raise ProviderResponseError(
                f"Ollama stopped at {limit}; the response is incomplete"
            )
        message = getattr(response, "message", None)
        output_text = getattr(message, "content", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ProviderResponseError(
                "Ollama response did not contain valid text"
            )

        model = getattr(response, "model", None) or request.model
        if not isinstance(model, str) or not model.strip():
            raise ProviderResponseError(
                "Ollama response contained invalid metadata"
            )

        try:
            return GenerationResult(
                text=normalize_output_text(output_text, request),
                provider="ollama",
                model=model,
                input_tokens=getattr(response, "prompt_eval_count", None),
                output_tokens=getattr(response, "eval_count", None),
                finish_status=getattr(response, "done_reason", None),
                request_id=None,
            )
        except (TypeError, ValueError) as error:
            raise ProviderResponseError(
                "Ollama response contained invalid metadata"
            ) from error

    def _get_client(self, timeout_seconds: float) -> Any:
        client = self._clients.get(timeout_seconds)
        if client is not None:
            return client
        with self._clients_lock:
            client = self._clients.get(timeout_seconds)
            if client is None:
                client = self._client_factory(
                    host=self._host,
                    timeout=timeout_seconds,
                )
                self._clients[timeout_seconds] = client
            return client
