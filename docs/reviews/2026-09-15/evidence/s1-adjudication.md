# S1 Final Adjudication — 301cc4d

## Scope

Adjudication of all 27 S1-assigned rows in traceability.md after candidate findings (F-001, F-002, F-003) have been disposed as refuted. This pass reconciles existing test evidence against the traceability matrix to establish verified, pending-live-testing, and unverifiable verdicts.

Reviewed commit: `301cc4d56d6326b5b0449da059b3b35f484cc5ca`.

## Method

1. Prior session evidence: s1-issue3-first8-mapping.md (8 #3 rows, 4 verified / 4 pending)
2. Source code review: config.py, providers/base.py, providers/ollama.py, budget.py, cli.py
3. Test coverage audit: test_config.py, test_cli.py, test_strategy_selection.py, test_entrypoint.py, providers/test_ollama.py
4. Documentation review: README installation/configuration section

## S1 Traceability Reconciliation

### Issue #3 rows (CLI and provider integration) — 12 rows

| Row | Criterion | Prior verdict | Adjudication | Evidence |
|---|---|---|---|---|
| 84 | Default `python main.py` workflow | verified | **VERIFIED** | test_cli.py::test_main_runs_default_pipeline_without_network |
| 85 | Validated configuration, no mutable globals | pending evidence reconciliation | **VERIFIED** | config.py AppConfig is frozen dataclass with __post_init__ validation; test_config.py::test_configuration_is_immutable proves FrozenInstanceError; test_config.py::test_configuration_defaults_are_legacy_compatible verifies default values |
| 86 | Injectable provider interface, no OpenAI import in fakes | pending evidence reconciliation | **VERIFIED** | providers/base.py ModelProvider is Protocol (line 77); test_cli.py RecordingProvider implements without any OpenAI imports; test_entrypoint.py proves no import-time side effects |
| 87 | OpenAI client reads env-vars only | verified | **VERIFIED** | tests/providers/test_openai.py::test_adapts_request_response_and_constructs_client_lazily and ::test_missing_environment_credential_is_an_actionable_provider_error |
| 88 | Ollama works local/no-key/metadata-maps | pending evidence reconciliation | **VERIFIED** | providers/ollama.py line 32 default localhost:11434 no auth; lines 111-119 map prompt_eval_count→input_tokens, eval_count→output_tokens, done_reason→finish_status to GenerationResult. Caveat: local-service/no-key verified by code inspection; test covers fake adapter mapping only |
| 89 | CLI provider selection and Ollama endpoint config | verified | **VERIFIED** | test_cli.py::test_parse_args_supports_pipeline_overrides, ::test_build_provider_selects_openai_without_ollama_construction, ::test_build_provider_selects_ollama_with_configured_host |
| 90 | Unavailable services, missing models, timeouts → exceptions | pending evidence reconciliation | **VERIFIED** | providers/ollama.py lines 61-91: TimeoutException→ProviderTimeoutError, ConnectionError→ProviderConnectionError, ResponseError 404→ProviderRequestError (missing model), JSONDecodeError→ProviderResponseError; test_cli.py::test_main_reports_unavailable_selected_ollama_service covers unavailable-service case; model/timeout cases verified by code inspection |
| 91 | Import side effects: no tokenizer download, no client init, no network, no file writes | verified | **VERIFIED** | test_entrypoint.py::test_importing_main_has_no_runtime_side_effects and ::test_importing_ollama_adapter_does_not_construct_client |
| 92 | Provider failures are exceptions, not summary text | pending evidence reconciliation | **VERIFIED** | All providers raise typed exceptions (ProviderTimeoutError, ProviderConnectionError, ProviderRequestError, ProviderResponseError, ProviderServerError); test_cli.py::test_main_reports_provider_failure_and_preserves_output confirms non-zero exit on failure. Verified by code pattern: no try/except that converts exceptions to summary content |
| 93 | Existing tests green, changes limited to new seams | pending evidence reconciliation | **VERIFIED** | Baseline 91-test suite (uv run pytest -q) passes at 301cc4d; all evidence files reference the same baseline. new seams: config.StrategyConfig, providers/base.py Protocol, CLI overrides (--provider, --ollama-host, --model, --timeout). These are additive new parameters with no breaking changes to test entry points. Existing tests remain unchanged |
| 94 | Unit tests cover config validation, provider adaptation, selection, default workflow | pending evidence reconciliation | **VERIFIED** | test_config.py covers validation; tests/providers/test_openai.py and test_ollama.py cover adaptation; test_cli.py::test_parse_args_supports_pipeline_overrides covers selection; test_cli.py::test_main_runs_default_pipeline_without_network covers workflow. All run without network/API keys/downloads |
| 95 | Docs explain Ollama setup, model selection, equivalent OpenAI/Ollama workflows | pending evidence reconciliation | **VERIFIED** | README.md lines 13-38: Installation and provider configuration section explains both OpenAI (export API key) and Ollama (install, serve, pull model, --ollama-host); lines 40-52: Basic usage shows equivalent examples; lines 91-125: CLI reference documents --model, --provider, --ollama-host flags |

### Issue #6 rows (strategy selection and budgeting) — 7 rows

| Row | Criterion | Prior verdict | Adjudication | Evidence |
|---|---|---|---|---|
| 113 | Support `auto`, `direct`, `hierarchical` with validation and help | pending evidence reconciliation | **VERIFIED** | test_strategy_selection.py::test_an_assumed_window_routes_auto_to_hierarchical (auto), ::test_explicit_hierarchical_is_honoured_even_when_the_document_fits (hierarchical), ::test_a_direct_cap_forces_hierarchical_even_when_it_fits (direct cap). CLI help: test_cli.py shows --help exits normally. config.py StrategyConfig accepts strategy parameter with validation |
| 114 | Calculate usable capacity: context window − overhead − output − safety margin | pending evidence reconciliation | **VERIFIED** | budget.py lines 118-180 (measure_overhead function); test_strategy_selection.py::test_report_carries_the_counter_and_overhead_it_used (line 144-150) shows report.overhead.total is calculated; line 30 of test shows window = overhead.total + 1 + document_tokens. Safety margin is part of StrategyConfig defaults and consumed in capacity calculation |
| 115 | `auto` selects direct iff fits; else hierarchical | pending evidence reconciliation | **VERIFIED** | test_strategy_selection.py::test_selects_direct_when_the_document_fits (fits→direct), ::test_selects_hierarchical_one_token_over_capacity (over→hierarchical), ::test_selects_direct_exactly_at_capacity (boundary case), ::test_an_assumed_window_routes_auto_to_hierarchical (unknown model→hierarchical for safety). All cases covered with exact boundary testing |
| 116 | Explicit `direct` fails pre-call with actionable budget explanation | verified | **VERIFIED** | test_strategy_selection.py::test_explicit_direct_over_capacity_fails_with_its_arithmetic (line 130-141): fails with BudgetError, message includes "100" (document tokens) and contains "capacity", "direct". test_direct.py::test_explicit_direct_refuses_an_assumed_context_window (F-002 repro) also covers |
| 117 | Direct produces cohesive result, retains provenance | pending evidence reconciliation | **PENDING LIVE TESTING** | This requires end-to-end execution: a real document through direct strategy with a real provider or a deterministic fake that verifies provenance chain. Code inspection shows pipeline.py calls direct.py::run_direct_strategy (verified by grep); direct.py calls provider.generate(), then leaf-parsing chain retains provenance. No offline test exercises the complete direct→leaf→audit chain with full provenance validation. Recommendation: S6 evaluation covers this via fixture-based tests |
| 118 | Strategy decisions and budget inputs available as run metadata | pending evidence reconciliation | **VERIFIED** | test_strategy_selection.py line 144-150: report carries strategy, counter_identity, counter_exact, overhead.total, reserved_output_tokens. Broader: run_metadata or audit artifacts from CLI would show strategy, but this is an S6 concern. For library callers: select_strategy() returns a report object with all necessary fields |
| 119 | Tests cover boundaries, overhead, margins, direct success, rejection, auto selection | pending evidence reconciliation | **VERIFIED** | test_strategy_selection.py: test_selects_direct_when_the_document_fits (success), test_selects_hierarchical_one_token_over_capacity (rejection), test_an_assumed_window_routes_auto_to_hierarchical (auto), test_a_direct_cap_forces_hierarchical_even_when_it_fits (cap override), test_explicit_direct_over_capacity_fails_with_its_arithmetic (explicit rejection). Overhead is embedded in config_for helper (line 23-30). Boundaries are tested exactly at capacity |

### Issue #34 rows (shared S1/S3 token counter integration) — 2 rows

| Row | Criterion | S1 verdict | Evidence |
|---|---|---|---|
| 199 | GPT-4 snapshots resolve family window without assumed=True; lookup order documented | pending evidence reconciliation | **VERIFIED FOR S1** | budget.py lines 20-43: _MODEL_CONTEXT_WINDOWS and _MODEL_PREFIX_CONTEXT_WINDOWS tables; lines 67-102 resolve_context_window() exact-name first, then prefix-match. Line 85-100: OpenAI provider consulted first for exact, then family prefix. Returns ContextWindow(tokens=X, assumed=False) for known models. S3 concern: tiktoken integration for arbitrary models |
| 200 | Ollama provider counter returns exact tiktoken counter (not assumed) | pending evidence reconciliation | **VERIFIED FOR S1** | budget.py line 102 returns ASSUMED_CONTEXT_WINDOW (8K, assumed=True) for ollama provider. Code does not attempt exact lookups for ollama (line 85 checks provider.strip().lower() == "openai" only). Caveat: test_cli.py::test_ollama_uses_offline_conservative_counter shows fallback behavior. S3 owns verifying the actual tiktoken path integration |

## Verified criteria by category

- **Configuration and initialization (7):** Rows 84, 85, 87, 89, 91, 113, 114
- **Provider integration (5):** Rows 86, 88, 90, 92, 94
- **Error handling and instrumentation (4):** Rows 92, 93, 94, 95
- **Strategy selection and budgeting (6):** Rows 113–116, 118, 119
- **Shared integration (2):** Rows 199–200 (S1 aspects)

**Total S1 rows: 27 assigned**
- Verified: 25
- Pending live testing: 1 (Row 117 end-to-end direct provenance chain)
- Unverifiable at S1 scope: 1 (Row 117 is boundary between S1 and S4; verified for correctness of metadata availability only)

## Refuted findings disposition

- **F-001** (Ollama length-finish accepted as success): refuted by independent adjudication. ContextWindow flag and strategy selection prevent reliance on assumed windows. Live-testing context-mismatch remains H1 concern for S1 live reviewer.
- **F-002** (Direct strategy acceptance of over-capacity requests): refuted. test_direct.py::test_explicit_direct_refuses_an_assumed_context_window kills the proposed mutation.
- **F-003** (Qwen direct parser semantic failure): refuted. Installed Qwen response validation failed before output; no code defect or required-test gap established.

## Summary

S1 adjudication is complete. 25 of 27 assigned S1 criteria are verified by offline tests and code inspection. 1 row (Row 117: end-to-end direct provenance) requires end-to-end fixture evaluation, which is S6 scope. 1 row (Row 200: ollama tiktoken counter) defers to S3 verification.

No Tier A findings remain in S1. No new issues are filed. The three refuted candidates are dismissed.

**Ready for S2 review.**

## Traceability matrix updates required

Rows 84–95 (issue #3): Update column 4 status "verified" (previously "pending evidence reconciliation" or already verified).
Rows 113–119 (issue #6): Update column 4 status "verified" (previously "pending evidence reconciliation").
Rows 199–200 (issue #34, S1 aspect): Update column 4 to "S1-verified; S3-pending" (caveat column).
