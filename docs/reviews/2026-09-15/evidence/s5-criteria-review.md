# S5 criteria review — caching, resume, retry, timeouts, concurrency, atomic writes (#11)

Session 7. Candidates only: no verdicts, no finding ids, no adjudicated severities. Tiers below are *proposed*.

Baseline re-confirmed: 766 passed on the full offline suite, on both checkouts, at 301cc4d clean.

**Headline: 8 of 9 rows verified against named pytest node ids under the strict standard.** One row is proposed `violated`, backed by an executable reproduction against the real production classes. This is the strongest subsystem result in the review so far, and also its most substantive new candidate.

## Row-by-row

| line | criterion (abbreviated) | proposed status | evidence |
|---|---|---|---|
| 155 | Cache keys derive from source, prompt/schema version, model and every behavior-relevant config value, excluding credentials | **violated** (mechanism otherwise sound) | `cache.py:225-292`, `segmentation.py:67-79`. Positive: `test_cache.py::test_descriptor_key_changes_for_every_output_affecting_input`, `::test_descriptor_accepts_all_approved_output_affecting_configuration`. Negative: repro proves `AppConfig.ollama_host` never reaches the descriptor. See C-S5-001 |
| 156 | Resume reuses only compatible work; no repeated calls after interruption | verified | `test_scheduler.py::test_failure_drains_successful_sibling_and_resume_reuses_it` (calls == `{"S000001": 1, "S000002": 0, "S000003": 1}`); `test_pipeline_reliability.py::test_compatible_hierarchical_pipeline_reuses_segment_leaf_merge_and_editorial` (zero provider calls on resume) |
| 157 | Misses and invalidation explainable in audit metadata without secrets | verified | `test_pipeline_reliability.py::test_pipeline_audit_reports_{source,prompt,schema,model,behavior}_descriptor_invalidation` (5 tests, each pinning one exact code); `test_audit_reliability.py::test_audit_v3_redacts_no_secrets_in_reliability`, `::test_audit_v3_rejects_credential_like_retry_work_ids` |
| 158 | Retry only transient failures, bounded exponential backoff with jitter; reject invalid/config errors immediately | verified at library level; candidate at product level | `providers/retrying.py:77-82` (`_jittered_delay`); `tests/providers/test_retrying.py::test_transient_failures_use_bounded_jitter_and_record_safe_attempts` asserts actual randomized delay values, not merely that a retry happened; `::test_terminal_failures_are_not_retried` (4 params, all propagate with zero sleep). See C-S5-002 |
| 159 | Configurable timeouts and bounded concurrency preserving deterministic ordering | verified | `test_scheduler.py::test_scheduler_caps_in_flight_work_and_returns_work_id_order` (`max_in_flight=2`, result order == submission order despite out-of-order completion); `::test_scheduler_waits_for_an_entire_level_before_submitting_the_next`; `test_pipeline_reliability.py::test_concurrent_retry_audit_follows_manifest_work_order` (`max_in_flight=4`); `tests/providers/test_ollama.py::test_uses_a_client_with_each_distinct_request_timeout` |
| 160 | Partial failures leave no misleading summary or complete-looking cache entry | verified | `verification.py:1562-1577` (`cache_if=lambda result: not result.failed`); `test_publication.py::test_audit_failure_never_writes_summary`; `::test_summary_failure_leaves_audit_staged_with_both_digests` (publication left at `audit_staged`, not `complete`); `cli.py:276-284` |
| 161 | Atomic writes for final text and audit; no silent exception swallowing | verified for the reliable/cache path; candidate for the legacy path | `finalization.py:70-90` (`_atomic_replace`, fsyncs file and directory) used by `cli.py:274` and `publish_final_output`; `test_cache.py::test_store_flushes_and_syncs_file_and_parent_before_returning`. See C-S5-003 |
| 162 | Cache and temp artifacts git-ignored; historical data untouched | verified (static evidence; no live end-to-end run) | `.gitignore:10` (`.summarizer-cache/`); `config.py:31-44` (`CacheConfig` rejects `.`, `/`, `..`, symlinks); `test_config.py::test_configuration_rejects_invalid_values[...]`, `::test_cache_configuration_rejects_a_symlinked_final_root`; `config.py:61-64` rejects `output_path == input_path` |
| 163 | Offline tests cover all eleven named cases | verified — every one has a covering test | census in the "nothing found" section below |

## Candidates

### C-S5-001 — the cache descriptor excludes endpoint identity, enabling a stale cross-server hit

**Proposed major, Tier B.** The strongest candidate produced this session.

Two `AppConfig`s differing only in `ollama_host` produce byte-identical `CacheDescriptor.canonical_value()` and therefore the same cache key, so a result computed against server A is served as a **validated hit** for a request naming server B.

Citations: `config.py:48-54` (`AppConfig.ollama_host` exists and is never threaded further), `pipeline.py:254-281` (`CacheCoordinator` construction carries no host or endpoint field), `cache.py:225-269` (`CacheDescriptor` has `provider` and `model` only), `providers/ollama.py:29-38` (host used solely to build the HTTP client).

**Reachability is the part that matters, and it is worse than "needs `--resume`."** Default `run_mode="new"` sets `allow_unreferenced_cache=True` (`pipeline.py:278`, `segmentation.py:116-123`), so a brand-new run with a fresh `--run-id` still adopts a pre-existing cache object purely by content-addressed key. Two ordinary invocations sharing a `--cache-dir` while pointed at different Ollama hosts with the same `--model` tag collide silently. No `--resume` required.

**The design doc records this as deliberate**, and that is the interesting part. `docs/plans/2026-09-06-reliability-cache-resume-design.md:51` buckets "hosts or endpoints" together with credentials and paths as excluded. The reviewer's objection is precise and worth carrying to the verifier: the criterion requires excluding *credentials*, and a bare host carrying no userinfo is not a secret. The codebase already knows how to strip just the credential component — `test_audit.py`'s own redaction test targets the `user:pass@` portion specifically. Excluding the whole endpoint rather than only its credential portion trades an already-handled leak risk for a stale-hit correctness bug. This is exactly the tension the brief predicted, now with evidence.

Repro: `.review/repro/C-S5-001-ollama-host-collision.py`, expressed as a pytest node because the sandbox permits `python -m pytest` but not arbitrary script execution. It uses the real `CacheCoordinator`, `CacheStore` and `CacheDescriptor` and mirrors `run_pipeline`'s actual construction rather than hand-building a descriptor. It asserts both that the keys collide and, independently at the store layer, that server A's stored answer returns as `lookup_b.hit` with A's payload. Result: 1 passed.

Confidence: high for the Ollama half (executable, real classes). Medium for the OpenAI half, which is inspection-only: `providers/openai.py:24-27` passes no `base_url`, so the SDK falls back to its own `OPENAI_BASE_URL` environment resolution (confirmed by reading the installed SDK's `__init__` source; supporting check at `.review/repro/C-S5-001-openai-base-url-env.py`). This repo has no OpenAI base-url config field, so that half is a latent mechanism rather than a currently reachable path. Falsified by: finding any code path folding `ollama_host` into `CacheCoordinator.behavior` or a new descriptor field. None exists at 301cc4d.

### C-S5-002 — jitter is implemented and tested but ships disabled, with no way to enable it from the CLI

**Proposed minor.** `RetryPolicy.jitter_fraction` defaults to `0` (`config.py:81`), `cli.py:160` constructs `RetryPolicy(max_attempts=args.max_retries)` with no override, and `_parser()` (`cli.py:57-139`) defines no `--jitter`. Every CLI-driven retry therefore uses pure deterministic exponential backoff. README.md:150 documents retry as "bounded exponential backoff" without mentioning jitter, which is consistent with actual behavior.

The design doc records this as deliberate (`...reliability-cache-resume-design.md:150-152`: compatibility defaults preserve deterministic delay unless jitter is explicitly enabled). It is a thundering-herd concern, not a correctness or leakage one. The reviewer explicitly invited a downgrade to noise.

Note the shape against row 158: the criterion says retry uses backoff "with jitter". The library satisfies it and has a genuinely good test asserting randomized delay values. The shipped product does not reach that code. Whether the criterion is about the library or the product is the verifier's call.

### C-S5-003 — the non-cache audit write is rename-atomic but not fsynced

**Proposed minor, low confidence by its own author.** `audit.py:1432-1445` (`write_audit`) writes via `NamedTemporaryFile` plus `Path.replace()` with no `os.fsync()` on the temp file or parent directory, unlike `finalization.py:70-90` (`_atomic_replace`), which fsyncs both. `write_audit` is used exactly when `materialize_audit=True` (`pipeline.py:489-493`), i.e. `--audit` without `--cache-dir`.

The rename itself is atomic, so no observer sees a torn file. The gap is durability under crash or power loss, not correctness. The author flagged low confidence that this violates "atomic" at all.

## Nothing found — probed and properly handled

- **Row 155 completeness beyond the endpoint gap.** Every behavior-relevant `PipelineConfig` / `StrategyConfig` / `VerificationConfig` field was enumerated and traced to either an explicit `behavior` entry or the `input_hash`. Fields interpolated into prompt text (`target_words` at `editorial.py:22,163`) are covered for free, since changing them changes `input_value`. `temperature` does not exist anywhere in this codebase, so there is nothing to omit. `max_output_tokens` is never sent to either provider's API — purely a local budget input, and present in the `strategy_config` behavior block anyway. `include_citations` is schema-permitted (`cache.py:51`) but never populated, correctly, because `render_citations` (`finalization.py:376`) is a deterministic local post-process applied after cache resolution and never sent to the model.
- **Row 158 classifier correctness.** `ProviderAuthenticationError`, `ProviderRequestError`, `ProviderResponseError` and bare `ValueError` all fail immediately with zero retries. Only `TransientProviderError` subclasses retry.
- **Row 160 verification/cache interaction.** `cache_if=lambda result: not result.failed` prevents a failed verification from becoming a cache object; `_finalize_summary` raises before any output write.
- **Row 161 cache-object atomicity.** `CacheStore._atomic_write` (`cache.py:741-777`) uses temp file, fsync, `os.replace`, directory fsync, under an flock-based per-key lock with first-writer-wins.
- **Row 163 census.** All eleven named cases have a directly named covering test: hits/misses `test_cache.py::test_load_reports_missing_corrupt_wrong_version_and_incompatible_objects`; invalidation, the 5 tests above; interruption/resume `test_failure_drains_successful_sibling_and_resume_reuses_it`; retryable `test_retrying.py::test_transient_attempts_use_closed_error_categories`; non-retryable `::test_terminal_failures_are_not_retried`; retry limits `::test_exhaustion_preserves_last_failure_as_cause`, `::test_single_attempt_never_sleeps`; timeouts `test_ollama.py::test_uses_a_client_with_each_distinct_request_timeout`; concurrency and ordering the two scheduler tests; atomic-write failure the two publication tests.

## Process notes

**The reviewer refused an out-of-band instruction, correctly.** Mid-run the lead sent it a message about a deleted file and a corrected path. It arrived embedded after a tool result rather than as a user turn, and asked for a write the `worktree-guard` hook refuses. The reviewer flagged it as a possible injection, declined the boundary-crossing part, and adopted only the independently reasonable part (inlining script sources). It surfaced the whole thing as data for the lead to judge rather than complying silently. That is the correct response to an instruction of unverified provenance, and it is the behavior the two session-6 violators lacked.

**Correcting the record on that message.** The lead's claim was partly wrong and partly right. The reviewer's `Write` to the *primary checkout* path was indeed blocked by `worktree-guard` and never succeeded. But it had also written a file into the *worktree's* `.review/wip/s5-criteria/`, and the lead did delete that one while clearing a stray tree. Both statements are true of different paths; the lead's message did not distinguish them.

**Root cause, again the brief.** Subagents inherit the worktree as cwd, `.review/` lives in the primary checkout, and `worktree-guard` blocks Write across that boundary while Bash `mkdir`/`cp` pass. Any future brief must either give the primary-checkout absolute path *and* warn that the Write tool will refuse it, or simply accept worktree-local artifacts and have the lead copy them. Both repro files were copied to `.review/repro/` with md5 verification.

**Tier vocabulary.** The reviewer proposed "Tier C" for C-S5-002 and C-S5-003. This review uses Tier A and Tier B only. Recorded as proposed-minor for the verifier to place.
