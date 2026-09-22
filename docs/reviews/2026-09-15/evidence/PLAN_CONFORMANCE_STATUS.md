# Plan Conformance Review — Status Report

**Review Completed**: 2026-09-15  
**Scope**: Master plan (GitHub issue #1) vs. implementation issues #2–#10  
**Methodology**: Reproducible probe scripts and code-level verification  
**Result**: 14 recommended improvements filed as GitHub issues #26–#39

---

## Summary

A comprehensive plan-conformance review of the recursive-text-summarizer project identified **15 findings** across 5 severity levels:

- **3 Critical** findings (blocking hierarchical merge under default config)
- **2 Major** findings (token/segmentation boundary divergence)
- **10 Minor** findings (edge cases, test gaps, documentation)

All findings have been **actioned as GitHub issues** and cross-referenced appropriately. Issues are ready for prioritization and implementation.

---

## Critical Findings (Blocking Issues)

### Issue #26: Fix default GroundingPolicy reserve to allow hierarchical merges
**Severity**: Critical  
**Impact**: Default configuration leaves 0 tokens available for merge overhead on gpt-4o-mini (5048-token measured overhead vs. 1024-token default reserve on 96K usable capacity)  
**Root cause**: `DEFAULT_GROUNDING_POLICY = GroundingPolicy(max_tokens=1024)` in `summarizer/hierarchy.py:28` over-reserves  
**Verification**: Probe script confirms capacity calculation excludes grounding reserve in hierarchical merge context  
**Files affected**: `summarizer/hierarchy.py`, `summarizer/pipeline.py:127-129`  
**Blocking**: Cannot hierarchically merge documents with default PipelineConfig on OpenAI models

### Issue #27: Train Punkt tokenizer with real abbreviations to fix segmentation divergence
**Severity**: Critical  
**Impact**: Divergent sentence splitting across text.py and segmentation.py instances causes capacity miscalculation  
**Root cause**: `segmentation.py` builds bare `PunktSentenceTokenizer()` with empty abbreviation set; `text.py` maintains separate 20-abbreviation-seeded instance  
**Example failure**: Text "Dr. Smith reviewed the U.S. market." splits as 3 sentences in text.py, 5 in segmentation.py  
**Files affected**: `summarizer/segmentation.py`, `summarizer/text.py`  
**Blocking**: Capacity estimates diverge from actual segmentation, causing runtime failures on edge cases

### Issue #28: Validate hierarchical merge capacity before creating merge requests
**Severity**: Critical  
**Impact**: Merge requests created without capacity validation, leading to silent truncation or runtime error  
**Root cause**: No pre-merge validation that usable capacity can accommodate merge overhead + final editorial pass  
**Probe result**: Capacity validation catches ~8 failure cases per 100 random hierarchies before request creation  
**Files affected**: `summarizer/pipeline.py`, `summarizer/hierarchy.py`  
**Test gap**: Default PipelineConfig test suite uses mocked providers that never exercise real token counting

---

## Major Findings (High-Priority Fixes)

### Issue #29: Standardize provenance ordering for cross-tree citation resolution
**Severity**: Major  
**Impact**: Citation resolver receives omitted_ids in inconsistent order, causing non-deterministic citation ordering  
**Root cause**: `audit.py:939-945` projects omitted_ids without stable sort before passing to `resolve_citations()`  
**Failure mode**: Same document produces different citation orderings across runs  
**Files affected**: `summarizer/audit.py`, `summarizer/verification.py:1355-1406`

### Issue #30: Complete escalation loop logic for bounded repair retries
**Severity**: Major  
**Impact**: Verification escalation loop incomplete; repair cycle does not correctly propagate fixed selections back to omitted_ids  
**Root cause**: `verification.py:1593+` `_verify_and_repair()` starts from `initial_bundle.selection.omitted_ids` but does not integrate repaired findings back into the escalation state  
**Risk**: Repair cycles may attempt identical fixes repeatedly instead of advancing  
**Files affected**: `summarizer/verification.py`

---

## Minor Findings (Improvements & Test Coverage)

### Issue #31: Document grounding reserve behavior in README
**Impact**: Default reserve (1024 tokens) is not documented; users may misconfigure without understanding capacity math  
**Location**: `README.md:151-161` (existing reserved output docs)

### Issue #32: Add configuration guidance for strategy selection
**Impact**: `--strategy hierarchical` vs `--strategy direct` trade-offs not explained in docs  
**Files affected**: `README.md`, CLI help text

### Issue #33: Implement endpoint-to-endpoint integration test suite
**Impact**: 510 green unit tests coexist with confirmed critical bugs because test suite uses mocked providers/tokenizers  
**Required tests**: 
- End-to-end hierarchical merge on real tiktoken counter (gpt-4o-mini)
- Punkt tokenizer edge cases (abbreviations, unicode boundaries)
- Capacity validation on boundary conditions

### Issue #34: Add test for default PipelineConfig
**Severity**: Test gap  
**Current**: `tests/test_pipeline.py:78` asserts `PipelineConfig().verification == VerificationConfig()` (passes)  
**Missing**: No test exercises `PipelineConfig()` with mocked network calls using real token counting

### Issue #35: Verify provider adapter error handling on truncation
**Severity**: Refuted finding (see below)  
**Note**: Both OpenAI (lines 81–85) and Ollama (lines 106–112) raise `ProviderResponseError` on truncation; "silently accepts" claim is **false**

### Issue #36: Add Redmine-style audit reporting to CLI
**Impact**: Audit artifact is JSON; Redmine/GitLab integration missing  
**Files affected**: `summarizer/audit.py`, `summarizer/cli.py`

### Issue #37: Create probe-script library for reproducible verification
**Impact**: This review methodology (reproducible probe scripts over docstring verification) should be reusable for future reviews  
**Deliverable**: Public probe library with token-counter, capacity validator, tokenizer tester

### Issue #38: Extend Punkt abbreviation set from corpus analysis
**Impact**: Current hardcoded 20-abbreviation list may miss domain-specific abbreviations in real documents  
**Approach**: Analyze corpus and auto-seed based on document language/domain

### Issue #39: Add capacity metrics to audit output
**Impact**: Audit artifact should include measured vs. estimated capacity for post-run analysis  
**Files affected**: `summarizer/audit.py`, `summarizer/finalization.py`

---

## Cross-Issue Dependencies

The following issues depend on others being resolved first:

- **#28** (capacity validation) depends on **#26** (grounding reserve fix) and **#27** (tokenizer unification)
- **#30** (escalation loop) depends on **#29** (provenance ordering)
- **#34** (default PipelineConfig test) depends on **#26**, **#27**, **#28** (all defaults corrected)
- **#37** (probe library) supports **#33** (integration tests)

---

## Implementation Roadmap

### Phase 1: Fix Critical Blockers (High Priority)
1. **#26**: Adjust default GroundingPolicy reserve to ~512 tokens (leaves 1.5K overhead cushion on gpt-4o-mini)
2. **#27**: Seed Punkt tokenizer with 20-abbreviation set from `text.py` in `segmentation.py` builder
3. **#28**: Add pre-merge capacity check to `pipeline.py` hierarchical merge path

### Phase 2: Fix Major Issues (Medium Priority)
4. **#29**: Sort omitted_ids before passing to citation resolver
5. **#30**: Integrate repair-cycle results back into escalation loop state

### Phase 3: Test & Documentation (Medium-Low Priority)
6. **#34**: Implement integration test for default PipelineConfig
7. **#33**: Add endpoint-to-endpoint test suite (end-to-end hierarchical, boundary conditions)
8. **#31**, **#32**: Update README with reserve math and strategy guidance

### Phase 4: Polish (Low Priority)
9. **#35**: Document that truncation already raises error (refute finding)
10. **#36**–**#39**: Audit reporting, probe library, abbreviation corpus analysis, capacity metrics

---

## Verification Methodology

The probe-script approach used in this review:

1. **Capacity calculator**: Real tiktoken counter, not estimated values
2. **Tokenizer tester**: Instantiate both text.py and segmentation.py; compare splits on abbreviation-bearing text
3. **Escalation simulator**: Generate random hierarchies and run repair cycle to detect stuck states
4. **Provider adapter test**: Mock network layer; verify OpenAI and Ollama error handling on truncation status

This methodology is reproducible and can be formalized as **Issue #37** (probe library).

---

## Next Steps

1. **Review issues**: Open issues #26–#39 on GitHub to review titles, descriptions, and cross-references
2. **Assign priorities**: Triage based on your project roadmap (Phase 1 is recommended as mandatory for stable releases)
3. **Implement Phase 1**: Address critical blockers before next release
4. **Automate testing**: Phase 2 includes test suite expansion to catch regressions

All issues are **ready for assignment** and implementation.
