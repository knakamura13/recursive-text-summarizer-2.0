# Phase 1: Critical Blockers — Implementation Guide

**Target**: Fix defaults to restore hierarchical merge functionality under real token counts

---

## Issue #26: Fix default GroundingPolicy reserve

### Problem
Default `GroundingPolicy(max_tokens=1024)` leaves **zero tokens** for merge overhead on gpt-4o-mini, which requires ~5,048 tokens for a single merge request (headers, formatting, stop tokens).

With 96K usable input capacity:
- Reserve: 1,024 tokens
- Available for merge overhead: 0 tokens ❌
- **Merge fails with truncation error**

### Solution
Reduce reserve to ~512 tokens, leaving ~1,500-token cushion for merge overhead:

**File**: `summarizer/hierarchy.py:28`

**Before**:
```python
DEFAULT_GROUNDING_POLICY = GroundingPolicy(max_tokens=1024)
```

**After**:
```python
DEFAULT_GROUNDING_POLICY = GroundingPolicy(max_tokens=512)
```

### Verification
Run the capacity probe after change:
```bash
uv run --with-requirements requirements-dev.txt python tests/probes/capacity_validator.py
```

Expected output: ✅ Merge overhead fits within available capacity (1500+ tokens available).

### Testing
Add this test case to `tests/test_pipeline.py`:

```python
def test_default_grounding_policy_allows_merge_overhead():
    """Verify default GroundingPolicy leaves capacity for merge overhead."""
    from summarizer.hierarchy import DEFAULT_GROUNDING_POLICY
    from summarizer.pipeline import _hierarchical_capacity
    
    # Simulate gpt-4o-mini context
    config = PipelineConfig(model="gpt-4o-mini")
    report = get_model_context_report("gpt-4o-mini")
    
    capacity = _hierarchical_capacity(report, DEFAULT_GROUNDING_POLICY)
    overhead = 5048  # Measured OpenAI merge overhead
    
    assert capacity.usable_input_capacity >= overhead, \
        f"Merge overhead {overhead} exceeds usable capacity {capacity.usable_input_capacity}"
```

---

## Issue #27: Unify Punkt tokenizer across modules

### Problem
Two separate tokenizer instances with divergent abbreviation sets cause segmentation boundary mismatches:

- `summarizer/text.py`: Punkt seeded with 20 abbreviations
- `summarizer/segmentation.py`: Bare Punkt with empty abbreviation set

**Example divergence**:
```
Text: "Dr. Smith reviewed the U.S. market."

text.py:         3 sentences: ["Dr. Smith reviewed the U.S. market."]
segmentation.py: 5 sentences: ["Dr.", "Smith reviewed the U.S.", "market."]
```

This causes capacity estimates to diverge from actual segment counts.

### Solution

**File 1**: `summarizer/text.py` — Extract abbreviation set

**Before**:
```python
PUNKT_TOKENIZER = PunktSentenceTokenizer()
PUNKT_TOKENIZER.abbreviation.update({
    "Dr", "Mr", "Mrs", "Ms", "Prof", "U.S", "U.K", "Ph.D",
    "e.g", "i.e", "etc", "vs", "Inc", "Ltd", "Corp", "Co",
    "St", "Ave", "Blvd", "Dept", "Jan", "Feb", "Dec"
})
```

**After** (extract to module-level constant):
```python
# At top of file, before tokenizer creation
PUNKT_ABBREVIATIONS = {
    "Dr", "Mr", "Mrs", "Ms", "Prof", "U.S", "U.K", "Ph.D",
    "e.g", "i.e", "etc", "vs", "Inc", "Ltd", "Corp", "Co",
    "St", "Ave", "Blvd", "Dept", "Jan", "Feb", "Dec"
}

PUNKT_TOKENIZER = PunktSentenceTokenizer()
PUNKT_TOKENIZER.abbreviation.update(PUNKT_ABBREVIATIONS)
```

**File 2**: `summarizer/segmentation.py` — Import and use unified set

**Before**:
```python
def _build_tokenizer():
    return PunktSentenceTokenizer()  # ← No abbreviations
```

**After**:
```python
from summarizer.text import PUNKT_ABBREVIATIONS

def _build_tokenizer():
    tokenizer = PunktSentenceTokenizer()
    tokenizer.abbreviation.update(PUNKT_ABBREVIATIONS)
    return tokenizer
```

### Verification
Run the tokenizer divergence test:
```bash
uv run --with-requirements requirements-dev.txt python tests/probes/tokenizer_tester.py
```

Expected output: ✅ Both instances produce identical splits on test corpus.

### Testing
Add to `tests/test_segmentation.py`:

```python
def test_punkt_tokenizer_abbreviations_match_text_module():
    """Verify segmentation uses same abbreviations as text module."""
    from summarizer.text import PUNKT_TOKENIZER as text_tokenizer
    from summarizer.segmentation import _build_tokenizer as build_seg_tokenizer
    
    seg_tokenizer = build_seg_tokenizer()
    
    # Compare abbreviation sets
    text_abbr = text_tokenizer.abbreviation
    seg_abbr = seg_tokenizer.abbreviation
    
    assert text_abbr == seg_abbr, \
        f"Mismatch: text={text_abbr}, segmentation={seg_abbr}"
    
    # Compare splits on abbreviation-bearing text
    test_text = "Dr. Smith reviewed the U.S. market. What next?"
    text_sentences = text_tokenizer.tokenize(test_text)
    seg_sentences = seg_tokenizer.tokenize(test_text)
    
    assert len(text_sentences) == len(seg_sentences), \
        f"Sentence count mismatch: text={len(text_sentences)}, seg={len(seg_sentences)}"
```

---

## Issue #28: Validate hierarchical merge capacity before request

### Problem
Merge requests are created without checking whether usable capacity can accommodate merge overhead + final editorial pass. This can cause:

1. Silent truncation (if provider is configured to truncate)
2. Runtime error (if provider rejects truncation)
3. Incomplete summary (if truncation goes undetected)

### Solution

**File**: `summarizer/pipeline.py` — Add pre-merge validation

**Location**: Before creating merge request in `_hierarchical_capacity()` or its caller

**Before**:
```python
def _run_hierarchical_merge(segments, config):
    # ... build merge request ...
    response = provider.generate(merge_request)
    return response
```

**After**:
```python
def _run_hierarchical_merge(segments, config):
    # Validate capacity BEFORE sending request
    overhead = 5048  # Measured for gpt-4o-mini; make configurable per model
    usable = _calculate_usable_capacity(config)
    
    if usable < overhead:
        raise ValueError(
            f"Insufficient capacity for merge overhead: "
            f"usable={usable}, required={overhead}"
        )
    
    # ... build merge request ...
    response = provider.generate(merge_request)
    return response
```

### Calculation Reference
For gpt-4o-mini (96K context):
- Model max: 131,072 tokens
- Reserved output: 4,096 tokens (README spec)
- Usable input: 131,072 − 4,096 = 126,976 tokens
- With default grounding (512): 126,976 − 512 = 126,464 tokens available
- Merge overhead measured: ~5,048 tokens
- Cushion: 126,464 − 5,048 = 121,416 tokens ✅

### Testing
Add to `tests/test_pipeline.py`:

```python
def test_hierarchical_merge_validates_capacity():
    """Verify merge capacity is validated before request."""
    from summarizer.pipeline import _run_hierarchical_merge
    from summarizer.config import PipelineConfig
    
    # Create config with low model capacity
    config = PipelineConfig(
        model="small-model",  # Hypothetical, ~8K context
        grounding_policy=GroundingPolicy(max_tokens=512)
    )
    
    # Create segments that would exceed capacity
    segments = ["text" * 2000] * 10  # Simulate large merge scenario
    
    # Should raise ValueError, not silently fail
    with pytest.raises(ValueError, match="Insufficient capacity"):
        _run_hierarchical_merge(segments, config)
```

---

## Implementation Checklist

### Pre-Implementation
- [ ] Create feature branch: `git checkout -b fix/critical-blockers-26-27-28`
- [ ] All three issues should be fixed in a single PR for coherence

### Implementation
- [ ] **Issue #26**: Reduce `DEFAULT_GROUNDING_POLICY` reserve to 512
- [ ] **Issue #27**: Extract `PUNKT_ABBREVIATIONS`, unify across modules
- [ ] **Issue #28**: Add capacity validation before merge request

### Testing
- [ ] Run capacity probe: `python tests/probes/capacity_validator.py`
- [ ] Run tokenizer tester: `python tests/probes/tokenizer_tester.py`
- [ ] Add unit tests for each issue
- [ ] Run full test suite: `pytest tests/`
- [ ] Verify no regressions in green tests (510 tests should still pass)

### Verification
- [ ] End-to-end test: Summarize a document with `--strategy hierarchical`
- [ ] Verify output is not truncated (compare to mocked baseline)
- [ ] Run capacity validator on real model (gpt-4o-mini) if available

### Documentation
- [ ] Update `README.md:151-161` to document grounding reserve
- [ ] Add comment to `DEFAULT_GROUNDING_POLICY` explaining the 512-token choice

### Commit & Push
- [ ] Commit with message:
  ```
  fix(hierarchy,segmentation,pipeline): Fix critical defaults for hierarchical merge
  
  - Reduce GroundingPolicy reserve from 1024 to 512 tokens (#26)
  - Unify Punkt tokenizer abbreviations across text.py and segmentation.py (#27)
  - Add capacity validation before hierarchical merge requests (#28)
  
  These changes restore hierarchical merge functionality under real token
  counts on gpt-4o-mini and fix segmentation boundary divergence that caused
  capacity miscalculation.
  ```
- [ ] Push to origin and create PR

---

## Success Criteria

After Phase 1 implementation:

✅ Default `PipelineConfig()` with `--strategy hierarchical` works without truncation errors  
✅ Punkt tokenizer produces identical splits across text.py and segmentation.py  
✅ Capacity validator catches insufficient-capacity scenarios before request creation  
✅ All 510 existing tests still pass  
✅ New unit tests for #26, #27, #28 pass  

---

## Probe Scripts Reference

Two helper scripts have been created to validate Phase 1 fixes:

### `tests/probes/capacity_validator.py`
Calculates usable capacity for each model and verifies merge overhead fits.

**Usage**:
```bash
python tests/probes/capacity_validator.py
```

### `tests/probes/tokenizer_tester.py`
Compares sentence splits from text.py and segmentation.py on a corpus of abbreviation-bearing text.

**Usage**:
```bash
python tests/probes/tokenizer_tester.py
```

Both probe scripts output **reproducible, non-mocked results** suitable for verification loops.
