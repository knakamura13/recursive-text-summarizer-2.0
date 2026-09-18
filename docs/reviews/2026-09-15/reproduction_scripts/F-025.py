"""C-S3M-002 verification repro.

Question: can `fencing` be negative pre-clamp (summarizer/budget.py:146,
`fencing=max(fencing, 0)`) with a REAL `TiktokenCounter`, reached through the
real `measure_overhead` call path?

This script does NOT hand-build a fake counter or a fake `monotonic=False`
stub. It uses the real `summarizer.tokenization.TiktokenCounter` class,
constructed through its real classmethods (`for_model` / `for_encoding`,
exactly as `resolve_token_counter` constructs it in production), and the real
`summarizer.budget._overhead_probe_segment` + `summarizer.leaf.build_leaf_request`
functions that `measure_overhead` itself calls.

Because `measure_overhead` does not expose the pre-clamp value directly (the
clamp is inline in the return statement), the pre-clamp value is reconstructed
by executing the *exact same expression* found at summarizer/budget.py:142
(`counter.count(request.input_text) - counter.count(probe.text)`) outside the
clamp, using the same real objects `measure_overhead` would have used. This is
disclosed as a reconstruction, not an instrumented read of the live function.
A cross-check against the real `measure_overhead()` return value is included:
whenever the reconstructed pre-clamp value is >= 0, it must equal
`overhead.fencing` exactly, which validates the reconstruction is faithful.

Coverage of "every model name the codebase can resolve": the only production
caller of `resolve_token_counter` is `summarizer/cli.py:54`, which passes
`model=config.model` and never `encoding_name`. `TiktokenCounter.for_model`
delegates to `tiktoken.encoding_for_model`, which (in the installed tiktoken
0.14.0) maps every recognized model string down to exactly one of 7 encodings
via `tiktoken.model.MODEL_TO_ENCODING` / `MODEL_PREFIX_TO_ENCODING`:
gpt2, r50k_base, p50k_base, p50k_edit, cl100k_base, o200k_base, o200k_harmony.
Since `TiktokenCounter.count` only calls `self.encoding.encode_ordinary`, the
result depends only on which of these 7 Encoding objects is selected, not on
the original model string. Testing all 7 encodings directly (via
`TiktokenCounter.for_encoding`, the same classmethod `resolve_token_counter`
uses for the `encoding_name` override path) is therefore exhaustive over every
encoding reachable through this codebase's production paths, regardless of
which model name a caller types.
"""

import json
import sys

sys.path.insert(
    0,
    "/Users/kylenakamura/documents-local/development-local/side-projects/recursive-text-summarizer",
)

import tiktoken

from summarizer.budget import _overhead_probe_segment, measure_overhead
from summarizer.leaf import build_leaf_request
from summarizer.tokenization import TiktokenCounter, TokenAccountingError

ALL_ENCODINGS = tiktoken.list_encoding_names()

# A representative model name per encoding, taken straight from
# tiktoken.model.MODEL_TO_ENCODING / MODEL_PREFIX_TO_ENCODING (0.14.0) and
# from this codebase's own _MODEL_CONTEXT_WINDOWS / _MODEL_PREFIX_CONTEXT_WINDOWS
# tables (summarizer/budget.py:20-43), so the `for_model` path is exercised
# too, not just `for_encoding`.
REPRESENTATIVE_MODELS = {
    "gpt2": "gpt2",
    "r50k_base": "davinci",
    "p50k_base": "text-davinci-003",
    "p50k_edit": "text-davinci-edit-001",
    "cl100k_base": "gpt-4",
    "o200k_base": "gpt-4o",
    "o200k_harmony": "gpt-oss-20b",
}

# Every model name literal this codebase's own tables name explicitly.
CODEBASE_MODEL_NAMES = [
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4.1",
    "gpt-5",
    "o1",
    "o1-mini",
    "o3",
    "o4",
    "o4-mini",
]


def reconstruct_pre_clamp_fencing(counter, *, with_overlap: bool) -> int:
    """Literally re-executes summarizer/budget.py:136-142, minus the clamp."""
    probe = _overhead_probe_segment(with_overlap=with_overlap)
    request = build_leaf_request(probe, model="probe", timeout_seconds=1)
    return counter.count(request.input_text) - counter.count(probe.text)


def main() -> None:
    print(f"tiktoken version: {tiktoken.__version__}")
    print(f"tiktoken.list_encoding_names(): {ALL_ENCODINGS}")
    print()

    results = []

    print("=== Part 1: every encoding tiktoken.list_encoding_names() names, ===")
    print("=== via TiktokenCounter.for_encoding (production classmethod) ===")
    for encoding_name in ALL_ENCODINGS:
        try:
            counter = TiktokenCounter.for_encoding(encoding_name)
        except TokenAccountingError as error:
            print(f"{encoding_name}: UNAVAILABLE OFFLINE ({error})")
            continue
        for with_overlap in (False, True):
            pre = reconstruct_pre_clamp_fencing(counter, with_overlap=with_overlap)
            post = measure_overhead(counter, with_overlap=with_overlap).fencing
            consistent = (post == pre) if pre >= 0 else (post == 0)
            results.append((encoding_name, with_overlap, pre, post, consistent))
            flag = "NEGATIVE <---" if pre < 0 else ""
            print(
                f"  encoding={encoding_name:<14} with_overlap={with_overlap!s:<5} "
                f"pre_clamp_fencing={pre:>4}  post_clamp_fencing={post:>4}  "
                f"reconstruction_consistent={consistent}  {flag}"
            )

    print()
    print("=== Part 2: every model name literal in this codebase's own tables, ===")
    print("=== via TiktokenCounter.for_model (the real resolve_token_counter path) ===")
    for model in CODEBASE_MODEL_NAMES:
        try:
            counter = TiktokenCounter.for_model(model)
        except TokenAccountingError as error:
            print(f"{model}: UNAVAILABLE / UNREGISTERED ({error})")
            continue
        for with_overlap in (False, True):
            pre = reconstruct_pre_clamp_fencing(counter, with_overlap=with_overlap)
            post = measure_overhead(counter, with_overlap=with_overlap).fencing
            flag = "NEGATIVE <---" if pre < 0 else ""
            print(
                f"  model={model:<16} encoding={counter.identity:<22} "
                f"with_overlap={with_overlap!s:<5} pre_clamp_fencing={pre:>4}  "
                f"post_clamp_fencing={post:>4}  {flag}"
            )

    print()
    print("=== Part 3: representative model per encoding, via for_model ===")
    for encoding_name, model in REPRESENTATIVE_MODELS.items():
        try:
            counter = TiktokenCounter.for_model(model)
        except TokenAccountingError as error:
            print(f"{model} ({encoding_name}): UNAVAILABLE / UNREGISTERED ({error})")
            continue
        assert counter.identity == f"tiktoken:{encoding_name}", (
            f"expected {encoding_name}, got {counter.identity}"
        )
        for with_overlap in (False, True):
            pre = reconstruct_pre_clamp_fencing(counter, with_overlap=with_overlap)
            flag = "NEGATIVE <---" if pre < 0 else ""
            print(
                f"  model={model:<20} encoding={encoding_name:<14} "
                f"with_overlap={with_overlap!s:<5} pre_clamp_fencing={pre:>4}  {flag}"
            )

    print()
    negatives = [r for r in results if r[2] < 0]
    print(f"Total (encoding, with_overlap) combinations tested in Part 1: {len(results)}")
    print(f"Combinations with NEGATIVE pre-clamp fencing: {len(negatives)}")
    all_consistent = all(r[4] for r in results)
    print(f"All reconstructions consistent with real measure_overhead() output: {all_consistent}")

    print()
    print("=== Part 4: magnitude / causal-chain context (assuming best case for candidate) ===")
    # Show the actual template strings being measured, so the character-count
    # vs token-count relationship is visible rather than asserted.
    for with_overlap in (False, True):
        probe = _overhead_probe_segment(with_overlap=with_overlap)
        request = build_leaf_request(probe, model="probe", timeout_seconds=1)
        print(f"--- with_overlap={with_overlap} ---")
        print(f"  probe.text = {probe.text!r} ({len(probe.text)} chars)")
        print(f"  request.input_text = {request.input_text!r} ({len(request.input_text)} chars)")
        print(
            f"  char delta (fenced - bare) = "
            f"{len(request.input_text) - len(probe.text)} chars added by fencing"
        )


if __name__ == "__main__":
    main()
