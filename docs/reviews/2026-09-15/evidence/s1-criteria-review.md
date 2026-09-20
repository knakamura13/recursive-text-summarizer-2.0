# S1 criteria-and-design review — 301cc4d

## Scope and method

Read the S1 traceability rows for #3, #6, #34, and #38, with #1/#12 treated
as cross-cutting only. Reviewed `pipeline.py`, `cli.py`, `config.py`,
`budget.py`, `tokenization.py`, `providers/base.py`, `providers/ollama.py`,
README, and the matching provider/strategy plans. This was a criteria/design
pass only: no tests were run, no mutation was made, and no live Ollama probe
was run (the S1 live reviewer owns H1/H2/H8).

The parent recorded a ready graph at the reviewed HEAD with no recorded skipped
or parse-partial source files. The graph MCP tools were not exposed to this
child session, so I used targeted source reads after that parent evidence; I
could not independently run `check_index_coverage`.

## Candidate returned for independent verification

### C-S1-CR-01 — Ollama length completion is accepted as a successful generation

* **Tier / proposed severity:** A / major
* **Claim:** A native Ollama response whose `done_reason` is `"length"` is
  accepted as a normal `GenerationResult`, so a silently context-truncated
  request can enter leaf/direct/hierarchy processing and produce a final
  summary instead of an actionable failure.
* **Evidence:** `summarizer/providers/ollama.py:93-118` requires only `done is
  True` and nonempty content, then stores `done_reason` as metadata;
  `summarizer/providers/base.py:38-60` validates neither an allowed terminal
  status nor its success meaning; `summarizer/pipeline.py:101-120,294-376`
  forwards successful results through normal execution; `docs/plans/2026-09-03-strategy-selection-design.md:27,135`
  documents silent Ollama prompt truncation as the reason direct fit must be
  proven.
* **Criterion:** #3 “Unavailable Ollama services, missing local models,
  malformed responses, and timeouts become actionable application exceptions”; #6
  “Direct … fails before a provider call when the request cannot fit”; #38
  default-pipeline failure behavior. This is specifically a post-request
  truncation failure that the documented budget defense cannot reliably prevent
  when the runtime context differs from the declared/explicit window.
* **Verifier repro:** Use the H1 live request with a deliberately smaller
  runtime `num_ctx` than the passed `--context-window`; verify
  `prompt_eval_count` is below the sent prompt and `done_reason="length"`, then
  observe whether the adapter/pipeline exits nonzero. A no-network unit refuter
  can inject an Ollama-shaped `done=True`, nonempty, `done_reason="length"`
  response and confirm whether `OllamaProvider.generate()` raises.
* **Why current tests missed it:** `tests/providers/test_ollama.py` pins
  mapping of `done_reason="stop"` but does not exercise `"length"`; no caller
  branches on `finish_status` (the audit only records it).

## Refuted or withheld ideas

* H1’s premise that the shipped budget uses an advertised 131K–262K window for
  arbitrary local Ollama tags is not supported by this source: `budget.py:67-102`
  resolves non-OpenAI tags to an *assumed* 8,192-token window, and
  `budget.py:286-355` prohibits explicit direct and selects hierarchy for an
  assumed window. The live reviewer should test the runtime-context mismatch,
  not report the premise unchanged.
* `--max-output-tokens` is only a reservation: `StrategyConfig` and the strategy
  plan expressly say it is not sent as an output cap. The CLI help likewise says
  “reserved … when sizing,” so this is an acknowledged design limitation, not a
  criteria/design finding from this pass.

## Limitations

No S1 product-wide criterion was marked verified: #1 and #12 are
cross-cutting, and this reviewer did not run the test, mutation, or live-probe
angles. The sole item above is a candidate, not a finding, until a separate
verifier writes and runs its repro. The live reviewer owns the actual Ollama
execution and should avoid changing the candidate’s claim without recording
the observed runtime context, prompt token count, and finish status.
