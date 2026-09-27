# Chapter comparison protocol

This study uses private, local source and response artifacts. Do not commit `artifact_root` or distribute the book and transcript extracts. The Tim Ferriss transcript page limits sharing to 500 words with attribution and restricts commercial reuse.

## Inputs and isolation

`manifest.json` pins the three source revisions, canonical-text SHA-256 values and word counts, model digest, Ollama version, 32,768-token context, sampling settings, and seeds. Source provenance, exact PDF page indices, visual extraction checks, and defects are in `/private/tmp/rts-study-20260926/corpus/<source>.json`; source-backed checklists were written before viewing generated summaries. Word targets use whitespace-delimited words excluding synthetic PDF page-marker lines. The canonical text, including page markers, is unchanged for all versions.

The managed checkouts are `/private/tmp/rts-study-20260926/checkouts/{legacy,pre_experiments,current}`. The runner rejects changed commits, tracked or untracked checkout changes, changed source hashes/counts, missing Python dependencies, changed Ollama version, or changed model digest. Each case has a unique directory and fresh summarizer cache. A separate loopback capture proxy pins temperature 1, top_k 64, top_p 0.95, `think=false`, common per-trial seed, context 32,768, and `num_predict=max(2048, 3*target_words+1024)` on every generation call. Its private trace records original/effective requests, responses, status, and elapsed time. Generation is sequential.

Run with a Python environment satisfying the repo's `requirements.txt`. The paths in the manifest are workstation-specific; to repeat on another machine, use a fresh private artifact root, reproduce the validated canonical extracts there, update the manifest hashes/paths and prepare clean pinned checkouts before generation. Do not reuse an existing case directory:

```sh
python3 -m venv --system-site-packages /private/tmp/rts-study-20260926/venv
/private/tmp/rts-study-20260926/venv/bin/python -m pip install -r requirements.txt
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/run_legacy.py --fidelity-check
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/run_study.py --phase pilot
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/run_study.py --phase baseline
# Freeze the development conclusions before running held-out sources:
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/run_study.py --phase validation
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/collect_results.py --phase pilot --phase baseline --phase validation --combine-pilot
```

This workstation's first baseline and some pilot cases stopped at environment/proxy setup without controlled model output. Their original directories remain intact; the study used `pilot-retry` and `baseline-retry` with the pinned venv for independent corrected cases. To inspect the **recorded** 54-case baseline and 24-case validation without generating again, run:

```sh
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/collect_results.py --phase pilot --phase pilot-retry --phase baseline-retry --phase validation --combine-pilot
```

The full evidence-backed report is private at `/private/tmp/rts-study-20260926/results/report.md`; source texts, trace bodies and full summaries are private beside it.
`collect_results.py` creates private aggregate metrics, an undisclosed A/B/C mapping, and local side-by-side published summaries. It never treats a failed candidate draft as published. Before scoring, read only the anonymous side-by-side outputs and frozen source checklists. Score faithfulness, concept coverage, explanatory usefulness, and readability independently of words/calls/time. Review rejected claims against the original PDF, not merely an application verdict.

The UI is limited to 5,000 target words; the statistical-learning 50% case and transcript 50% case therefore use the library harness. The frontend NVC smoke used isolated app-data roots and browsers, not existing user runs. The first 25% current-version UI smoke omitted `num_predict` at transport because the adapter did not set it; the harness and subsequent UI smokes enforce the formula in the proxy, and that first UI smoke is not counted as a controlled trial.

The output-allowance rule is intentionally not capped by the context window. For the transcript 50% case, the 13,286-word target yields a 40,882-token allowance, which alone exceeds the 32,768-token context. Both modern versions therefore raise `BudgetError` at strategy selection before any model request. Those two held-out trials are recorded preflight-infeasibility outcomes, not missing data. Capping the allowance would silently lower the requested target and change the frozen protocol. Issues #105 and #107 own feasibility classification and any context-compatible budgeting.
