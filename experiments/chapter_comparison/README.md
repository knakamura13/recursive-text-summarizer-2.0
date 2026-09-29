# Chapter comparison protocol

This study uses private, local source and response artifacts. Do not commit `artifact_root` or distribute the book and transcript extracts. The Tim Ferriss transcript page limits sharing to 500 words with attribution and restricts commercial reuse.

## Inputs and isolation

`manifest.json` pins the three source revisions, canonical-text SHA-256 values and word counts, model digest, Ollama version, 32,768-token context, sampling settings, and seeds. Source provenance, exact PDF page indices, visual extraction checks, and defects are in `/private/tmp/rts-study-20260926/corpus/<source>.json`; source-backed checklists were written before viewing generated summaries. Word targets use whitespace-delimited words excluding synthetic PDF page-marker lines. The canonical text, including page markers, is unchanged for all versions.

The managed checkouts are `/private/tmp/rts-study-20260926/checkouts/{legacy,pre_experiments,current}`. The runner rejects changed commits, tracked or untracked checkout changes, changed source hashes/counts, missing Python dependencies, changed Ollama version, or changed model digest. Each case has a unique directory and fresh summarizer cache. A separate loopback capture proxy pins temperature 1, top_k 64, top_p 0.95, `think=false`, common per-trial seed and context 32,768 on every generation call. For the pinned revisions it fills in `num_predict=max(2048, 3*target_words+1024)` when a request carries none; a candidate's own `num_predict` is forwarded unchanged. Its private trace records original/effective requests, responses, status, and elapsed time. Beside it, `requests.jsonl` holds one text-free row per request: original and effective options, overridden keys, stop reason, token counts, Ollama error text, repeats of an identical earlier request and elapsed time. Generation is sequential.

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

## Working-tree replays (#107)

`--phase d5-replay` and `--phase d5-e2e` run this repository's working tree (`--version worktree`) with the pipeline's default request allowances, not a pinned checkout. `run.json` records the revision, whether `summarizer/` or `experiments/chapter_comparison/` had uncommitted changes, and the SHA-256 of `run_modern.py`, `capture_proxy.py` and `run_study.py`. `d5-replay` hands each saved failing trial's cached segmentation and leaf summaries to the current merge stage, so no leaf is regenerated: Gathering 25%, ISLR 25% and 50%, transcript 25%, all seed 101. Each saved cache object must match its payload digest. The segmentation must pass the cached-segmentation decoder's strict checks under the token counter and segment budget it was made with, and every leaf must pass the current provenance check. Replay requires `--strategy hierarchical`, the only path that uses saved leaves, and `replayed_leaves` is recorded only when the pipeline takes them. The saved transcript 50% trial stopped before any leaf, so it runs fresh and records the configuration refusal. `run.json` adds the terminal stage, a request budget failure's class and stage, and the words of the audited published sentences. `collect_results.py` writes working-tree trials to `results/effective-requests.{json,md}` and keeps them out of the blinded comparison outputs:

```sh
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/run_study.py --phase d5-replay
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/run_study.py --phase d5-e2e
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/collect_results.py --phase d5-replay --phase d5-e2e
```

`d5-e2e` holds the diagnostics for a hierarchical publication. Only Gathering 10% (980 words) and Snow Fall at 300 words on the automatic strategy keep the study configuration. The Snow Fall sample document is not a manifest source, and it routes direct on its own. Its hierarchical cases are forced, and the two 65,536-context cases double the study's 32,768 pin. Each case passes its context to both the proxy (`--num-ctx`) and `run_modern.py` (`--context-window`). The proxy refuses, with HTTP 400, any request whose `num_ctx` differs from its pin.

## Exact local token counts (#120)

For an Ollama host on this machine, the working tree counts tokens with the model's own tokenizer, read from its local GGUF file (`run.json` records `counter_identity: gguf:<digest prefix>`). For gemma4 this matched Ollama's `prompt_eval_count` on all 1,742 recorded #107 requests at system + user tokens + 14 template tokens. The pinned revisions and the byte counter are unchanged. A replayed segmentation is checked under the counter that made it, and the merge stage onwards uses the current counter. `d11-replay` replays the same four saved merge inputs as `d5-replay`. With the exact count, only the transcript routes hierarchical on its own at 32,768, so `d11-e2e` runs transcript 25% fresh on the automatic strategy:

```sh
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/run_study.py --phase d11-replay
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/run_study.py --phase d11-e2e
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/collect_results.py --phase d11-replay --phase d11-e2e
```

## Editorial content units (#109)

Each working-tree trial saves the record the editorial step receives, after any compression, as `editorial_root.json` in its trial directory, together with the direct summary it was prepared from. `--editorial-root-from <trial>` replays both in a new trial of the same case (same source and target). The saved direct summary replaces the direct request, and the saved record replaces compression, so the audit describes the same root. `--editorial-units none` removes the record's content units, and `root` keeps them. Only the editorial and verification requests are new. Replay supports direct runs only, and a saved file without the direct summary is refused. `run.json` records the saved record it used as `replayed_editorial_root`. Stage 1 of #109 compares the two on the development cases, which all route direct at 32,768. Run `d7-a` first, because `d7-c` reuses its records:

```sh
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/run_study.py --phase d7-a
/private/tmp/rts-study-20260926/venv/bin/python experiments/chapter_comparison/run_study.py --phase d7-c
```
