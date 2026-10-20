# Verifying a tokenizer.json counter against Ollama (#190)

Ollama models pulled in the newer tensor layout (for example `gemma4:31b` and `gemma4:31b-mlx`) have no GGUF file, so the exact counter of #120 cannot read them. `TokenizerJsonCounter` reads the model's Hugging Face `tokenizer.json` layer instead, but it is refused until its counts have been checked against Ollama's `prompt_eval_count`. Until then the summarizer uses the byte counter and says so in preflight, in the Run and on the command line.

`verify_ollama_token_counts.py` makes that check. **It runs real model inference**: one warm-up request and 19 fixed probes to an Ollama server on this machine, each limited to one generated token and sent in the summarizer's own request shape (`think: false`). Nothing in the repository runs it. Its tests use a scripted server and a synthetic tokenizer. Its runs against a real server are recorded at the end of this file.

## Run it

From the repository root, with the model installed and `ollama serve` running:

```sh
uv run --with-requirements requirements.txt python experiments/token_counter_verification/verify_ollama_token_counts.py --model gemma4:31b --output /private/tmp/rts-token-verification/gemma4-31b.json
```

To count the probes and check the counter loads without contacting Ollama:

```sh
uv run --with-requirements requirements.txt python experiments/token_counter_verification/verify_ollama_token_counts.py --model gemma4:31b --dry-run
```

`--host` defaults to `http://localhost:11434` and must be this machine, because the tokenizer is read from the local model files. A cold 31B load can take minutes; `--timeout` is per request (default 900 seconds). Exit status is 0 for a pass, 1 for a failed comparison and 2 when the check could not be made (no tokenizer layer, host not this machine, server unreachable).

## Long requests (`--set long`)

The short probes are a few dozen tokens each. `--set long` swaps them for requests of a few thousand tokens, the size of the summarizer's own leaf segments: each is a quarter of the usable input capacity, about 5,000 to 6,000 tokens at a 32,768-token window. It makes **9 requests**: the short warm-up and eight probes, about 26,300 counted prompt tokens in all (counted with the `gemma4:31b` tokenizer, before the template). Passages come from the public-domain novels committed under `sample-documents/` (Alice, Frankenstein, Moby Dick) and from synthetic non-ASCII text, so nothing private is read, and no text is written to the report. `--documents-dir` points elsewhere if they move.

| Probe | Kind | What it is | Counted tokens |
| --- | --- | --- | --- |
| `long-leaf-2k` | clean | the app's leaf request (`build_leaf_request`) for a 2,000-token passage, with the Ollama wire schema | 2,454 |
| `long-plain-4k` | clean | a 4,000-token passage, no schema | 4,004 |
| `long-leaf-4k` | clean | leaf request, 4,000-token passage, schema | 4,409 |
| `long-non-ascii-3k` | clean | synthetic text in Latin accents, CJK, Hangul and Greek | 2,997 |
| `long-leaf-6k` | clean | leaf request, 6,000-token passage, schema | 6,445 |
| `long-trailing-blank-lines` | boundary | a 2,000-token passage plus `\n\n` at the end | 1,992 |
| `long-leading-blank-lines` | boundary | the same passage with `\n\n` first | 1,993 |
| `long-markers` | marker | a 2,000-token passage with `<bos>`, `<|turn>`, `<eos>` and `<|video|>` between paragraphs | 1,997 |

The leaf probes carry the app's real instructions (about 420 tokens) and the schema the app sends Ollama (about 600 tokens as JSON, a term the budget charges as prompt tokens), so they test at scale that the schema is not in the prompt count. Requests have the same shape as the summarizer's (`think: false`, `format` when a schema is set, one generated token) and the same clean, boundary and marker rules apply. The overhead is re-measured from the five clean probes, which must agree exactly. No two leaf probes are adjacent, so neighbours never share the leaf instructions as a cached prefix; the app's own back-to-back leaf requests do share it, and that case is not measured here.

Predictions, written before the run, from the second short run: the five clean probes agree exactly (at 18 if the overhead is unchanged); both boundary probes are 1 lower (`\n\n` is one token under the counter and Ollama trims message edges); `long-markers` is 10 lower (the four markers are 2, 3, 2 and 3 tokens as pieces and one token each as special tokens). A clean probe that disagrees, above all one of the later leaf probes by hundreds of tokens, would point to a reused prompt cache lowering `prompt_eval_count`, or to a long-input effect the short probes could not show. Run time was not measured; it is the prefill of about 26,300 tokens plus a possible cold load.

```sh
uv run --with-requirements requirements.txt python experiments/token_counter_verification/verify_ollama_token_counts.py --model gemma4:31b --set long --output /private/tmp/rts-token-verification/gemma4-31b-long.json
```

`--set long --dry-run` counts the eight probes and sends nothing.

## What it checks

For each probe, the counter counts the system message and the user message. Ollama wraps both in the model's chat template, so the check is `counted + template overhead == prompt_eval_count`, the method of #120. #120 measured 14 template tokens for gemma4 on Ollama 0.34.4; the overhead is re-measured on every run, never assumed, and 14 is only a reference the script reports a difference from. The requests carry `think: false` because `OllamaProvider` does; the first run left it out and measured a different template.

Probes come in three kinds, so that edge whitespace, which a server may trim, cannot hide a real mismatch or fail an exact one:

- **Clean** (11): no whitespace at either edge of either message. They set the overhead, which is observed, not assumed: every clean probe must show the same `reported - counted`, or the run fails and names the probes in each group. They include internal runs of spaces, tabs, newlines and indentation, non-ASCII text, markup, long text, and two probes with a `format` schema as the summarizer's requests have. The schema probes share the overhead only if Ollama does not count the schema as prompt tokens, as #120 found on the GGUF path.
- **Boundary** (7): `Count these words.` with one run of one kind of whitespace at one edge, each alone: trailing and leading newline, space and tab in the user message (the leading space is two, because one merges into the first word and costs no token), and a trailing newline in the system message. They only need `reported <= counted + overhead`. A probe Ollama counts lower is reported with its gap, which is how the run shows whether a server trims and which characters.
- **Marker** (1): special-token markers such as `<bos>` and `<|turn>`. The counter splits a marker into pieces, which can only count more than Ollama, so Ollama may count fewer here and never more.

Any probe that Ollama counts higher than counter plus overhead fails the run, and a missing `prompt_eval_count` is a failure. Each system message opens with a different word so a reused prefix is only the template header, and a throwaway warm-up request goes first so that header is cached the same way for every probe.

Output has no document text: probe names, counts and the verdict. The JSON report adds the Ollama version, the counter identity and the counting fingerprint.

## After a pass

A pass prints the counting fingerprint. To enable the counter, add it to `_VERIFIED_TOKENIZER_JSON_FINGERPRINTS` in `summarizer/tokenization.py` in a commit that says what was compared. The fingerprint covers what decides a count (the model, normalizer, pre-tokenizer and ordinary added tokens), so `gemma4:31b` and `gemma4:31b-mlx` share one even though their manifests list a second `tokenizer.json` each. A tokenizer that counts differently has another fingerprint and stays refused.

The fixed probes are a smoke check of a few dozen tokens each. #120 compared 1,742 recorded requests; for a model that will run large baselines, also compare counts on a few long recorded or fresh requests before relying on the counter.

## Run on 2026-10-19 (`gemma4:31b`, Ollama 0.40.6): did not pass

The script exited 1 on its constant-overhead rule. Reported minus counted was +16 on nine of the eleven plain probes and +15 on `whitespace` and `code`, the only two whose user text ends in a newline. The special-marker probe was 7 lower than the plain overhead of 16. Fingerprint `63e8558fe07c03e6...` was not added.

Offline follow-up on the same counts, using the real tokenizer, the model files and the server log, with no further requests:

- **The request shape differed from the summarizer's.** The script sent no `think` field. `OllamaProvider` sends `think: False` on every request and #120's capture proxy pinned it, so the 16 was measured on a request the summarizer never makes. With the model's own special tokens, the bare gemma4 turn skeleton (`<bos>`, then system, user and model turns with their role names and newlines) counts exactly 14, the figure of #120. Adding `<|think|>` and a newline gives 16, the figure observed. That suggested 14 against 16 was thinking being switched on. The script now sends `think: false`, and the second run below showed that reading was only half right. The server log shows `gemma4:31b` was served by the MLX runner with the built-in `gemma4` renderer and a loaded draft model (the second tokenizer layer is the draft's); `gemma4:e2b-it-qat` used the same renderer on the llama runner, so no chat template text exists in the model files to read.
- **The one-token difference.** `whitespace` and `code` are the only probes with a trailing newline, and also the only ones with runs of spaces and internal newlines. Removing the trailing newline changes both counts by exactly one and no other probe. No internal construct fits: collapsing space runs gives -3 and -2, collapsing newline runs and replacing tabs give 0, and removing the space before a newline or the indent after one give -1/0 and 0/-1. A merge of the trailing newline with a newline in the template is ruled out if the skeleton above is right, because user text is followed by the special token `<turn|>`. What remains is Ollama trimming trailing whitespace from message text (the Gemma 3 chat template does so with `trim`; not checked for gemma4's built-in renderer) or a tokenizer quirk in the MLX runner. Both leave the counter counting one more, the observed direction. They differ on trailing spaces: a user text of `Count these words.  ` (two trailing spaces) would report 15 more than counted if whitespace is trimmed and 16 more if only newlines behave differently.
- **Markers.** The marker probe equals the counter's count with `<bos>`, `<eos>`, `<pad>`, `<|turn>` and `<|video|>` each matched as one special token, so Ollama reads such text as special tokens. The counter counts their pieces, which is more, so this is safe. The two schema probes matched the plain ones exactly, so the schema was not part of the prompt count on this runner.
- **What to change.** The tokenization of message text matched on every probe except by that one token, so the constant-overhead rule is the likelier thing to change, not the counter: probes with no leading or trailing whitespace would set the overhead and agree exactly, and probes with boundary whitespace and markers would only need `reported <= counted + overhead`. That reading was found after the run, so a run on probes written beforehand (trailing and leading newline, space and tab, each alone, with `think: false`) comes before the rule is changed or the fingerprint is added. Not checked: the renderer's source, leading whitespace, the system-message boundary, long recorded requests, and the cause of the 14-to-16 step.

## Second run (2026-10-19, `gemma4:31b`, Ollama 0.40.6): passed, with one prediction wrong

The script sent `think: false` and the grouped probes. Predictions were committed before the run (4ab0cbd). The first run's reading predicted:

- Clean probes: `reported - counted` is 14 for all eleven, the figure of #120, because `think: false` removes the two thinking tokens. If it is 16 again, the thinking marker is not the cause and the step from 14 stays unexplained. If clean probes disagree, internal whitespace or content is the cause and the counter is not yet exact.
- Boundary probes: each costs exactly one token under the counter, so each is 1 lower than the overhead if Ollama trims that whitespace and 0 if it keeps it. `trailing-newline` is 1 lower, as in the first run. If the renderer trims message whitespace, all seven are 1 lower, including `system-trailing-newline`. If only trailing newlines behave differently, the other six are 0.
- `special-markers`: 7 lower than the overhead.
- Verdict: pass if no gap is positive.

Result, from `/private/tmp/rts-token-verification/gemma4-31b-run2.{log,json}`. The script passed and the fingerprint was not added.

- **Clean probes: all eleven +18, exact. The prediction of 14 was wrong.** The internal-whitespace, markup, non-ASCII, long and schema probes agree exactly. The overhead is 18 with `think: false`, 16 with `think` unset (first run). Token arithmetic fits both: the bare turn skeleton is 14, `<|think|>` and a newline make 16, and an empty thought channel (`<|channel>thought`, newline, `<channel|>`) makes 18. So on this runner the thinking setting moves the overhead between 16 and 18. That reading comes from counting the model's own special tokens; the renderer's source was not read. It does not explain #120's 14, measured on `gemma4:latest`, Ollama 0.34.4 and the llama runner. A different Ollama version, model or runner may account for it; none was tested.
- **Boundary probes: all seven exactly 1 lower.** Ollama drops a leading or trailing newline, space or tab from the user message and a trailing newline from the system message. The first run's +15 was this. The counter counts those edge tokens, so it can only count more than Ollama, never fewer.
- **Markers: 7 lower**, as predicted. Ollama reads `<bos>`, `<eos>`, `<pad>`, `<|turn>` and `<|video|>` as special tokens. The schema probes matched the other clean probes, so the schema is not in the prompt count on this runner.
- **What this does not show.** The probes are a smoke check of a few dozen tokens each. Long recorded requests (#120 compared 1,742), other Ollama versions or runners, and edge whitespace of mixed kinds were not compared. The budget code's comment that the template is 14 tokens (`GgufTokenCounter`) describes the llama runner; on this runner it is 18, and the margin that relied on the schema count exceeding the template by at least 19 tokens was not rechecked here.

## Long run (`--set long`, 2026-10-19, `gemma4:31b`, Ollama 0.40.6): passed, every prediction held

One run, 9 requests (`/private/tmp/rts-token-verification/gemma4-31b-long.{json,log}`). The script took 264 seconds; the server log shows 20 to 51 seconds per long request (about 120 prompt tokens per second) after a cold load that the warm-up absorbed.

| Probe | Kind | Counted | Reported | Reported minus counted |
| --- | --- | --- | --- | --- |
| `long-leaf-2k` | clean | 2,454 | 2,472 | +18 |
| `long-plain-4k` | clean | 4,004 | 4,022 | +18 |
| `long-leaf-4k` | clean | 4,409 | 4,427 | +18 |
| `long-non-ascii-3k` | clean | 2,997 | 3,015 | +18 |
| `long-leaf-6k` | clean | 6,445 | 6,463 | +18 |
| `long-trailing-blank-lines` | boundary | 1,992 | 2,009 | +17 |
| `long-leading-blank-lines` | boundary | 1,993 | 2,010 | +17 |
| `long-markers` | marker | 1,997 | 2,005 | +8 |

- The five clean probes agree exactly at an overhead of 18, the same as the short set, from 2,454 to 6,445 tokens. The three leaf probes carry the app's instructions and the Ollama wire schema (about 600 tokens as JSON), and the schema is not in the prompt count at that size.
- Both blank-line probes are 1 lower, so Ollama trims `\n\n` at a message edge as it trimmed the short whitespace probes. `long-markers` is 10 lower, as predicted.
- No request was counted higher by Ollama than counter plus overhead. The overhead and every boundary and marker gap match what the short runs showed.
- Still not compared: requests above 6,445 tokens (merges and the editorial reach about 20,000), back-to-back requests that share the leaf instructions as a prefix (the leaf probes were kept apart), the app's real schemas for the other stages, other Ollama versions or runners, `gemma4:31b-mlx` itself (same fingerprint, not run), thinking enabled, and anything near #120's 1,742 requests. The fingerprint was not added.

## Not covered

- GGUF models whose tokenizer is not yet in `_VERIFIED_GGUF_TOKENIZERS`, such as `gemma4:e2b-it-qat` in #190. The script reads `tokenizer.json` layers only.
- Requests far longer than a few hundred tokens, and any model other than the one named.
