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

## What it checks

For each probe, the counter counts the system message and the user message. Ollama wraps both in the model's chat template, so the check is `counted + template overhead == prompt_eval_count`, the method of #120 (14 template tokens for gemma4 with `think` false). The requests carry `think: false` because `OllamaProvider` does; the first run left it out and measured a different template.

Probes come in three kinds, so that edge whitespace, which a server may trim, cannot hide a real mismatch or fail an exact one:

- **Clean** (11): no whitespace at either edge of either message. They set the overhead, which is observed, not assumed: every clean probe must show the same `reported - counted`, or the run fails and names the probes in each group. They include internal runs of spaces, tabs, newlines and indentation, non-ASCII text, markup, long text, and two probes with a `format` schema as the summarizer's requests have. The schema probes share the overhead only if Ollama does not count the schema as prompt tokens, as #120 found on the GGUF path.
- **Boundary** (7): `Count these words.` with one run of one kind of whitespace at one edge, each alone: trailing and leading newline, space and tab in the user message, and a trailing newline in the system message. They only need `reported <= counted + overhead`. A probe Ollama counts lower is reported with its gap, which is how the run shows whether a server trims and which characters.
- **Marker** (1): special-token markers such as `<bos>` and `<|turn>`. The counter splits a marker into pieces, which can only count more than Ollama, so Ollama may count fewer here and never more.

Any probe that Ollama counts higher than counter plus overhead fails the run, and a missing `prompt_eval_count` is a failure. Each system message opens with a different word so a reused prefix is only the template header, and a throwaway warm-up request goes first so that header is cached the same way for every probe.

Output has no document text: probe names, counts and the verdict. The JSON report adds the Ollama version, the counter identity and the counting fingerprint.

## After a pass

A pass prints the counting fingerprint. To enable the counter, add it to `_VERIFIED_TOKENIZER_JSON_FINGERPRINTS` in `summarizer/tokenization.py` in a commit that says what was compared. The fingerprint covers what decides a count (the model, normalizer, pre-tokenizer and ordinary added tokens), so `gemma4:31b` and `gemma4:31b-mlx` share one even though their manifests list a second `tokenizer.json` each. A tokenizer that counts differently has another fingerprint and stays refused.

The fixed probes are a smoke check of a few dozen tokens each. #120 compared 1,742 recorded requests; for a model that will run large baselines, also compare counts on a few long recorded or fresh requests before relying on the counter.

## Run on 2026-10-19 (`gemma4:31b`, Ollama 0.40.6): did not pass

The script exited 1 on its constant-overhead rule. Reported minus counted was +16 on nine of the eleven plain probes and +15 on `whitespace` and `code`, the only two whose user text ends in a newline. The special-marker probe was 7 lower than the plain overhead of 16. Fingerprint `63e8558fe07c03e6...` was not added.

Offline follow-up on the same counts, using the real tokenizer, the model files and the server log, with no further requests:

- **The request shape differed from the summarizer's.** The script sent no `think` field. `OllamaProvider` sends `think: False` on every request and #120's capture proxy pinned it, so the 16 was measured on a request the summarizer never makes. With the model's own special tokens, the bare gemma4 turn skeleton (`<bos>`, then system, user and model turns with their role names and newlines) counts exactly 14, the figure of #120. Adding `<|think|>` and a newline gives 16, the figure observed. So 14 against 16 is most likely thinking being switched on. This is arithmetic, not a confirmed cause, and the script should send `think: false` before it is run again. The server log shows `gemma4:31b` was served by the MLX runner with the built-in `gemma4` renderer and a loaded draft model (the second tokenizer layer is the draft's); `gemma4:e2b-it-qat` used the same renderer on the llama runner, so no chat template text exists in the model files to read.
- **The one-token difference.** `whitespace` and `code` are the only probes with a trailing newline, and also the only ones with runs of spaces and internal newlines. Removing the trailing newline changes both counts by exactly one and no other probe. No internal construct fits: collapsing space runs gives -3 and -2, collapsing newline runs and replacing tabs give 0, and removing the space before a newline or the indent after one give -1/0 and 0/-1. A merge of the trailing newline with a newline in the template is ruled out if the skeleton above is right, because user text is followed by the special token `<turn|>`. What remains is Ollama trimming trailing whitespace from message text (the Gemma 3 chat template does so with `trim`; not checked for gemma4's built-in renderer) or a tokenizer quirk in the MLX runner. Both leave the counter counting one more, the observed direction. They differ on trailing spaces: a user text of `Count these words.  ` (two trailing spaces) would report 15 more than counted if whitespace is trimmed and 16 more if only newlines behave differently.
- **Markers.** The marker probe equals the counter's count with `<bos>`, `<eos>`, `<pad>`, `<|turn>` and `<|video|>` each matched as one special token, so Ollama reads such text as special tokens. The counter counts their pieces, which is more, so this is safe. The two schema probes matched the plain ones exactly, so the schema was not part of the prompt count on this runner.
- **What to change.** The tokenization of message text matched on every probe except by that one token, so the constant-overhead rule is the likelier thing to change, not the counter: probes with no leading or trailing whitespace would set the overhead and agree exactly, and probes with boundary whitespace and markers would only need `reported <= counted + overhead`. That reading was found after the run, so a run on probes written beforehand (trailing and leading newline, space and tab, each alone, with `think: false`) comes before the rule is changed or the fingerprint is added. Not checked: the renderer's source, leading whitespace, the system-message boundary, long recorded requests, and the cause of the 14-to-16 step.

## Second run: predictions written before it, with `think: false` and the grouped probes

Reading of the first run (above) predicts:

- Clean probes: `reported - counted` is 14 for all eleven, the figure of #120, because `think: false` removes the two thinking tokens. If it is 16 again, the thinking marker is not the cause and the step from 14 stays unexplained. If clean probes disagree, internal whitespace or content is the cause and the counter is not yet exact.
- Boundary probes: `trailing-newline` is 1 lower than the overhead, as in the first run. If the renderer trims message whitespace, every other boundary probe is also 1 lower, including `system-trailing-newline`. If only trailing newlines behave differently, the others are 0.
- `special-markers`: 7 lower than the overhead.
- Verdict: pass if no gap is positive.

## Not covered

- GGUF models whose tokenizer is not yet in `_VERIFIED_GGUF_TOKENIZERS`, such as `gemma4:e2b-it-qat` in #190. The script reads `tokenizer.json` layers only.
- Requests far longer than a few hundred tokens, and any model other than the one named.
