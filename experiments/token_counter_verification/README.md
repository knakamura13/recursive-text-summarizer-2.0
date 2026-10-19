# Verifying a tokenizer.json counter against Ollama (#190)

Ollama models pulled in the newer tensor layout (for example `gemma4:31b` and `gemma4:31b-mlx`) have no GGUF file, so the exact counter of #120 cannot read them. `TokenizerJsonCounter` reads the model's Hugging Face `tokenizer.json` layer instead, but it is refused until its counts have been checked against Ollama's `prompt_eval_count`. Until then the summarizer uses the byte counter and says so in preflight, in the Run and on the command line.

`verify_ollama_token_counts.py` makes that check. **It runs real model inference**: one warm-up request and 12 fixed probes to an Ollama server on this machine, each limited to one generated token. Nothing in the repository runs it, and it has not been run against a real server. Its tests use a scripted server and a synthetic tokenizer.

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

For each probe, the counter counts the system message and the user message. Ollama wraps both in the model's chat template, so the check is `counted + template overhead == prompt_eval_count`, the method of #120 (14 template tokens for gemma4 on its GGUF path).

- **The overhead is observed, not assumed.** Every plain probe must show the same `reported - counted`. If they differ, the run fails and names the probes in each group. Two causes are likely: the counter does not match Ollama's tokenization on those texts, or a reused prompt cache lowered a count. Each system message opens with a different word so a reused prefix is only the template header, and a throwaway warm-up request goes first so that header is cached the same way for every probe.
- **Never fewer.** Any probe that Ollama counts higher than counter plus overhead fails the run.
- **Schema is not prompt text.** Two probes carry a `format` schema as the summarizer's requests do. They share the plain overhead only if Ollama does not count the schema as prompt tokens, which is what the budget code assumes.
- **Special-token markers.** One probe quotes markers such as `<bos>` and `<|turn>`. The counter splits a marker into pieces, which can only count more than Ollama does, so Ollama may count fewer here and never more.
- A missing `prompt_eval_count` is a failure, not a pass.

Output has no document text: probe names, counts and the verdict. The JSON report adds the Ollama version, the counter identity and the counting fingerprint.

## After a pass

A pass prints the counting fingerprint. To enable the counter, add it to `_VERIFIED_TOKENIZER_JSON_FINGERPRINTS` in `summarizer/tokenization.py` in a commit that says what was compared. The fingerprint covers what decides a count (the model, normalizer, pre-tokenizer and ordinary added tokens), so `gemma4:31b` and `gemma4:31b-mlx` share one even though their manifests list a second `tokenizer.json` each. A tokenizer that counts differently has another fingerprint and stays refused.

The fixed probes are a smoke check of a few dozen tokens each. #120 compared 1,742 recorded requests; for a model that will run large baselines, also compare counts on a few long recorded or fresh requests before relying on the counter.

## Not covered

- GGUF models whose tokenizer is not yet in `_VERIFIED_GGUF_TOKENIZERS`, such as `gemma4:e2b-it-qat` in #190. The script reads `tokenizer.json` layers only.
- Requests far longer than a few hundred tokens, and any model other than the one named.
