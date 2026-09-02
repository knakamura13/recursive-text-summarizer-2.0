# Legacy Summarizer Baseline

This document records the observable behavior of the legacy summarizer before the generalized hierarchical pipeline is introduced. The characterization suite protects compatibility where useful and makes known defects visible. These findings describe the current implementation; they are not the target behavior for the rebuilt pipeline.

## Running the offline suite

From the repository root, create and activate a virtual environment, install the development dependencies, and run the full suite:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

The suite blocks outbound socket connections. It replaces OpenAI, NLTK, dotenv, tqdm, and requests behavior with deterministic test doubles, so the characterization tests do not need network access, credentials, or downloaded tokenizer data.

## Compatibility constraints

The executable workflow currently provides these observable guarantees:

- `python main.py` reads UTF-8 text from `input.txt` in the current working directory.
- A successful run writes UTF-8 text to `output.txt` in the current working directory.
- The workflow makes one provider call for each source chunk, preserving source order.
- Provider responses are joined with exactly one newline between adjacent responses.

## Known legacy limitations

The characterization suite preserves evidence for these limitations:

- The chunk character budget can exceed the configured limit because the separator added before a sentence is not included in the budget check.
- A sentence longer than the configured chunk size remains unsplit.
- Importing `main.py` downloads the NLTK `punkt` resource and constructs an OpenAI client.
- Prompts and model aliases are fixed in source code.
- Intermediate summaries are concatenated directly, without recursive synthesis.
- Provider errors can become output text, while a terminal generic provider failure can raise `UnboundLocalError`.
- Fatal executable errors are logged but do not produce a nonzero process exit.
- Module constants require source edits rather than runtime configuration.

These are findings about legacy behavior, not requirements or target behavior for the new pipeline.

## Fixture corpus

The offline corpus supplies five compact, fictional source genres for future evaluation:

- `article.txt` is a municipal news article with dated actions, attributed claims, and an unsigned funding qualification.
- `report.txt` is a quarterly operations report with metrics, causes, recommendations, and preliminary figures.
- `transcript.txt` is a meeting exchange with corrections, proposals, and a decision that remains pending.
- `structured.md` is a migration plan with headings, nested lists, scoped work, rollback details, and an unresolved warning.
- `narrative.txt` is a chronological field account whose observations do not establish a final cause.

The existing `omscs-ml-lectures/` historical lecture data remains unchanged by this baseline harness.
