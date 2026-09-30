# SPOC quad recall spike (#150)

Question: does extracting Subject-Predicate-Object-Context quads from a source, then checking how many a summary still entails, give a usable coverage and qualifier-loss signal? Nothing here is wired into the pipeline, audit or CLI, and published output is unchanged.

## Pieces

- `quads.py`: few-shot JSON prompt and schema, defensive parsing, an `anchored` flag (the quoted evidence is found verbatim in the segment), and centrality weights.
- `score.py`: classifies each quad as `full` (S, P, O and context kept), `partial` (S, P, O kept, context dropped or strengthened) or `missed`. `LLMJudge` is the real measurement. `LexicalJudge` is a word-overlap stand-in that only tests the plumbing offline.
- `degrade.py`: controlled degradations (drop a qualifier phrase, drop sentences naming an entity).
- `run_spike.py`: `extract`, `score` and `check` commands.
- `reference/`: hand-authored quads and summary for `tests/fixtures/article.txt`. Written by a person reading the fixture, not model output.

## Run

```sh
# Offline plumbing check (no model)
python -m experiments.quad_recall.run_spike check --judge lexical \
  --quads experiments/quad_recall/reference/article.quads.json \
  --summary experiments/quad_recall/reference/article.summary.txt \
  --drop-phrase "but the award is not yet signed" --target-contains "not yet been signed"

# Live: extract, then score a summary, with a hosted or local model
python -m experiments.quad_recall.run_spike extract --source tests/fixtures/article.txt --out article.quads.json --provider ollama --model gemma3:4b
python -m experiments.quad_recall.run_spike score --quads article.quads.json --summary summary.txt --provider ollama --model gemma3:4b
```

## Status against #150's acceptance criteria

Done and tested offline (`tests/test_quad_recall.py`): extraction and judging code, the three-way classification, centrality weighting, the degradation check, and request counting.

Not done, because it needs a live model: extraction on all five fixtures with a hosted and a local model, the 40-quad hand-judged extraction sample, recall against the curated must-retain claims, and the go/no-go decision.

The offline `article` check shows only that the scoring and degradation logic works: with the lexical stand-in, removing the "not yet signed" qualifier moves exactly one quad from `full` to `partial` and changes nothing else. It is not evidence that an LLM extracts usable context fields or judges paraphrase reliably, which is what the spike has to answer.
