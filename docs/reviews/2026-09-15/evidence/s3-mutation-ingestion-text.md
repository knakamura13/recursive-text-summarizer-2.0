# S3 mutation pass — `ingestion.py` and `text.py` (the known coverage hole)

Session 7. Candidates only: no verdicts, no finding ids, no adjudicated severities. Every tier below is *proposed*, pending an independent verifier.

Reviewed commit `301cc4d`, confirmed clean before and after. Disposable copy at `/tmp/m-ingestion-text/repo`, deleted at the end. Baseline on the untouched copy: 766 passed. Each mutation applied in isolation with `.orig` snapshots restored via `finally` before the next one, so no mutation compounded.

## 1. A methodology defect found mid-run, and why it matters beyond this pass

`cp -R` carried over pre-existing `__pycache__/*.pyc` files for the unmutated modules. The first sweep produced an attribution that was impossible on its face: a mutation confined to `SourceReadError` appeared to be killed by a `SourceDecodeError` assertion, and those two paths do not overlap.

The cause is stale bytecode. Under a rapid mutate/test/restore loop, Python can reuse a `.pyc` compiled from the *unmutated* source, so the mutation never executes. The reviewer deleted the stale `.pyc` files, set `PYTHONDONTWRITEBYTECODE=1`, and reran the **entire** 19-mutation sweep. Every other row reproduced identically; only that one row's attribution changed.

**This generalizes to the rest of the review.** The dominant failure direction is a false SURVIVOR: if the interpreter runs cached unmutated bytecode, the tests pass and the mutation is recorded as having survived when it was never actually applied. No earlier mutation pass in this review — S1's 4, S2's 21, or session 6's 20 on `budget.py`/`segmentation.py`/`tokenization.py` — is documented as having taken this precaution. Their survivor counts should be treated as upper bounds until spot-checked. Two mitigating facts: several survivors were later exercised by independent verifiers through repros that called the real code directly, which would not be fooled this way, and a false survivor inflates candidate counts rather than hiding defects. Recorded as process item **P-004**.

A second red herring was ruled out: intermittent `765 passed, 1 skipped` instead of 766 traces to `tests/test_cache.py:281`, which skips when the subprocess does not inherit the full shell environment (reproducible with `env -i`). Unrelated to these modules.

## 2. `text.py` reachability — the prior claim holds, independently re-derived

Session 6 skipped `text.py` on the reasoning that it is legacy code reachable only through `legacy_workflow.py`. That claim was re-derived from scratch rather than inherited, because #1's acceptance criteria promise the `python main.py` default workflow and a path from `main.py` into the legacy chunker would have made this live code.

- Importers of `summarizer.text`: only `summarizer/legacy_workflow.py` and `tests/test_text.py`.
- References to `legacy_workflow` / `LegacyWorkflow`: only `summarizer/config.py` (which defines the plain `LegacyWorkflowConfig` dataclass), `tests/test_config.py`, `tests/test_legacy_workflow.py`, and `docs/plans/`. No other production module.
- `main.py` calls `summarizer.cli.main()` only. `cli.py`'s full import list contains no reference to `LegacyWorkflow`, `LegacyWorkflowConfig`, or `summarizer.text`; its `main()` calls `read_source` then `run_pipeline`.
- No `pyproject.toml`, `setup.py` or `setup.cfg` exists anywhere in the tree, so no declared entry point can route into legacy code.
- `.github/workflows/tests.yml` runs only `pytest -q`. `subtitles_to_txt.py` never imports `summarizer`.
- #1's `input.txt`/`output.txt` criterion is satisfied through `cli.py`'s argparse defaults flowing into `read_source` then `run_pipeline`: the filenames are preserved, but the code path is the new pipeline, not `LegacyWorkflow`.

**`text.py` and `legacy_workflow.py` are dead code, reachable only from their own dedicated tests.** No mutation was run against `text.py`; the budget went to `ingestion.py`.

Note for S1: this bears on the #38 row asking whether `summarizer/legacy_workflow.py` is "removed or explicitly retained as a documented compatibility mode." It is retained and tested but, on this evidence, not reachable from any entry point. Whether that counts as documented retention is S1's call, not this pass's.

## 3. Mutation table — `summarizer/ingestion.py`

19 mutations, one at a time. Mutations affecting the *values* these functions produce ran against the full 766-test suite, because `ingest_text` and `source_id` are consumed well outside `tests/test_ingestion.py` (`budget.py`, `direct.py`, `pipeline.py`, `segmentation.py`, 13 test files, and format validation in `audit.py` and `cache.py`). Mutations confined to `read_source`'s own error formatting ran narrow, with automatic escalation on any narrow survivor. None escalated.

| # | line@301cc4d | mutation | selection | result |
|---|---|---|---|---|
| 1 | :37 | drop BOM stripping | full | KILLED — `test_ingestion.py::test_normalization_preserves_structure_and_unicode`, `::test_source_identity_is_based_on_canonical_utf8_text` |
| 2 | :37 | drop dedicated CRLF handling | full | KILLED — same two |
| 3 | :37 | drop lone-`\r` to `\n` replace | full | KILLED — `::test_normalization_preserves_structure_and_unicode` |
| 4 | :38 | `rstrip(" \t")` to `rstrip()` | full | **SURVIVED** |
| 5 | :38 | `rstrip(" \t")` to `rstrip(" ")` | full | KILLED — `::test_normalization_preserves_structure_and_unicode` |
| 6 | :39-40 | leading blank-line collapse `while` to `if` | full | **SURVIVED** |
| 7 | :41-42 | trailing blank-line collapse `while` to `if` | full | KILLED — `::test_normalization_preserves_structure_and_unicode` |
| 8 | :41-42 | trailing collapse pops wrong end | full | KILLED — 13 failures incl. `test_evaluation.py`, `test_leaf_stage.py` |
| 9 | :43 | join separator `"\n"` to `" "` | full | KILLED — 8 failures incl. `test_segmentation.py` |
| 10 | :49 | drop `.strip()` in emptiness check | full | KILLED — all 5 `test_unicode_whitespace_only_input_is_rejected[...]` |
| 11 | :6,51 | hash `sha256` to `sha3_256` (still 64 hex) | full | **SURVIVED** |
| 12 | :6,51 | hash `sha256` to `md5` (32 hex) | full | KILLED — 123 failures incl. `test_audit.py`, `test_verification_integration.py` |
| 13 | :55 | default encoding `"utf-8"` to `"latin-1"` | narrow | KILLED — 3 ingestion tests |
| 14 | :61-63 | decode-error message drops path | narrow | KILLED — `::test_decode_error_mentions_path_and_encoding` |
| 15 | :61-63 | decode-error message drops encoding | narrow | KILLED — same |
| 16 | :60-63 | decode error loses `from error` chaining | narrow | KILLED — same |
| 17 | :64-67 | read error loses `from error` chaining | narrow | KILLED — `::test_read_error_mentions_path_and_encoding` (this is the row the bytecode fix corrected) |
| 18 | :68 | `read_source` drops `path=` | narrow | KILLED — `::test_read_source_decodes_utf8_and_records_path` |
| 19 | :26 | `SourceDocument` no longer frozen | full | KILLED — `::test_source_document_is_immutable` |

**16 killed, 3 survived.**

## 4. Candidates

**C-S3M-101 — leading blank-line collapse is pinned only for a single leading blank line.**
`while lines and not lines[0]: lines.pop(0)` at `ingestion.py:39-40@301cc4d` collapses to a single `pop` under an `if` and survives all 766 tests, because every fixture in the suite has at most one leading blank line. The current code is correct, confirmed by direct execution: `normalize_source_text("\n\n# H\nBody") == "# H\nBody"`. Reachable unconditionally on the CLI's default path (`cli.py:250`). Proposed tier B, **minor**. Confidence high on the mechanics (two clean full-suite runs), lower on the severity ceiling. Falsifying experiment, not run: put a three-leading-blank-line document through `--audit --citations` and check for offset misalignment.

**C-S3M-102 — the trailing-strip's character scope is unpinned.**
`rstrip(" \t")` at `ingestion.py:38@301cc4d` broadens to bare `rstrip()`, stripping NBSP, vertical tab and form feed at line ends, and survives all 766 tests. The current code correctly preserves a trailing NBSP (`"Title\xa0\nBody"` is unchanged); the sibling test covers only *interior* NBSP. Proposed tier B, **minor but borderline** — the module's own docstring disclaims "flattening document structure," and silently altering content is precisely the failure mode it means to avoid. Falsifying experiment, not run: search the fixture corpus for trailing NBSP or other exotic whitespace.

**C-S3M-103 — `source_id`'s hash algorithm is unpinned; only its format is checked.**
Swapping `sha256` for `sha3_256` at `ingestion.py:6,51@301cc4d`, which still yields 64 hex characters, survives all 766 tests, because `audit.py:123-126` and `cache.py:126-129,257-258` validate only "64 lowercase hex" and never the algorithm. `md5` at 32 characters is caught immediately (row 12), which proves format checking works but not algorithm pinning. Blast radius inside this codebase is narrow: `checkpoint.py` and `scheduler.py` only compare stored `source_id` values against each other and never recompute independently, and no README or doc claims `source_id` is an externally verifiable SHA-256. Proposed tier B, **minor**, the weakest of the three. Falsifying experiment, not run: check `hierarchy.py` and `verification.py` for any independent hash recomputation across a resume or checkpoint boundary.

## 5. Nothing found — properly pinned

BOM stripping, CRLF collapse, lone-CR normalization, tab inclusion in the trailing strip, trailing multi-blank collapse, correct-end popping, newline-based join, the `.strip()`-based emptiness check, hash *format* validation, default UTF-8 decoding, both error messages' path and encoding content, both exceptions' `from error` chaining, path propagation into the returned document, and dataclass immutability. Killing node ids are in the table.

**The prior reviewer's "well-covered on inspection" judgment of `ingestion.py` largely holds, and the reason is more interesting than the verdict.** The single dense fixture `test_normalization_preserves_structure_and_unicode` does far more work than one input/output pair suggests: its interior structure — multiple CRLF sequences, an embedded lone `\r`, two trailing blank lines — jointly pins five of the eight normalization rules. The two real gaps are both *asymmetries* between what the fixture happens to contain and what the rule accepts: it has one leading blank line but two trailing ones, so the trailing collapse is pinned and the leading one is not; and it contains interior but not trailing NBSP, so the strip's character scope is unpinned. That is the general shape to look for when a single fixture is carrying a whole module's coverage.
