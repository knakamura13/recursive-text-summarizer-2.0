# Repro: rows 190 & 193 (#31) — regression tests fail against 174f54b

Goal: confirm, by actually running it, that
`tests/test_verification_repair.py::test_verify_and_repair_keeps_independent_repairs_in_returned_text`
and `::test_verify_and_repair_discards_nested_repairs_when_continuation_fails`
are genuine regression tests for issue #31 — i.e. that they fail against the
pre-fix baseline commit `174f54b` and pass at `301cc4d`.

Method: extract the `174f54b` tree with `git archive` into a scratch
directory. This is a read-only export — it does not check out a branch in any
worktree and does not mutate the reviewed repo or the main checkout. The
current test file (unmodified from `301cc4d`) is then copied on top of that
extracted tree, and the target tests are run against the *old*
`summarizer/verification.py`.

## Commands run

```bash
MAIN=/Users/kylenakamura/documents-local/development-local/side-projects/recursive-text-summarizer
rm -rf /tmp/s4-criteria/174f54b-snapshot
mkdir -p /tmp/s4-criteria/174f54b-snapshot
git -C "$MAIN" archive 174f54b | tar -x -C /tmp/s4-criteria/174f54b-snapshot

cp tests/test_verification_repair.py /tmp/s4-criteria/174f54b-snapshot/tests/test_verification_repair.py

cd /tmp/s4-criteria/174f54b-snapshot
UV_OFFLINE=1 uv run --with-requirements requirements-dev.txt python -m pytest -q \
  "tests/test_verification_repair.py::test_verify_and_repair_keeps_independent_repairs_in_returned_text" \
  "tests/test_verification_repair.py::test_verify_and_repair_discards_nested_repairs_when_continuation_fails"
```

(`git -C "$MAIN" cat-file -t 174f54b` confirms it resolves to a real, reachable
commit: `Merge pull request #24 from knakamura13/knakamura/issue-10-claim-verification`,
the same pre-fix baseline other subsystems' traceability rows already cite for
#26/#27/#29/#30/#32.)

## Result at 174f54b (pre-fix)

```
FAILED tests/test_verification_repair.py::test_verify_and_repair_keeps_independent_repairs_in_returned_text
FAILED tests/test_verification_repair.py::test_verify_and_repair_discards_nested_repairs_when_continuation_fails
2 failed in 0.33s
```

Failure detail for the first test:

```
>       assert result.text == outer_repaired
E       AssertionError: assert 'Claim A is w...m B is wrong.' == 'Claim A is f...m B is wrong.'
E
E         - Claim A is fixed. Claim B is wrong.
E         + Claim A is wrong. Claim B is wrong.
```

This matches the design doc's own description of the pre-fix bug exactly
(`docs/plans/2026-09-14-issue-31-repair-lineage-design.md`: "a post-repair
verifier failure that returns the pre-repair draft" / recursion "starts again
from the original draft" rather than the already-repaired one) — the outer
level's committed fix ("Claim A is fixed.") is lost.

## Result at 301cc4d (current, in the reviewed worktree)

Already confirmed passing as part of the full 766-test baseline run, and
individually:

```
tests/test_verification_repair.py::test_verify_and_repair_keeps_independent_repairs_in_returned_text PASSED
tests/test_verification_repair.py::test_verify_and_repair_discards_nested_repairs_when_continuation_fails PASSED
```

## Cleanup

`/tmp/s4-criteria/174f54b-snapshot` is scratch and was deleted at the end of
this review. No file inside the reviewed repo (worktree or main checkout) was
touched by this experiment — `git archive` reads the object database only.
