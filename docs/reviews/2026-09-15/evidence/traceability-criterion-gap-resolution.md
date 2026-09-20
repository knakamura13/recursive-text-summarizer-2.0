# Why six findings map to no traceability criterion

Session 6. Reviewed commit `301cc4d`.

## The question

The session-6 traceability sync folded F-004 through F-018 into the matrix and could map only two of them to rows. Six findings named no transcribed acceptance criterion at all:

- F-004 (refuted) zero-valued grounding reserve rejected at construction
- F-005 (confirmed, minor) `MergeGrounding.request_capacity_tokens` zero/negative guard
- F-006 (refuted) `TreeNode` empty `covered_segments` guard
- F-007 (confirmed, minor) `group_children` `count <= 0` guard
- F-008 (confirmed, minor) `select_source_passages` empty-children precondition
- F-010 (downgraded, minor) `grounding.py:118` `cost <= max_tokens` exact-boundary inclusivity

Two explanations were possible, and they have opposite consequences:

1. The criteria exist in the issue bodies and were never transcribed. That would mean the 216-row matrix has holes and its completeness claim is false.
2. No acceptance criterion covers defensive validation at all. That would mean the matrix is faithful and these findings simply have nothing to be verified against.

## The answer: explanation 2

Checked the bodies of the two issues these findings cluster under.

**Issue #26** ("Size the merge grounding reserve from merge capacity so default hierarchical runs complete") carries exactly four acceptance criteria, and the matrix carries exactly those four as rows 173 to 176:

- a default-config hierarchical run above usable input capacity completes with at least one merge
- the reserve is derived from merge capacity, or documented as fixed with segmentation bounded below it
- a reserve that cannot hold any passage raises `BudgetError` naming reserve size, smallest candidate size, and segment id
- an offline regression test fails against `174f54b` and passes with the fix

None concerns rejecting a zero or negative parameter, and none concerns boundary inclusivity. The issue's own Evidence section does cite `grounding.py:110-118`, which is where F-010 lives, but it cites it to explain the packing behavior, not to impose a criterion on the comparison operator.

**Issue #7** contains one criterion mentioning validation at all ("Each merge returns a validated structured summary node suitable for another merge level"), which is about the merge result's shape, not about defensive parameter guards.

## Consequence

The matrix is faithful to the issue bodies. No transcription holes were found, and the 216-row count stands.

These six findings are therefore correctly recorded with an empty `criteria_violated`, consistent with the F-009 and F-011 correction: a guard that currently behaves correctly violates no present-tense criterion, and a missing test for it is a Tier B coverage gap rather than a defect. They can never reach `verified` status against a criterion because no criterion describes them.

This is itself a reviewable observation about the backlog rather than about the code: the issue set specifies behavior and largely does not specify defensive validation, so guards written beyond the criteria are unspecified surface. Whether that surface should be specified, tested, or removed is a judgment for the repository owner and is not a defect claim.

## Scope of this check

Two issue bodies (#7, #26) were read in full. The finding-to-issue clustering came from the session-6 traceability sync. This does not re-verify the other 214 rows, and it does not rule out transcription gaps elsewhere in the matrix; it answers only why these six findings had no home.
