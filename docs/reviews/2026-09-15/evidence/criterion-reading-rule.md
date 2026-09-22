# Criterion-reading rule: how a traceability row and a finding's `criteria_violated` relate

Decided by the repository owner in session 7, on a question raised by an independent verifier and left open since session 6. Recorded here because it changes published counts and because the rejected alternative was well argued.

## The question

Some acceptance criteria describe *behavior* ("fail clearly instead of looping indefinitely"). Others describe *test coverage* ("offline tests cover every classification, malformed verifier output, successful and failed repair, stale-span refusal, and repair-pass exhaustion").

When behavior is correct but no test covers the named case, is the criterion violated?

The review answered this inconsistently. F-020 (#32's abbreviation criterion, which names "Dr.", "e.g." and "U.S." and whose test covers only two) was recorded `violated` in the traceability matrix while its finding file kept `criteria_violated: []`. `report.md` flagged the disagreement as open rather than resolving it.

## The rule

**The two fields answer different questions and are allowed to differ.**

- A **traceability row status** scores whether the criterion is satisfied. For a coverage-type criterion naming a required test, no test means `violated`, even where behavior is correct.
- A **finding's `criteria_violated`** records present-tense code defects only. A guard that behaves correctly violates nothing, so the field is `[]`.

The reasoning for the first half: reading "offline tests cover X" as satisfied because the behavior happens to be right empties the clause of meaning. That clause exists precisely because the issue author wanted the behavior pinned, not merely present. A criterion that can never fail is not a criterion.

The reasoning for the second half is the F-009/F-011 correction established in session 5, which stands unchanged: two candidates framed as "does not validate" turned out, on their own reproductions, to be guards that do validate. Labelling those as criterion violations misrepresented working code as broken.

## Consequences

- F-020's row 194 stays `violated`.
- #10 line 153 (repair-pass exhaustion, F-028) becomes `violated`.
- Both findings keep `criteria_violated: []`.
- **The violated count must be reported split**: behavior violations separately from coverage-clause violations. A single aggregated number invites exactly the misreading the split is meant to prevent — a reader seeing "N violated" will assume N defects.

## The rejected alternative, recorded because it was well argued

The independent verifier that raised the question recommended the opposite: align the traceability row to the finding file, changing F-020's row from `violated` to `verified* — see F-020`, and scoring #10 line 153 the same way. Its case:

- The F-009/F-011 correction rule was deliberately generalized in session 5 across 7 of the review's then-10 confirmed Tier B findings, making it the review's settled later convention.
- F-020's own finding file already answers this shape of question with `criteria_violated: []`, so traceability.md's `violated` is the single outlier.
- `report.md` flags the disagreement as unresolved rather than as two deliberate rules for two criterion types, and there is no textual signal in the review's history that coverage-type criteria were meant to be read differently from behavior-type ones.

That last point is fair and is the reason this document exists: the distinction was not previously stated anywhere, so the verifier was right that the inconsistency looked like an oversight rather than a rule. It is now a rule.

The reason the recommendation was not adopted: adopting it drops the review to zero violated rows and makes every "offline tests cover X" clause unfalsifiable while behavior is correct. Given that this review has now found four separate instances of correct code that no test would defend (F-021, F-028, F-030, C-S2H-001), a scoring rule that renders coverage clauses permanently satisfied would suppress precisely the signal the review has been most productive at finding.

## A third option, considered and not taken

Introducing `coverage-gap` as a distinct traceability status alongside `violated` and `verified` would be the most precise encoding and the hardest to misread. It was declined because it adds a sixth status to the matrix vocabulary and would require re-examining earlier rows that might qualify — work with a poor ratio of clarity gained to churn introduced, at a point where the matrix already carries a `not-a-criterion` status added this session for a different reason. The split-reporting requirement above achieves most of the same clarity without the re-examination.
