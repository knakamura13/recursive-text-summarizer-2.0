# Issue #65 Audit Durability Design

## Goal

Give every `--audit` artifact the same crash-durability guarantee, regardless of
whether the run uses a cache-backed publication session.

## Decision

Move the existing durable byte-replacement primitive from
`summarizer.finalization` to `summarizer.audit`. `write_audit()` will serialize
and delegate to that primitive; `finalization` will import it under its existing
`_atomic_replace` name. This preserves the current publication seam and the
CLI's internal import while giving both write routes one implementation.

The primitive keeps the existing behavior: create a same-directory temporary
file, write and flush it, fsync the file, replace the destination, then fsync
the parent directory. Any `OSError` cleans up the temporary file where possible
and propagates unchanged. This changes durability only; it does not claim that
the summary/audit pair becomes a single atomic transaction.

## Alternatives considered

- Copy the fsync sequence into `write_audit()`: smaller diff, but duplicate
  logic can drift again.
- Create a new generic I/O module: cleaner if more durable writers are imminent,
  but more scope than this issue requires.

## Verification and documentation

Focused unit tests will assert the `file`, then `directory`, fsync sequence for
both standalone `write_audit()` and the publication replacement helper. A README
sentence will distinguish the uniform audit durability guarantee from the
cache-backed manifest-witnessed publication protocol. Historical review evidence
under `docs/reviews/2026-09-15/` remains unchanged.
