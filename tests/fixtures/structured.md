# Atlas Inventory Migration

The fictional Atlas team plans to move warehouse inventory records from LedgerFox 4 to Meridian Cloud during the October maintenance window. The plan covers product counts and location codes, but archived purchase orders remain out of scope.

## Readiness

- Data preparation
  - Remove duplicate location aliases.
  - Preserve the original item identifier in `legacy_id`.
- Validation
  - Compare record totals before and after import.
  - Sample fifty serialized items from each warehouse.
- Operations
  - Freeze manual adjustments at 18:00 Friday.
  - Reopen access only after the reconciliation report passes.

## Rollback

The database team will retain a read-only LedgerFox snapshot for thirty days. A rollback may still require several hours because scanner configurations must be restored separately.

> **Warning:** Meridian's staging export omitted seven inactive locations during the August rehearsal. The cause appears to be a status filter, but that explanation is not yet confirmed. Do not start production migration until the corrected export has been reviewed.

The migration lead will publish a go or no-go recommendation by October 9.
