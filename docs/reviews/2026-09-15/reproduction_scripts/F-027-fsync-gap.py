"""S6 independent-verifier repro for C-S5-003: does the real write_audit()
call os.fsync() the way finalization._atomic_replace() does?

Read-only against the tracked tree; all writes go to a tempfile.TemporaryDirectory.
Reuses the existing `_artifact()` test helper (tests/test_verification_audit.py)
to build a real, schema-valid AuditArtifact rather than a hand-rolled fake, then
calls the actual production `write_audit` and the actual production
`_atomic_replace` with os.fsync wrapped in a call counter.

Run with (from a checkout at 301cc4d):
  UV_OFFLINE=1 uv run --with-requirements requirements-dev.txt python \
    .review/wip/v-misc/C-S5-003-fsync-gap.py
"""
from __future__ import annotations

import inspect
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, ".")

from summarizer import audit as audit_module
from summarizer import finalization as finalization_module
from tests.test_verification_audit import _artifact

fsync_calls = {"count": 0}
real_fsync = os.fsync


def counting_fsync(fd: int) -> None:
    fsync_calls["count"] += 1
    real_fsync(fd)


artifact = _artifact()

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    # --- Path 1: the real production write_audit() ---
    fsync_calls["count"] = 0
    target1 = tmp_path / "audit1.json"
    with mock.patch("os.fsync", counting_fsync):
        audit_module.write_audit(target1, artifact)
    write_audit_fsync_calls = fsync_calls["count"]
    write_audit_wrote_file = target1.exists()

    # --- Path 2: the real production _atomic_replace() (used by publish_final_output) ---
    fsync_calls["count"] = 0
    target2 = tmp_path / "audit2.json"
    with mock.patch("os.fsync", counting_fsync):
        finalization_module._atomic_replace(target2, b'{"probe": "_atomic_replace path"}')
    atomic_replace_fsync_calls = fsync_calls["count"]
    atomic_replace_wrote_file = target2.exists()

print("write_audit()      -> file written:", write_audit_wrote_file, " os.fsync() calls:", write_audit_fsync_calls)
print("_atomic_replace()  -> file written:", atomic_replace_wrote_file, " os.fsync() calls:", atomic_replace_fsync_calls)
print()
print("Static confirmation -- 'fsync' token present in each function's own source:")
write_audit_src = inspect.getsource(audit_module.write_audit)
atomic_replace_src = inspect.getsource(finalization_module._atomic_replace)
print("  'fsync' in write_audit source?      ", "fsync" in write_audit_src)
print("  'fsync' in _atomic_replace source?  ", "fsync" in atomic_replace_src)
