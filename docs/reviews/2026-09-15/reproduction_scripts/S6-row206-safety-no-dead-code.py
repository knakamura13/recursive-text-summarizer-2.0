"""S6 repro for traceability row 206 (#37): no unreferenced function remains
in summarizer/safety.py.

For every top-level `def` in summarizer/safety.py, greps the rest of the
`summarizer/` package (excluding safety.py's own definition line) for a call
site. Reports each function and its caller files/lines. Exit code is nonzero
if any function has zero callers outside its own definition.

Usage: python repro_row206_safety_no_dead_code.py <repo_root>
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
safety_path = repo_root / "summarizer" / "safety.py"
source_root = repo_root / "summarizer"

tree = ast.parse(safety_path.read_text(encoding="utf-8"), filename=str(safety_path))
top_level_functions = [
    node.name for node in ast.iter_child_nodes(tree) if isinstance(node, ast.FunctionDef)
]

print(f"Top-level functions in {safety_path}: {top_level_functions}")

exit_code = 0
for name in top_level_functions:
    callers: list[str] = []
    for py_file in source_root.rglob("*.py"):
        if py_file == safety_path:
            continue
        text = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if name in line and ("(" in line or "import" in line):
                # crude but sufficient: real usage (call or import), not just
                # a substring coincidence in an unrelated identifier
                if (
                    f"{name}(" in line
                    or f"import {name}" in line
                    or f"import safety" in line
                    and name in line
                ):
                    callers.append(f"{py_file.relative_to(repo_root)}:{lineno}: {line.strip()}")
    status = "REFERENCED" if callers else "**UNREFERENCED**"
    if not callers:
        exit_code = 1
    print(f"\n{name}: {status}")
    for caller in callers:
        print(f"  {caller}")

raise SystemExit(exit_code)
