"""Mutation-testing harness for the S2H (issue #7 hierarchical merging)
investigation of da77a67 / 3359e74's five claims.

For each of claims 1-4 (the code-level claims; claim 5 is about the test
suite itself and is what this script measures), this reverts main's fix for
that one claim in an isolated copy of the tree, then runs the real pytest
suite against the mutated copy. If the suite goes red, the claim is actively
guarded today. If the suite stays green, that specific defect shape could
regress silently - the "test hole" claim 5 describes.

This never touches the real checkout: it works on a `git archive` export of
HEAD into a scratch directory.

Usage (from the repo root):
    UV_OFFLINE=1 python .review/wip/issue7/mutation_test_claims.py

Requires `uv` on PATH and network-free `--with-requirements requirements-dev.txt`
resolution (same as the project's own baseline command).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_TARGETS = (
    "tests/test_hierarchy.py",
    "tests/test_merge_prompt.py",
    "tests/test_budget.py",
)


def export_head(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", "HEAD"], cwd=REPO_ROOT, capture_output=True, check=True
    )
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)


def run_pytest(tree: Path, *, full_suite: bool) -> subprocess.CompletedProcess:
    args = [
        "uv", "run", "--with-requirements", "requirements-dev.txt",
        "python", "-m", "pytest", "-q",
    ]
    if not full_suite:
        args.extend(TEST_TARGETS)
    return subprocess.run(
        args,
        cwd=tree,
        capture_output=True,
        text=True,
        env={"UV_OFFLINE": "1", "PATH": __import__("os").environ["PATH"],
             "HOME": __import__("os").environ.get("HOME", "")},
    )


def mutate_1_fanout_floor(path: Path) -> None:
    """Reintroduce: merge_fanout floors its result at two."""
    hierarchy = path / "summarizer" / "hierarchy.py"
    src = hierarchy.read_text()
    old = '''    costs = [measure_child_tokens(child, counter) for child in children]
    largest = max(costs)
    measured = capacity // largest if largest else len(children)

    # A merge level cannot be narrower than a pair, so a capacity that admits
    # only one child admits no merge at all. Reporting a fanout of 2 here -
    # which an earlier revision did - assembles a request of twice the size it
    # was sized against, and on a provider that truncates silently that is
    # undetectable content loss rather than an error.
    if measured < 2:
        worst = costs.index(largest)
        raise BudgetError(
            f"a merge request cannot hold two summaries: child {worst} costs "
            f"{largest} tokens and a pair costs {2 * largest} against a "
            f"capacity of {capacity}"
        )

    if ceiling is not None and ceiling < measured:
        return ceiling, (
            f"the configured ceiling of {ceiling} is below the measured "
            f"fanout of {measured}"
        )
    return measured, (
        f"a largest child of {largest} tokens fits {measured} times in a "
        f"capacity of {capacity}"
    )'''
    assert old in src, "mutation 1 anchor not found - has hierarchy.py changed?"
    new = '''    costs = [measure_child_tokens(child, counter) for child in children]
    largest = max(costs)
    # MUTATION 1: reintroduce the pre-fix floor-at-2 behavior.
    if largest > capacity:
        worst = costs.index(largest)
        raise BudgetError(
            f"a single summary does not fit a merge request: child {worst} "
            f"costs {largest} tokens against a capacity of {capacity}, "
            f"including delimiters"
        )

    measured = capacity // largest
    if ceiling is not None and ceiling < measured:
        return max(ceiling, 2), (
            f"the configured ceiling of {ceiling} is below the measured "
            f"fanout of {measured}"
        )
    return max(measured, 2), (
        f"a largest child of {largest} tokens fits {measured} times in a "
        f"capacity of {capacity}"
    )'''
    hierarchy.write_text(src.replace(old, new))


def mutate_2a_hardcoded_fence(path: Path) -> None:
    """Reintroduce: hardcoded rather than measured per-child delimiter cost."""
    hierarchy = path / "summarizer" / "hierarchy.py"
    src = hierarchy.read_text()
    old = '''def measure_child_tokens(node: SummaryNode, counter: TokenCounter) -> int:
    """Measure what one child costs inside a merge request, delimiters included."""
    return counter.count(serialize_child(node)) + child_fence_tokens(counter)'''
    assert old in src, "mutation 2a anchor not found"
    new = '''_CHILD_FENCE_TOKENS = 48


def measure_child_tokens(node: SummaryNode, counter: TokenCounter) -> int:
    """Measure what one child costs inside a merge request, delimiters included."""
    # MUTATION 2a: hardcoded rather than measured delimiter cost.
    return counter.count(serialize_child(node)) + _CHILD_FENCE_TOKENS'''
    hierarchy.write_text(src.replace(old, new))


def mutate_2b_trust_leaf_capacity(path: Path) -> None:
    """Reintroduce: merge stage trusts a capacity computed for a leaf request."""
    hierarchy = path / "summarizer" / "hierarchy.py"
    src = hierarchy.read_text()
    old = (
        "        overhead = measure_merge_overhead(counter, level=level)\n"
        "        child_capacity = usable_tokens - overhead\n"
    )
    assert old in src, "mutation 2b anchor not found"
    new = (
        "        overhead = 0  # MUTATION 2b: no merge overhead subtracted.\n"
        "        child_capacity = usable_tokens - overhead\n"
    )
    hierarchy.write_text(src.replace(old, new))


def mutate_3_stale_passthrough_level(path: Path) -> None:
    """Reintroduce: a pass-through node keeps reporting its child's level."""
    hierarchy = path / "summarizer" / "hierarchy.py"
    lines = hierarchy.read_text().splitlines(keepends=True)
    target = next(
        i for i, line in enumerate(lines)
        if "summary=only.summary.model_copy(update=" in line
    )
    lines[target] = (
        "                            summary=only.summary,  "
        "# MUTATION 3: stale level carried through\n"
    )
    hierarchy.write_text("".join(lines))


def mutate_4_coverage_concatenation(path: Path) -> None:
    """Reintroduce: coverage computed by concatenation rather than union."""
    hierarchy = path / "summarizer" / "hierarchy.py"
    lines = hierarchy.read_text().splitlines(keepends=True)
    start = next(
        i for i, line in enumerate(lines)
        if line.strip() == "covered = tuple(" and "dict.fromkeys(" in lines[i + 1]
    )
    # Splice out the dict.fromkeys(...) wrapper, keep a bare concatenation.
    end = start + 1
    while "dict.fromkeys(" not in lines[end]:
        end += 1
    # lines[start] = "covered = tuple(\n"; lines[start+1] = "dict.fromkeys(\n"
    # find the matching closing ")\n" for the dict.fromkeys( call, then the
    # outer tuple(...) close one line after that.
    close = start + 1
    depth = 0
    for i in range(start + 1, len(lines)):
        depth += lines[i].count("(") - lines[i].count(")")
        if depth <= -1:
            close = i
            break
    new_block = [
        "    # MUTATION 4: concatenation rather than union - duplicates possible.\n",
        "    covered = tuple(\n",
        "        identifier\n",
        "        for member in members\n",
        "        for identifier in member.covered_segments\n",
        "    )\n",
    ]
    lines[start:close + 1] = new_block
    hierarchy.write_text("".join(lines))


MUTATIONS = {
    "1_fanout_floor": mutate_1_fanout_floor,
    "2a_hardcoded_fence": mutate_2a_hardcoded_fence,
    "2b_trust_leaf_capacity": mutate_2b_trust_leaf_capacity,
    "3_stale_passthrough_level": mutate_3_stale_passthrough_level,
    "4_coverage_concatenation": mutate_4_coverage_concatenation,
}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="issue7-mutants-") as tmp:
        base = Path(tmp) / "pristine"
        export_head(base)

        for name, mutate in MUTATIONS.items():
            mutant = Path(tmp) / name
            shutil.copytree(base, mutant)
            mutate(mutant)
            full_suite = name == "4_coverage_concatenation"
            result = run_pytest(mutant, full_suite=full_suite)
            tail = "\n".join(result.stdout.strip().splitlines()[-6:])
            verdict = "CAUGHT (suite failed)" if result.returncode != 0 else "NOT CAUGHT (suite green)"
            scope = "full suite" if full_suite else "hierarchy/merge/budget tests"
            print(f"\n=== mutation: {name} ({scope}) -> {verdict} ===")
            print(tail)


if __name__ == "__main__":
    main()
