"""Run pinned local chapter comparisons sequentially with independent traces.

Corpus and generated content remain in the artifact root outside the repository.
Run --phase pilot first; freeze changes before --phase validation.

The d5 phases run this repository's working tree, not a pinned checkout.
``d5-replay`` feeds saved failing trials' segments and leaves to the current
merge stage without regenerating leaves; ``d5-e2e`` runs one fresh case.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
ROOT = Path(MANIFEST["artifact_root"])


def verify_environment() -> None:
    missing = [name for name in ("requests", "pydantic", "ollama", "nltk") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"study Python {sys.executable} lacks required packages: {', '.join(missing)}")
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=20) as response:
        models = json.load(response)["models"]
    digest = next((model["digest"] for model in models if model["name"] == MANIFEST["model"]), None)
    if digest != MANIFEST["model_digest"]:
        raise RuntimeError(f"model digest changed: {digest}")
    with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=20) as response:
        version = json.load(response)["version"]
    if version != MANIFEST["ollama_version"]:
        raise RuntimeError(f"Ollama version changed: {version}")
    for name, revision in MANIFEST["versions"].items():
        checkout = Path(MANIFEST["checkout_root"]) / name
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        if actual != revision:
            raise RuntimeError(f"{name} checkout changed: {actual}")
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=checkout, text=True
        )
        if status:
            raise RuntimeError(f"{name} checkout has tracked or untracked changes: {status[:200]}")


# One working-tree case: document, target (a fraction of the source's words, or
# an absolute word count for a non-manifest source), seed, the saved phase whose
# merge inputs are replayed (None runs fresh), strategy and context window.
# Replay requires the hierarchical strategy, the only path that uses saved leaves.
# An optional seventh item, (saved phase, units), hands the editorial step the
# record saved by that phase's trial of the same case, with its content units
# kept (`root`) or removed (`none`).
WORKTREE_CASES = {
    "d5-replay": (
        ("gathering", 0.25, 101, "validation", "hierarchical", 32_768),
        ("isl", 0.25, 101, "baseline-retry", "hierarchical", 32_768),
        ("isl", 0.5, 101, "baseline-retry", "hierarchical", 32_768),
        ("transcript", 0.25, 101, "validation", "hierarchical", 32_768),
        # That trial stopped before any leaf, so it runs fresh.
        ("transcript", 0.5, 101, None, "auto", 32_768),
    ),
    # Diagnostics for a hierarchical publication. Only the first two keep the
    # study's 32,768 context and automatic strategy; Snow Fall routes direct on
    # its own, so its hierarchical cases are forced.
    "d5-e2e": (
        ("gathering", 0.1, 101, None, "auto", 32_768),
        ("snowfall", 300, 101, None, "auto", 32_768),
        ("snowfall", 300, 101, None, "hierarchical", 32_768),
        ("gathering", 0.1, 101, None, "hierarchical", 65_536),
        ("snowfall", 300, 101, None, "hierarchical", 65_536),
    ),
    # #120: the same saved merge inputs under the exact local tokenizer count.
    "d11-replay": (
        ("gathering", 0.25, 101, "validation", "hierarchical", 32_768),
        ("isl", 0.25, 101, "baseline-retry", "hierarchical", 32_768),
        ("isl", 0.5, 101, "baseline-retry", "hierarchical", 32_768),
        ("transcript", 0.25, 101, "validation", "hierarchical", 32_768),
    ),
    # With the exact count only the transcript routes hierarchical on its own
    # at the study's context; every other study case fits one direct request.
    "d11-e2e": (("transcript", 0.25, 101, None, "auto", 32_768),),
    # #109 stage 1: development cases, which all route direct at 32,768. `d7-a`
    # runs each case in full and saves its editorial record; `d7-c` reuses that
    # record without its content units, so only the units differ.
    "d7-a": (
        ("atomic_habits", 0.25, 101, None, "auto", 32_768),
        ("nvc", 0.25, 101, None, "auto", 32_768),
        ("isl", 0.25, 101, None, "auto", 32_768),
    ),
    "d7-c": (
        ("atomic_habits", 0.25, 101, None, "auto", 32_768, ("d7-a", "none")),
        ("nvc", 0.25, 101, None, "auto", 32_768, ("d7-a", "none")),
        ("isl", 0.25, 101, None, "auto", 32_768, ("d7-a", "none")),
    ),
}
PHASES = ("pilot", "pilot-retry", "baseline", "baseline-retry", "validation", *WORKTREE_CASES)
# Sources outside the manifest; they are repository sample documents.
EXTRA_SOURCES = {
    "snowfall": HERE.parents[1] / "sample-documents" / "article_Snow Fall The Avalanche at Tunnel Creek by John Branch.txt",
}


def _sources(documents):
    for document in documents:
        source = ROOT / "corpus" / f"{document}.txt"
        if not source.is_file():
            raise FileNotFoundError(source)
        text = source.read_text(encoding="utf-8")
        # Structural page markers help attribution but are not source words.
        words = sum(
            len(line.split())
            for line in text.splitlines()
            if not line.startswith(("=== PDF PAGE ", "===== PDF page ", "===== PDF PAGE "))
        )
        expected = MANIFEST["sources"][document]
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if words != expected["words"] or actual_hash != expected["sha256"]:
            raise RuntimeError(f"canonical source changed for {document}: {words} words, {actual_hash}")
        if words == 0:
            raise ValueError(f"empty source: {source}")
        yield document, source, words


def _saved_trial(saved_phase: str, document: str, fraction: float, seed: int, version: str = "current") -> Path:
    label = f"{document}-{int(fraction * 100)}-seed{seed}-{version}"
    trials = sorted((ROOT / "runs" / saved_phase / label).glob(f"{version}-*"))
    if len(trials) != 1:
        raise RuntimeError(f"expected one saved trial under {saved_phase}/{label}, found {len(trials)}")
    return trials[0]


def cases(phase: str, document_filter: str | None = None):
    """Yield the phase's cases, only for `document_filter` when given.

    The filter applies before saved trials are looked up, so one document's
    cases can run while another's saved trials don't exist yet.
    """
    if phase in WORKTREE_CASES:
        chosen = [case for case in WORKTREE_CASES[phase] if document_filter in (None, case[0])]
        documents = sorted({case[0] for case in chosen})
        sources = {document: (source, words) for document, source, words in _sources(
            [document for document in documents if document not in EXTRA_SOURCES])}
        for document in documents:
            if document in EXTRA_SOURCES:
                path = EXTRA_SOURCES[document]
                sources[document] = (path, len(path.read_text(encoding="utf-8").split()))
        for document, amount, seed, saved_phase, strategy, num_ctx, *editorial in chosen:
            source, words = sources[document]
            fraction = amount if isinstance(amount, float) else None
            target = round(words * amount) if fraction is not None else amount
            replay = None if saved_phase is None else _saved_trial(saved_phase, document, fraction, seed)
            editorial_from, units = (None, "root")
            if editorial:
                editorial_phase, units = editorial[0]
                editorial_from = _saved_trial(editorial_phase, document, fraction, seed, "worktree")
            yield (document, source, words, fraction, target, seed, "worktree", replay, strategy, num_ctx,
                   editorial_from, units)
        return
    documents = {
        "pilot": ["nvc"],
        "pilot-retry": ["nvc"],
        "baseline": ["atomic_habits", "nvc", "isl"],
        "baseline-retry": ["atomic_habits", "nvc", "isl"],
        "validation": ["gathering", "all_statistics", "transcript", "news"],
    }[phase]
    if document_filter is not None:
        documents = [document for document in documents if document == document_filter]
    seeds = MANIFEST["seeds"] if phase.startswith("baseline") else MANIFEST["seeds"][:1]
    sources = list(_sources(documents))
    # Finish every document and length at one seed before beginning the next trial.
    for seed in seeds:
        for document, source, words in sources:
            for fraction in MANIFEST["target_fractions"]:
                target = round(words * fraction)
                for version in MANIFEST["versions"]:
                    yield document, source, words, fraction, target, seed, version, None, "auto", 32_768


def run_one(phase: str, document: str, source: Path, words: int, fraction: float | None, target: int, seed: int,
            version: str, replay_from: Path | None, strategy: str, num_ctx: int,
            editorial_from: Path | None = None, editorial_units: str = "root") -> dict:
    amount = f"{int(fraction * 100)}" if fraction is not None else f"{target}"
    label = f"{document}-{amount}-seed{seed}-{version}"
    if strategy != "auto":
        label += f"-{strategy}"
    if num_ctx != 32_768:
        label += f"-ctx{num_ctx}"
    run_dir = ROOT / "runs" / phase / label
    if run_dir.exists():
        raise FileExistsError(f"independent trial already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    trace_dir = run_dir / "trace"
    proxy_command = [sys.executable, str(HERE / "capture_proxy.py"), "--seed", str(seed),
                     "--trace-dir", str(trace_dir), "--num-ctx", str(num_ctx)]
    if version != "worktree":
        # Pinned revisions keep the frozen protocol allowance where they send none.
        proxy_command.extend(["--max-output-tokens", str(max(2048, 3 * target + 1024))])
    proxy = subprocess.Popen(proxy_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started = time.time()
    try:
        assert proxy.stdout is not None
        proxy_url = proxy.stdout.readline().strip()
        if not proxy_url.startswith("http://127.0.0.1:"):
            raise RuntimeError(f"capture proxy failed to start: {proxy_url}")
        runner = HERE / ("run_legacy.py" if version == "legacy" else "run_modern.py")
        command = [sys.executable, str(runner), "--source", str(source), "--target", str(target),
                   "--seed", str(seed), "--output-dir", str(run_dir),
                   "--ollama-url" if version == "legacy" else "--proxy-url", proxy_url]
        if version != "legacy":
            command.extend(["--version", version])
        if replay_from is not None:
            command.extend(["--replay-from", str(replay_from)])
        if strategy != "auto":
            command.extend(["--strategy", strategy])
        if num_ctx != 32_768:
            command.extend(["--context-window", str(num_ctx)])
        if editorial_from is not None:
            command.extend(["--editorial-root-from", str(editorial_from), "--editorial-units", editorial_units])
        with (run_dir / "runner.stdout").open("w", encoding="utf-8") as out, (run_dir / "runner.stderr").open("w", encoding="utf-8") as err:
            result = subprocess.run(command, stdout=out, stderr=err, check=False)
        run = next((json.loads(path.read_text(encoding="utf-8")) for path in run_dir.glob("*/run.json")), {})
        record = {"label": label, "phase": phase, "version": version,
                  "revision": MANIFEST["versions"].get(version, run.get("revision")), "source": document,
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_words": words,
                  "target_fraction": fraction, "target_words": target, "seed": seed,
                  "strategy": strategy, "num_ctx": num_ctx,
                  "max_output_tokens": run.get("max_output_tokens", max(2048, 3 * target + 1024)),
                  "replay_from": None if replay_from is None else str(replay_from),
                  "editorial_root_from": None if editorial_from is None else str(editorial_from),
                  "editorial_units": editorial_units,
                  "exit_code": result.returncode, "elapsed_seconds": time.time() - started,
                  "trace_count": len(list(trace_dir.glob("request-*.json")))}
        (run_dir / "trial.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record
    finally:
        proxy.terminate()
        try:
            proxy.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proxy.kill()
            proxy.communicate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--document", choices=(*MANIFEST["sources"], *EXTRA_SOURCES), help="restrict to one document")
    parser.add_argument("--version", choices=(*MANIFEST["versions"], "worktree"), help="restrict to one version")
    parser.add_argument("--seed", type=int, help="restrict to one seed")
    args = parser.parse_args()
    verify_environment()
    selected = [case for case in cases(args.phase, args.document)
                if (args.seed is None or case[5] == args.seed)
                and (args.version is None or case[6] == args.version)]
    if not selected:
        parser.error("no cases match the requested filters")
    failed = False
    for case in selected:
        record = run_one(args.phase, *case)
        print(json.dumps(record), flush=True)
        failed |= record["exit_code"] != 0
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
