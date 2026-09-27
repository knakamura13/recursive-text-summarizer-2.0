"""Run pinned local chapter comparisons sequentially with independent traces.

Corpus and generated content remain in the artifact root outside the repository.
Run --phase pilot first; freeze changes before --phase validation.
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


def cases(phase: str):
    documents = {
        "pilot": ["nvc"],
        "pilot-retry": ["nvc"],
        "baseline": ["atomic_habits", "nvc", "isl"],
        "baseline-retry": ["atomic_habits", "nvc", "isl"],
        "validation": ["gathering", "all_statistics", "transcript", "news"],
    }[phase]
    seeds = MANIFEST["seeds"] if phase.startswith("baseline") else MANIFEST["seeds"][:1]
    sources = []
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
        sources.append((document, source, words))
    # Finish every document and length at one seed before beginning the next trial.
    for seed in seeds:
        for document, source, words in sources:
            for fraction in MANIFEST["target_fractions"]:
                target = round(words * fraction)
                for version in MANIFEST["versions"]:
                    yield document, source, words, fraction, target, seed, version


def run_one(phase: str, document: str, source: Path, words: int, fraction: float, target: int, seed: int, version: str) -> dict:
    label = f"{document}-{int(fraction * 100)}-seed{seed}-{version}"
    run_dir = ROOT / "runs" / phase / label
    if run_dir.exists():
        raise FileExistsError(f"independent trial already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    trace_dir = run_dir / "trace"
    proxy = subprocess.Popen(
        [sys.executable, str(HERE / "capture_proxy.py"), "--seed", str(seed),
         "--max-output-tokens", str(max(2048, 3 * target + 1024)),
         "--trace-dir", str(trace_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
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
        with (run_dir / "runner.stdout").open("w", encoding="utf-8") as out, (run_dir / "runner.stderr").open("w", encoding="utf-8") as err:
            result = subprocess.run(command, stdout=out, stderr=err, check=False)
        record = {"label": label, "phase": phase, "version": version,
                  "revision": MANIFEST["versions"][version], "source": document,
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_words": words,
                  "target_fraction": fraction, "target_words": target, "seed": seed,
                  "max_output_tokens": max(2048, 3 * target + 1024),
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
    parser.add_argument("--phase", choices=("pilot", "pilot-retry", "baseline", "baseline-retry", "validation"), required=True)
    parser.add_argument("--document", choices=MANIFEST["sources"], help="restrict to one document")
    parser.add_argument("--version", choices=MANIFEST["versions"], help="restrict to one version")
    parser.add_argument("--seed", type=int, help="restrict to one seed")
    args = parser.parse_args()
    verify_environment()
    selected = [case for case in cases(args.phase)
                if (args.document is None or case[0] == args.document)
                and (args.seed is None or case[5] == args.seed)
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
