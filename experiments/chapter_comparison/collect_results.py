#!/usr/bin/env python3
"""Collect chapter-comparison metrics into private, blinded JSON and Markdown.

Reads existing artifacts only; never contacts a model service. Generated summaries,
source text, prompts, raw trace bodies, revisions, and the blind key stay private.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
DEFAULT_ROOT = Path(MANIFEST["artifact_root"])


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return rows, 0
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
            else:
                invalid += 1
        except json.JSONDecodeError:            invalid += 1
    return rows, invalid


def _number(*values: Any) -> int | float | None:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return value
    return None


def blind_map(root: Path, versions: list[str], documents: list[str]) -> dict[str, dict[str, str]]:
    """Load or create stable secret-random per-document permutations."""
    private_dir = root / "results"
    private_dir.mkdir(parents=True, exist_ok=True)
    path = private_dir / ".blind_mapping.json"
    stored = read_json(path)
    if stored is not None and isinstance(stored.get("mapping"), dict) and isinstance(stored.get("key"), str):
        key = bytes.fromhex(stored["key"])
        mapping = stored["mapping"]
    else:
        key = secrets.token_bytes(32)
        mapping = {}
    labels = ("A", "B", "C")
    if len(versions) != len(labels):
        raise ValueError("blind comparison requires exactly three manifest versions")
    for document in documents:
        current = mapping.get(document)
        if isinstance(current, dict) and set(current) == set(versions) and set(current.values()) == set(labels):
            continue
        ordered = sorted(versions, key=lambda version: hmac.new(key, f"{document}\0{version}".encode(), hashlib.sha256).digest())
        mapping[document] = dict(zip(ordered, labels, strict=True))
    payload = {"key": key.hex(), "mapping": mapping}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return mapping


def _trace_summary(trial_dir: Path) -> dict[str, Any]:
    files = sorted(trial_dir.rglob("request-*.json"))
    statuses: list[int] = []
    elapsed: list[float] = []
    cap_hit = False
    for path in files:
        record = read_json(path)
        if not record:
            continue
        status = _number(record.get("status"))
        if status is not None:
            statuses.append(int(status))
        time_s = _number(record.get("elapsed_seconds"))
        if time_s is not None:
            elapsed.append(float(time_s))
        response = record.get("response")
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                response = {}
        if isinstance(response, dict) and response.get("done_reason") in ("length", "max_tokens"):
            cap_hit = True
    return {
        "request_attempts": len(files),
        "successful_requests": sum(200 <= status < 300 for status in statuses),
        "failed_requests": sum(status < 200 or status >= 300 for status in statuses),
        "trace_elapsed_seconds": round(sum(elapsed), 3) if elapsed else None,
        "output_capacity_hit": cap_hit if files else None,
    }


def _audit_summary(path: Path) -> dict[str, Any]:
    audit = read_json(path)
    if audit is None:
        return {"audit_present": path.exists(), "audit_valid": False, "audit_truncated": None}
    truncated: bool | None = None
    def walk(value: Any) -> None:
        nonlocal truncated
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = str(key).casefold().replace("-", "_")
                if normalized in {"truncated", "was_truncated", "output_truncated", "truncation_detected"} and isinstance(item, bool):
                    truncated = item if truncated is None else truncated or item
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(audit)
    return {"audit_present": True, "audit_valid": True, "audit_truncated": truncated}


def collect_trial(trial_path: Path, mapping: dict[str, dict[str, str]]) -> dict[str, Any]:
    trial = read_json(trial_path) or {}
    trial_dir = trial_path.parent
    nested_runs = sorted(trial_dir.rglob("run.json"))
    run_path = next((p for p in nested_runs if read_json(p) is not None), None)
    run = read_json(run_path) if run_path else {}
    run = run or {}
    phase = trial.get("phase")
    source_id = trial.get("source")
    # Older metadata sometimes stores a source path rather than corpus id.
    if isinstance(source_id, str):
        source_id = Path(source_id).stem
    source_id = str(source_id or trial.get("document") or "unknown")
    actual_version = str(trial.get("version") or run.get("version") or "unknown")
    label = mapping.get(source_id, {}).get(actual_version)
    if label is None:
        # Unknown/malformed versions remain explicitly unlabelled, never expose their name.
        label = "unmapped"
    passes_path = trial_dir / "passes.jsonl"
    calls_path = trial_dir / "calls.jsonl"
    passes, invalid_pass_rows = read_jsonl(passes_path)
    calls, invalid_call_rows = read_jsonl(calls_path)
    trace = _trace_summary(trial_dir)
    audit_path = next((p for p in trial_dir.rglob("audit.json") if p.is_file()), trial_dir / "audit.json")
    audit = _audit_summary(audit_path)
    status = run.get("status")
    if status is None:
        code = _number(trial.get("exit_code"))
        status = "completed" if code == 0 else ("failed" if code is not None else "incomplete")
    failures = sum(1 for call in calls if call.get("status") == "failed") + trace["failed_requests"]
    if status == "failed" and failures == 0:
        failures = 1
    explicit_retry_counts = [_number(call.get("retry_count")) for call in calls]
    explicit_retry_counts = [int(v) for v in explicit_retry_counts if v is not None]
    retry_count = sum(explicit_retry_counts) if explicit_retry_counts else None
    source_words = _number(trial.get("source_words"), run.get("source_words"), run.get("raw_text_word_count"))
    target_words = _number(trial.get("target_words"), run.get("target_words"))
    final_words = _number(run.get("final_words"), run.get("word_count"))
    if final_words is None and passes:
        final_words = _number(passes[-1].get("output_words"))
    capacity = _number(trial.get("max_output_tokens"), run.get("max_output_tokens"), run.get("num_predict"))
    context = _number(run.get("context_window"), run.get("effective_context_window"), run.get("num_ctx"))
    elapsed = _number(trial.get("elapsed_seconds"), run.get("elapsed_seconds"))
    if elapsed is None and calls:
        values = [_number(call.get("elapsed_ms")) for call in calls]
        elapsed = round(sum(v for v in values if v is not None) / 1000, 3) if any(v is not None for v in values) else None
    return {
        "phase": str(phase or "unknown"), "document": source_id,
        "target_fraction": _number(trial.get("target_fraction")), "target_words": target_words,
        "seed": _number(trial.get("seed"), run.get("seed")), "condition": label,
        "status": str(status), "failure_count": failures,
        "failure_present": bool(run.get("failure") or failures),
        "calls": len(calls) if calls else None,
        "request_attempts": trace["request_attempts"],
        "successful_requests": trace["successful_requests"], "failed_requests": trace["failed_requests"],
        "retry_count": retry_count, "elapsed_seconds": elapsed,
        "trace_elapsed_seconds": trace["trace_elapsed_seconds"],
        "source_words": source_words, "target_words_observed": target_words,
        "final_words": final_words, "context_tokens": context,
        "output_capacity_tokens": capacity,
        "output_capacity_hit": trace["output_capacity_hit"],
        "audit_truncated": audit["audit_truncated"],
        "audit_present": audit["audit_present"], "audit_valid": audit["audit_valid"],
        "pass_count": _number(run.get("pass_count")) if run.get("pass_count") is not None else (len(passes) or None),
        "passes": [{k: v for k, v in p.items() if k in {"pass", "input_words", "output_words", "reduction_percent", "chunk_count", "retry_count", "elapsed_ms"}} for p in passes],
        "artifact_completeness": {
            "trial_metadata": True, "run_metadata": run_path is not None,
            "passes_file": passes_path.exists(), "calls_file": calls_path.exists(),
            "valid_pass_rows": len(passes), "invalid_pass_rows": invalid_pass_rows,
            "valid_call_rows": len(calls), "invalid_call_rows": invalid_call_rows,
            "trace_files": trace["request_attempts"], "audit_present": audit["audit_present"],
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def make_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Blinded chapter comparison results", "", "Only aggregate metrics are included; raw summaries, source text, prompts, traces, and version identities are excluded.", ""]
    docs = sorted({r["document"] for r in rows})
    phases = sorted({r["phase"] for r in rows})
    for phase in phases:
        lines += [f"## Phase: {phase}", ""]
        for doc in docs:
            subset = [r for r in rows if r["phase"] == phase and r["document"] == doc]
            if not subset:
                continue
            lines += [f"### {doc}", "", "Each cell lists status / failures / calls / request attempts / retries / elapsed seconds / final words / capacity tokens / cap hit / truncated. Missing values remain —.", "", "| Target | Seed | A | B | C |", "|---:|---:|:---|:---|:---|"]
            by_case: dict[tuple[Any, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
            for r in subset:
                by_case[(r["target_words"], r["seed"])][r["condition"]] = r
            for (target, seed), conditions in sorted(by_case.items(), key=lambda x: (x[0][0] or 0, x[0][1] or 0)):
                cells = []
                for condition in ("A", "B", "C"):
                    r = conditions.get(condition)
                    if r is None:
                        cells.append("absent")
                    else:
                        vals = [r["status"], r["failure_count"], r["calls"], r["request_attempts"], r["retry_count"], r["elapsed_seconds"], r["final_words"], r["output_capacity_tokens"], r["output_capacity_hit"], r["audit_truncated"]]
                        cells.append(" / ".join(_fmt(v) for v in vals))
                lines.append("| " + " | ".join([_fmt(target), _fmt(seed), *cells]) + " |")
            lines += ["", "#### Three-seed variation", "", "| Target | Condition | Seeds observed | Final words (mean; range) | Elapsed seconds (mean; range) | Calls (mean; range) | Retries (mean; range) |", "|---:|:---:|---:|---:|---:|---:|---:|"]
            groups: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
            for row in subset:
                groups[(row["target_words"], row["condition"])].append(row)
            for (target, condition), group in sorted(groups.items(), key=lambda x: (x[0][0] or 0, x[0][1])):
                metrics = ("final_words", "elapsed_seconds", "calls", "retry_count")
                values = []
                for metric in metrics:
                    numbers = [float(r[metric]) for r in group if isinstance(r.get(metric), (int, float)) and not isinstance(r.get(metric), bool)]
                    values.append(f"{statistics.mean(numbers):.2f}; {min(numbers):.2f}–{max(numbers):.2f}" if numbers else "—")
                seeds = sorted({str(r["seed"]) for r in group if r["seed"] is not None})
                lines.append("| " + " | ".join([_fmt(target), condition, f"{len(seeds)} ({', '.join(seeds)})", *values]) + " |")
            lines.append("")
    if not rows:
        lines += ["No trial.json files were found under the selected runs directory.", ""]
    return "\n".join(lines)


def _published_summary(trial_path: Path) -> tuple[str | None, str]:
    """Load only a successful final output artifact contained in this trial."""
    trial_dir = trial_path.parent.resolve()
    run_path = next((p for p in sorted(trial_dir.rglob("run.json")) if read_json(p) is not None), None)
    run = read_json(run_path) if run_path else {}
    run = run or {}
    trial = read_json(trial_path) or {}
    exit_code = _number(trial.get("exit_code"))
    fallback_status = "completed" if exit_code == 0 else ("failed" if exit_code is not None else "incomplete")
    status = str(run.get("status") or fallback_status)
    if status.casefold() in {"failed", "failure", "running", "incomplete"}:
        return None, status
    base = run_path.parent if run_path else trial_dir
    candidates: list[Path] = []
    for field in ("summary_path", "output_path"):
        value = run.get(field)
        if isinstance(value, str) and value.strip():
            candidate = Path(value)
            candidates.append(candidate if candidate.is_absolute() else base / candidate)
    candidates.extend(sorted(trial_dir.rglob("summary.txt")))
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(trial_dir)
            if not resolved.is_file():
                continue
            content = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            continue
        if content.strip():
            return content, status
    return None, status


def write_side_by_side(root: Path, entries: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    output = root / "results" / "side_by_side"
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, Any, Any], dict[str, list[tuple[str | None, str]]]] = defaultdict(lambda: defaultdict(list))
    for trial_path, row in entries:
        content, status = _published_summary(trial_path)
        key = (row["phase"], row["document"], row["target_words"], row["seed"])
        grouped[key][row["condition"]].append((content, status))
    filename_counts: dict[str, int] = defaultdict(int)
    names: dict[tuple[str, str, Any, Any], str] = {}
    for phase, document, target, seed in grouped:
        fraction = next((row.get("target_fraction") for _, row in entries if row["phase"] == phase and row["document"] == document and row["target_words"] == target and row["seed"] == seed), None)
        pct = f"{round(fraction * 100):02d}" if isinstance(fraction, (int, float)) else f"target{target if target is not None else 'unknown'}"
        safe_document = re.sub(r"[^A-Za-z0-9_-]+", "_", document).strip("_") or "unknown"
        stem = f"{safe_document}-{pct}-seed{seed if seed is not None else 'unknown'}"
        filename_counts[stem] += 1
        names[(phase, document, target, seed)] = stem
    for key, count in filename_counts.items():
        if count > 1:
            for group_key in names:
                if names[group_key] == key:
                    names[group_key] = f"{key}-{group_key[0]}"
    written: list[str] = []
    for group_key, conditions in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or 0, item[0][3] or 0)):
        phase, document, target, seed = group_key
        name = names[group_key] + ".md"
        destination = output / name
        checklist = root / "corpus" / f"{document}.checklist.md" if document in MANIFEST["sources"] else root / "corpus" / "__missing__"
        checklist_link = f"[source-backed checklist](../../corpus/{document}.checklist.md)" if checklist.is_file() else "source-backed checklist unavailable"
        lines = [f"# {document} — target {target if target is not None else 'unknown'} words — seed {seed if seed is not None else 'unknown'}", "", f"Phase: {phase}. Source: {checklist_link}.", "", "Version identities are intentionally withheld until evaluation is complete.", ""]
        for condition in ("A", "B", "C"):
            lines.extend([f"## {condition}", ""])
            trials = conditions.get(condition, [])
            if not trials:
                lines.extend(["**No trial artifact found for this condition.**", ""])
                continue
            for content, status in trials:
                if content is None:
                    if status.casefold() == "failed":
                        lines.extend([f"**Failure — no output published** (run status: {status}).", ""])
                    else:
                        lines.extend([f"**No final output available** (run status: {status}; summary artifact missing or empty).", ""])
                else:
                    lines.extend([content.rstrip(), ""])
        destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        written.append(f"side_by_side/{name}")
    return written


def write_combined_pilot(root: Path, mapping: dict[str, dict[str, str]]) -> list[str]:
    """Build seed-101 NVC pilot comparisons, substituting only setup-failed retries."""
    pilot_root = root / "runs" / "pilot"
    retry_root = root / "runs" / "pilot-retry"
    pilot_paths = sorted(pilot_root.rglob("trial.json")) if pilot_root.exists() else []
    retry_paths = sorted(retry_root.rglob("trial.json")) if retry_root.exists() else []
    originals: dict[tuple[int, int], dict[str, tuple[Path, dict[str, Any]]]] = defaultdict(dict)
    retries: dict[tuple[int, int], dict[str, tuple[Path, dict[str, Any]]]] = defaultdict(dict)
    for path in pilot_paths + retry_paths:
        metadata = read_json(path)
        if not metadata or metadata.get("source") != "nvc" or metadata.get("seed") != 101:
            continue
        fraction = metadata.get("target_fraction")
        target = metadata.get("target_words")
        if fraction not in (0.25, 0.5) or not isinstance(target, int):
            continue
        version = str(metadata.get("version") or "unknown")
        row = collect_trial(path, mapping)
        key = (round(fraction * 100), target)
        if path.is_relative_to(pilot_root):
            if metadata.get("phase") == "pilot":
                originals[key][version] = (path, row)
        elif metadata.get("phase") == "pilot-retry":
            content, status = _published_summary(path)
            if content is not None and status.casefold() in {"completed", "succeeded"}:
                retries[key][version] = (path, row)
    output = root / "results" / "side_by_side"
    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for (pct, target), by_version in sorted(originals.items()):
        chosen: dict[str, tuple[Path, dict[str, Any]]] = {}
        for version, original in by_version.items():
            path, row = original
            replacement = retries.get((pct, target), {}).get(version)
            if row["status"] == "failed" and row["request_attempts"] == 0 and replacement is not None:
                chosen[version] = replacement
            else:
                chosen[version] = original
        filename = f"pilot-combined-nvc-{pct:02d}-seed101.md"
        destination = output / filename
        checklist = root / "corpus" / "nvc.checklist.md"
        checklist_link = "[source-backed checklist](../../corpus/nvc.checklist.md)" if checklist.is_file() else "source-backed checklist unavailable"
        lines = [f"# Combined pilot — NVC — target {target} words — seed 101", "", f"Source: {checklist_link}.", "", "This private comparison uses successful retry output only where the original pilot condition failed before making any model request. Original trial records and failure metrics remain in the aggregate results.", "", "Version identities are intentionally withheld until evaluation is complete.", ""]
        for condition in ("A", "B", "C"):
            lines.extend([f"## {condition}", ""])
            version = next((name for name, label in mapping.get("nvc", {}).items() if label == condition), None)
            selected = chosen.get(version) if version is not None else None
            if selected is None:
                lines.extend(["**No trial artifact found for this condition.**", ""])
                continue
            selected_path, selected_row = selected
            content, status = _published_summary(selected_path)
            if content is None:
                if status.casefold() == "failed":
                    lines.extend([f"**Failure — no output published** (run status: {status}).", ""])
                else:
                    lines.extend([f"**No final output available** (run status: {status}; summary artifact missing or empty).", ""])
            else:
                lines.extend([content.rstrip(), ""])
        destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        written.append(f"side_by_side/{filename}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="private study artifact root")
    parser.add_argument("--phase", action="append", help="limit to phase (repeatable)")
    parser.add_argument("--combine-pilot", action="store_true", help="write combined NVC pilot comparisons using successful setup-failure retries")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    runs_root = root / "runs"
    trial_paths = sorted(runs_root.rglob("trial.json")) if runs_root.exists() else []
    trials = [(path, read_json(path)) for path in trial_paths]
    versions = list(MANIFEST["versions"])
    documents = sorted({str((item or {}).get("source") or (item or {}).get("document") or "unknown") for _, item in trials})
    mapping = blind_map(root, versions, documents or list(MANIFEST["sources"]))
    row_entries = [(path, collect_trial(path, mapping)) for path, item in trials if item is not None and (not args.phase or item.get("phase") in args.phase)]
    row_entries.sort(key=lambda pair: (pair[1]["phase"], pair[1]["document"], pair[1]["target_words"] or 0, pair[1]["seed"] or 0, pair[1]["condition"]))
    rows = [row for _, row in row_entries]
    output = root / "results"
    output.mkdir(parents=True, exist_ok=True)
    summary_files = write_side_by_side(root, row_entries)
    if args.combine_pilot:
        summary_files.extend(write_combined_pilot(root, mapping))
    (output / "results.json").write_text(json.dumps({"study": MANIFEST["study"], "rows": rows, "side_by_side_files": summary_files}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "results.md").write_text(make_markdown(rows), encoding="utf-8")
    print(f"Collected {len(rows)} trial(s); outputs: {output / 'results.json'} and {output / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

