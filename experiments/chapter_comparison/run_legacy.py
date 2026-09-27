#!/usr/bin/env python3
"""Run the pinned 2008717 summarizer against local Ollama without importing main.py.

Only the original function definitions and literal settings are imported from the
pinned source. Import-time downloads, client initialization, logging setup, and the
legacy command-line file workflow are deliberately not executed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import requests

LEGACY_REVISION = "2008717136ed235dec000ea281b31c4f4aeed43b"
MODEL = "gemma4:latest"
MODEL_DIGEST = "c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb"
DEFAULT_LEGACY_MAIN = Path("/private/tmp/rts-study-20260926/checkouts/legacy/main.py")
MAX_PASSES = 8
STALL_REDUCTION_PERCENT = 2.0
NUM_CTX = 32768


def _safe_legacy_namespace(
    source_path: Path,
    *,
    save_file: Callable[[str, str], None] | None = None,
    sent_tokenize: Callable[[str], list[str]] | None = None,
) -> dict[str, Any]:
    """Compile only selected original definitions, never the module's top-level code."""
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    wanted = {"remove_extra_whitespace", "chunk_text_by_sentences", "summarize_with_gpt"}
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    }
    missing = wanted - definitions.keys()
    if missing:
        raise ValueError(f"Pinned main.py is missing legacy functions: {', '.join(sorted(missing))}")
    if any(not isinstance(definitions[name], ast.FunctionDef) for name in wanted):
        raise ValueError("Pinned main.py uses an unsupported async legacy function")

    settings: dict[str, Any] = {}
    setting_names = {"DRY_RUN", "DELIMITER", "CHUNK_SIZE", "GPT_REQUEST_TIMEOUT", "GPT_REQUEST_MAX_RETRY"}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        for name in names & setting_names:
            settings[name] = ast.literal_eval(node.value)
    if setting_names - settings.keys():
        raise ValueError(f"Pinned main.py is missing literal settings: {', '.join(sorted(setting_names - settings.keys()))}")

    if sent_tokenize is None:
        # Import only the tokenizer; never call nltk.download or load main.py imports.
        from nltk.tokenize import sent_tokenize as nltk_sent_tokenize

        sent_tokenize = nltk_sent_tokenize

    namespace: dict[str, Any] = {
        "__name__": "_pinned_legacy_main_functions",
        "__file__": str(source_path),
        "re": re,
        "requests": requests,
        "logging": logging,
        "time": time.time,
        "sleep": time.sleep,
        "sent_tokenize": sent_tokenize,
        "save_file": save_file or (lambda _content, _path: None),
        "client": None,
        **settings,
    }
    selected = ast.Module(body=[definitions[name] for name in (
        "remove_extra_whitespace", "chunk_text_by_sentences", "summarize_with_gpt"
    )], type_ignores=[])
    exec(compile(ast.fix_missing_locations(selected), str(source_path), "exec"), namespace)
    return namespace


def _api_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/api/chat") else normalized + "/api/chat"


class OllamaChatClient:
    """Small OpenAI-shaped transport required by the untouched legacy function."""

    def __init__(
        self,
        *,
        base_url: str,
        seed: int,
        num_predict: int,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self.endpoint = _api_endpoint(base_url)
        self.seed = seed
        self.num_predict = num_predict
        self._post = post or requests.post
        self.calls: list[dict[str, Any]] = []
        self.current_call: dict[str, Any] | None = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def begin_call(self, *, pass_number: int, chunk_number: int) -> dict[str, Any]:
        record: dict[str, Any] = {
            "pass": pass_number,
            "chunk": chunk_number,
            "attempts": [],
            "started_at": time.monotonic(),
        }
        self.current_call = record
        return record

    def create(self, *, model: str, messages: list[dict[str, str]], timeout: float) -> Any:
        del model  # The original OpenAI model alias is intentionally replaced by the pinned local model.
        record = self.current_call
        if record is None:
            record = self.begin_call(pass_number=0, chunk_number=0)
        request_body = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": NUM_CTX,
                "num_predict": self.num_predict,
                "temperature": 1,
                "top_k": 64,
                "top_p": 0.95,
                "seed": self.seed,
            },
        }
        attempt: dict[str, Any] = {"number": len(record["attempts"]) + 1}
        started = time.monotonic()
        try:
            response = self._post(self.endpoint, json=request_body, timeout=timeout)
            attempt["http_status"] = getattr(response, "status_code", None)
            response.raise_for_status()
            payload = response.json()
            content = payload["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Ollama response message.content was not text")
            attempt["ok"] = True
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        except Exception as exc:
            attempt["ok"] = False
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            if "http_status" not in attempt:
                attempt["http_status"] = getattr(locals().get("response"), "status_code", None)
            raise
        finally:
            attempt["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            record["attempts"].append(attempt)

    def end_call(self) -> dict[str, Any]:
        record = self.current_call
        if record is None:
            raise RuntimeError("No active legacy model call")
        record["elapsed_ms"] = round((time.monotonic() - record.pop("started_at")) * 1000, 3)
        record["retry_count"] = max(0, len(record["attempts"]) - 1)
        record["successful_attempts"] = sum(1 for attempt in record["attempts"] if attempt["ok"])
        self.calls.append(record)
        self.current_call = None
        return record


def _word_count(text: str) -> int:
    return len(text.split())


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()


class LegacyRun:
    def __init__(
        self,
        *,
        source: Path,
        target_words: int,
        seed: int,
        output_dir: Path,
        ollama_url: str,
        legacy_main: Path = DEFAULT_LEGACY_MAIN,
    ) -> None:
        self.source = source.resolve()
        self.target_words = target_words
        self.seed = seed
        self.output_dir = output_dir.resolve()
        self.ollama_url = ollama_url
        self.legacy_main = legacy_main.resolve()
        self.num_predict = max(2048, math.ceil(3 * target_words) + 1024)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.output_dir / "calls.jsonl"
        self.passes_path = self.output_dir / "passes.jsonl"
        self.calls_path.write_text("", encoding="utf-8")
        self.passes_path.write_text("", encoding="utf-8")
        self.transport = OllamaChatClient(
            base_url=ollama_url, seed=seed, num_predict=self.num_predict
        )
        self.legacy = _safe_legacy_namespace(self.legacy_main)
        self.legacy["client"] = self.transport
        self.legacy["save_file"] = lambda _content, _path: None


    def _summarize_chunk(self, text: str, pass_number: int, chunk_number: int) -> str:
        self.transport.begin_call(pass_number=pass_number, chunk_number=chunk_number)
        summary: str | None = None
        failure: Exception | None = None
        try:
            # Keep the original argument/model alias; transport maps it to gemma4:latest.
            summary = self.legacy["summarize_with_gpt"](text, _model="gpt-4")
        except Exception as exc:
            failure = exc
        record = self.transport.end_call()
        if summary is not None:
            record["normalized_output"] = summary
        call_artifact = {
            "pass": pass_number,
            "chunk": chunk_number,
            "input_word_count": _word_count(text),
            "output_word_count": _word_count(summary or ""),
            "normalized_output": summary,
            "attempt_count": len(record["attempts"]),
            "retry_count": record["retry_count"],
            "status": "failed" if failure is not None or not record["successful_attempts"] else "ok",
            "failed_attempt_count": len(record["attempts"]) - record["successful_attempts"],
            "successful_attempts": record["successful_attempts"],
            "elapsed_ms": record["elapsed_ms"],
        }
        if failure is not None:
            call_artifact["exception"] = f"{type(failure).__name__}: {failure}"
        if not record["successful_attempts"]:
            call_artifact["failure"] = "No model attempt succeeded; see the capture-proxy trace for details."
        _append_jsonl(self.calls_path, call_artifact)
        if failure is not None:
            raise failure
        if not record["successful_attempts"]:
            raise RuntimeError(
                f"Legacy request failed after {len(record['attempts'])} attempts for pass {pass_number}, chunk {chunk_number}"
            )
        return summary or ""

    def _run_pass(self, text: str, pass_number: int) -> tuple[str, dict[str, Any]]:
        try:
            chunks = self.legacy["chunk_text_by_sentences"](text, self.legacy["CHUNK_SIZE"])
        except LookupError as exc:
            raise RuntimeError(
                "Legacy NLTK sentence-tokenizer data is unavailable; the runner never downloads it."
            ) from exc
        summaries = [
            self._summarize_chunk(chunk, pass_number, index)
            for index, chunk in enumerate(chunks, start=1)
        ]
        # This is the original main.py concatenation, without an added synthesis pass.
        combined = "\n".join(summaries)
        pass_path = self.output_dir / f"pass_{pass_number:02d}_output.txt"
        input_path = "source.txt" if pass_number == 1 else f"pass_{pass_number - 1:02d}_output.txt"
        pass_path.write_text(combined, encoding="utf-8")
        before_words = _word_count(text)
        after_words = _word_count(combined)
        reduction = ((before_words - after_words) / before_words * 100) if before_words else 0.0
        record = {
            "pass": pass_number,
            "input_words": before_words,
            "output_words": after_words,
            "reduction_percent": round(reduction, 6),
            "chunk_count": len(chunks),
            "retry_count": sum(call["retry_count"] for call in self.transport.calls if call["pass"] == pass_number),
            "elapsed_ms": round(sum(call["elapsed_ms"] for call in self.transport.calls if call["pass"] == pass_number), 3),
            "input_path": input_path,
            "output_path": pass_path.name,
        }
        _append_jsonl(self.passes_path, record)
        return combined, record

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        source_text = self.source.read_text(encoding="utf-8")
        (self.output_dir / "source.txt").write_text(source_text, encoding="utf-8")
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        current = source_text
        pass_records: list[dict[str, Any]] = []
        above_text: str | None = None
        below_text: str | None = None
        consecutive_stalls = 0
        stop_reason = "max_passes"
        error: Exception | None = None
        try:
            for pass_number in range(1, MAX_PASSES + 1):
                before_text = current
                current, pass_record = self._run_pass(before_text, pass_number)
                pass_records.append(pass_record)
                after_words = pass_record["output_words"]
                before_words = pass_record["input_words"]
                if before_words > self.target_words and after_words <= self.target_words:
                    above_text = before_text
                    below_text = current
                    stop_reason = "target_reached"
                    break
                if after_words <= self.target_words:
                    stop_reason = "target_reached"
                    break
                if pass_record["reduction_percent"] < STALL_REDUCTION_PERCENT:
                    consecutive_stalls += 1
                else:
                    consecutive_stalls = 0
                if consecutive_stalls >= 2:
                    stop_reason = "stalled"
                    break
        except Exception as exc:
            error = exc
            stop_reason = "failed"

        if above_text is not None:
            (self.output_dir / "immediately_above_target.txt").write_text(above_text, encoding="utf-8")
        if below_text is not None:
            (self.output_dir / "immediately_below_target.txt").write_text(below_text, encoding="utf-8")
        result_path = self.output_dir / "summary.txt"
        result_path.write_text(current, encoding="utf-8")
        final_words = _word_count(current)
        run_record: dict[str, Any] = {
            "status": "failed" if error else "completed",
            "version": 1,
            "revision": LEGACY_REVISION,
            "source": str(self.source),
            "source_sha256": source_hash,
            "source_words": _word_count(source_text),
            "raw_text_word_count": _word_count(source_text),
            "raw_text_word_count_basis": "Whitespace-delimited count of source text; page-marker tokens are retained.",
            "target_words": self.target_words,
            "seed": self.seed,
            "model": MODEL,
            "model_digest": MODEL_DIGEST,
            "ollama_url": self.ollama_url,
            "calls_path": self.calls_path.name,
            "passes_path": self.passes_path.name,
            "raw_trace_owner": "capture_proxy.py records raw attempts when --ollama-url is its trial proxy URL",
            "selected_strategy": "legacy_chunked_concat",
            "num_ctx": NUM_CTX,
            "num_predict": self.num_predict,
            "passes": pass_records,
            "pass_count": len(pass_records),
            "final_words": final_words,
            "stop_reason": stop_reason,
            "stalled": stop_reason == "stalled",
            "overshoot_words": max(0, self.target_words - final_words) if stop_reason == "target_reached" else 0,
            "immediately_above_target_path": "immediately_above_target.txt" if above_text is not None else None,
            "immediately_below_target_path": "immediately_below_target.txt" if below_text is not None else None,
            "output_path": result_path.name,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "failure": f"{type(error).__name__}: {error}" if error else None,
            "traceback": "".join(traceback.format_exception(error)) if error else None,
        }
        (self.output_dir / "run.json").write_text(
            json.dumps(run_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return run_record


def fidelity_check(legacy_main: Path = DEFAULT_LEGACY_MAIN) -> dict[str, Any]:
    """Prove prompt, response normalization, chunking, and concat match pinned source."""
    legacy_main = legacy_main.resolve()
    ref_logs: list[str] = []
    adapted_logs: list[str] = []
    reference = _safe_legacy_namespace(legacy_main, save_file=lambda content, _path: ref_logs.append(content))
    adapted = _safe_legacy_namespace(legacy_main, save_file=lambda content, _path: adapted_logs.append(content))
    raw_answer = "  Mock   summary\nfor fidelity.  "

    class DummyReferenceClient:
        def __init__(self) -> None:
            self.kwargs: list[dict[str, Any]] = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs: Any) -> Any:
            self.kwargs.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw_answer))])

    reference_client = DummyReferenceClient()
    reference["client"] = reference_client
    reference_text = "Alpha. Beta."
    expected_summary = reference["summarize_with_gpt"](reference_text, _model="gpt-4")

    posts: list[dict[str, Any]] = []

    class DummyHttpResponse:
        status_code = 200
        ok = True

        def __init__(self) -> None:
            self.payload = {"message": {"content": raw_answer}}
            self.text = json.dumps(self.payload)

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self.payload

    def dummy_post(url: str, *, json: dict[str, Any], timeout: float) -> DummyHttpResponse:
        posts.append({"url": url, "json": json, "timeout": timeout})
        return DummyHttpResponse()

    adapter_transport = OllamaChatClient(
        base_url="http://127.0.0.1:11434", seed=19, num_predict=2048, post=dummy_post
    )
    adapted["client"] = adapter_transport
    adapted["save_file"] = lambda content, _path: adapted_logs.append(content)
    adapter_transport.begin_call(pass_number=1, chunk_number=1)
    adapter_summary = adapted["summarize_with_gpt"](reference_text, _model="gpt-4")
    call_record = adapter_transport.end_call()

    reference_chunks = reference["chunk_text_by_sentences"]("Alpha. Beta. Gamma.", 12)
    adapted_chunks = adapted["chunk_text_by_sentences"]("Alpha. Beta. Gamma.", 12)
    expected_chunks = [" Alpha. Beta.", "Gamma."]
    whitespace_input = "  Alpha\n\tBeta   Gamma  "
    reference_whitespace = reference["remove_extra_whitespace"](whitespace_input)
    adapted_whitespace = adapted["remove_extra_whitespace"](whitespace_input)
    reference_concat = "\n".join([expected_summary, "Second."])
    adapted_concat = "\n".join([adapter_summary, "Second."])

    reference_messages = reference_client.kwargs[0]["messages"] if reference_client.kwargs else []
    adapter_messages = posts[0]["json"]["messages"] if posts else []
    expected_messages = [
        {
            "role": "system",
            "content": "You are a writing assistant, skilled in revising and summarizing complex technical writing with accuracy and precision.",
        },
        {
            "role": "user",
            "content": (
                "Provide an executive summary of the following text (delimited by triple quotes). "
                "Present the key ideas and findings directly, without bullet points, "
                "as if for a busy professional who needs to grasp the essential points quickly. "
                "Ignore complete sentences and grammatical correctness. "
                "Abbreviate long and repetitive words. "
                + reference["DELIMITER"] + reference_text + reference["DELIMITER"]
            ),
        },
    ]
    checks = {
        "chunk_reference_literal": reference_chunks == expected_chunks,
        "adapter_chunk_matches_reference": adapted_chunks == reference_chunks,
        "whitespace_reference_literal": reference_whitespace == "Alpha Beta Gamma",
        "adapter_whitespace_matches_reference": adapted_whitespace == reference_whitespace,
        "prompt_roles_match_original_reference": [message["role"] for message in adapter_messages] == ["system", "user"] == [message["role"] for message in reference_messages],
        "prompt_content_matches_original_reference": [message["content"] for message in adapter_messages] == [message["content"] for message in expected_messages] == [message["content"] for message in reference_messages],
        "full_prompt_matches_original_reference": bool(reference_messages) and adapter_messages == reference_messages == expected_messages,
        "normalized_output_matches_reference": adapter_summary == expected_summary == "Mock summary for fidelity.",
        "concat_matches_reference": adapted_concat == reference_concat,
        "concat_original_literal": adapted_concat == "Mock summary for fidelity.\nSecond.",
        "safe_original_log_matches": bool(ref_logs) and adapted_logs == ref_logs,
        "sampling_and_context_options": posts[0]["json"]["options"] == {
            "num_ctx": NUM_CTX, "num_predict": 2048, "temperature": 1,
            "top_k": 64, "top_p": 0.95, "seed": 19,
        },
        "single_successful_adapter_attempt": len(call_record["attempts"]) == 1 and call_record["successful_attempts"] == 1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {"passed": not failed, "checks": checks, "failed_checks": failed, "live_generation": False}
    if failed:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pinned legacy chapter summarizer through local Ollama.")
    parser.add_argument("--source", type=Path, help="Canonical chapter/corpus text file")
    parser.add_argument("--target", type=int, help="Target word count")
    parser.add_argument("--seed", type=int, help="One trial seed shared by all chat requests")
    parser.add_argument("--output-dir", type=Path, help="Unique trial artifact directory")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama or capture-proxy base URL")
    parser.add_argument("--legacy-main", type=Path, default=Path(os.environ.get("RTS_LEGACY_MAIN", str(DEFAULT_LEGACY_MAIN))), help=argparse.SUPPRESS)
    parser.add_argument("--fidelity-check", "--check-fidelity", dest="fidelity_check", action="store_true", help="Run an offline original-source fidelity check; no model request")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.fidelity_check:
        try:
            print(json.dumps(fidelity_check(args.legacy_main), indent=2))
            return 0
        except Exception as exc:
            print(f"Fidelity check failed: {exc}", file=sys.stderr)
            return 1
    missing = [name for name in ("source", "target", "seed", "output_dir") if getattr(args, name) is None]
    if missing:
        parser.error("required for a run: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if args.target <= 0:
        parser.error("--target must be a positive word count")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    try:
        run = LegacyRun(
            source=args.source, target_words=args.target, seed=args.seed,
            output_dir=args.output_dir, ollama_url=args.ollama_url, legacy_main=args.legacy_main,
        ).run()
    except Exception as exc:
        print(f"Legacy run setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(run, ensure_ascii=False, indent=2))
    return 1 if run["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
