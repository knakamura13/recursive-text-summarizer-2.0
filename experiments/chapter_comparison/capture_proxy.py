"""Record complete Ollama requests/responses while pinning controlled sampling.

Run only on localhost. Captures can contain entire copyrighted source passages;
keep --trace-dir outside the repository and do not distribute it.

Sampling (seed, temperature, top_k, top_p), the context window and ``think`` are
pinned on every chat request. A candidate's own ``num_predict`` is forwarded
unchanged; ``--max-output-tokens`` only fills it in when a candidate omits it,
as the legacy transport does. Besides the full private trace, each chat
request appends one text-free row to ``requests.jsonl``: original and effective
options, overridden keys, status, stop reason, token counts, Ollama error text,
repeats of an identical earlier request and elapsed time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Options the proxy reports for every request, pinned or not.
REPORTED_OPTIONS = ("seed", "temperature", "top_k", "top_p", "num_ctx", "num_predict")
TRUNCATION_REASONS = frozenset({"length", "max_tokens"})
_ERROR_LIMIT = 500


def pin_request(
    data: dict[str, Any], *, seed: int, num_ctx: int, fill_num_predict: int | None
) -> list[str]:
    """Apply the protocol's pins in place; return the keys whose value changed."""
    options = data.setdefault("options", {})
    before = {key: options.get(key) for key in REPORTED_OPTIONS}
    before_think = data.get("think")
    options.update(seed=seed, temperature=1, top_k=64, top_p=0.95, num_ctx=num_ctx)
    if fill_num_predict is not None:
        options.setdefault("num_predict", fill_num_predict)
    # The model's default thinking mode is on; the modern provider
    # explicitly disables it. Match that behavior for legacy transport.
    data["think"] = False
    changed = [key for key in REPORTED_OPTIONS if options.get(key) != before[key]]
    if before_think is not False:
        changed.append("think")
    return changed


def _final_response(response: bytes) -> dict[str, Any]:
    """Return an Ollama response's JSON object; both candidates send stream=false."""
    try:
        value = json.loads(response)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def request_metadata(
    *,
    index: int,
    started: float,
    elapsed: float,
    status: int,
    requested: dict[str, Any],
    effective: dict[str, Any],
    effective_body: bytes,
    overridden: list[str],
    response: bytes,
    transport_error: str | None,
    first_index_by_hash: dict[str, int],
) -> dict[str, Any]:
    """Describe one exchange without any prompt, source or response text."""
    original_options = requested.get("options") or {}
    effective_options = effective.get("options") or {}
    final = _final_response(response)
    error = transport_error
    if error is None and not 200 <= status < 300:
        error = str(final.get("error") or response.decode("utf-8", errors="replace"))
    digest = hashlib.sha256(effective_body).hexdigest()
    repeat_of = first_index_by_hash.setdefault(digest, index)
    schema = effective.get("format")
    effective_predict = effective_options.get("num_predict")
    eval_count = final.get("eval_count")
    return {
        "index": index,
        "started": started,
        "elapsed_seconds": elapsed,
        "status": status,
        "model": effective.get("model"),
        "schema_title": schema.get("title") if isinstance(schema, dict) else None,
        "request_sha256": digest,
        "repeat_of": None if repeat_of == index else repeat_of,
        "request_bytes": len(effective_body),
        "original_options": {key: original_options.get(key) for key in REPORTED_OPTIONS},
        "effective_options": {key: effective_options.get(key) for key in REPORTED_OPTIONS},
        "original_think": requested.get("think"),
        "effective_think": effective.get("think"),
        "overridden": overridden,
        "done_reason": final.get("done_reason"),
        "truncated": final.get("done_reason") in TRUNCATION_REASONS,
        "output_at_limit": (
            isinstance(eval_count, int)
            and isinstance(effective_predict, int)
            and effective_predict > 0
            and eval_count >= effective_predict
        ),
        "prompt_eval_count": final.get("prompt_eval_count"),
        "eval_count": eval_count,
        "error": None if error is None else error[:_ERROR_LIMIT],
    }


class CaptureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def _forward(self) -> None:
        server = self.server
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        requested = body
        chat = self.path == "/api/chat" and self.command == "POST"
        original: dict[str, Any] = {}
        effective: dict[str, Any] = {}
        overridden: list[str] = []
        if chat:
            original = json.loads(body)
            effective = json.loads(body)
            overridden = pin_request(
                effective, seed=server.seed, num_ctx=server.num_ctx,
                fill_num_predict=server.max_output_tokens,
            )
            body = json.dumps(effective, ensure_ascii=False).encode("utf-8")
        started = time.time()
        status = 0
        response = b""
        transport_error: str | None = None
        try:
            request = urllib.request.Request(
                server.upstream + self.path,
                data=body if self.command == "POST" else None,
                headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
                method=self.command,
            )
            with urllib.request.urlopen(request, timeout=server.timeout) as upstream:
                status = upstream.status
                response = upstream.read()
        except urllib.error.HTTPError as error:
            status = error.code
            response = error.read()
        except (TimeoutError, OSError) as error:
            status = 502
            transport_error = f"{type(error).__name__}: {error}"
            response = transport_error.encode("utf-8")
        completed = time.time()
        if chat:
            with server.trace_lock:
                server.sequence += 1
                index = server.sequence
                metadata = request_metadata(
                    index=index,
                    started=started,
                    elapsed=completed - started,
                    status=status,
                    requested=original,
                    effective=effective,
                    effective_body=body,
                    overridden=overridden,
                    response=response,
                    transport_error=transport_error,
                    first_index_by_hash=server.first_index_by_hash,
                )
                (server.trace_dir / f"request-{index:05d}.json").write_text(
                    json.dumps(
                        {
                            "index": index,
                            "path": self.path,
                            "started": started,
                            "elapsed_seconds": completed - started,
                            "status": status,
                            "metadata": metadata,
                            "original_request": requested.decode("utf-8", errors="replace"),
                            "effective_request": body.decode("utf-8", errors="replace"),
                            "response": response.decode("utf-8", errors="replace"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                with (server.trace_dir / "requests.jsonl").open("a", encoding="utf-8") as rows:
                    rows.write(json.dumps(metadata) + "\n")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        try:
            self.wfile.write(response)
        except BrokenPipeError:
            pass

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=32768,
        help="context window pinned on every chat request (study protocol: 32768)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="num_predict for requests that carry none; a candidate's own value is kept",
    )
    parser.add_argument("--trace-dir", type=Path, required=True)
    args = parser.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=False)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), CaptureHandler)
    server.upstream = args.upstream.rstrip("/")
    server.seed = args.seed
    server.max_output_tokens = args.max_output_tokens
    server.num_ctx = args.num_ctx
    server.timeout = args.timeout
    server.trace_dir = args.trace_dir
    server.trace_lock = threading.Lock()
    server.sequence = 0
    server.first_index_by_hash = {}
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
