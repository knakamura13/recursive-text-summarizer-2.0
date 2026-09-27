"""Record complete Ollama requests/responses while pinning controlled sampling.

Run only on localhost. Captures can contain entire copyrighted source passages;
keep --trace-dir outside the repository and do not distribute it.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class CaptureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def _forward(self) -> None:
        server = self.server
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        requested = body
        if self.path == "/api/chat" and self.command == "POST":
            data = json.loads(body)
            options = data.setdefault("options", {})
            options.update(seed=server.seed, temperature=1, top_k=64, top_p=0.95, num_ctx=32768)
            if server.max_output_tokens is not None:
                options["num_predict"] = server.max_output_tokens
            # The model's default thinking mode is on; the modern provider
            # explicitly disables it. Match that behavior for legacy transport.
            data["think"] = False
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        started = time.time()
        status = 0
        response = b""
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
            response = str(error).encode("utf-8")
        completed = time.time()
        if self.path == "/api/chat":
            with server.trace_lock:
                server.sequence += 1
                index = server.sequence
                path = server.trace_dir / f"request-{index:05d}.json"
                path.write_text(
                    json.dumps(
                        {
                            "index": index,
                            "path": self.path,
                            "started": started,
                            "elapsed_seconds": completed - started,
                            "status": status,
                            "original_request": requested.decode("utf-8", errors="replace"),
                            "effective_request": body.decode("utf-8", errors="replace"),
                            "response": response.decode("utf-8", errors="replace"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
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
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--trace-dir", type=Path, required=True)
    args = parser.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=False)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), CaptureHandler)
    server.upstream = args.upstream.rstrip("/")
    server.seed = args.seed
    server.max_output_tokens = args.max_output_tokens
    server.timeout = args.timeout
    server.trace_dir = args.trace_dir
    server.trace_lock = threading.Lock()
    server.sequence = 0
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
