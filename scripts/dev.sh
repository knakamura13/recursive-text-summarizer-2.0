#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SUMMARIZER_DATA_DIR="${SUMMARIZER_DATA_DIR:-$ROOT/.local-data}"

cd "$ROOT"

if [[ ! -d frontend/node_modules ]]; then
  (cd frontend && pnpm install)
fi

uvicorn summarizer_web.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

trap 'kill $BACKEND_PID 2>/dev/null || true' EXIT

cd frontend
pnpm dev --host 127.0.0.1 --port 5173
