#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/frontend/src/lib/api/generated.ts"

python - <<'PY'
from summarizer_web.main import create_app
import json
from pathlib import Path

app = create_app()
schema = app.openapi()
Path("/tmp/openapi.json").write_text(json.dumps(schema), encoding="utf-8")
print("Wrote /tmp/openapi.json")
PY

if command -v npx >/dev/null 2>&1; then
  npx --yes openapi-typescript /tmp/openapi.json -o "$OUT"
  echo "Generated $OUT"
else
  echo "openapi-typescript not available; skipping generation"
fi
