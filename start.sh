#!/usr/bin/env bash
# Boots the detector sidecar and the Node server together.
set -euo pipefail
cd "$(dirname "$0")"

cleanup() {
  echo "[start] stopping children…"
  [[ -n "${DET_PID:-}" ]] && kill "$DET_PID" 2>/dev/null || true
  [[ -n "${NODE_PID:-}" ]] && kill "$NODE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[start] launching detector on :8000"
python3 -m uvicorn detector.service:app --port 8000 --host 127.0.0.1 &
DET_PID=$!

echo "[start] launching Node server on :3000"
node server.js &
NODE_PID=$!

wait -n
