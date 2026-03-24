#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACK_PID=""

cleanup() {
  if [ -n "$BACK_PID" ]; then
    kill "$BACK_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Запускаю backend..."
(
  cd "$ROOT/backend"
  source .venv/bin/activate
  uvicorn main:app --reload --host 127.0.0.1 --port 8000
) &
BACK_PID=$!

sleep 4

echo "Запускаю frontend..."
cd "$ROOT/web"
npm run dev
