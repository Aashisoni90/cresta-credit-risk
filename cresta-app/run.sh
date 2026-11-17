#!/usr/bin/env bash
# Start CRESTA locally. Backend on :8000, frontend on :3002.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"        # esg_distress_project/
VENV="$ROOT/.venv"

[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -r "$ROOT/cresta-app/backend/requirements.txt"

( cd "$ROOT/cresta-app/backend" && "$VENV/bin/python" -m uvicorn main:app --port 8000 ) &
BACK=$!
( cd "$ROOT/cresta-app/frontend" && PORT=3002 npm run dev ) &
FRONT=$!
trap "kill $BACK $FRONT 2>/dev/null" EXIT
wait
