#!/usr/bin/env bash
# Stop EMPTracker API + Streamlit frontend.

set -euo pipefail

ROOT="/data/home/emplyee-tracking/custom_sys"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(ts) Stopping EMPTracker"

# Prefer PIDs from last start, then fall back to pattern match.
if [[ -f "$LOG_DIR/api.pid" ]]; then
  kill "$(cat "$LOG_DIR/api.pid")" 2>/dev/null || true
  rm -f "$LOG_DIR/api.pid"
fi
if [[ -f "$LOG_DIR/frontend.pid" ]]; then
  kill "$(cat "$LOG_DIR/frontend.pid")" 2>/dev/null || true
  rm -f "$LOG_DIR/frontend.pid"
fi

# Kill whatever still holds the EMPTracker ports, then clean leftovers.
if command -v fuser >/dev/null 2>&1; then
  fuser -k 8000/tcp 8501/tcp 2>/dev/null || true
fi
pkill -f 'streamlit run app.py --server.port 8501' 2>/dev/null || true
pkill -9 -f 'python -m api.hub' 2>/dev/null || true
pkill -9 -f 'python -m api.worker' 2>/dev/null || true
pkill -f 'python -m api' 2>/dev/null || true
sleep 1
pkill -9 -f 'python -m api.hub' 2>/dev/null || true
pkill -9 -f 'python -m api' 2>/dev/null || true
rm -f /data/home/emplyee-tracking/custom_sys/resnet-pipeline/data/api/hub/hub.pid
rm -f /data/home/emplyee-tracking/custom_sys/resnet-pipeline/data/api/hub/active/*.json 2>/dev/null || true

sleep 1
if ss -ltn 2>/dev/null | grep -qE ':8000 |:8501 '; then
  echo "$(ts) WARNING: something is still listening on 8000/8501"
  ss -ltnp 2>/dev/null | grep -E ':8000 |:8501 ' || true
else
  echo "$(ts) Ports 8000 and 8501 are free"
fi
echo "$(ts) Stop done"
