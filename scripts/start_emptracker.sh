#!/usr/bin/env bash
# Start EMPTracker API (8000) + Streamlit frontend (8501).
# Safe to run when already up: skips duplicates.

set -euo pipefail

ROOT="/data/home/emplyee-tracking/custom_sys"
PIPELINE="$ROOT/resnet-pipeline"
FRONTEND="$ROOT/frontend"
LOG_DIR="$ROOT/logs"
CONDA_SH="/home/ai-dw/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="y3"

mkdir -p "$LOG_DIR"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$ENV_NAME"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

if ss -ltn 2>/dev/null | grep -q ':8000 '; then
  echo "$(ts) API already listening on :8000"
else
  echo "$(ts) Starting API on :8000"
  (
    cd "$PIPELINE"
    nohup python -m api >>"$LOG_DIR/api.log" 2>&1 &
    echo $! >"$LOG_DIR/api.pid"
  )
fi

if ss -ltn 2>/dev/null | grep -q ':8501 '; then
  echo "$(ts) Frontend already listening on :8501"
else
  echo "$(ts) Starting Streamlit on :8501"
  (
    cd "$FRONTEND"
    nohup streamlit run app.py \
      --server.port 8501 \
      --server.address 0.0.0.0 \
      >>"$LOG_DIR/frontend.log" 2>&1 &
    echo $! >"$LOG_DIR/frontend.pid"
  )
fi

echo "$(ts) Start check done"
