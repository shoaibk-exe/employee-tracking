#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prefer the project's tested conda env when available.
if [[ -x "/home/ai-dw/miniconda3/envs/y3/bin/python" ]]; then
  PY="/home/ai-dw/miniconda3/envs/y3/bin/python"
else
  PY="${PYTHON:-python3}"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Small matrices only: big BLAS/OpenMP pools busy-spin and starve the RTSP decoder threads.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export MTMCT_SETTINGS="${MTMCT_SETTINGS:-config/settings.yaml}"

if [[ ! -f sample_data/cam_01.mp4 ]]; then
  echo "Generating sample videos..."
  "$PY" scripts/generate_sample_videos.py --out sample_data
fi

# Optional: reuse a local YOLO26 ReID ONNX if present next door without modifying it.
if [[ ! -f models/yolo26x-reid.onnx && -f ../resnet-pipeline/models/yolo26x-reid.onnx ]]; then
  echo "Copying ReID model into multitrack_cam/models (read-only copy)..."
  cp -n ../resnet-pipeline/models/yolo26x-reid.onnx models/yolo26x-reid.onnx || true
fi

exec "$PY" -m backend.main "$@"
