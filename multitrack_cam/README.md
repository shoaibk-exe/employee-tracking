# MTMCT — Multi-Target Multi-Camera Tracking

Production-oriented **4-camera** person tracking with **persistent global identities**.

This package lives entirely under `multitrack_cam/` and does not modify sibling projects.

## What it does

- Ingests 4 RTSP / USB / file sources independently (reconnect, freeze detection, bounded queues)
- Per-camera YOLO + BoT-SORT local tracking (local IDs are never persistent identities)
- Shared ReID encoder + quality-gated crops
- Conservative global association (Hungarian + topology + ambiguity margins)
- Persistent identity gallery in SQLite (default) or PostgreSQL
- Live 2×2 dashboard with global IDs, camera health, and identity history

## Important limitations (by design)

- Appearance ReID **cannot guarantee** identity after clothing changes or multi-day returns.
- Uncertain matches stay `PROVISIONAL` / `UNKNOWN` — the system will **not** force merges.
- Wrong merges are treated as worse than temporary unknowns.

## Hardware note (this host)

- GPU: NVIDIA RTX 3060 12GB
- Recommended env: conda `y3` (Python 3.11, PyTorch CUDA, Ultralytics)

## Quick start

```bash
cd multitrack_cam
chmod +x scripts/run_server.sh
./scripts/run_server.sh
```

Open **http://127.0.0.1:8090/**

Sample MP4s are auto-generated into `sample_data/` if missing.

### Configure live RTSP later

Edit `config/cameras.yaml`:

```yaml
source: "rtsp://${CAM1_USER}:${CAM1_PASS}@192.168.1.10:554/stream1"
```

Copy `.env.example` → `.env` and set credentials. Credentials are never returned by the API.

### PostgreSQL (optional)

```bash
docker compose up -d postgres
export MTMCT_DATABASE_URL="postgresql+psycopg://mtmct:mtmct@localhost:5432/mtmct"
./scripts/run_server.sh
```

## Layout

```text
multitrack_cam/
  backend/          # cameras, detection, tracking, reid, association, identity, api
  frontend/         # 2×2 live dashboard
  config/           # settings, cameras, topology, botsort
  scripts/          # startup + sample video generator
  tests/            # unit/integration tests
  sample_data/      # prerecorded test clips
```

## Pipeline

```text
Cameras → StreamManager → YOLO+BoT-SORT (per cam)
        → quality crops → shared ReID
        → tracklets → GlobalAssociator → GlobalIdentityManager
        → SQLite/Postgres + gallery → FastAPI / WebSocket / MJPEG dashboard
```

## Tests

```bash
cd multitrack_cam
PYTHONPATH=. /home/ai-dw/miniconda3/envs/y3/bin/python -m pytest -q
```

## Models

| Role | Default | Notes |
|------|---------|-------|
| Detector/tracker | `yolo11n.pt` | Ultralytics auto-download; swap to larger models if VRAM allows |
| ReID | `models/yolo26x-reid.onnx` | Copied from local assets if available; fallback embeddings are **test-only** |

Ultralytics components are AGPL — review licensing before commercial redistribution.

## API

- `GET /api/health`
- `GET /api/stats`
- `GET /api/cameras`
- `GET /api/identities`
- `GET /api/identities/{global_id}`
- `POST /api/identities/correct`
- `GET /api/stream/{camera_id}` (MJPEG)
- `WS /ws/live`

## Calibration

Thresholds in `config/settings.yaml` (`association.*`) must be calibrated on **your** camera footage. Defaults are conservative starters, not universal truth.
