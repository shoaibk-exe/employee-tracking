# Employee monitoring

Observations come from cameras. State machines turn those observations into sessions. Reports sum the sessions.

The Streamlit app in the parent folder is a separate prototype. This package is the attendance and desk pipeline.

## Run

```bash
cd employee_monitoring
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `ENTRANCE_CAMERA_RTSP` and `OFFICE_CAMERA_RTSP` in `.env`. Do not put passwords in the YAML files.

Postgres:

```bash
docker compose up -d db
```

API and pipeline:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Send header `X-API-Key` with the value from `.env`.

Camera check, before face recognition:

```bash
python scripts/test_camera.py --camera entrance_01
```

Enroll:

```bash
python scripts/enroll_employee.py --employee-code EMP001 --name "Ali" --images path\to\photos --desk-id desk_001
```

Draw a desk:

```bash
python scripts/draw_zones.py --camera office_01 --desk-id desk_001 --employee-id EMP001
```

Tests that do not load models:

```bash
pytest
```

## V1 scope

Entrance identity is confirmed on a track, then a line crossing opens or closes an attendance session. The office camera detects Working or Notworking, associates the body base with a desk polygon, and only then starts or ends a desk session. Notworking does not count as being at the desk.

A dead camera becomes `UNKNOWN_CAMERA_FAILURE`. It does not mark the employee away.

Phone detection is not wired into the live loop. `app/detection/phone_detector.py` raises if called.

## Models

- Entrance person tracking: `models.entrance_person_model` (defaults to `yolov8n.pt`)
- Desk posture: `models/sitting_model/best.pt` with classes `Notworking` (0) and `Working` (1). Only `Working` starts desk time.
- Faces: InsightFace behind `app/face/recognizer.py`

Match thresholds in `configs/recognition.yaml` must be tuned on the deployment cameras.

## Clock

Store UTC. `LOCAL_TIMEZONE` is used only when the API prints a clock time. Sync servers with NTP.
