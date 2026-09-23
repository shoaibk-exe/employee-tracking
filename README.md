# Face Attendance + Chair Occupancy

Two-camera workplace CV system:

- **Camera 1** — InsightFace `buffalo_l` face recognition. First match of the day clocks the employee in.
- **Camera 2** — YOLO + ByteTrack chair occupancy. Empty-chair time is accumulated as away-from-desk.

The Streamlit app is for enrollment, kiosk attendance, the daily sheet, and a live occupancy view. All models and capture logic live in `pipeline/` and do not import Streamlit.

## Setup

Python 3.10–3.12 recommended (InsightFace is not reliable on 3.13).

```bash
cd employee
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

- `SOURCE_MODE=rtsp` and paste CCTV URLs into `FACE_RTSP_URL` / `OCCUPANCY_RTSP_URL`
- or `SOURCE_MODE=mp4` and paste file paths into `FACE_MP4_PATH` / `OCCUPANCY_MP4_PATH`

```bash
streamlit run app/Home.py
```

First face run downloads `buffalo_l` into the InsightFace cache. First occupancy run downloads `yolov8n.pt` if `models/chair.pt` is not present.

## Custom chair model

Train YOLO on your chairs, then set:

```
YOLO_MODEL_PATH=models/chair.pt
YOLO_CHAIR_CLASS=chair
YOLO_PERSON_CLASS=person
```

If the custom weights detect chairs only, a person model (`PERSON_MODEL_PATH` or `yolov8n.pt`) is used for sit/stand overlap.

## Layout

| Path | Role |
| --- | --- |
| `pipeline/` | Capture, face engine, occupancy engine, SQLite |
| `app/` | Streamlit UI |
| `data/` | Database and enrolled photos |
| `models/` | Custom YOLO weights |

## Attendance rule

The first recognition of a calendar day marks **Present** with a clock-in timestamp. Later detections that day are ignored.
