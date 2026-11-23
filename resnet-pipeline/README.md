# Employee Presence Monitoring

Upgrading from ROI-only occupancy toward a person-centric SaaS pipeline.

**Current modes** (`pipeline.mode` / `--mode`):

| Mode | What it does |
|---|---|
| `occupancy` | Existing ResNet-18 on manual workstation crops (unchanged) |
| `tracking` | YOLO26m person detection + BoT-SORT (Milestone 2) |
| `full` | Person-centric presence: tracking + posture + person→workstation association + ResNet-18 support + identity + seated time |

It measures **observed presence** (seated at a workstation / standing / away / unknown),
**not productivity or "working"**. A BoT-SORT track ID is **never** an employee ID.
Identity is `UNKNOWN_EMPLOYEE` until a verified source (badge, admin label, audited
verification service) is integrated; no face recognition or embeddings are used.

## Installation

```bash
conda activate y3                         # or employee-monitoring / a venv with requirements.txt
pip install -r requirements.txt
```

CUDA is used automatically when available (`model.device: auto`), otherwise CPU.
`cv2.imshow` and ROI selection need a desktop session (or `ssh -X`); without a display the
live window is disabled automatically and video/CSV output still works.

Place the sitting/standing detector (`models/yolov8m_sitting_standing.pt`),
`models/yolo26x-reid.onnx`, and `models/resnet18_empty_occupied.pth` under `models/`.
VideoMAE weights live under `training/videomae/checkpoints/`.

**License note:** Ultralytics YOLO is AGPL-licensed. Review licensing before commercial SaaS distribution.

## Run API + frontend (live dashboard)

Use **two terminals**. Keep only one API and one Streamlit process (duplicate instances fight over the same RTSP and leave the camera stuck on Reconnecting).

```bash
# Terminal 1 — detection API (port 8000)
cd /data/home/emplyee-tracking/custom_sys/resnet-pipeline
conda activate y3
python -m api

# Terminal 2 — Streamlit frontend (port 8501)
cd /data/home/emplyee-tracking/custom_sys/frontend
conda activate y3
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Then open http://localhost:8501 (API docs: http://localhost:8000/docs).

- Cameras that already have workstation ROIs **auto-start detection when the API boots**.
- First live frames take ~20–60s while YOLO / ResNet / VideoMAE / ReID load onto GPU.
- Optional env: `MONITOR_API_KEY` (default `your-key-here`), `API_HOST`, `API_PORT`.

Stop with Ctrl+C in each terminal (or `pkill -f 'python -m api'` / `pkill -f 'streamlit run app.py'`).

## Workflow (CLI / offline video)

Run these commands from the project root (`resnet-pipeline/`).

```bash
# --- Full presence pipeline ---
# Workstations = your admin ROIs in config/workstations.json (scene.use_admin_rois: true by default).
# Automatic desk discovery is off for production accuracy; draw ROIs once with select_rois.
python -m scripts.scene.select_rois --source testing/input2.mp4   # only if you need to (re)draw desks
python main.py --mode full --source testing/input2.mp4 --show --save-video
python main.py --mode full --source testing/input2.mp4 --no-show --save-video --debug-overlay
python main.py --mode full --source "rtsp://user:pass@host:554/stream" --show --save-video
# Optional: scene.use_admin_rois: false loads a saved scene map instead (advanced / review tooling).

# --- Milestone 2: person tracking (no ROIs required) ---
python main.py --mode tracking --source testing/input2.mp4 --show --save-video

# Live RTSP tracking
python main.py --mode tracking --source "rtsp://user:pass@host:554/stream" --show --save-video

# --- Existing occupancy path (ResNet-18 + manual ROIs) ---
python -m scripts.occupancy.inspect_checkpoint --checkpoint models/resnet18_empty_occupied.pth
python -m scripts.scene.select_rois --source testing/input.mp4
python -m scripts.occupancy.test_classifier --image path/to/workstation_crop.jpg
python main.py --mode occupancy --source testing/input.mp4 --show --save-video
```

Useful `main.py` options: `--config`, `--checkpoint`, `--rois`, `--output-dir`, `--interval`,
`--device`, `--rotation`, `--recording-start-utc 2026-10-01T07:26:00Z`, `--max-frames N`
(stop after N classified frames), `--no-show`, `--no-save-video`, `--log-level DEBUG`.
Window keys: **Q**/Esc quit, **P** pause/resume, **N** step one frame while paused.

## Full mode: how decisions are made

1. **Tracking**: YOLO26m + BoT-SORT on every frame (ReID off). Composite track ID
   `tenant:camera:session:local_id`; reset on outage/reconnect.
2. **Posture** per track: MOVING if speed > 0.5 box-heights/s; otherwise the sitting/standing
   detector class; otherwise box height vs the expected seated height at that image row
   (linear perspective model fitted from the map).
3. **Association**: gated score (seat overlap, distance to anchor, context containment, head
   height) + Hungarian assignment with a "no workstation" option; near-ties are AMBIGUOUS.
   3 s of consistent evidence to associate (start backdated), 2 s of contradiction or 5 s without
   support to end (end backdated to the last support). One person per workstation.
4. **Occupancy fusion** (per workstation, every `inference.interval_seconds`): seated person →
   occupied; standing person in the crop → neutral (never "occupied"); ResNet "occupied" with
   nobody visible needs ≥ 0.95; otherwise ResNet. Disagreements are logged. The existing
   workstation state machine then confirms/holds states and produces UNKNOWN on gaps.
5. **Identity**: only verified evidence; conflicts → UNKNOWN_EMPLOYEE. Employee time = seated
   intervals ∩ verified identity windows; everything else stays UNKNOWN_EMPLOYEE.

### Full mode outputs (`output.directory`)

| File | Content |
|---|---|
| `summary.csv`, `intervals.csv`, `events.csv`, `predictions.csv` | workstation occupancy (same schema as occupancy mode) |
| `fusion_<session>.csv` | per workstation per tick: evidence source, reason, ResNet result, disagreement |
| `tracks_<session>.csv` | per frame per track: box, posture, association state, workstation, employee |
| `association_events_<session>.csv` | person↔workstation start/end (event time + decision time) |
| `seated_intervals_<session>.csv` | seated time per track per workstation (employee split) |
| `employee_summary_<session>.csv` | seated seconds per employee per workstation |
| `annotated_full_<session>.mp4` | overlay video (`--debug-overlay` adds decision reasons) |
| `data/store/<tenant>/*.jsonl` | idempotent tenant-scoped history (sessions, map versions, intervals, events) |

### Validated on `testing/input2.mp4` (60 s, 13 workstations, RTX 3060)

* Runs in real time (~20 FPS incl. tracking, 1 Hz ResNet).
* 9 workstations get confirmed seated occupants; occupancy starts backdated to evidence start.
* Standing person beside WS04 → UNKNOWN (not occupied); a seated person at an unmapped desk is
  reported seated but not associated; ResNet false "occupied" at WS13 (empty chair, nobody
  visible) is suppressed.

### Known limitations

* Automatic discovery on this scene matched 8/13 admin workstations (occlusion hides chairs,
  edge chairs are proposed) — **admin review is mandatory**; seeding from admin ROIs is preferred.
* `WS06/WS07` seat evidence is ambiguous (one person overlaps both admin boxes); it is flagged
  `needs_review` and must be fixed in `review_scene_map` — until then that person is credited
  to WS06.
* WS10/WS11 had no chair/seated evidence during mapping, so their seat zones equal the admin
  box; review them too.
* Fully occluded seated people (behind monitors) yield UNKNOWN rather than occupied.
* Track IDs fragment under occlusion (ReID off), which splits seated intervals per track but
  does not change workstation occupancy totals.

## Project layout

```text
main.py                         application controller (MonitoringApp / TrackingApp / FullApp)
core/
  config.py                     settings loading and validation
  tracking/                     YOLO detect + TrackTrack/BoT-SORT + ReID
  vlm/                          clothing feature extractors (Qwen / Moondream)
  identity/                     P## gallery, hybrid identity resolver, verified employees
  association/                  person↔desk, occupancy fusion, working/presence time
  occupancy/                    ResNet-18 ROI classifier, state machine, time intervals, ROIs
  scene/                        workstation discovery / scene map
  io/                           video reader, visualizer, CSV / Excel reports
storage/repository.py           tenant-scoped idempotent JSONL repository
config/
  settings.yaml                 all settings (validated at startup; unknown keys are errors)
  tracking/                     TrackTrack / BoT-SORT YAML
  workstations.json             workstation ROIs written by select_rois
scripts/
  scene/                        select_rois, discover_scene, review_scene_map
  occupancy/                    inspect_checkpoint, test_classifier
tests/                          pytest suite mirrored by package (vlm/, tracking/, …)
```

## Checkpoint compatibility

The classifier loads checkpoints with `torch.load(weights_only=True)` and supports:

* a dict with `state_dict` or `model_state_dict`, or a plain state dict;
* DataParallel `module.` prefixes;
* class metadata as `class_to_idx`, `idx_to_class`, `class_names`, or `classes`.

The output dimension is read from `fc.weight`, a torchvision `resnet18` with a matching `fc`
layer is built, and weights are loaded with `strict=True`. Class order comes from the
checkpoint; `model.class_names` in settings is only needed when the checkpoint has no mapping,
and it is rejected if it contradicts the checkpoint. `unknown` cannot be a model class
because it is the internal monitoring state.

**Full pickled models** (`torch.save(model)`) are not loaded, because unpickling can execute
arbitrary code. In a trusted environment, convert once:

```python
import torch
model = torch.load("model.pth", map_location="cpu", weights_only=False)
torch.save({"state_dict": model.state_dict(), "class_to_idx": {"empty": 0, "occupied": 1}}, "converted.pth")
```

### Preprocessing

The checkpoint stores `image_size`, `mean`, and `std`, but not the resize step, so
`model.preprocessing_config` must describe the training **eval** transform. The provided
values are copied from `training/resnet-18/train.py`:

`RGB -> Resize(256, bilinear, shorter side) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)`

OpenCV crops (BGR) are converted to RGB and passed through the same torchvision PIL
transforms. `crop_size`, `mean`, and `std` are cross-checked against the checkpoint at
startup. `scripts.occupancy.test_classifier --compare-training` verifies equality with the training
pipeline on real images.

Because of `CenterCrop`, long thin ROIs lose their edges. `select_rois` outlines the region
the model actually sees in magenta; frame each box like the training crops.

## ROI calibration

```bash
python -m scripts.scene.select_rois --source testing/input.mp4 [--rotation 180] [--frame-time 30] [--edit]
```

`--source` accepts a video, an RTSP URL, or a full-resolution still frame.

| Action | Key / mouse |
|---|---|
| Draw a box | left-drag |
| Accept / reject the pending box | Enter, Space, A / X, Backspace |
| Delete a box to redraw it | right-click inside it |
| Undo last box | U |
| Inspect crops | N |
| Move through the video | `[` / `]` (`--step` seconds) |
| Rotate (before any box exists) | O |
| Save / cancel | S / Q, Esc |

The file stores boxes in original-resolution pixels plus `reference_resolution`, `rotation`,
and a reference frame image (`config/workstations_reference.jpg`). An existing file is backed
up before it is overwritten, and canceling never writes anything. At runtime, ROIs that fall
outside the frame are rejected. A different resolution with the same aspect ratio is scaled;
other mismatches or a rotation mismatch are errors. **If the camera moves, recalibrate.**

## Timestamps

* **MP4**: per-frame presentation timestamps from OpenCV (`CAP_PROP_POS_MSEC` after `grab()`),
  which handle variable frame rate. If a backend returns non-increasing timestamps, the
  reader falls back to `1/fps` steps and logs a warning; re-mux the file
  (`ffmpeg -i in.mp4 -c copy out.mp4`) or decode with PyAV if that happens.
  `wall_clock_utc` is empty unless `camera.recording_start_utc` is set.
* **RTSP**: a background thread keeps only the newest frame (no growing latency).
  `timestamp_seconds` is monotonic time since session start, taken when the frame arrived;
  `wall_clock_utc` is the session's UTC start plus that offset.
* Processing speed never affects durations: durations come from source time only.

## Temporal logic

Configured under `state:`.

* Every workstation starts as `UNKNOWN`.
* A new state is confirmed once predictions of that class with confidence
  `>= confidence_threshold` persist for `confirmation_seconds`. Gaps between them must
  not exceed `max_observation_gap_seconds`, and nothing may contradict them in between.
* Hysteresis: predictions of the current state with confidence
  `>= hold_confidence_threshold` keep the state and cancel a pending change.
* Low-confidence predictions are neutral and never mean `empty`. After
  `uncertain_timeout_seconds` without supporting evidence, the state becomes `UNKNOWN`.
* An observation gap longer than `max_observation_gap_seconds`, or an RTSP outage, makes the
  workstation `UNKNOWN` from its last observation. An outage never counts as occupied.

**Transition timestamp policy:**

* A confirmed change is backdated to the first observation of the sustained new state.
  `confirmation_timestamp` in `events.csv` records when it was actually confirmed, so the
  uncertainty window is the difference between the two.
* A change to `UNKNOWN` is backdated to the last supporting observation.

Intervals are contiguous and never overlap. The last interval ends at the last valid
observation and has an `end_reason` such as `truncated_end_of_video`, `truncated_user_stop`,
`truncated_max_frames`, `truncated_source_lost`, or `truncated_interrupted`. Intervals that
ended with a confirmed change have `end_reason` `transition`.

## Outputs

Each run gets a `session_id` (e.g. `20261022T054832Z_e5f939`). Rows are appended to shared
CSVs in `output.directory` and flushed every `flush_interval_seconds`, so a crash loses at most
a few seconds of records. Ctrl+C and SIGTERM finalize intervals and summaries cleanly. The
annotated video is `annotated_<session_id>.mp4`, and logs are written to
`logs/monitor_<session_id>.log`.

| File | One row per | Columns |
|---|---|---|
| `predictions.csv` | workstation per classified frame | camera_id, session_id, timestamp_seconds, wall_clock_utc, workstation_id, predicted_class, confidence, probabilities_json |
| `events.csv` | state change (incl. initial `session_start` → unknown) | camera_id, session_id, workstation_id, previous_state, new_state, transition_timestamp, confirmation_timestamp, reason |
| `intervals.csv` | completed state interval | camera_id, session_id, workstation_id, state, start_timestamp, end_timestamp, duration_seconds, end_reason |
| `summary.csv` | workstation per session | camera_id, session_id, workstation_id, occupied_seconds, empty_seconds, standing_near_seconds, unknown_seconds, observed_seconds, transition_count |
| `summary_daily.csv` | workstation, day, state (only when wall-clock time is known) | camera_id, session_id, report_date, workstation_id, state, duration_seconds |

`observed_seconds` is the time in a confirmed model state (everything except `unknown`).
`standing_near_seconds` stays 0 because this model has no such class. `reason` values are
`session_start`, `confirmed`, `low_confidence`, `observation_gap`, and `source_outage`.
Daily summaries split at local midnight of `output.report_timezone`.

## Tests

```bash
python -m pytest
```

The suite covers ROI cropping and validation, classifier loading, mapping, BGR→RGB,
preprocessing, batching, CPU/CUDA and bad checkpoints, the state machine, the time tracker
(durations, gaps, day boundaries, reconnects), the video reader (PTS, sampling, rotation,
simulated RTSP outage and reconnect), and a synthetic end-to-end run with a mock classifier.
`test_pipeline_with_real_checkpoint` runs your actual checkpoint, ROIs, and video. It is
skipped automatically if any of them is missing.
