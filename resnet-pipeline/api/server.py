"""API between the Streamlit app and occupancy detection.

Run from ``resnet-pipeline``::

    python -m api

The frontend sends an RTSP link and a rotate flag. A new stream must have
workstation boxes drawn on a rotated snapshot before detection starts. Those
boxes are stored in rotated-frame pixels and reused unchanged by the detector.

On API startup, every camera that already has saved ROIs starts detection
automatically so the dashboard does not need a manual Start click after reboot.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from api.session import DetectionSession
from api.snapshot import grab_frame
from api.store import Catalog, CatalogError
from core.activity.time_tracker import ACTIVITY_CLASSES
from core.io.day_ledger import DayLedger
from core.io.video_reader import VideoSourceError

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "config" / "settings.yaml"
CATALOG = Catalog(ROOT / "data" / "api")
API_KEY = os.environ.get("MONITOR_API_KEY", "your-key-here")
logger = logging.getLogger("api.server")

_sessions: dict[str, DetectionSession] = {}
_sessions_lock = threading.Lock()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _autostart_cameras()
    try:
        yield
    finally:
        _shutdown_sessions()


app = FastAPI(title="EMPTracker API", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


class CameraIn(BaseModel):
    name: str
    rtsp: str
    rotate: bool = False


class WorkstationBox(BaseModel):
    name: str | None = None
    bbox: list[int] = Field(min_length=4, max_length=4)


class RoiIn(BaseModel):
    width: int
    height: int
    workstations: list[WorkstationBox]


def require_key(
    x_api_key: str | None = Header(default=None),
    key: str | None = Query(default=None),
) -> None:
    if API_KEY and (x_api_key or key) != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _camera_or_404(camera_id: str) -> dict[str, Any]:
    camera = CATALOG.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return camera


def _rotation(camera: dict[str, Any]) -> int:
    return 180 if camera.get("rotate") else 0


def _status(camera_id: str, status: str, error: str) -> None:
    CATALOG.update_camera(camera_id, status=status, error=error)


def _base(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _stop_session(camera_id: str) -> None:
    with _sessions_lock:
        session = _sessions.pop(str(camera_id), None)
    if session is not None:
        session.stop()


def _seed_live_preview(camera_id: str) -> None:
    """Publish the ROI reference frame to live.jpg so View detection has a picture immediately."""
    reference = CATALOG.snapshot_path(camera_id)
    if not reference.is_file():
        return
    try:
        import cv2

        image = cv2.imread(str(reference))
        if image is None:
            return
        rois = CATALOG.read_rois(camera_id) or {}
        for workstation_id, entry in (rois.get("workstations") or {}).items():
            bbox = entry.get("bbox") if isinstance(entry, dict) else None
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (52, 211, 153), 3)
            cv2.putText(
                image,
                str(workstation_id),
                (x1 + 8, max(24, y1 + 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        target = CATALOG.session_dir_for(camera_id) / "live.jpg"
        temporary = target.with_suffix(".jpg.tmp")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(target)
    except Exception:
        logger.exception("Failed to seed live preview for camera %s", camera_id)


def _begin_detection(camera: dict[str, Any]) -> None:
    """Spawn or refresh the full detector worker for one catalog camera."""
    camera_id = str(camera["id"])
    if not CATALOG.roi_path(camera_id).is_file():
        raise HTTPException(
            status_code=400,
            detail="Draw a workstation ROI before starting detection.",
        )
    # Show the drawn ROI on View detection immediately while the hub warms up.
    _seed_live_preview(camera_id)
    CATALOG.update_camera(
        camera_id,
        status="reconnecting",
        error="Starting detection on the saved workstation ROI…",
    )

    with _sessions_lock:
        existing = _sessions.get(camera_id)

    if existing is not None and existing.running:
        # ROI file mtime already changed; refresh launch so the hub restarts that cam only.
        existing.refresh_launch()
        existing.reload_preview()
        return

    _stop_session(camera_id)
    session = DetectionSession(
        camera_id=camera_id,
        source=camera["rtsp"],
        rotation=_rotation(camera),
        roi_path=CATALOG.roi_path(camera_id),
        settings_path=SETTINGS,
        output_dir=CATALOG.session_dir_for(camera_id),
        on_status=_status,
    )
    with _sessions_lock:
        _sessions[camera_id] = session
    session.start()


def _autostart_cameras() -> None:
    """Resume detection for every camera that already has workstation ROIs."""
    for camera in CATALOG.list_cameras():
        camera_id = str(camera["id"])
        if not CATALOG.roi_path(camera_id).is_file():
            CATALOG.update_camera(camera_id, status="offline", error="")
            continue
        try:
            _begin_detection(camera)
            logger.info("Auto-started detection for camera %s", camera_id)
        except Exception:
            logger.exception("Failed to auto-start camera %s", camera_id)
            CATALOG.update_camera(
                camera_id,
                status="offline",
                error="Auto-start failed. Use Start detection in the UI.",
            )


def _shutdown_sessions() -> None:
    for camera in CATALOG.list_cameras():
        camera_id = str(camera["id"])
        _stop_session(camera_id)
        CATALOG.update_camera(camera_id, status="offline", error="")


def _progress_path(camera_id: Any, suffix: str) -> Path:
    return CATALOG.session_dir / str(camera_id) / f"progress.{suffix}"


def _load_progress(camera: dict[str, Any]) -> dict[str, Any]:
    with _sessions_lock:
        session = _sessions.get(str(camera["id"]))
    if session is not None:
        live = session.latest_progress()
        if live:
            return live
    path = _progress_path(camera["id"], "json")
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"workstations": [], "timestamp_seconds": 0}


def _minutes(seconds: float) -> int:
    return int(round(float(seconds) / 60.0))


def _report_tz() -> Any:
    cached = getattr(_report_tz, "value", None)
    if cached is None:
        from core.config import load_settings

        cached = load_settings(SETTINGS).report_timezone
        setattr(_report_tz, "value", cached)
    return cached


def _today() -> date:
    return datetime.now(_report_tz()).date()


def _parse_day(selected_date: str | None) -> date | None:
    if not selected_date:
        return _today()
    try:
        return date.fromisoformat(selected_date)
    except ValueError:
        return None


def _camera_day(camera: dict[str, Any], day: date) -> dict[str, Any]:
    directory = CATALOG.session_dir / str(camera["id"])
    ledger = DayLedger(directory, _report_tz())
    ledger.import_summary(directory / "summary.csv")
    return ledger.combined(day)


def _activity_share(seconds: dict[str, float]) -> dict[str, Any]:
    total = sum(max(0.0, float(value or 0.0)) for value in seconds.values())
    percent = {
        name: (0 if total <= 0 else round(100.0 * float(seconds.get(name, 0.0) or 0.0) / total, 1))
        for name in ACTIVITY_CLASSES
    }
    return {
        "classes": list(ACTIVITY_CLASSES),
        "seconds": {name: round(float(seconds.get(name, 0.0) or 0.0), 3) for name in ACTIVITY_CLASSES},
        "percent": percent,
        "total_seconds": round(total, 3),
    }


def _empty_progress(message: str) -> dict[str, Any]:
    return {
        "live": True,
        "message": message,
        "working": 0,
        "not_working": 0,
        "attendance": {"Occupied": 0, "Empty": 0, "Unknown": 0},
        "timeline": [],
        "weekly": [],
        "workstations": [],
        "files": [],
        "session_minutes": 0,
        "videomae": _activity_share({name: 0.0 for name in ACTIVITY_CLASSES}),
    }


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/cameras", dependencies=[Depends(require_key)])
def list_cameras() -> list[dict[str, Any]]:
    return CATALOG.list_cameras()


@app.post("/cameras", dependencies=[Depends(require_key)])
def add_camera(body: CameraIn) -> dict[str, Any]:
    try:
        camera = CATALOG.add_camera(body.name, body.rtsp, body.rotate)
    except CatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "id": camera["id"], "needs_roi": camera["needs_roi"], "camera": camera}


@app.post("/cameras/{camera_id}/reconnect", dependencies=[Depends(require_key)])
def reconnect_camera(camera_id: str) -> dict[str, Any]:
    return start_camera(camera_id)


@app.delete("/cameras/{camera_id}", dependencies=[Depends(require_key)])
def delete_camera(camera_id: str) -> dict[str, bool]:
    _stop_session(camera_id)
    if not CATALOG.delete_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found.")
    return {"ok": True}


@app.get("/cameras/{camera_id}/snapshot", dependencies=[Depends(require_key)])
def camera_snapshot(camera_id: str) -> dict[str, Any]:
    camera = _camera_or_404(camera_id)
    rotation = _rotation(camera)
    try:
        jpeg, width, height = grab_frame(camera["rtsp"], rotation)
    except VideoSourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    CATALOG.save_snapshot(camera_id, jpeg)
    import base64

    encoded = base64.b64encode(jpeg).decode("ascii")
    return {
        "width": width,
        "height": height,
        "rotation": rotation,
        "image": f"data:image/jpeg;base64,{encoded}",
    }


@app.get("/cameras/{camera_id}/reference.jpg")
def camera_reference(camera_id: str, _: None = Depends(require_key)) -> FileResponse:
    """Last ROI reference frame for this camera, if one was saved."""
    _camera_or_404(camera_id)
    path = CATALOG.snapshot_path(camera_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No reference frame saved for this camera yet.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/cameras/{camera_id}/rois", dependencies=[Depends(require_key)])
def get_rois(camera_id: str) -> dict[str, Any]:
    _camera_or_404(camera_id)
    return CATALOG.read_rois(camera_id) or {
        "reference_resolution": None,
        "rotation": _rotation(_camera_or_404(camera_id)),
        "workstations": {},
    }


@app.post("/cameras/{camera_id}/rois", dependencies=[Depends(require_key)])
def save_rois(camera_id: str, body: RoiIn) -> dict[str, Any]:
    camera = _camera_or_404(camera_id)
    snapshot = CATALOG.snapshot_path(camera_id)
    if snapshot.is_file():
        import cv2

        image = cv2.imread(str(snapshot))
        if image is not None:
            height, width = image.shape[:2]
            if (body.width, body.height) != (width, height):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Boxes were drawn for {body.width}x{body.height}, "
                        f"but the snapshot is {width}x{height}."
                    ),
                )
    try:
        saved = CATALOG.save_rois(
            camera_id,
            body.width,
            body.height,
            _rotation(camera),
            [box.model_dump() for box in body.workstations],
            camera["rtsp"],
        )
    except CatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Always restart detection so View detection / VideoMAE use the new boxes.
    started = False
    try:
        _begin_detection(camera)
        started = True
    except HTTPException:
        raise
    except Exception:
        logger.exception("Saved ROI for camera %s but failed to restart detection", camera_id)
    return {"ok": True, "rois": saved, "detection_restarted": started}


@app.post("/cameras/{camera_id}/start", dependencies=[Depends(require_key)])
def start_camera(camera_id: str) -> dict[str, Any]:
    camera = _camera_or_404(camera_id)
    _begin_detection(camera)
    return {"ok": True, "status": "reconnecting"}


@app.post("/cameras/{camera_id}/stop", dependencies=[Depends(require_key)])
def stop_camera(camera_id: str) -> dict[str, bool]:
    _camera_or_404(camera_id)
    _stop_session(camera_id)
    CATALOG.update_camera(camera_id, status="offline", error="")
    return {"ok": True}


@app.get("/stream/{camera_id}")
def stream_camera(camera_id: str, _: None = Depends(require_key)) -> StreamingResponse:
    _camera_or_404(camera_id)

    def frames():
        while CATALOG.get_camera(camera_id) is not None:
            with _sessions_lock:
                session = _sessions.get(str(camera_id))
            jpeg = None if session is None else session.latest_jpeg()
            if jpeg:
                header = (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                )
                yield header + jpeg + b"\r\n"
            time.sleep(0.05)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/cameras/{camera_id}/frame.jpg")
def latest_frame(camera_id: str, _: None = Depends(require_key)) -> Response:
    """One processed JPEG. The detection page polls this instead of a multipart stream."""
    _camera_or_404(camera_id)
    with _sessions_lock:
        session = _sessions.get(str(camera_id))
    jpeg = None if session is None else session.latest_jpeg()
    if not jpeg:
        # Fall back to the last file on disk so the UI does not go blank during RTSP outages.
        disk = CATALOG.session_dir_for(camera_id) / "live.jpg"
        if disk.is_file():
            jpeg = disk.read_bytes()
    if not jpeg:
        raise HTTPException(status_code=404, detail="No processed frame yet.")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def _office_progress(selected_date: str | None, request: Request) -> dict[str, Any]:
    day = _parse_day(selected_date)
    if day is None:
        return _empty_progress("No detection report is stored for that date.")
    # Office totals must not sum duplicate cameras on the same RTSP (e.g. c200_2 ×3
    # test cams) — that triples working / VideoMAE hours. Keep per-camera rows for
    # the table, but roll up one contribution per (rtsp, workstation_id).
    best_by_desk: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    files: list[dict[str, str]] = []
    for camera in CATALOG.list_cameras():
        rtsp = str(camera.get("rtsp") or "").strip()
        combined = _camera_day(camera, day)
        for row in combined.get("workstations") or []:
            occ = float(row.get("occupied_seconds") or 0.0)
            emp = float(row.get("empty_seconds") or 0.0)
            unk = float(row.get("unknown_seconds") or 0.0)
            desk_activity = {
                name: float(row.get(f"{name}_seconds") or 0.0) for name in ACTIVITY_CLASSES
            }
            share = _activity_share(desk_activity)
            rows.append({
                "camera_id": camera["id"],
                "camera": camera["name"],
                "workstation_id": row.get("workstation_id"),
                "state": row.get("state") or "unknown",
                "confidence": row.get("confidence"),
                "bbox": row.get("bbox") or [],
                "occupied_seconds": occ,
                "empty_seconds": emp,
                "unknown_seconds": unk,
                "yolo_activity": row.get("yolo_activity"),
                "activity_class": row.get("activity_class"),
                "activity_confidence": row.get("activity_confidence"),
                "videomae_seconds": share["seconds"],
                "videomae_percent": share["percent"],
            })
            ws_id = str(row.get("workstation_id") or "")
            key = (rtsp, ws_id)
            wall = occ + emp + unk + sum(desk_activity.values())
            prev = best_by_desk.get(key)
            if prev is None or wall > float(prev["_wall"]):
                best_by_desk[key] = {
                    "_wall": wall,
                    "occupied": occ,
                    "empty": emp,
                    "unknown": unk,
                    "activity": desk_activity,
                }
        entry: dict[str, str] = {"camera_id": str(camera["id"]), "camera": camera["name"]}
        if _progress_path(camera["id"], "xlsx").is_file():
            entry["excel"] = f"/cameras/{camera['id']}/progress.xlsx"
        if _progress_path(camera["id"], "json").is_file():
            entry["json"] = f"/cameras/{camera['id']}/progress.json"
        if "excel" in entry or "json" in entry:
            files.append(entry)
    if not rows:
        return _empty_progress("No detection report is stored for that date.")
    occupied = empty = unknown = 0.0
    activity_totals = {name: 0.0 for name in ACTIVITY_CLASSES}
    for slot in best_by_desk.values():
        occupied += float(slot["occupied"])
        empty += float(slot["empty"])
        unknown += float(slot["unknown"])
        for name, value in slot["activity"].items():
            activity_totals[name] += float(value)
    work = _minutes(occupied)
    idle = _minutes(empty)
    unknown_minutes = _minutes(unknown)
    weekday = day.strftime("%a")
    return {
        "live": True,
        "message": "",
        "day": day.isoformat(),
        "working": work,
        "not_working": idle,
        "attendance": {"Occupied": work, "Empty": idle, "Unknown": unknown_minutes},
        "timeline": [["working", work], ["not_working", idle]],
        "weekly": [{
            "day": weekday,
            "working": round(occupied / 3600.0, 2),
            "not_working": round(empty / 3600.0, 2),
        }],
        "workstations": rows,
        "files": files,
        "session_minutes": work + idle + unknown_minutes,
        "videomae": _activity_share(activity_totals),
        "base_url": _base(request),
    }


@app.get("/progress", dependencies=[Depends(require_key)])
def progress(
    request: Request,
    employee: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    del employee  # Occupancy results are per workstation, not per enrolled employee.
    return _office_progress(date, request)


@app.get("/report", dependencies=[Depends(require_key)])
def report(selected_date: str | None = Query(default=None, alias="date")) -> list[dict[str, Any]]:
    day = _parse_day(selected_date)
    if day is None:
        return []
    table = []
    for camera in CATALOG.list_cameras():
        for row in _camera_day(camera, day).get("workstations") or []:
            occupied = float(row.get("occupied_seconds") or 0.0)
            empty = float(row.get("empty_seconds") or 0.0)
            total = occupied + empty
            table.append({
                "Camera": camera["name"],
                "Workstation": row.get("workstation_id"),
                "State": row.get("state") or "unknown",
                "Occupied": f"{_minutes(occupied) // 60}h {_minutes(occupied) % 60:02d}m",
                "Empty": f"{_minutes(empty) // 60}h {_minutes(empty) % 60:02d}m",
                "Work %": round(occupied * 100 / total) if total else 0,
            })
    return table


@app.get("/cameras/{camera_id}/progress.xlsx")
def download_excel(camera_id: str, _: None = Depends(require_key)) -> FileResponse:
    _camera_or_404(camera_id)
    path = _progress_path(camera_id, "xlsx")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Excel report is not ready yet.")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"camera_{camera_id}_progress.xlsx",
    )


@app.get("/cameras/{camera_id}/progress.json")
def download_json(camera_id: str, _: None = Depends(require_key)) -> FileResponse:
    _camera_or_404(camera_id)
    path = _progress_path(camera_id, "json")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="JSON report is not ready yet.")
    return FileResponse(path, media_type="application/json", filename=f"camera_{camera_id}_progress.json")


@app.get("/employees", dependencies=[Depends(require_key)])
def list_employees(request: Request) -> list[dict[str, Any]]:
    rows = []
    for employee in CATALOG.list_employees():
        images = []
        for image in employee.get("images", []):
            images.append({
                "id": image["id"],
                "filename": image["filename"],
                "url": (
                    f"{_base(request)}/employees/{employee['id']}/images/{image['id']}?key={API_KEY}"
                ),
            })
        rows.append({"id": employee["id"], "name": employee["name"], "images": images})
    return rows


@app.post("/employees", dependencies=[Depends(require_key)])
async def create_employee(name: str = Form(...), image: UploadFile = File(...)) -> dict[str, Any]:
    try:
        return CATALOG.enroll(name, image.filename or "image.jpg", await image.read())
    except CatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/employees/{employee_id}/images", dependencies=[Depends(require_key)])
async def add_employee_image(employee_id: str, image: UploadFile = File(...)) -> dict[str, Any]:
    employee = next(
        (item for item in CATALOG.list_employees() if str(item["id"]) == str(employee_id)),
        None,
    )
    if employee is None:
        raise HTTPException(status_code=404, detail="Employee not found.")
    try:
        return CATALOG.enroll(employee["name"], image.filename or "image.jpg", await image.read())
    except CatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/employees/{employee_id}", dependencies=[Depends(require_key)])
def remove_employee(employee_id: str) -> dict[str, bool]:
    if not CATALOG.delete_employee(employee_id):
        raise HTTPException(status_code=404, detail="Employee not found.")
    return {"ok": True}


@app.delete("/employees/{employee_id}/images/{image_id}", dependencies=[Depends(require_key)])
def remove_employee_image(employee_id: str, image_id: str) -> dict[str, bool]:
    if not CATALOG.delete_employee_image(employee_id, image_id):
        raise HTTPException(status_code=404, detail="Image not found.")
    return {"ok": True}


@app.get("/employees/{employee_id}/images/{image_id}")
def employee_image(employee_id: str, image_id: str, _: None = Depends(require_key)) -> FileResponse:
    path = CATALOG.employee_image(employee_id, image_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media)


@app.get("/cameras/{camera_id}/desks", dependencies=[Depends(require_key)])
def get_desks(camera_id: str) -> dict[str, Any]:
    """Workstation boxes, exposed under the desks path the frontend already calls."""
    return get_rois(camera_id)
