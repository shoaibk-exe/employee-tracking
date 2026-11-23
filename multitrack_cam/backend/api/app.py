"""FastAPI application: REST, WebSocket metadata, MJPEG camera streams, dashboard."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import load_app_config
from backend.pipeline.orchestrator import TrackingPipeline
from backend.streaming.mjpeg import mjpeg_generator, mjpeg_live_generator
from backend.utils.logging_setup import setup_logging
from backend.utils.paths import PROJECT_ROOT, resolve_path

logger = logging.getLogger(__name__)


class ManualCorrectionRequest(BaseModel):
    compound_key: str
    global_id: str


class CalibrationRequest(BaseModel):
    camera_id: str
    # Normalized [0..1] points in the rotated processing frame.
    points_camera: list[list[float]]
    points_reference: list[list[float]]
    # "homography" (>= 4 ground points, global plane) or "anchors" (verified spots such as seats).
    mode: str = "homography"
    labels: list[str] | None = None
    # Anchor mode only: max match radius around each anchor in the member view / in the reference view.
    anchor_radius: float | None = None
    reference_radius: float | None = None


def create_app(settings_path: str | None = None) -> FastAPI:
    config = load_app_config(settings_path)
    log_dir = resolve_path(config.get("system", "log_directory", default="data/logs"))
    setup_logging(str(config.get("system", "log_level", default="INFO")), log_dir)

    state: dict[str, Any] = {"pipeline": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipeline = TrackingPipeline(config)
        state["pipeline"] = pipeline
        pipeline.start()
        logger.info("API lifespan: pipeline started")
        try:
            yield
        finally:
            pipeline.stop()
            state["pipeline"] = None
            logger.info("API lifespan: pipeline stopped")

    api_cfg = config.section("api")
    app = FastAPI(title="MTMCT Multi-Camera Tracking", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(api_cfg.get("cors_origins") or ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    frontend_dir = PROJECT_ROOT / "frontend"
    if frontend_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    def pipeline() -> TrackingPipeline:
        p = state.get("pipeline")
        if p is None:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        return p

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        index = frontend_dir / "index.html"
        if not index.is_file():
            return HTMLResponse("<h1>MTMCT</h1><p>frontend/index.html missing</p>")
        return HTMLResponse(index.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        p = pipeline()
        stats = p.system_stats()
        return {
            "status": "ok",
            "uptime_seconds": round(p.health.uptime_seconds(), 1),
            "live_cameras": stats.live_cameras,
            "total_cameras": stats.total_cameras,
            "gpu_memory_mb": stats.gpu_memory_mb,
        }

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        s = pipeline().system_stats()
        return {
            "live_cameras": s.live_cameras,
            "total_cameras": s.total_cameras,
            "active_global_identities": s.active_global_identities,
            "total_retained_identities": s.total_retained_identities,
            "unresolved_tracks": s.unresolved_tracks,
            "gpu_memory_mb": s.gpu_memory_mb,
            "gpu_util_percent": s.gpu_util_percent,
            "detection_latency_ms": round(s.detection_latency_ms, 1),
            "reid_latency_ms": round(s.reid_latency_ms, 1),
            "association_latency_ms": round(s.association_latency_ms, 1),
            "end_to_end_latency_ms": round(s.end_to_end_latency_ms, 1),
            "timestamp": s.timestamp,
        }

    @app.get("/api/cameras")
    def cameras() -> list[dict[str, Any]]:
        return pipeline().camera_views()

    @app.get("/api/identities")
    def identities() -> list[dict[str, Any]]:
        return pipeline().identity.list_identities()

    @app.get("/api/identities/{global_id}")
    def identity_detail(global_id: str) -> dict[str, Any]:
        detail = pipeline().identity.identity_detail(global_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        return detail

    @app.post("/api/identities/correct")
    def correct_identity(body: ManualCorrectionRequest) -> dict[str, str]:
        pipeline().identity.manual_correct(body.compound_key, body.global_id)
        return {"status": "ok"}

    @app.get("/api/stream/{camera_id}")
    def stream(camera_id: str) -> StreamingResponse:
        p = pipeline()
        if p.streams.get_worker(camera_id) is None:
            raise HTTPException(status_code=404, detail="Unknown camera")

        if p.display_decoupled and p.mode == "live":
            return StreamingResponse(
                mjpeg_live_generator(lambda: p.live_jpeg(camera_id), max_fps=p.mjpeg_fps),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
            )

        def frames():
            return p.annotated_frame(camera_id)

        return StreamingResponse(
            mjpeg_generator(frames, fps=p.mjpeg_fps, quality=p.jpeg_quality),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/snapshot/{camera_id}")
    def snapshot(camera_id: str) -> Response:
        """Un-annotated, rotated processing frame (used by the calibration page)."""
        p = pipeline()
        if p.streams.get_worker(camera_id) is None:
            raise HTTPException(status_code=404, detail="Unknown camera")
        frame = p.raw_snapshot(camera_id)
        if frame is None:
            raise HTTPException(status_code=503, detail="No frame yet")
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise HTTPException(status_code=500, detail="Encode failed")
        return Response(content=buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/calibration")
    def calibration_status() -> dict[str, Any]:
        p = pipeline()
        return {"groups": p.calibration_status(), "resolver": p.resolver_stats()}

    @app.post("/api/calibration")
    def set_calibration(body: CalibrationRequest) -> dict[str, Any]:
        try:
            return pipeline().set_calibration(
                body.camera_id, body.points_camera, body.points_reference, mode=body.mode, labels=body.labels,
                anchor_radius=body.anchor_radius, reference_radius=body.reference_radius,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/calibration/{camera_id}")
    def clear_calibration(camera_id: str) -> dict[str, str]:
        pipeline().clear_calibration(camera_id)
        return {"status": "cleared"}

    @app.get("/calibrate", response_class=HTMLResponse)
    def calibrate_page() -> HTMLResponse:
        page = frontend_dir / "calibrate.html"
        if not page.is_file():
            return HTMLResponse("<h1>calibrate.html missing</h1>")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                p = state.get("pipeline")
                if p is None:
                    await ws.send_json({"error": "pipeline_unavailable"})
                else:
                    await ws.send_json(
                        {
                            "stats": {
                                **{
                                    k: getattr(p.system_stats(), k)
                                    for k in [
                                        "live_cameras",
                                        "total_cameras",
                                        "active_global_identities",
                                        "total_retained_identities",
                                        "unresolved_tracks",
                                        "gpu_memory_mb",
                                        "detection_latency_ms",
                                        "reid_latency_ms",
                                        "association_latency_ms",
                                        "end_to_end_latency_ms",
                                    ]
                                }
                            },
                            "cameras": p.camera_views(),
                        }
                    )
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("WebSocket closed: %s", exc)

    return app
