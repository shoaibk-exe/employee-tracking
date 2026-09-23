"""Camera inventory and live health. Health is about the stream, not the employee."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_service, require_api_key
from app.database.session import session_scope

router = APIRouter(prefix="/cameras", tags=["cameras"], dependencies=[Depends(require_api_key)])


@router.get("")
def list_cameras(request: Request) -> list[dict]:
    service = get_service(request)
    with session_scope(service._session_factory) as db:
        rows = service.repository.list_cameras(db)
    return [
        {
            "camera_id": row.id,
            "name": row.name,
            "type": row.type,
            "location": row.location,
            "enabled": row.enabled,
        }
        for row in rows
    ]


@router.get("/health")
def camera_health(request: Request) -> list[dict]:
    service = get_service(request)
    payload = []
    for camera in service.config.cameras:
        reader = service._readers.get(camera.camera_id)
        inference = service.camera_monitors.get(camera.camera_id)
        if reader is None:
            payload.append(
                {
                    "camera_id": camera.camera_id,
                    "last_frame_time": None,
                    "fps": 0,
                    "connection_status": "OFFLINE",
                    "reconnect_count": 0,
                    "process_fps": 0,
                    "inference_errors": 0,
                }
            )
            continue
        health = reader.health_snapshot()
        infer = inference.snapshot() if inference else None
        payload.append(
            {
                "camera_id": health.camera_id,
                "last_frame_time": None if health.last_frame_time is None else health.last_frame_time.isoformat(),
                "fps": health.fps,
                "connection_status": health.connection_status.value,
                "reconnect_count": health.reconnect_count,
                "process_fps": health.process_fps,
                "detail": health.detail,
                "inference_errors": 0 if infer is None else infer.error_count,
                "last_inference_error": "" if infer is None else infer.last_error,
            }
        )
    return payload
