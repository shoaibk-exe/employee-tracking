"""Desk status and desk-time totals. Away-from-desk is not the same as outside the office."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_service, require_api_key
from app.database.session import session_scope
from app.utils.timestamps import to_local, utc_now

router = APIRouter(prefix="/desk", tags=["desk"], dependencies=[Depends(require_api_key)])


@router.get("/status")
def desk_status(request: Request) -> list[dict]:
    service = get_service(request)
    rows = []
    seen = set()
    for desk in service.config.desks:
        seen.add(desk.employee_id)
        runtime = service.runtime.get(desk.employee_id)
        rows.append(
            {
                "employee_id": desk.employee_id,
                "desk_id": desk.desk_id,
                "status": runtime.desk_state.value if runtime else "UNKNOWN",
                "camera_id": desk.camera_id,
                "desk_confidence": None if runtime is None else runtime.desk_confidence,
            }
        )
    for employee_id, runtime in service.runtime.items():
        if employee_id in seen:
            continue
        rows.append(
            {
                "employee_id": employee_id,
                "desk_id": runtime.desk_id,
                "status": runtime.desk_state.value,
                "camera_id": runtime.current_camera_id,
                "desk_confidence": runtime.desk_confidence,
            }
        )
    return rows


@router.get("/{employee_id}/summary")
def desk_summary(employee_id: str, request: Request, day: date | None = None) -> dict:
    service = get_service(request)
    target = day or to_local(utc_now(), service.config.local_timezone).date()
    with session_scope(service._session_factory) as db:
        if service.repository.get_employee(db, employee_id) is None:
            raise HTTPException(status_code=404, detail="Employee not found")
        summary = service.summary_for(employee_id, target, db)
    return {
        "employee_id": employee_id,
        "status": summary["desk_status"],
        "desk_time_seconds": summary["total_desk_seconds"],
        "away_time_seconds": summary["total_away_seconds"],
        "number_of_desk_absences": summary["number_of_desk_absences"],
        "longest_away_seconds": summary["longest_away_seconds"],
        "date": summary["date"],
    }
