"""Attendance totals come from sessions, not from how many frames matched a face."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_service, require_api_key
from app.database.session import session_scope
from app.utils.timestamps import to_local, utc_now

router = APIRouter(prefix="/attendance", tags=["attendance"], dependencies=[Depends(require_api_key)])


def _today(service) -> date:
    return to_local(utc_now(), service.config.local_timezone).date()


@router.get("/today")
def attendance_today(request: Request) -> list[dict]:
    service = get_service(request)
    day = _today(service)
    with session_scope(service._session_factory) as db:
        return [service.summary_for(row.employee_code, day, db) for row in service.repository.list_employees(db)]


@router.get("/{employee_id}/summary")
def attendance_summary(employee_id: str, request: Request, day: date | None = None) -> dict:
    service = get_service(request)
    target = day or _today(service)
    with session_scope(service._session_factory) as db:
        if service.repository.get_employee(db, employee_id) is None:
            raise HTTPException(status_code=404, detail="Employee not found")
        return service.summary_for(employee_id, target, db)


@router.get("/{employee_id}")
def attendance_history(employee_id: str, request: Request) -> dict:
    service = get_service(request)
    zone = service.config.local_timezone
    with session_scope(service._session_factory) as db:
        if service.repository.get_employee(db, employee_id) is None:
            raise HTTPException(status_code=404, detail="Employee not found")
        rows = service.repository.attendance_sessions_for(db, employee_id)
    return {
        "employee_id": employee_id,
        "sessions": [
            {
                "entry_time": row.entry_time.isoformat(),
                "exit_time": None if row.exit_time is None else row.exit_time.isoformat(),
                "entry_camera": row.entry_camera,
                "exit_camera": row.exit_camera,
                "status": row.status,
                "local_entry_date": to_local(row.entry_time, zone).date().isoformat(),
            }
            for row in rows
        ],
    }
