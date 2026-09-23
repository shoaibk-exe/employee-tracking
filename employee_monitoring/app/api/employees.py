"""Employee records. Face images are not returned."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.api.deps import get_service, require_api_key
from app.database.session import session_scope

router = APIRouter(prefix="/employees", tags=["employees"], dependencies=[Depends(require_api_key)])


class EmployeeIn(BaseModel):
    employee_code: str
    name: str
    active: bool = True
    assigned_desk_id: str | None = None


class EmployeeUpdate(BaseModel):
    name: str | None = None
    active: bool | None = None
    assigned_desk_id: str | None = None


class EmployeeOut(BaseModel):
    employee_id: str
    name: str
    active: bool
    assigned_desk_id: str | None


def _out(row) -> EmployeeOut:
    return EmployeeOut(
        employee_id=row.employee_code,
        name=row.name,
        active=row.active,
        assigned_desk_id=row.assigned_desk_id,
    )


@router.get("")
def list_employees(request: Request) -> list[EmployeeOut]:
    service = get_service(request)
    with session_scope(service._session_factory) as db:
        return [_out(row) for row in service.repository.list_employees(db)]


@router.get("/{employee_id}")
def get_employee(employee_id: str, request: Request) -> EmployeeOut:
    service = get_service(request)
    with session_scope(service._session_factory) as db:
        row = service.repository.get_employee(db, employee_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Employee not found")
        return _out(row)


@router.post("", status_code=201)
def create_employee(body: EmployeeIn, request: Request) -> EmployeeOut:
    service = get_service(request)
    with session_scope(service._session_factory) as db:
        row = service.repository.ensure_employee(db, body.employee_code, body.name, body.assigned_desk_id)
        row.active = body.active
        return _out(row)


@router.put("/{employee_id}")
def update_employee(employee_id: str, body: EmployeeUpdate, request: Request) -> EmployeeOut:
    service = get_service(request)
    with session_scope(service._session_factory) as db:
        row = service.repository.get_employee(db, employee_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Employee not found")
        if body.name is not None:
            row.name = body.name
        if body.active is not None:
            row.active = body.active
        if body.assigned_desk_id is not None:
            row.assigned_desk_id = body.assigned_desk_id
        return _out(row)
