"""Excel progress report: per-workstation empty/occupied/working + P## attribution.

Written on every finalize (end of video, quit, interrupt) so a mid-run stop
still leaves a usable ``.xlsx`` in the output directory.

Times are shown as ``N min`` under 60 minutes, or ``H:MM`` (e.g. ``1:30``) at/above
60 minutes — no raw seconds columns.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """``59 min`` if under 1 hour; ``1:30`` (hours:minutes) once total minutes ≥ 60."""
    total_min = max(0, int(round(float(seconds) / 60.0)))
    if total_min < 60:
        return f"{total_min} min"
    hours, minutes = divmod(total_min, 60)
    return f"{hours}:{minutes:02d}"


def _pct(part: float, total: float) -> float:
    if total <= 1e-9:
        return 0.0
    return round(100.0 * float(part) / float(total), 1)


def _join_person_attribution(entries: Sequence[tuple[str, float]]) -> str:
    """``P01 (12 min); P03 (1:05)`` sorted by time descending."""
    if not entries:
        return ""
    ordered = sorted(entries, key=lambda kv: (-kv[1], kv[0]))
    return "; ".join(f"{pid} ({_fmt_duration(secs)})" for pid, secs in ordered if secs > 0.05)


def write_progress_workbook(
    path: Path,
    *,
    session_id: str,
    tenant_id: str,
    camera_id: str,
    end_reason: str,
    last_timestamp: float | None,
    # employee_id → workstation_id → seated seconds
    employee_seated: Mapping[str, Mapping[str, float]],
    # workstation_id → seated seconds (all people)
    workstation_seated: Mapping[str, float],
    # anonymous P## rows: dict with person_id, workstation_id?, seated/working/seen
    person_rows: Sequence[Mapping[str, Any]] = (),
    # track-level working snapshots
    working_rows: Sequence[Mapping[str, Any]] = (),
    # desk empty/working totals (pose+occupied "working" clock)
    desk_rows: Sequence[Mapping[str, Any]] = (),
    # ResNet occupancy per workstation (occupied/empty/unknown seconds)
    occupancy_rows: Sequence[Mapping[str, Any]] = (),
    # P## seated time broken out per workstation
    person_desk_rows: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Write ``path`` (.xlsx). Returns the path written."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()

    total_time = max(0.0, float(last_timestamp or 0.0))

    occ_map = {str(r.get("workstation_id")): r for r in occupancy_rows if r.get("workstation_id")}
    desk_map = {str(r.get("workstation_id")): r for r in desk_rows if r.get("workstation_id")}

    person_at_ws: dict[str, list[tuple[str, float]]] = {}
    ws_persons: dict[str, list[tuple[str, float]]] = {}
    for row in person_desk_rows:
        pid = str(row.get("person_id") or "")
        ws_id = str(row.get("workstation_id") or "")
        secs = float(row.get("seated_seconds", 0.0) or 0.0)
        if not pid or not ws_id or secs <= 0:
            continue
        person_at_ws.setdefault(pid, []).append((ws_id, secs))
        ws_persons.setdefault(ws_id, []).append((pid, secs))

    all_ws = sorted(
        set(workstation_seated)
        | set(desk_map)
        | set(occ_map)
        | set(ws_persons)
    )

    # ---- Progress (primary)
    progress = wb.active
    progress.title = "Progress"
    progress.append(["Workstation Progress"])
    progress["A1"].font = Font(bold=True, size=14)
    progress.append([
        "total_session", _fmt_duration(total_time),
        "end_reason", end_reason,
    ])
    progress.append([])
    headers = [
        "workstation_id",
        "total",
        "empty", "empty_pct",
        "occupied", "occupied_pct",
        "working", "working_pct",
        "primary_person_id", "persons_at_desk",
        "seated", "unknown",
    ]
    progress.append(headers)
    for col in range(1, len(headers) + 1):
        progress.cell(4, col).font = Font(bold=True)

    sum_empty = sum_occ = sum_work = 0.0
    for ws_id in all_ws:
        occ = occ_map.get(ws_id, {})
        desk = desk_map.get(ws_id, {})
        empty = float(occ.get("empty_seconds", desk.get("empty_seconds", 0.0)) or 0.0)
        occupied = float(occ.get("occupied_seconds", 0.0) or 0.0)
        if ws_id not in occ_map:
            occupied = max(occupied, float(workstation_seated.get(ws_id, 0.0)))
        unknown = float(occ.get("unknown_seconds", 0.0) or 0.0)
        working = float(desk.get("working_seconds", 0.0) or 0.0)
        seated = float(workstation_seated.get(ws_id, 0.0))
        denom = total_time if total_time > 1e-9 else max(empty + occupied + unknown, seated, 1e-9)
        people = ws_persons.get(ws_id, [])
        primary = max(people, key=lambda kv: kv[1])[0] if people else ""
        progress.append([
            ws_id,
            _fmt_duration(denom),
            _fmt_duration(empty), _pct(empty, denom),
            _fmt_duration(occupied), _pct(occupied, denom),
            _fmt_duration(working), _pct(working, denom),
            primary, _join_person_attribution(people),
            _fmt_duration(seated), _fmt_duration(unknown),
        ])
        sum_empty += empty
        sum_occ += occupied
        sum_work += working

    if all_ws:
        n = len(all_ws)
        site_denom = total_time * n if total_time > 1e-9 else max(sum_empty + sum_occ, 1e-9)
        seated_all = sum(float(v) for v in workstation_seated.values())
        progress.append([])
        progress.append([
            "ALL_WORKSTATIONS",
            _fmt_duration(total_time),
            _fmt_duration(sum_empty), _pct(sum_empty, site_denom),
            _fmt_duration(sum_occ), _pct(sum_occ, site_denom),
            _fmt_duration(sum_work), _pct(sum_work, site_denom),
            "", f"{n} desks",
            _fmt_duration(seated_all), "",
        ])
        for col in range(1, len(headers) + 1):
            progress.cell(progress.max_row, col).font = Font(bold=True)
    else:
        progress.append(["(no workstation data yet)"] + [""] * (len(headers) - 1))

    # ---- Summary
    summary = wb.create_sheet("Summary", 1)
    summary.append(["Employee Presence Progress"])
    summary["A1"].font = Font(bold=True, size=14)
    summary.append(["tenant_id", tenant_id])
    summary.append(["camera_id", camera_id])
    summary.append(["session_id", session_id])
    summary.append(["end_reason", end_reason])
    summary.append(["total_session", _fmt_duration(total_time)])
    summary.append(["workstations", len(all_ws)])
    summary.append(["total_empty", _fmt_duration(sum_empty)])
    summary.append(["total_occupied", _fmt_duration(sum_occ)])
    summary.append(["total_working", _fmt_duration(sum_work)])
    if total_time > 1e-9 and all_ws:
        site_denom = total_time * len(all_ws)
        summary.append(["site_empty_pct", _pct(sum_empty, site_denom)])
        summary.append(["site_occupied_pct", _pct(sum_occ, site_denom)])
        summary.append(["site_working_pct", _pct(sum_work, site_denom)])
    summary.append(["saved_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")])
    summary.append([])
    summary.append([
        "Note",
        "Times: 'N min' under 60 minutes, 'H:MM' (e.g. 1:30) at/above 60 minutes. "
        "empty/occupied = ResNet; working = occupied + sitting; pct = share of session per desk.",
    ])

    # ---- Persons at each workstation
    pdesk = wb.create_sheet("Persons_at_WS")
    pdesk.append([
        "workstation_id", "person_id", "seated", "share_of_desk_seated_pct",
        "tenant_id", "camera_id", "session_id",
    ])
    for col in range(1, 8):
        pdesk.cell(1, col).font = Font(bold=True)
    any_pdesk = False
    for ws_id in sorted(ws_persons):
        people = ws_persons[ws_id]
        desk_total = sum(s for _, s in people) or 1e-9
        for pid, secs in sorted(people, key=lambda kv: (-kv[1], kv[0])):
            any_pdesk = True
            pdesk.append([
                ws_id, pid, _fmt_duration(secs), _pct(secs, desk_total),
                tenant_id, camera_id, session_id,
            ])
    if not any_pdesk:
        pdesk.append(["(none yet)", "", "0 min", 0, tenant_id, camera_id, session_id])

    # ---- Employees
    emp = wb.create_sheet("Employees")
    emp.append([
        "employee_id", "workstation_id", "seated",
        "tenant_id", "camera_id", "session_id",
    ])
    for col in range(1, 7):
        emp.cell(1, col).font = Font(bold=True)
    for employee, per_ws in sorted(employee_seated.items()):
        for ws_id, seconds in sorted(per_ws.items()):
            emp.append([
                employee, ws_id, _fmt_duration(seconds),
                tenant_id, camera_id, session_id,
            ])
    if emp.max_row == 1:
        emp.append(["(none yet)", "", "0 min", tenant_id, camera_id, session_id])

    # ---- Persons_P
    people = wb.create_sheet("Persons_P")
    people.append([
        "person_id", "home_workstation", "at_workstations", "seen",
        "seen_pct_of_session", "desks_detail", "samples", "anchor_track_T",
        "tenant_id", "camera_id", "session_id",
    ])
    for col in range(1, 12):
        people.cell(1, col).font = Font(bold=True)
    for row in person_rows:
        pid = str(row.get("person_id", ""))
        ws_secs = float(row.get("workstation_seconds", 0.0) or 0.0)
        seen = float(row.get("seen_seconds", 0.0) or 0.0)
        desks = person_at_ws.get(pid, [])
        people.append([
            pid,
            row.get("home_workstation", "") or "",
            _fmt_duration(ws_secs),
            _fmt_duration(seen),
            _pct(seen, total_time if total_time > 1e-9 else 1.0),
            _join_person_attribution(desks),
            int(row.get("samples", 0) or 0),
            row.get("anchor_track_T", "") or "",
            tenant_id, camera_id, session_id,
        ])
    if people.max_row == 1:
        people.append(
            ["(none yet)", "", "0 min", "0 min", 0, "", 0, "", tenant_id, camera_id, session_id]
        )

    # ---- Workstations (compact)
    desks = wb.create_sheet("Workstations")
    desks.append([
        "workstation_id",
        "empty", "empty_pct",
        "occupied", "occupied_pct",
        "working", "working_pct",
        "primary_person_id", "persons_at_desk",
        "seated", "total",
        "tenant_id", "camera_id", "session_id",
    ])
    for col in range(1, 15):
        desks.cell(1, col).font = Font(bold=True)
    for ws_id in all_ws:
        occ = occ_map.get(ws_id, {})
        desk = desk_map.get(ws_id, {})
        empty = float(occ.get("empty_seconds", desk.get("empty_seconds", 0.0)) or 0.0)
        occupied = float(occ.get("occupied_seconds", 0.0) or 0.0)
        if ws_id not in occ_map:
            occupied = max(occupied, float(workstation_seated.get(ws_id, 0.0)))
        working = float(desk.get("working_seconds", 0.0) or 0.0)
        seated = float(workstation_seated.get(ws_id, 0.0))
        denom = total_time if total_time > 1e-9 else max(empty + occupied, seated, 1e-9)
        people_list = ws_persons.get(ws_id, [])
        primary = max(people_list, key=lambda kv: kv[1])[0] if people_list else ""
        desks.append([
            ws_id,
            _fmt_duration(empty), _pct(empty, denom),
            _fmt_duration(occupied), _pct(occupied, denom),
            _fmt_duration(working), _pct(working, denom),
            primary, _join_person_attribution(people_list),
            _fmt_duration(seated), _fmt_duration(denom),
            tenant_id, camera_id, session_id,
        ])
    if desks.max_row == 1:
        desks.append(
            ["(none yet)", "0 min", 0, "0 min", 0, "0 min", 0, "", "", "0 min", "0 min",
             tenant_id, camera_id, session_id]
        )

    # ---- Working by track
    work = wb.create_sheet("Working_by_track")
    work.append([
        "local_track_id", "composite_track_id", "person_id", "employee_id",
        "workstation_id", "activity_class", "working", "working_pct_of_session",
        "active", "tenant_id", "camera_id", "session_id",
    ])
    for col in range(1, 13):
        work.cell(1, col).font = Font(bold=True)
    for row in working_rows:
        secs = float(row.get("working_seconds", 0.0) or 0.0)
        work.append([
            row.get("local_track_id", ""),
            row.get("composite_track_id", ""),
            row.get("person_id", "") or "",
            row.get("employee_id", "") or "",
            row.get("workstation_id", "") or "",
            row.get("activity_class", "") or "",
            _fmt_duration(secs),
            _pct(secs, total_time if total_time > 1e-9 else 1.0),
            int(bool(row.get("active"))),
            tenant_id, camera_id, session_id,
        ])

    for sheet in wb.worksheets:
        for column in sheet.columns:
            letter = column[0].column_letter
            width = min(48, max(10, max(len(str(c.value or "")) for c in column) + 2))
            sheet.column_dimensions[letter].width = width
            for cell in column:
                cell.alignment = Alignment(vertical="center")

    tmp = path.with_suffix(".tmp.xlsx")
    wb.save(tmp)
    tmp.replace(path)
    logger.info("Wrote progress Excel %s (end_reason=%s)", path, end_reason)
    return path
