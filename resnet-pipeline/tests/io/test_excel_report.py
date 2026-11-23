"""Progress Excel is written even for a short / interrupted session."""

from pathlib import Path

from core.io.excel_report import _fmt_duration, write_progress_workbook


def test_fmt_duration_minutes_then_hours():
    assert _fmt_duration(0) == "0 min"
    assert _fmt_duration(59) == "1 min"       # rounds to nearest minute
    assert _fmt_duration(30 * 60) == "30 min"
    assert _fmt_duration(59 * 60) == "59 min"
    assert _fmt_duration(60 * 60) == "1:00"
    assert _fmt_duration(90 * 60) == "1:30"
    assert _fmt_duration(125 * 60) == "2:05"


def test_write_progress_workbook(tmp_path: Path):
    path = tmp_path / "progress.xlsx"
    out = write_progress_workbook(
        path,
        session_id="sess1",
        tenant_id="demo",
        camera_id="CAM01",
        end_reason="interrupted",
        last_timestamp=100.0,  # ~2 min session
        employee_seated={"E01": {"WS01": 12.0}, "UNKNOWN_EMPLOYEE": {"WS02": 3.0}},
        workstation_seated={"WS01": 40.0, "WS02": 10.0},
        person_rows=[{
            "person_id": "P01", "home_workstation": "WS01",
            "workstation_seconds": 40.0, "seen_seconds": 80.0, "samples": 8, "anchor_track_T": "T3",
        }],
        working_rows=[{
            "local_track_id": 3, "composite_track_id": "cam:3", "person_id": "P01",
            "employee_id": "E01", "workstation_id": "WS01", "activity_class": "working",
            "working_seconds": 25.0, "active": True,
        }],
        desk_rows=[
            {"workstation_id": "WS01", "empty_seconds": 20.0, "working_seconds": 25.0},
            {"workstation_id": "WS02", "empty_seconds": 50.0, "working_seconds": 0.0},
        ],
        occupancy_rows=[
            {"workstation_id": "WS01", "empty_seconds": 30.0, "occupied_seconds": 60.0, "unknown_seconds": 10.0},
            {"workstation_id": "WS02", "empty_seconds": 70.0, "occupied_seconds": 20.0, "unknown_seconds": 10.0},
        ],
        person_desk_rows=[
            {"person_id": "P01", "workstation_id": "WS01", "seated_seconds": 40.0},
            {"person_id": "P02", "workstation_id": "WS01", "seated_seconds": 5.0},
            {"person_id": "P03", "workstation_id": "WS02", "seated_seconds": 10.0},
        ],
    )
    assert out.is_file()
    from openpyxl import load_workbook
    wb = load_workbook(out)
    assert set(wb.sheetnames) >= {
        "Progress", "Summary", "Persons_at_WS", "Employees", "Persons_P", "Workstations", "Working_by_track",
    }

    prog = wb["Progress"]
    assert prog["A4"].value == "workstation_id"
    assert list(c.value for c in prog[4])[:8] == [
        "workstation_id", "total", "empty", "empty_pct", "occupied", "occupied_pct", "working", "working_pct",
    ]
    assert prog["A5"].value == "WS01"
    assert prog["B5"].value == "2 min"     # total session ~100s
    assert prog["C5"].value == "0 min"     # empty 30s → rounds to 0 min
    assert prog["D5"].value == 30.0        # empty_pct
    assert prog["E5"].value == "1 min"     # occupied 60s
    assert prog["F5"].value == 60.0        # occupied_pct
    assert prog["G5"].value == "0 min"     # working 25s → 0 min
    assert prog["H5"].value == 25.0        # working_pct
    assert prog["I5"].value == "P01"
    assert "P01" in str(prog["J5"].value)

    assert wb["Summary"]["B5"].value == "interrupted"
    assert wb["Employees"]["A2"].value == "E01"
    assert wb["Persons_P"]["A2"].value == "P01"
    assert wb["Persons_at_WS"]["A2"].value == "WS01"
    assert wb["Workstations"]["A2"].value == "WS01"
    assert wb["Workstations"]["D2"].value == "1 min"  # occupied


def test_progress_hour_format_and_fallback(tmp_path: Path):
    path = tmp_path / "progress2.xlsx"
    out = write_progress_workbook(
        path,
        session_id="s2",
        tenant_id="demo",
        camera_id="CAM01",
        end_reason="max_frames",
        last_timestamp=90 * 60,  # 1:30 session
        employee_seated={},
        workstation_seated={"WS01": 12.0},
        desk_rows=[{"workstation_id": "WS01", "empty_seconds": 8.0, "working_seconds": 5.0}],
        occupancy_rows=[{
            "workstation_id": "WS01",
            "empty_seconds": 30 * 60,
            "occupied_seconds": 60 * 60,
            "unknown_seconds": 0.0,
        }],
        person_desk_rows=[{"person_id": "P01", "workstation_id": "WS01", "seated_seconds": 12.0}],
    )
    from openpyxl import load_workbook
    wb = load_workbook(out)
    prog = wb["Progress"]
    assert prog["A5"].value == "WS01"
    assert prog["B5"].value == "1:30"   # total
    assert prog["C5"].value == "30 min"  # empty
    assert prog["E5"].value == "1:00"   # occupied
    assert prog["I5"].value == "P01"
