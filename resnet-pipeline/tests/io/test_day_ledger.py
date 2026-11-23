from datetime import date
from zoneinfo import ZoneInfo

from core.io.day_ledger import DayLedger


KARACHI = ZoneInfo("Asia/Karachi")


def _summary(path, rows):
    path.write_text(
        "camera_id,session_id,workstation_id,occupied_seconds,empty_seconds,standing_near_seconds,"
        "unknown_seconds,observed_seconds,transition_count\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_restart_adds_to_the_same_day_and_a_second_update_does_not_double_count(tmp_path):
    ledger = DayLedger(tmp_path, KARACHI)
    day = date(2026, 11, 6)
    ledger.record_session(day, "20261106T072357Z_aaaaaa", {
        "WS01": {"occupied_seconds": 100, "empty_seconds": 5, "state": "occupied", "bbox": [1, 2, 3, 4]},
    }, closed=True)
    ledger.record_session(day, "20261106T090000Z_bbbbbb", {
        "WS01": {"occupied_seconds": 10, "empty_seconds": 1, "state": "empty", "bbox": [1, 2, 3, 4]},
    }, closed=False)
    combined = ledger.combined(day)
    row = combined["workstations"][0]
    assert row["occupied_seconds"] == 110
    assert row["empty_seconds"] == 6
    assert row["state"] == "empty"
    assert combined["session_count"] == 2

    ledger.record_session(day, "20261106T090000Z_bbbbbb", {
        "WS01": {"occupied_seconds": 25, "empty_seconds": 1, "state": "occupied"},
    }, closed=False)
    again = ledger.combined(day)["workstations"][0]
    assert again["occupied_seconds"] == 125
    assert again["bbox"] == [1, 2, 3, 4]


def test_a_later_calendar_day_does_not_include_the_previous_day(tmp_path):
    ledger = DayLedger(tmp_path, KARACHI)
    ledger.record_session(date(2026, 11, 6), "20261106T120000Z_aaaaaa", {
        "WS01": {"occupied_seconds": 50, "empty_seconds": 0},
    }, closed=True)
    ledger.record_session(date(2026, 11, 7), "20261106T200000Z_aaaaaa", {
        "WS01": {"occupied_seconds": 8, "empty_seconds": 0},
    }, closed=False)
    assert ledger.combined(date(2026, 11, 6))["workstations"][0]["occupied_seconds"] == 50
    assert ledger.combined(date(2026, 11, 7))["workstations"][0]["occupied_seconds"] == 8


def test_combined_drops_workstations_removed_from_latest_session(tmp_path):
    ledger = DayLedger(tmp_path, KARACHI)
    day = date(2026, 11, 10)
    ledger.record_session(day, "20261110T010000Z_old", {
        "WS01": {"occupied_seconds": 100, "empty_seconds": 5, "state": "occupied"},
        "WS02": {"occupied_seconds": 70, "empty_seconds": 2, "state": "occupied"},
    }, closed=True)
    ledger.record_session(day, "20261110T120000Z_new", {
        "WS01": {"occupied_seconds": 10, "empty_seconds": 1, "state": "empty", "bbox": [1, 2, 3, 4]},
    }, closed=False)
    rows = {row["workstation_id"]: row for row in ledger.combined(day)["workstations"]}
    assert set(rows) == {"WS01"}
    assert rows["WS01"]["occupied_seconds"] == 110
    assert "WS02" not in rows


def test_summary_import_uses_the_office_calendar_and_does_not_replace_a_live_session(tmp_path):
    ledger = DayLedger(tmp_path, KARACHI)
    summary = tmp_path / "summary.csv"
    _summary(summary, [
        "1,20261105T180000Z_aaaaaa,WS01,10,1,0,0,10,1",
        "1,20261105T200000Z_bbbbbb,WS01,30,2,0,0,30,1",
        "1,20261106T072357Z_cccccc,WS01,40,4,0,0,40,1",
    ])
    assert ledger.import_summary(summary) == 3
    assert ledger.import_summary(summary) == 0
    nov5 = ledger.combined(date(2026, 11, 5))["workstations"][0]
    nov6 = {row["workstation_id"]: row for row in ledger.combined(date(2026, 11, 6))["workstations"]}
    assert nov5["occupied_seconds"] == 10
    assert nov6["WS01"]["occupied_seconds"] == 70

    ledger.record_session(date(2026, 11, 6), "20261106T072357Z_cccccc", {
        "WS01": {"occupied_seconds": 41, "empty_seconds": 4, "state": "occupied"},
    }, closed=True)
    assert ledger.import_summary(summary) == 0
    assert ledger.combined(date(2026, 11, 6))["workstations"][0]["occupied_seconds"] == 71
