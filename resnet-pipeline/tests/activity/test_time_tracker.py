"""Unit tests for VideoMAE day-time accumulation."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from core.activity.time_tracker import ActivityTimeTracker


class ActivityTimeTrackerTests(unittest.TestCase):
    def test_accumulates_and_switches_classes(self) -> None:
        tracker = ActivityTimeTracker(
            "WS01",
            origin_utc=datetime(2026, 11, 10, 5, 0, tzinfo=timezone.utc),
            report_tz=ZoneInfo("Asia/Karachi"),
        )
        tracker.observe(0.0, "computer_interaction")
        tracker.observe(10.0, "computer_interaction")
        tracker.observe(10.0, "phone_using")
        tracker.observe(25.0, "phone_using")
        totals = tracker.totals_at(25.0)
        self.assertAlmostEqual(totals["computer_interaction"], 10.0)
        self.assertAlmostEqual(totals["phone_using"], 15.0)
        self.assertAlmostEqual(totals["neither"], 0.0)
        fields = tracker.as_seconds_fields(totals)
        self.assertEqual(fields["computer_interaction_seconds"], 10.0)
        self.assertEqual(fields["phone_using_seconds"], 15.0)

    def test_seconds_by_day_keeps_growing_while_class_unchanged(self) -> None:
        tracker = ActivityTimeTracker(
            "WS01",
            origin_utc=datetime(2026, 11, 10, 5, 0, tzinfo=timezone.utc),
            report_tz=ZoneInfo("Asia/Karachi"),
        )
        tracker.observe(0.0, "neither")
        tracker.observe(5.0, "neither")
        tracker.observe(12.0, "neither")
        by_day = tracker.seconds_by_day(12.0)
        day = next(iter(by_day))
        self.assertAlmostEqual(by_day[day]["neither"], 12.0)
        self.assertAlmostEqual(tracker.totals_at(12.0)["neither"], 12.0)


if __name__ == "__main__":
    unittest.main()
