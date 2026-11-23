"""Unit tests for multi-cam scalability helpers (no GPU / models required)."""

from __future__ import annotations

import time

from core.scalability import CoalescingWorker, OverloadGuard, ScalabilityConfig, phase_offset_seconds
from core.config import load_settings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_phase_offset_is_stable_and_bounded():
    period = 1.0
    a = phase_offset_seconds("1", period)
    b = phase_offset_seconds("2", period)
    assert 0.0 <= a < period
    assert 0.0 <= b < period
    assert a == phase_offset_seconds("1", period)
    assert a != b


def test_coalescing_worker_keeps_newest_only():
    seen: list[int] = []
    ready = []

    def handler(job: int) -> None:
        time.sleep(0.05)
        seen.append(job)
        ready.append(job)

    worker = CoalescingWorker("test-coalesce", handler)
    try:
        worker.submit(1)
        time.sleep(0.01)
        worker.submit(2)
        worker.submit(3)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and 3 not in ready:
            time.sleep(0.01)
        assert 3 in seen
        assert 2 not in seen  # superseded while 1 was running or before start
    finally:
        worker.close()


def test_overload_guard_trips_and_recovers():
    guard = OverloadGuard(budget_seconds=0.1, consecutive=3)
    assert not guard.should_shed()
    guard.record(0.2)
    guard.record(0.2)
    assert not guard.should_shed()
    guard.record(0.2)
    assert guard.should_shed()
    guard.record(0.01)
    assert not guard.should_shed()


def test_settings_load_includes_scalability_defaults():
    settings = load_settings(ROOT / "config" / "settings.yaml", {
        "pipeline": {"mode": "occupancy"},
        "camera": {"source": str(ROOT / "testing" / "input.mp4")},
        "visualization": {"show_window": False, "save_video": False},
    })
    assert isinstance(settings.scalability, ScalabilityConfig)
    assert settings.scalability.enabled is True
    assert settings.scalability.async_videomae is True
