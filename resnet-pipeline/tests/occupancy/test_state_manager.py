import pytest

from core.occupancy.state_manager import StateConfig, WorkstationStateManager

CLASSES = ["empty", "occupied"]
CFG = StateConfig(confidence_threshold=0.8, hold_confidence_threshold=0.6, confirmation_seconds=3.0,
                  max_observation_gap_seconds=5.0, uncertain_timeout_seconds=10.0)


def feed(manager, observations):
    transitions = []
    for ts, cls, conf in observations:
        transitions += manager.observe(ts, cls, conf)
    return transitions


def confirmed(transitions):
    return [(t.previous_state, t.new_state, t.transition_timestamp, t.confirmation_timestamp, t.reason)
            for t in transitions if t.previous_state is not None]


def test_starts_empty_and_confirms_stable_occupied():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    tr = feed(m, [(t, "occupied", 0.95) for t in range(6)])
    assert tr[0].previous_state is None and tr[0].new_state == "empty" and tr[0].reason == "session_start"
    assert confirmed(tr) == [("empty", "occupied", 0, 3, "confirmed")]
    assert m.snapshot().state == "occupied" and m.snapshot().state_since == 0


def test_stable_empty():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    assert confirmed(feed(m, [(t, "empty", 0.9) for t in range(5)])) == []
    assert m.snapshot().state == "empty"


def test_single_frame_misclassification_ignored():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    obs = [(0, "occupied", .96), (1, "occupied", .95), (2, "occupied", .95), (3, "occupied", .95),
           (4, "empty", .58), (5, "occupied", .97), (6, "empty", .95), (7, "occupied", .96), (8, "occupied", .9)]
    tr = confirmed(feed(m, obs))
    assert tr == [("empty", "occupied", 0, 3, "confirmed")]
    assert m.snapshot().state == "occupied"


def test_low_confidence_returns_desk_to_empty():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    feed(m, [(t, "occupied", 0.95) for t in range(4)])
    tr = confirmed(feed(m, [(t, "empty", 0.55) for t in range(4, 20)]))
    assert [row[4] for row in tr] == ["low_confidence"]
    assert tr[-1][:3] == ("occupied", "empty", 3) and tr[-1][4] == "low_confidence"


def test_hold_threshold_keeps_state_alive():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    feed(m, [(t, "occupied", 0.95) for t in range(4)])
    assert confirmed(feed(m, [(t, "occupied", 0.65) for t in range(4, 30)])) == []
    assert m.snapshot().state == "occupied"


def test_sustained_transition_backdated_to_first_evidence():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    feed(m, [(t, "occupied", 0.95) for t in range(5)])
    tr = confirmed(feed(m, [(t, "empty", 0.9) for t in range(5, 10)]))
    assert tr == [("occupied", "empty", 5, 8, "confirmed")]


def test_neutral_low_confidence_does_not_reset_candidate():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    feed(m, [(t, "occupied", 0.95) for t in range(5)])
    tr = confirmed(feed(m, [(5, "empty", .9), (6, "empty", .5), (7, "empty", .9), (8, "empty", .9)]))
    assert tr == [("occupied", "empty", 5, 8, "confirmed")]


def test_supporting_observation_cancels_candidate():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    feed(m, [(t, "occupied", 0.95) for t in range(5)])
    tr = confirmed(feed(m, [(5, "empty", .9), (6, "empty", .9), (7, "occupied", .9), (8, "empty", .9),
                            (9, "empty", .9), (10, "empty", .9)]))
    assert tr == []
    assert confirmed(m.observe(11, "empty", .9)) == [("occupied", "empty", 8, 11, "confirmed")]


def test_long_observation_gap_returns_to_empty_from_last_observation():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    feed(m, [(t, "occupied", 0.95) for t in range(5)])
    tr = confirmed(m.observe(20, "occupied", 0.95))
    assert tr == [("occupied", "empty", 4, 20, "observation_gap")]
    tr = confirmed(feed(m, [(t, "occupied", 0.95) for t in range(21, 24)]))
    assert tr == [("empty", "occupied", 20, 23, "confirmed")]


def test_missing_frames_within_gap_tolerated():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    feed(m, [(t, "occupied", 0.95) for t in range(4)])
    assert m.observe_missing(5) == []
    assert confirmed(m.observe(7, "occupied", 0.95)) == []
    assert m.observe_missing(13)[0].new_state == "empty"


def test_outage_marks_empty():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    feed(m, [(t, "occupied", 0.95) for t in range(4)])
    tr = m.mark_unavailable(6.5)
    assert confirmed(tr) == [("occupied", "empty", 3, 6.5, "source_outage")]
    assert m.mark_unavailable(7) == []


def test_out_of_order_and_duplicate_ignored():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    feed(m, [(t, "occupied", 0.95) for t in range(4)])
    assert m.observe(2, "empty", 0.99) == [] and m.observe(3, "empty", 0.99) == []
    assert m.dropped_out_of_order == 2


def test_independent_workstations():
    a = WorkstationStateManager("W01", CLASSES, CFG)
    b = WorkstationStateManager("W02", CLASSES, CFG)
    for t in range(5):
        a.observe(t, "occupied", 0.95)
        b.observe(t, "empty", 0.95)
    assert a.snapshot().state == "occupied" and b.snapshot().state == "empty"


def test_unknown_class_and_invalid_config():
    m = WorkstationStateManager("W01", CLASSES, CFG)
    with pytest.raises(ValueError):
        m.observe(0, "standing_near", 0.9)
    with pytest.raises(ValueError):
        WorkstationStateManager("W01", CLASSES, StateConfig(confidence_threshold=0.5, hold_confidence_threshold=0.7))
