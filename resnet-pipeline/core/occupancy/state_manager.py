"""Per-workstation temporal stabilization of classifier predictions.

State machine rules (all time-based, never frame-count based):

* A desk is ``empty`` until ``occupied`` is confirmed. There is no unknown workstation state.
* A new state is confirmed only after observations of that class with confidence
  >= ``confidence_threshold`` span at least ``confirmation_seconds``, with no
  qualifying-observation gap longer than ``max_observation_gap_seconds`` and no
  contradicting evidence in between.
* Hysteresis: while in a state, observations of that same class with confidence
  >= ``hold_confidence_threshold`` keep it alive and cancel any pending candidate.
* Low-confidence observations are neutral: they neither confirm nor cancel. If an
  occupied desk receives no supporting evidence for ``uncertain_timeout_seconds``,
  it becomes ``empty``.
* A gap longer than ``max_observation_gap_seconds``, a missing crop, or a source
  outage returns an occupied desk to ``empty`` from its last supported observation.

Transition timestamp policy: a confirmed transition is backdated to the first
observation of the sustained new state (``evidence_start``); ``confirmation_timestamp``
records when it was actually confirmed. A return to ``empty`` is backdated to the
last observation that supported occupied, so unsupported time is empty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)

# Kept so older imports still resolve. Workstation occupancy does not use this state:
# a desk that is not occupied is empty.
UNKNOWN = "unknown"
EMPTY = "empty"


@dataclass(frozen=True)
class StateConfig:
    confidence_threshold: float = 0.80
    hold_confidence_threshold: float = 0.60
    confirmation_seconds: float = 3.0
    max_observation_gap_seconds: float = 5.0
    uncertain_timeout_seconds: float = 10.0

    def validate(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("state.confidence_threshold must be in (0, 1]")
        if not 0.0 < self.hold_confidence_threshold <= self.confidence_threshold:
            raise ValueError("state.hold_confidence_threshold must be in (0, confidence_threshold]")
        if self.confirmation_seconds < 0:
            raise ValueError("state.confirmation_seconds must be >= 0")
        if self.max_observation_gap_seconds <= 0:
            raise ValueError("state.max_observation_gap_seconds must be > 0")
        if self.uncertain_timeout_seconds <= 0:
            raise ValueError("state.uncertain_timeout_seconds must be > 0")


@dataclass(frozen=True)
class StateTransition:
    workstation_id: str
    previous_state: str | None
    new_state: str
    transition_timestamp: float
    confirmation_timestamp: float
    reason: str
    evidence_start: float | None = None
    evidence_end: float | None = None


@dataclass(frozen=True)
class StateSnapshot:
    workstation_id: str
    state: str
    state_since: float | None
    last_observation: float | None
    last_predicted_class: str | None
    last_confidence: float | None
    candidate_state: str | None
    candidate_since: float | None


class WorkstationStateManager:
    def __init__(self, workstation_id: str, class_names: Sequence[str], config: StateConfig) -> None:
        config.validate()
        if UNKNOWN in class_names:
            raise ValueError(f"'{UNKNOWN}' is reserved for the monitoring state")
        self.workstation_id = workstation_id
        self.class_names = list(class_names)
        self.config = config
        self.state: str | None = None
        self.state_since: float | None = None
        self.last_observation: float | None = None
        self.last_support: float | None = None
        self.last_predicted_class: str | None = None
        self.last_confidence: float | None = None
        self._candidate: str | None = None
        self._candidate_first: float | None = None
        self._candidate_last: float | None = None
        self.dropped_out_of_order = 0

    def _transition(self, new_state: str, at: float, confirmed_at: float, reason: str,
                    evidence: tuple[float, float] | None = None) -> StateTransition:
        at = max(at, self.state_since) if self.state_since is not None else at
        transition = StateTransition(
            self.workstation_id, self.state, new_state, at, confirmed_at, reason,
            evidence[0] if evidence else None, evidence[1] if evidence else None,
        )
        self.state, self.state_since = new_state, at
        if new_state != EMPTY:
            self.last_support = confirmed_at
        logger.debug("%s: %s -> %s at %.3f (%s)", self.workstation_id, transition.previous_state,
                     new_state, at, reason)
        return transition

    def _reset_candidate(self) -> None:
        self._candidate = self._candidate_first = self._candidate_last = None

    def _accept_timestamp(self, ts: float) -> bool:
        if self.last_observation is not None and ts <= self.last_observation:
            self.dropped_out_of_order += 1
            logger.warning("%s: ignoring out-of-order/duplicate observation at %.3f (last %.3f)",
                           self.workstation_id, ts, self.last_observation)
            return False
        return True

    def _open_empty(self, ts: float) -> StateTransition:
        self.state_since = ts
        self.state = EMPTY
        transition = StateTransition(self.workstation_id, None, EMPTY, ts, ts, "session_start")
        return transition

    def _start_or_gap(self, ts: float) -> list[StateTransition]:
        if self.state is None:
            return [self._open_empty(ts)]
        if self.last_observation is not None and ts - self.last_observation > self.config.max_observation_gap_seconds:
            self._reset_candidate()
            if self.state != EMPTY:
                return [self._transition(EMPTY, self.last_observation, ts, "observation_gap")]
        return []

    def observe(self, ts: float, predicted_class: str, confidence: float) -> list[StateTransition]:
        """Feed one prediction; returns any transitions it caused (in time order)."""
        if predicted_class not in self.class_names:
            raise ValueError(f"{self.workstation_id}: unknown class {predicted_class!r}")
        if not self._accept_timestamp(ts):
            return []
        cfg = self.config
        transitions = self._start_or_gap(ts)
        self.last_predicted_class, self.last_confidence = predicted_class, confidence

        if predicted_class == self.state and confidence >= cfg.hold_confidence_threshold:
            if self.state != EMPTY:
                self.last_support = ts
            self._reset_candidate()
        elif confidence >= cfg.confidence_threshold and predicted_class != self.state:
            if (self._candidate == predicted_class and self._candidate_last is not None
                    and ts - self._candidate_last <= cfg.max_observation_gap_seconds):
                self._candidate_last = ts
            else:
                self._candidate, self._candidate_first, self._candidate_last = predicted_class, ts, ts
            assert self._candidate_first is not None
            if ts - self._candidate_first >= cfg.confirmation_seconds:
                transitions.append(self._transition(
                    predicted_class, self._candidate_first, ts, "confirmed", (self._candidate_first, ts)))
                self._reset_candidate()

        if (self.state != EMPTY and self.last_support is not None
                and ts - self.last_support > cfg.uncertain_timeout_seconds):
            transitions.append(self._transition(EMPTY, self.last_support, ts, "low_confidence"))
        self.last_observation = ts
        return transitions

    def observe_missing(self, ts: float) -> list[StateTransition]:
        """No usable prediction at ``ts`` (e.g. classifier skipped this crop). The desk stays empty."""
        if self.last_observation is not None and ts <= self.last_observation:
            return []
        if self.state is None:
            self.last_observation = ts
            return [self._open_empty(ts)]
        if ts - self.last_observation > self.config.max_observation_gap_seconds and self.state != EMPTY:
            self._reset_candidate()
            return [self._transition(EMPTY, self.last_observation, ts, "observation_gap")]
        return []

    def mark_unavailable(self, detected_at: float | None = None, reason: str = "source_outage") -> list[StateTransition]:
        """Source lost: an occupied desk becomes empty from the last observation."""
        self._reset_candidate()
        if self.state is None or self.state == EMPTY or self.last_observation is None:
            return []
        confirmed_at = detected_at if detected_at is not None else self.last_observation
        return [self._transition(EMPTY, self.last_observation, max(confirmed_at, self.last_observation), reason)]

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            self.workstation_id, self.state or EMPTY, self.state_since, self.last_observation,
            self.last_predicted_class, self.last_confidence, self._candidate, self._candidate_first,
        )
