"""Identity activity/verification lifecycle helpers."""

from __future__ import annotations

from backend.types import IdentityActivityStatus, VerificationStatus


def next_activity_status(
    current: IdentityActivityStatus,
    *,
    observed: bool,
    seconds_since_seen: float,
    temporarily_lost_after: float = 5.0,
    out_of_view_after: float = 30.0,
    inactive_after: float = 3600.0,
) -> IdentityActivityStatus:
    if current == IdentityActivityStatus.ARCHIVED:
        return current
    if current == IdentityActivityStatus.PROVISIONAL:
        return IdentityActivityStatus.ACTIVE if observed else current

    if observed:
        return IdentityActivityStatus.ACTIVE

    if seconds_since_seen < temporarily_lost_after:
        return IdentityActivityStatus.TEMPORARILY_LOST
    if seconds_since_seen < out_of_view_after:
        return IdentityActivityStatus.OUT_OF_VIEW
    if seconds_since_seen < inactive_after:
        return IdentityActivityStatus.INACTIVE
    return IdentityActivityStatus.INACTIVE


def can_promote_verification(
    current: VerificationStatus,
    target: VerificationStatus,
) -> bool:
    order = [
        VerificationStatus.UNVERIFIED,
        VerificationStatus.PROVISIONAL,
        VerificationStatus.VERIFIED,
        VerificationStatus.MANUAL,
    ]
    return order.index(target) >= order.index(current)
