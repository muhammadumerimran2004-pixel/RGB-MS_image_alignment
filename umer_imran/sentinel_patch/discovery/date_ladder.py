"""Date gap between the flight and a Sentinel acquisition, and the tier it falls in (blueprint 6.2.1)."""

from __future__ import annotations

from datetime import date, datetime

from sentinel_patch.config.schema import DiscoveryConfig


def date_gap_days(acquired_utc: datetime, flight_date: date) -> int:
    """Signed gap in whole days: positive when the satellite passed AFTER the flight."""
    return (acquired_utc.date() - flight_date).days


def date_tier(gap_days: int, cfg: DiscoveryConfig) -> str:
    gap = abs(gap_days)
    if gap <= cfg.preferred_gap_days:
        return "same_period"
    if gap <= cfg.acceptable_gap_days:
        return "near"
    return "stale"
