"""Walk the date ladder and pick a scene (blueprint 6.2.1). Provider-agnostic: works on ``SceneCandidate``s.

Dates are visited nearest first (gap 0, 1, 2, ...). On one date every candidate is measured and the one
with the lowest masked fraction wins; a tie goes to the scene acquired after the flight, then to
provider order. The first date with a scene at or below the cut-off ends the search. If no date passes,
the clearest scene at or below the fallback limit is returned as "partial".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Generic, TypeVar

from sentinel_patch.config.schema import DiscoveryConfig
from sentinel_patch.discovery.date_ladder import date_gap_days
from sentinel_patch.discovery.provider import SceneCandidate
from sentinel_patch.errors import SentinelPatchError

P = TypeVar("P")  # whatever ``prepare`` returns; it must expose ``.stats`` (masked_fraction, clear_pixels)

CLEAN, PARTIAL = "clean", "partial"


@dataclass
class Selection(Generic[P]):
    prepared: P | None
    quality: str | None  # "clean" | "partial" | None
    records: list[dict]


def select_scene(
    candidates: list[SceneCandidate], prepare: Callable[[SceneCandidate], P], flight_date: date, cfg: DiscoveryConfig,
) -> Selection[P]:
    gaps = {id(c): date_gap_days(c.datetime_utc, flight_date) for c in candidates}
    rungs: dict[int, list[SceneCandidate]] = {}
    for candidate in candidates:
        rungs.setdefault(abs(gaps[id(candidate)]), []).append(candidate)

    records: list[dict] = []
    best_partial: tuple[float, P, dict] | None = None
    evaluated = 0

    def usable(stats) -> bool:
        return stats.clear_pixels >= cfg.min_clear_pixels

    for gap in sorted(rungs):
        passing: list[tuple[tuple, P, dict]] = []
        group = sorted(rungs[gap], key=lambda c: 0 if gaps[id(c)] >= 0 else 1)  # stable: provider order inside
        for candidate in group:
            record = {"provider": candidate.provider, "item_id": candidate.item_id,
                      "datetime_utc": candidate.datetime_utc, "date_gap_days": gaps[id(candidate)]}
            records.append(record)
            if evaluated >= cfg.max_candidates_evaluated:
                record.update(accepted=False, reason_code="EVALUATION_LIMIT")
                continue
            evaluated += 1
            try:
                prepared = prepare(candidate)
            except SentinelPatchError as exc:
                record.update(accepted=False, reason_code=exc.code, message=str(exc))
                continue
            stats = prepared.stats
            record.update(aoi_masked_fraction=stats.masked_fraction, aoi_clear_pixels=stats.clear_pixels)
            if stats.masked_fraction > cfg.max_aoi_masked_fraction:
                record.update(accepted=False, reason_code="AOI_TOO_CLOUDY")
                if usable(stats) and stats.masked_fraction <= cfg.fallback_max_aoi_masked_fraction and (
                    best_partial is None or stats.masked_fraction < best_partial[0]
                ):  # strict '<': on equal fractions the nearer date, visited first, stays
                    best_partial = (stats.masked_fraction, prepared, record)
                continue
            if not usable(stats):
                record.update(accepted=False, reason_code="TOO_FEW_CLEAR_PIXELS")
                continue
            passing.append(((stats.masked_fraction, len(passing)), prepared, record))
            record.update(accepted=False, reason_code="CLEARER_SCENE_ON_SAME_DATE")
        if passing:
            _, winner, winner_record = min(passing, key=lambda entry: entry[0])
            winner_record.update(accepted=True, reason_code=None)
            return Selection(winner, CLEAN, records)

    if best_partial is not None:
        _, prepared, record = best_partial
        record.update(accepted=True, reason_code="PARTIAL_ACCEPTED")
        return Selection(prepared, PARTIAL, records)
    return Selection(None, None, records)
