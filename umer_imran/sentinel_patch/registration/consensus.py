"""Multi-scene consensus: accept a shift only when several independent acquisitions agree on it.

Why: for a small field (1-4 ha = 100-400 Sentinel pixels) one scene's correlation surface is often a ridge
with no clear peak, so the single-scene peak-ratio gate refuses it. Noise moves the best point along the
ridge from scene to scene, while a real geolocation offset stays put. Agreement among independent
acquisitions is therefore evidence the single scene cannot give.

Limits (stated in the report and in docs): agreement guards against noise, cloud and crop change. It cannot
guard against a *static* look-alike in the scene (the same wrong match appearing in every acquisition), so
each vote must still show a real correlation peak, away from the search border and within the plausible shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from sentinel_patch.config.schema import ConsensusConfig

ConsensusStatus = Literal["agreed", "too_few_votes", "no_agreement", "ambiguous"]


@dataclass(frozen=True)
class Vote:
    item_id: str
    acquisition_key: str  # one vote per acquisition: tiles of one datatake are not independent
    gap_days: int
    shift_m: tuple[float, float]  # (east, north) metres, the shift to ADD to the Sentinel geotransform origin
    shift_px: tuple[float, float]  # (col, row) pixels of the target content
    peak: float
    peak_ratio: float
    overlap_px: int

    def as_dict(self) -> dict:
        return {"item_id": self.item_id, "gap_days": self.gap_days, "shift_m": list(self.shift_m),
                "shift_px": list(self.shift_px), "peak": self.peak, "peak_ratio": self.peak_ratio,
                "overlap_px": self.overlap_px}


@dataclass(frozen=True)
class ConsensusResult:
    status: ConsensusStatus
    votes: int
    members: tuple[int, ...] = ()  # indices into the vote list
    shift_m: tuple[float, float] | None = None  # median of the members
    shift_px: tuple[float, float] | None = None
    spread_m: float | None = None  # largest member distance from the median
    runner_up_size: int = 0
    needed: int = 0
    reason: str = ""
    member_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def agreed(self) -> bool:
        return self.status == "agreed"

    def as_dict(self) -> dict:
        return {"status": self.status, "votes": self.votes, "members": list(self.member_ids),
                "shift_m": None if self.shift_m is None else list(self.shift_m),
                "shift_px": None if self.shift_px is None else list(self.shift_px),
                "spread_m": self.spread_m, "runner_up_size": self.runner_up_size, "needed": self.needed,
                "reason": self.reason}


def _median(points: np.ndarray) -> np.ndarray:
    return np.median(points, axis=0)


def _cluster_around(points: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    return np.flatnonzero(np.hypot(*(points - centre).T) <= radius + 1e-9)


def largest_cluster(points: np.ndarray, radius: float) -> np.ndarray:
    """Indices of the biggest group of points that sit within ``radius`` of the group's own median.

    Every point is tried as a seed; the group is re-centred on its median once so a single outlier seed
    cannot define it. Ties go to the tighter group, then to the earlier seed (deterministic).
    """
    best: tuple[int, float, np.ndarray] | None = None
    for seed in range(len(points)):
        members = _cluster_around(points, points[seed], radius)
        members = _cluster_around(points, _median(points[members]), radius)
        if members.size == 0:
            continue
        spread = float(np.hypot(*(points[members] - _median(points[members])).T).max())
        if best is None or (members.size, -spread) > (best[0], -best[1]):
            best = (int(members.size), spread, members)
    return best[2] if best else np.array([], dtype=int)


def decide(votes: Sequence[Vote], cfg: ConsensusConfig, pixel_m: float, max_shift_m: float) -> ConsensusResult:
    """Apply the agreement rules to the collected votes."""
    n = len(votes)
    needed = max(cfg.min_scenes, math.ceil(cfg.min_agree_fraction * n))
    if n < cfg.min_scenes:
        return ConsensusResult("too_few_votes", n, needed=needed,
                               reason=f"{n} usable scene(s); at least {cfg.min_scenes} independent acquisitions are needed.")
    points = np.array([vote.shift_m for vote in votes], dtype=float)
    radius = cfg.agreement_radius_px * pixel_m
    members = largest_cluster(points, radius)
    rest = np.setdiff1d(np.arange(n), members)
    runner_up = int(largest_cluster(points[rest], radius).size) if rest.size else 0
    median = _median(points[members])
    spread = float(np.hypot(*(points[members] - median).T).max())
    px = _median(np.array([votes[i].shift_px for i in members], dtype=float))
    common = dict(votes=n, members=tuple(int(i) for i in members), shift_m=(float(median[0]), float(median[1])),
                  shift_px=(float(px[0]), float(px[1])), spread_m=spread, runner_up_size=runner_up, needed=needed,
                  member_ids=tuple(votes[i].item_id for i in members))
    if members.size < needed:
        return ConsensusResult("no_agreement", reason=f"only {members.size} of {n} scenes agree within "
                               f"{radius:.0f} m; {needed} are needed.", **common)
    if runner_up >= members.size:
        return ConsensusResult("ambiguous", reason=f"a second shift is supported by {runner_up} scenes, as many as "
                               f"the first ({members.size}).", **common)
    if math.hypot(*median) > max_shift_m:
        return ConsensusResult("no_agreement", reason=f"the agreed shift ({math.hypot(*median):.1f} m) exceeds "
                               f"max_shift_m ({max_shift_m:.0f} m).", **common)
    return ConsensusResult("agreed", reason=f"{members.size} of {n} scenes agree within {radius:.0f} m.", **common)
