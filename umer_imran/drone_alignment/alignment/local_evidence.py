from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np

class CellMatchStatus(str, Enum):
    ACCEPTED = "accepted"
    NO_FEATURES = "no_features"
    AMBIGUOUS_ROADS = "ambiguous_roads"
    SEARCH_BOUNDARY = "search_boundary"
    ROAD_TREE_CONFLICT = "road_tree_conflict"
    LOW_CONFIDENCE = "low_confidence"
    SPATIAL_OUTLIER = "spatial_outlier"
    
    # New additions for cell correlation
    INSUFFICIENT_VALID_DATA = "insufficient_valid_data"
    LOW_TEXTURE = "low_texture"
    LOW_PHASE_RESPONSE = "low_phase_response"
    AMBIGUOUS_PEAK = "ambiguous_peak"
    CHANNEL_CONFLICT = "channel_conflict"

@dataclass(frozen=True)
class LocalMatchSample:
    row: int
    col: int
    center_xy: tuple[float, float]
    source_ms_xy: tuple[float, float]
    predicted_rgb_xy: tuple[float, float]
    residual_dx_dy: tuple[float, float]
    displacement_dx_dy: tuple[float, float]
    confidence: float
    road_score: float | None
    second_road_score: float | None
    tree_residual_dx_dy: tuple[float, float] | None
    status: CellMatchStatus
    evidence: str = "none"
    validation_global_score: float | None = None
    validation_local_score: float | None = None
    channel: str | None = None
    phase_response: float | None = None
    peak_sharpness: float | None = None
    valid_fraction: float | None = None
    phase_shift_dx_dy: tuple[float, float] | None = None
    rejection_reason: str | None = None

@dataclass(frozen=True)
class SparseDisplacementResult:
    samples: tuple[LocalMatchSample, ...]
    rgb_features: object | None
    ms_features: object | None
    global_translation_px: tuple[float, float]
    trusted_count: int
    coverage: float

    @property
    def accepted(self) -> tuple[LocalMatchSample, ...]:
        return tuple(item for item in self.samples if item.status == CellMatchStatus.ACCEPTED)

    @property
    def validation_summary(self) -> dict[str, float | int | None]:
        """Independent held-out structural distance scores for accepted samples."""
        scored = [item for item in self.accepted
                  if item.validation_global_score is not None and item.validation_local_score is not None]
        if not scored:
            return {"count": 0, "global_score": None, "local_score": None, "improvement": None}
        global_score = float(np.mean([item.validation_global_score for item in scored]))
        local_score = float(np.mean([item.validation_local_score for item in scored]))
        return {"count": len(scored), "global_score": global_score, "local_score": local_score,
                "improvement": global_score - local_score}
