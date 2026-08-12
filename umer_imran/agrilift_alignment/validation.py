import cv2
import numpy as np
from .core import AlignmentError, Candidate, Correspondences, Transform

def spatial_coverage(candidate: Candidate, valid_mask: np.ndarray, grid=4) -> float:
    pts=candidate.correspondences.rgb[candidate.inlier_mask]
    if len(pts)<15: raise AlignmentError("ERR_INSUFFICIENT_MATCHES","Fewer than 15 RANSAC inliers.")
    hull=cv2.convexHull(pts.astype(np.float32))
    hull_mask=np.zeros(valid_mask.shape,np.uint8); cv2.fillConvexPoly(hull_mask,hull.astype(np.int32),1)
    coverage=(hull_mask.astype(bool)&valid_mask).sum()/max(1,valid_mask.sum())
    quadrants=set(candidate.correspondences.tiles[candidate.inlier_mask]//grid*grid + candidate.correspondences.tiles[candidate.inlier_mask]%grid)
    represented={(q//grid>=grid//2,q%grid>=grid//2) for q in quadrants}
    if coverage < .40 or len(represented)<3:
        raise AlignmentError("ERR_SPATIAL_COVERAGE_FAILED",f"coverage={coverage:.2%}, quadrants={len(represented)}")
    return float(coverage)

def split_by_tile(c: Correspondences) -> tuple[Correspondences, Correspondences]:
    """Keep whole tiles out of fitting; no same-region leakage into geometric QA."""
    hold = (c.tiles % 5) == 0
    if hold.sum() < 8 or (~hold).sum() < 20:
        raise AlignmentError("ERR_GEOMETRIC_QA_FAILED", "Insufficient independent estimation/verification tile evidence.")
    return (Correspondences(c.ms[~hold], c.rgb[~hold], c.tiles[~hold]),
            Correspondences(c.ms[hold], c.rgb[hold], c.tiles[hold]))

def held_out_improvement(transform: Transform, held_out: Correspondences) -> float:
    after=np.linalg.norm(transform.apply(held_out.ms)-held_out.rgb,axis=1)
    before=np.linalg.norm(held_out.ms-held_out.rgb,axis=1)
    if np.sqrt(np.mean(after**2)) >= np.sqrt(np.mean(before**2)):
        raise AlignmentError("ERR_NO_BASELINE_IMPROVEMENT","Candidate does not beat identity baseline.")
    if np.sqrt(np.mean(after**2)) >= 1.0:
        raise AlignmentError("ERR_GEOMETRIC_QA_FAILED","Held-out native-equivalent RMSE >= 1 registration pixel.")
    return float(np.sqrt(np.mean(after**2)))
