"""V2 orchestrator: phase is proposal-only; no output is published without tiled geometric QA."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from drone_alignment.config.schema import AlignmentConfig
from drone_alignment.io.validators import validate_inputs
from drone_alignment.alignment.coarse import coarse_align
from .core import AlignmentError, Status
from .preprocessing import erode_valid_mask, sobel_map, log_map, orientation_map, gabor_orientation_map, gabor_energy_map
from .registration import phase_proposal, tiled_sift, tiled_local_displacements, tiled_mim_descriptors, fit_affine, fit_similarity
from .validation import spatial_coverage, held_out_improvement, split_by_tile

@dataclass(frozen=True)
class V2Result:
    status: Status
    transform: object | None
    held_out_rmse: float | None
    coverage: float | None
    error_code: str | None
    diagnostics: tuple[str, ...] = ()

def align(rgb_path: Path, ms_path: Path, config: AlignmentConfig | None = None) -> V2Result:
    """Run registration evidence only. Native output is intentionally not available until Phase 9 is implemented."""
    cfg=config or AlignmentConfig()
    rgb_meta,ms_meta=validate_inputs(rgb_path,ms_path,cfg)
    coarse=coarse_align(rgb_meta,ms_meta,cfg)
    mask=erode_valid_mask(coarse.rgb_valid_mask & coarse.ms_valid_mask,8)
    last_error="ERR_INSUFFICIENT_MATCHES"
    diagnostics=[]
    for channel in cfg.registration_channel_priority:
      name=channel.value
      for mapper in (sobel_map,log_map,orientation_map,gabor_orientation_map,gabor_energy_map):
       try:
        rgb=mapper(coarse.registration_bands_rgb[name],mask)
        ms=mapper(coarse.registration_bands_ms[name],mask)
        proposal=phase_proposal(rgb,ms,mask,max_shift_px=cfg.transform.max_translation_m/coarse.registration_gsd)
        # A locally plausible template candidate is not sufficient reason to skip
        # descriptor evidence.  Each evidence source must independently clear the
        # same held-out geometric gates; neither can override the other.
        matchers=[
            ("local", lambda: tiled_local_displacements(rgb,ms,mask,proposal,grid=7,radius=32)),
            ("sift", lambda: tiled_sift(rgb,ms,mask,proposal,grid=7)),
            ("mim", lambda: tiled_mim_descriptors(rgb,ms,mask,proposal,grid=7)),
        ]
        for matcher_name,matcher in matchers:
          try:
            correspondences=matcher()
            estimation,held_out=split_by_tile(correspondences)
            candidates=[]
            for fitter in (fit_similarity,fit_affine):
                candidate=fitter(estimation)
                coverage=spatial_coverage(candidate,mask,grid=7)
                rmse=held_out_improvement(candidate.transform,held_out)
                candidates.append((rmse,coverage,candidate))
            rmse,coverage,candidate=min(candidates,key=lambda x:x[0])
            diagnostics.append(f"{name}/{mapper.__name__}/{matcher_name}:PASS")
            return V2Result(Status.PASS,candidate.transform,rmse,coverage,None,tuple(diagnostics))
          except AlignmentError as exc:
            last_error=exc.code
            diagnostics.append(f"{name}/{mapper.__name__}/{matcher_name}:{exc.code}")
       except AlignmentError as exc:
        last_error=exc.code
        diagnostics.append(f"{name}/{mapper.__name__}/proposal:{exc.code}")
    return V2Result(Status.FAIL,None,None,None,last_error,tuple(diagnostics))
