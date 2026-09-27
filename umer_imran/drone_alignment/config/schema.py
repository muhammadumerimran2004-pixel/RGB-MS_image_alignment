from enum import Enum
import math
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class ResolutionMode(str, Enum):
    MS_NATIVE = "ms"
    RGB_NATIVE = "rgb"


class AlignmentMode(str, Enum):
    MANUAL = "manual"
    AUTOMATED = "automated"
    ROAD_GRID = "road_grid"
    LOCAL_MESH = "local_mesh"
    LOCAL_CORRELATION = "local_correlation"


class DetectorType(str, Enum):
    ORB = "orb"
    SIFT = "sift"


class RepresentationType(str, Enum):
    """Single-channel views used to make cross-spectral structure comparable."""
    NORMALIZED = "normalized"
    GABOR_ENERGY = "gabor_energy"


class TransformType(str, Enum):
    AFFINE = "affine"
    HOMOGRAPHY = "homography"


class InterpolationType(str, Enum):
    BILINEAR = "bilinear"
    NEAREST = "nearest"


class RegistrationChannel(str, Enum):
    GREEN = "green"
    RED = "red"


class ManualCoordinateMode(str, Enum):
    """Coordinate system used by manually entered GCPs."""
    MAP = "map"
    PIXEL = "pixel"


class CoarseAlignmentConfig(BaseModel):
    target_gsd_m: Optional[float] = Field(
        None,
        description="Target GSD in meters. None = preserve reference native resolution."
    )
    overlap_margin_m: float = Field(
        10.0,
        ge=0.0,
        description="Margin in metres added around the pre-alignment RGB/MS overlap when building the "
                    "working and output grid, clamped to the RGB reference's extent. It must exceed the "
                    "real RGB-to-MS shift (GPS-only ODM runs have shown 2-6 m), otherwise a correctly "
                    "aligned MS is clipped at the grid edge."
    )
    registration_max_dimension: int = Field(
        3072, ge=512, le=8192,
        description="Maximum side length of the low-resolution grid used only for registration."
    )


class FeatureDetectionConfig(BaseModel):
    detector: DetectorType = Field(
        DetectorType.ORB,
        description="Feature detector algorithm (ORB by default, SIFT as fallback)."
    )
    max_keypoints: int = Field(
        10000,
        ge=500,
        le=50000,
        description="Max keypoints to detect on downsampled red image."
    )
    downsample_max_dim: int = Field(
        4000,
        ge=1000,
        le=16000,
        description="Max dimension for downsampled image during keypoint detection."
    )
    lowe_ratio: float = Field(
        0.75,
        gt=0.0,
        lt=1.0,
        description="Lowe's ratio test threshold for descriptor matching."
    )
    min_good_matches: int = Field(
        20,
        ge=4,
        description="Minimum matched keypoint count after Lowe ratio test."
    )
    ransac_reproj_threshold: float = Field(
        3.0,
        gt=0.0,
        description="RANSAC reprojection error threshold in downsampled pixels."
    )
    ransac_max_iters: int = Field(
        5000,
        ge=100,
        description="Max RANSAC iterations."
    )
    ransac_confidence: float = Field(
        0.999,
        gt=0.0,
        le=1.0,
        description="RANSAC confidence probability."
    )
    apply_clahe: bool = Field(True, description="Apply CLAHE after valid-pixel normalization.")
    clahe_clip_limit: float = Field(2.0, gt=0.0, le=20.0)
    clahe_grid_size: int = Field(8, ge=2, le=64)
    percentile_low: float = Field(2.0, ge=0.0, lt=100.0)
    percentile_high: float = Field(98.0, gt=0.0, le=100.0)
    min_valid_pixels: int = Field(500, ge=16)


class ArosicsBandPair(BaseModel):
    """One ordered, semantically comparable reference/target band pair for AROSICS.

    Band indices are one-indexed GeoTIFF band numbers.  The reference raster
    may be a multispectral image even though the CLI continues to call it the
    RGB reference.
    """

    name: str = Field(
        ..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
        description="Human-readable pair label written to the alignment report.",
    )
    reference_band: int = Field(..., ge=1, description="1-indexed matching band in the reference raster.")
    target_band: int = Field(..., ge=1, description="1-indexed matching band in the target raster.")


class ArosicsGlobalCandidateConfig(BaseModel):
    """COREG (global): one feature-free candidate inside global estimation.

    This is tried only if every feature-based candidate (ORB/SIFT) is
    rejected. Like those, its candidate undergoes footprint and residual
    validation before publication - it is never trusted on its own.
    """
    enabled: bool = Field(True, description="Try AROSICS global COREG as a feature-free candidate.")
    window_size: tuple[int, int] = Field((256, 256), description="Matching window size in pixels.")
    max_shift_px: int = Field(
        50, gt=1, le=500,
        description="Max allowed shift in registration pixels. Large offsets are this stage's job; "
                    "COREG_LOCAL (ArosicsLocalConfig) refines the verified result with a much smaller search.",
    )
    min_reliability: float = Field(
        60.0, ge=0.0, le=100.0,
        description="Minimum shift_reliability percentage (0-100). AROSICS' own default is 60.",
    )


class ArosicsLocalConfig(BaseModel):
    """COREG_LOCAL refinement of the already-verified global result.

    AROSICS performs the warp itself (its own DESHIFTER, or an equivalent
    streaming GDAL thin-plate-spline warp for large rasters - both apply the
    exact same GCP-based transform). This config exists to keep AROSICS
    close to its own documented defaults (small max_shift, min_reliability
    60, its own tie-point filtering) and to add the independent holdout
    verification the AROSICS docs themselves recommend, which this pipeline
    automates rather than leaving to manual inspection.
    """
    enabled: bool = Field(True, description="Refine the verified global result with AROSICS COREG_LOCAL.")

    # --- tie-point grid ---
    target_points_per_axis: int = Field(
        16, ge=4, le=128,
        description="Used to derive grid_res_px when grid_res_px is not set explicitly: "
                    "grid_res_px = ceil(max(staged target W, H) / target_points_per_axis).",
    )
    grid_res_px: Optional[int] = Field(None, ge=16, description="Explicit tie-point grid resolution override, in staged-target pixels.")
    window_size: tuple[int, int] = Field((256, 256), description="Matching window size in reference pixels.")
    max_shift_m: Optional[float] = Field(
        None, gt=0.0, le=5.0,
        description="Max local search radius in metres. None = derive automatically from the verified "
                    "global result's residual RMSE (clamped to [0.10, 0.30] m), respecting periodic_texture_period_m.",
    )
    max_iter: int = Field(5, ge=1, le=20, description="Max COREG_LOCAL matching iterations per tie point.")

    # --- AROSICS tie-point filtering (kept close to AROSICS' own defaults) ---
    min_reliability: float = Field(60.0, ge=0.0, le=100.0, description="AROSICS' own default is 60.")
    tie_point_filter_level: Literal[0, 1, 2, 3] = Field(
        3, description="0=none, 1=+reliability, 2=+SSIM, 3=+RANSAC (AROSICS' own default and maximum).",
    )
    rs_max_outlier: float = Field(10.0, gt=0.0, le=50.0, description="RANSAC: expected outlier percentage.")
    rs_tolerance: float = Field(2.5, gt=0.0, le=20.0, description="RANSAC: tolerance around rs_max_outlier.")

    # --- our additional filter, beyond AROSICS' own ---
    neighbour_filter: bool = Field(
        True,
        description="Flag tie points whose shift disagrees with their nearest neighbours' median shift. "
                    "AROSICS' own RANSAC filter fits one global model and can tolerate a locally "
                    "inconsistent (e.g. one-crop-row-aliased) vector that this catches.",
    )
    neighbour_k: int = Field(6, ge=3, le=16)
    max_neighbour_deviation_px: float = Field(2.0, gt=0.0, description="In target (MS) pixels.")

    # --- pre-warp gates (evaluated on tie points before any GCP warp is attempted) ---
    min_valid_tie_points: int = Field(12, ge=6)
    coverage_grid: int = Field(4, ge=2, le=16, description="Grid density for the occupancy/hull-coverage check.")
    min_cell_occupancy: float = Field(0.5, gt=0.0, le=1.0)
    min_hull_coverage: float = Field(0.30, gt=0.0, le=1.0)

    # --- holdout split and verification (never trusts AROSICS' own success flag alone) ---
    holdout_fraction: float = Field(0.20, gt=0.0, lt=0.5)
    min_holdout_points: int = Field(4, ge=3)
    no_gain_floor_px: float = Field(0.5, ge=0.0, description="In target (MS) pixels.")
    max_holdout_ratio: float = Field(0.70, gt=0.0, le=1.0, description="median(after) <= ratio * median(before) required.")
    min_holdout_win_fraction: float = Field(0.60, ge=0.0, le=1.0)
    max_holdout_after_px: float = Field(2.0, gt=0.0, description="In target (MS) pixels.")
    full_grid_verification: bool = Field(True, description="Re-run COREG_LOCAL on the published output as a sanity check.")
    max_verification_p90_px: float = Field(1.5, gt=0.0)

    # --- warp ---
    warp_engine: Literal["auto", "deshifter", "gdal_tps"] = "auto"
    max_in_memory_warp_gb: float = Field(4.0, gt=0.0, description="Above this estimate, prefer the streaming gdal_tps engine.")
    gdal_warp_memory_mb: int = Field(512, ge=64)
    resampling: Literal["bilinear", "cubic", "nearest"] = "bilinear"

    # --- domain guard against repeating crop-row texture ---
    periodic_texture_period_m: Optional[float] = Field(
        None, gt=0.0,
        description="E.g. crop row spacing in metres. When set, the local search radius is capped "
                    "below half this period so a phase-correlation peak cannot lock onto the wrong row.",
    )

    export_tie_points: bool = Field(True, description="Write the tie-point table alongside the aligned output.")
    cpus: Optional[int] = Field(None, ge=1)

    @model_validator(mode="after")
    def _validate_local_config(self) -> "ArosicsLocalConfig":
        wx, wy = self.window_size
        if wx < 32 or wy < 32 or wx % 2 or wy % 2:
            raise ValueError(f"window_size ({wx}, {wy}) must have even elements >= 32.")
        fit_floor = self.min_valid_tie_points - math.ceil(self.min_valid_tie_points * self.holdout_fraction)
        if fit_floor < 6:
            raise ValueError(
                f"min_valid_tie_points ({self.min_valid_tie_points}) and holdout_fraction "
                f"({self.holdout_fraction}) leave only {fit_floor} fit points; at least 6 are required "
                "for a non-degenerate thin-plate-spline warp."
            )
        return self


class ArosicsConfig(BaseModel):
    """Optional AROSICS geospatial co-registration.

    The verified global result (ORB/SIFT/phase, or AROSICS' own global
    COREG as a last resort - see ``global_candidate``) is estimated first.
    AROSICS' local COREG_LOCAL then refines that result and performs the
    warp itself, subject to the gates in ``local``. This mirrors AROSICS'
    own coarse-to-fine design: it corrects small residual misregistration,
    not the original large offset.
    """
    enabled: bool = Field(True, description="Master switch for every AROSICS candidate (global and local).")
    band_pairs: list[ArosicsBandPair] = Field(
        default_factory=list,
        description=(
            "Ordered AROSICS reference/target band pairs. If supplied, these replace the legacy "
            "green-green/red-red candidates; the first pair is the primary AROSICS benchmark."
        ),
    )
    global_candidate: ArosicsGlobalCandidateConfig = Field(default_factory=ArosicsGlobalCandidateConfig)
    local: ArosicsLocalConfig = Field(default_factory=ArosicsLocalConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data):
        """Migrate the pre-split flat AROSICS config keys with a warning.

        Older config files/callers may still set ``window_size``,
        ``max_shift_px``, ``min_reliability``, ``grid_res`` or
        ``min_local_tie_points`` directly on ``arosics:``. These moved to
        ``arosics.global_candidate`` and/or ``arosics.local``.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        legacy_global: dict = {}
        legacy_local: dict = {}
        import warnings

        if "window_size" in data:
            value = data.pop("window_size")
            legacy_global["window_size"] = value
            legacy_local["window_size"] = value
            warnings.warn(
                "arosics.window_size is deprecated; set arosics.global_candidate.window_size and/or "
                "arosics.local.window_size instead.", DeprecationWarning, stacklevel=2,
            )
        if "max_shift_px" in data:
            legacy_global["max_shift_px"] = data.pop("max_shift_px")
            warnings.warn(
                "arosics.max_shift_px is deprecated; set arosics.global_candidate.max_shift_px instead.",
                DeprecationWarning, stacklevel=2,
            )
        if "min_reliability" in data:
            value = data.pop("min_reliability")
            legacy_global["min_reliability"] = value
            legacy_local["min_reliability"] = value
            warnings.warn(
                "arosics.min_reliability is deprecated; set arosics.global_candidate.min_reliability "
                "and/or arosics.local.min_reliability instead.", DeprecationWarning, stacklevel=2,
            )
        if "grid_res" in data:
            legacy_local["grid_res_px"] = int(data.pop("grid_res"))
            warnings.warn(
                "arosics.grid_res is deprecated; set arosics.local.grid_res_px instead.",
                DeprecationWarning, stacklevel=2,
            )
        if "min_local_tie_points" in data:
            legacy_local["min_valid_tie_points"] = data.pop("min_local_tie_points")
            warnings.warn(
                "arosics.min_local_tie_points is deprecated; set arosics.local.min_valid_tie_points instead.",
                DeprecationWarning, stacklevel=2,
            )
        if "ignore_errors" in data:
            data.pop("ignore_errors")
            warnings.warn(
                "arosics.ignore_errors is deprecated and has been dropped: AROSICS' internal "
                "ignore_errors=True is always used for local refinement.", DeprecationWarning, stacklevel=2,
            )

        def _merge(key: str, legacy: dict) -> None:
            if not legacy:
                return
            existing = data.get(key) or {}
            if isinstance(existing, BaseModel):
                existing = existing.model_dump()
            data[key] = {**legacy, **existing}

        _merge("global_candidate", legacy_global)
        _merge("local", legacy_local)
        return data



class TransformValidationConfig(BaseModel):
    transform_type: TransformType = Field(
        TransformType.AFFINE,
        description="Transform model type (AFFINE default, HOMOGRAPHY fallback)."
    )
    max_translation_m: float = Field(
        50.0,
        gt=0.0,
        description="Max allowed translation in meters."
    )
    max_rotation_deg: float = Field(
        5.0,
        gt=0.0,
        le=45.0,
        description="Max allowed rotation in degrees."
    )
    max_scale_deviation: float = Field(
        0.1,
        gt=0.0,
        le=0.5,
        description="Max allowed scale deviation from 1.0 (e.g. 0.1 allows 0.9..1.1)."
    )
    min_inlier_ratio: float = Field(
        0.3,
        gt=0.0,
        le=1.0,
        description="Min RANSAC inlier fraction."
    )
    min_phase_correlation_response: float = Field(0.15, ge=0.0, le=1.0)
    min_phase_verification_correlation: float = Field(0.30, ge=-1.0, le=1.0)
    min_retained_valid_ratio: float = Field(
        0.80,
        gt=0.0,
        le=1.0,
        description="Minimum fraction of the MS footprint (inside the output frame) that the correction "
                    "keeps on the grid. The only footprint gate: RGB/MS coverage overlap is reported, "
                    "not gated, because the two flights routinely cover different ground.",
    )
    min_inlier_coverage_ratio: float = Field(0.05, ge=0.0, le=1.0)


class ManualAlignmentConfig(BaseModel):
    """Manual ground-control-point alignment: model selection and QA gates.

    AROSICS is not involved in manual mode; the pipeline's own tiled warper
    performs the resample so leave-one-out and Jacobian QA can run against
    the exact field that gets applied.
    """
    model: Literal["auto", "translation", "similarity", "affine", "tps"] = "auto"
    min_tps_points: int = Field(8, ge=4, description="Below this many control points, TPS is not attempted.")
    min_hull_coverage: float = Field(
        0.35, gt=0.0, le=1.0,
        description="Minimum fraction of the registration valid area the control-point convex hull must "
                    "cover before TPS is attempted; below this, TPS would extrapolate past its evidence.",
    )
    min_tps_gain: float = Field(
        0.15, ge=0.0, lt=1.0,
        description="TPS is only selected when its leave-one-out RMSE beats affine's by at least this fraction.",
    )
    tps_smoothing_m: float = Field(0.0, ge=0.0, description="SciPy RBF smoothing parameter, in metres.")
    outlier_threshold_mad: float = Field(3.5, gt=0.0, description="Robust z-score threshold (MAD-based) for flagging a control point.")
    outlier_policy: Literal["reject", "warn"] = "reject"
    max_loo_rmse_px: float = Field(2.0, gt=0.0, description="Leave-one-out RMSE gate, in registration pixels.")
    max_checkpoint_rmse_px: float = Field(2.0, gt=0.0, description="Check-point RMSE gate, in registration pixels.")
    max_field_gradient: float = Field(0.15, gt=0.0, description="Max ||grad(residual)|| on the TPS Jacobian lattice.")
    min_jacobian_det: float = Field(0.5, gt=0.0, le=1.0, description="Min det(I + grad(residual)) on the TPS Jacobian lattice.")
    taper_fallback_weight: float = Field(0.0, ge=0.0, description="Fallback weight to the affine baseline outside control points (0.0 = pure TPS).")
    taper_support_radius_px: float = Field(500.0, gt=0.0, description="Support radius in registration pixels for TPS tapering.")
    pixel_convention: Literal["corner", "center"] = Field(
        "center",
        description="For pixel-mode control points: 'center' treats an integer pixel index as denoting "
                    "the pixel's centre (+0.5 on both axes) before applying the raster transform.",
    )


class WarpConfig(BaseModel):
    tile_size: int = Field(
        2048,
        ge=256,
        le=8192,
        description="Tile dimension for windowed tiled warping."
    )
    reflectance_interpolation: InterpolationType = Field(
        InterpolationType.BILINEAR,
        description="Resampling interpolation for continuous reflectance bands."
    )
    mask_interpolation: InterpolationType = Field(
        InterpolationType.NEAREST,
        description="Resampling interpolation for categorical mask bands."
    )
    nodata_value: float = Field(
        0.0,
        description="Nodata value for unmapped exterior pixels."
    )


class QualityConfig(BaseModel):
    num_checkpoints: int = Field(
        50,
        ge=10,
        le=500,
        description="Number of verification points sampled for residual checking."
    )
    max_acceptable_rmse_px: float = Field(
        2.0,
        gt=0.0,
        description="Max global acceptable RMSE in pixels."
    )
    max_acceptable_edge_rmse_px: float = Field(
        3.5,
        gt=0.0,
        description="Max acceptable edge/corner region RMSE in pixels."
    )
    checkpoint_grid_density: int = Field(
        7,
        ge=3,
        le=15,
        description="Grid density (N x N) for spatial residual error partitioning."
    )
    holdout_gross_mismatch_px: float = Field(
        10.0,
        gt=0.0,
        description="Automated feature-match verification only (never manual GCPs): a held-out "
                    "correspondence farther than this from the fitted transform is a false match, not a "
                    "measurement of the transform's accuracy, so it is excluded from the RMSE gates. It "
                    "still counts against min_holdout_agreement.",
    )
    min_holdout_agreement: float = Field(
        0.7,
        gt=0.0,
        le=1.0,
        description="Automated feature-match verification only: minimum fraction of held-out "
                    "correspondences within holdout_gross_mismatch_px of the fitted transform. A wrong "
                    "transform agrees with almost none of them.",
    )
    preview_max_dimension: int = Field(1600, ge=256, le=4096)


class RoadGridConfig(BaseModel):
    blur_kernel_size: int = Field(
        15, ge=3, le=101,
        description="Gaussian blur kernel size (must be odd). Larger values merge fragmented road pixels "
                    "into continuous axial mass, improving projection profile quality. Important when "
                    "roads are not perfectly axis-aligned.",
    )
    bright_percentile: float = Field(
        90.0, gt=50.0, le=99.9,
        description="Percentile threshold for road (bright) mask. Pixels above this percentile "
                    "in the normalized grayscale are classified as road.",
    )
    dark_percentile: float = Field(
        10.0, ge=0.1, lt=50.0,
        description="Percentile threshold for tree-row (dark) mask. Pixels below this percentile "
                    "in the normalized grayscale are classified as tree rows.",
    )
    num_strips: int = Field(
        20, ge=4, le=100,
        description="Number of horizontal/vertical strips for local offset refinement.",
    )
    guardrail_px: int = Field(
        30, ge=5, le=200,
        description="Maximum pixel deviation from the road-derived primary offset allowed "
                    "during strip refinement. Prevents aliasing onto adjacent crop rows.",
    )
    min_road_pixel_fraction: float = Field(
        0.02, gt=0.0, le=0.5,
        description="Minimum fraction of valid pixels that must be detected as road for the "
                    "strategy to proceed. Below this, InsufficientRoadFeatureError is raised.",
    )
    min_strip_road_pixels: int = Field(
        50, ge=10,
        description="Minimum road pixels in a sub-strip for it to contribute to the consensus. "
                    "Strips below this are discarded to avoid NaN from flat cross-correlation signals.",
    )
    use_dark_refinement: bool = Field(
        True,
        description="Whether to also cross-correlate tree-row (dark) masks as secondary confirmation.",
    )


class LocalMeshConfig(BaseModel):
    """Experimental controls for sparse, road-constrained local registration.

    This configuration deliberately stops before raster warping: it produces and
    validates sparse registration-grid displacement observations only.  A local
    field may be used for output warping only after it passes its own QA gate.
    """

    enabled: bool = Field(False, description="Enable experimental sparse local-mesh sampling.")
    allow_legacy_checker_mesh_export: bool = Field(
        False,
        description="Unsafe legacy switch. The fixed-grid/TPS exporter is disabled after repeated-road aliasing was observed.",
    )
    framing_rails_enabled: bool = Field(
        True,
        description="Use full-canvas bright-road rails as primary structural boundaries.",
    )
    exterior_rail_max_deviation_px: int = Field(
        12, ge=1, le=256,
        description="Absolute deviation allowed when matching exterior framing rails.",
    )
    internal_rail_max_deviation_px: int = Field(
        20, ge=1, le=256,
        description="Maximum local deviation allowed for internal bright-road rail matches.",
    )
    rail_min_length_fraction: float = Field(
        0.20, gt=0.0, le=1.0,
        description="Minimum canvas-span fraction for a bright component to qualify as a structural rail.",
    )
    rail_min_width_px: int = Field(3, ge=1, le=128)
    rail_merge_gap_px: int = Field(8, ge=0, le=128)
    rail_orientation_tolerance_deg: float = Field(15.0, gt=0.0, le=45.0)
    dark_boundary_min_length_fraction: float = Field(0.15, gt=0.0, le=1.0)
    min_region_size_px: int = Field(64, ge=16, le=2048)
    rail_deviation_deadband_px: float = Field(
        1.0, ge=0.0, le=10.0,
        description="Matched rail deviations below this are treated as zero to avoid harmful sub-pixel resampling.",
    )
    region_local_max_deviation_px: int = Field(
        3, ge=0, le=32,
        description="Hard search limit for grayscale correction inside one rail-bounded region.",
    )
    region_boundary_margin_px: int = Field(
        8, ge=0, le=128,
        description="Margin removed from a region before local matching so enclosing rails cannot dominate its interior score.",
    )
    region_min_holdout_improvement: float = Field(
        0.01, ge=0.0, le=1.0,
        description="Minimum independent correlation improvement required to accept a region-local correction.",
    )
    grid_rows: int = Field(5, ge=2, le=32, description="Number of local sampling rows.")
    grid_cols: int = Field(5, ge=2, le=32, description="Number of local sampling columns.")
    cell_overlap_fraction: float = Field(
        0.25, ge=0.0, lt=1.0,
        description="Fractional overlap used only as a feature halo around each cell core.",
    )
    cell_halo_px: int = Field(16, ge=0, le=512, description="Additional global-feature halo around each cell.")
    max_search_radius_px: int = Field(
        12, ge=1, le=256,
        description="Maximum residual search around the verified global translation.",
    )
    competitor_spacing_fraction: float = Field(
        0.45, gt=0.0, lt=0.5,
        description="Reserved guardrail fraction below half the separation of competing road hypotheses.",
    )
    ambiguity_separation_px: int = Field(
        4, ge=1, le=64,
        description="Candidate displacements closer than this are one local peak, not competing roads.",
    )
    orientation_tolerance_deg: float = Field(25.0, gt=0.0, le=90.0)
    min_road_points: int = Field(20, ge=5, le=100000)
    min_tree_points: int = Field(20, ge=5, le=100000)
    allow_tree_only: bool = Field(
        True,
        description="Permit a stricter tree-row-only sample when a cell lacks usable bright-road evidence.",
    )
    tree_only_min_confidence: float = Field(
        0.55, ge=0.0, le=1.0,
        description="Minimum confidence for a tree-row-only sample; normally stricter than road-supported evidence.",
    )
    tree_ambiguity_margin: float = Field(
        0.12, gt=0.0, le=1.0,
        description="Required relative gap between the best and second distinct tree-row displacement.",
    )
    min_match_confidence: float = Field(0.45, ge=0.0, le=1.0)
    ambiguity_margin: float = Field(
        0.08, gt=0.0, le=1.0,
        description="Minimum relative score advantage required over a distinct nearby-road hypothesis.",
    )
    road_tree_agreement_px: float = Field(5.0, gt=0.0, le=128.0)
    max_neighbor_difference_px: float = Field(8.0, gt=0.0, le=256.0)
    min_trusted_cells: int = Field(6, ge=1, le=1024)
    min_spatial_coverage: float = Field(0.30, gt=0.0, le=1.0)
    field_support_radius_px: float = Field(
        256.0, gt=1.0, le=4096.0,
        description="Distance over which sparse local vectors influence a fitted displacement field.",
    )
    field_fallback_weight: float = Field(
        0.20, ge=0.0, le=10.0,
        description="Weight that pulls unsupported mesh nodes back to the global correction.",
    )
    rbf_smoothing: float = Field(
        0.50, ge=0.0, le=1000.0,
        description="Regularization used by the experimental thin-plate spline field fitter.",
    )
    max_field_gradient: float = Field(
        0.15, gt=0.0, le=5.0,
        description="Maximum permitted displacement change per registration pixel in a fitted field.",
    )
    max_field_loo_rmse_px: float = Field(
        2.0, gt=0.0,
        description="Maximum leave-one-out displacement error for an accepted local field.",
    )


class CellCorrelationConfig(BaseModel):
    enabled: bool = True
    grid_rows: int = Field(7, ge=2, le=32)
    grid_cols: int = Field(7, ge=2, le=32)

    cell_halo_px: int = Field(16, ge=4, le=256)
    max_search_radius_px: int = Field(12, ge=1, le=128)
    interpolation_margin_px: int = Field(2, ge=1, le=8)

    min_valid_fraction: float = Field(0.40, ge=0.0, le=1.0)
    min_texture_std: float = Field(0.02, ge=0.0)
    min_phase_response: float = Field(0.10, ge=0.0, le=1.0)
    min_peak_sharpness: float = Field(1.30, ge=1.0)
    full_confidence_peak_sharpness: float = Field(2.0, gt=1.0)
    peak_exclusion_radius_px: int = Field(3, ge=1, le=16)
    min_match_confidence: float = Field(0.35, ge=0.0, le=1.0)
    max_channel_disagreement_px: float = Field(2.0, ge=0.0, le=32.0)

    max_neighbor_difference_px: float = Field(8.0, gt=0.0)
    min_trusted_cells: int = Field(6, ge=4)
    min_spatial_coverage: float = Field(0.30, gt=0.0, le=1.0)

    field_support_radius_px: float = Field(256.0, gt=1.0, le=4096.0)
    field_fallback_weight: float = Field(0.20, ge=0.0, le=10.0)
    rbf_smoothing: float = Field(0.50, ge=0.0, le=1000.0)
    max_field_gradient: float = Field(0.15, gt=0.0, le=5.0)
    max_field_loo_rmse_px: float = Field(2.0, gt=0.0)

    min_holdout_cells: int = Field(3, ge=2)
    max_holdout_cells: int = Field(12, ge=1, le=64)
    min_holdout_improvement: float = Field(0.005, ge=0.0, le=1.0)
    min_holdout_win_fraction: float = Field(0.60, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_correlation_config(self) -> CellCorrelationConfig:
        if self.cell_halo_px < self.max_search_radius_px + self.interpolation_margin_px:
            raise ValueError(f"cell_halo_px ({self.cell_halo_px}) must be >= max_search_radius_px + interpolation_margin_px")
        
        total_cells = self.grid_rows * self.grid_cols
        if self.min_trusted_cells > total_cells:
            raise ValueError(f"min_trusted_cells ({self.min_trusted_cells}) cannot exceed total grid cells ({total_cells})")
        if self.min_holdout_cells > total_cells:
            raise ValueError(f"min_holdout_cells ({self.min_holdout_cells}) cannot exceed total grid cells ({total_cells})")
        if self.max_holdout_cells < self.min_holdout_cells:
            raise ValueError(
                f"max_holdout_cells ({self.max_holdout_cells}) must be >= min_holdout_cells ({self.min_holdout_cells})"
            )
        if self.min_trusted_cells < 4:
            raise ValueError(f"min_trusted_cells ({self.min_trusted_cells}) must be at least 4 for leave-one-out fitting")
        if self.full_confidence_peak_sharpness <= self.min_peak_sharpness:
            raise ValueError(f"full_confidence_peak_sharpness ({self.full_confidence_peak_sharpness}) must be > min_peak_sharpness ({self.min_peak_sharpness})")
            
        return self


class AlignmentConfig(BaseModel):
    alignment_mode: AlignmentMode = Field(
        AlignmentMode.AUTOMATED,
        description="Alignment mode: 'automated' for pipeline detection or 'manual' for user GCP control points."
    )
    resolution_mode: ResolutionMode = Field(
        ResolutionMode.MS_NATIVE,
        description="Target GSD mode: 'ms' preserves MS pixel resolution, 'rgb' upsamples to RGB."
    )
    coarse: CoarseAlignmentConfig = Field(default_factory=CoarseAlignmentConfig)
    features: FeatureDetectionConfig = Field(default_factory=FeatureDetectionConfig)
    arosics: ArosicsConfig = Field(default_factory=ArosicsConfig)
    transform: TransformValidationConfig = Field(default_factory=TransformValidationConfig)
    manual: ManualAlignmentConfig = Field(default_factory=ManualAlignmentConfig)
    warp: WarpConfig = Field(default_factory=WarpConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    road_grid: RoadGridConfig = Field(default_factory=RoadGridConfig)
    local_mesh: LocalMeshConfig = Field(default_factory=LocalMeshConfig)
    cell_correlation: CellCorrelationConfig = Field(default_factory=CellCorrelationConfig)
    
    rgb_red_band_index: int = Field(
        1,
        ge=1,
        description="1-indexed Red band number in RGB file (ODM default: 1)."
    )
    ms_red_band_index: int = Field(
        1,
        ge=1,
        description="1-indexed Red band number in MS file."
    )
    rgb_green_band_index: int = Field(2, ge=1, description="1-indexed Green band in RGB file.")
    ms_green_band_index: int = Field(2, ge=1, description="1-indexed Green band in MS file.")
    registration_channel_priority: list[RegistrationChannel] = Field(
        default_factory=lambda: [RegistrationChannel.GREEN, RegistrationChannel.RED],
        min_length=1,
        description="Semantic channel pairs attempted in priority order."
    )
    min_common_valid_fraction: float = Field(0.20, gt=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_manual_tps_keys(cls, data):
        """Migrate the pre-split ``transform.manual_tps_*`` keys with a warning.

        These moved to ``manual.*`` when manual alignment was rewritten around
        an explicit :class:`ManualAlignmentConfig`. ``manual_tps_smoothing``
        cannot be converted unit-for-unit (it was in registration-pixel units;
        ``manual.tps_smoothing_m`` is in metres, which needs the registration
        GSD not known at config-load time), so a non-zero legacy value is
        dropped with a warning rather than silently reinterpreted. Also warns
        about the retired footprint keys, whose meaning changed so no value is carried over.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        import warnings

        coarse_data = data.get("coarse")
        if isinstance(coarse_data, dict) and "overlap_margin_px" in coarse_data:
            warnings.warn(
                "coarse.overlap_margin_px has been replaced by coarse.overlap_margin_m (metres) and is "
                "ignored.", DeprecationWarning, stacklevel=2,
            )
        transform_data = data.get("transform")
        if isinstance(transform_data, BaseModel):
            transform_data = transform_data.model_dump()
        if not isinstance(transform_data, dict):
            return data
        transform_data = dict(transform_data)
        for retired in ("min_reference_overlap_ratio", "min_target_overlap_ratio"):
            if retired in transform_data:
                warnings.warn(
                    f"transform.{retired} has been retired and is ignored: RGB/MS coverage overlap is "
                    "reported but no longer gated, because the two flights routinely cover different "
                    "ground. transform.min_retained_valid_ratio is the remaining footprint gate.",
                    DeprecationWarning, stacklevel=2,
                )
        legacy_manual: dict = {}

        if "manual_tps_smoothing" in transform_data:
            value = transform_data.pop("manual_tps_smoothing")
            if value == 0.0:
                legacy_manual["tps_smoothing_m"] = 0.0
                warnings.warn(
                    "transform.manual_tps_smoothing is deprecated; set manual.tps_smoothing_m "
                    "(in metres) instead.", DeprecationWarning, stacklevel=2,
                )
            else:
                warnings.warn(
                    "transform.manual_tps_smoothing is deprecated and was expressed in registration-pixel "
                    f"units; the legacy value ({value}) cannot be converted to metres without the "
                    "registration GSD and has been ignored. Set manual.tps_smoothing_m (in metres) "
                    "explicitly.", DeprecationWarning, stacklevel=2,
                )
        if "manual_tps_fallback_weight" in transform_data:
            legacy_manual["taper_fallback_weight"] = transform_data.pop("manual_tps_fallback_weight")
            warnings.warn(
                "transform.manual_tps_fallback_weight is deprecated; set manual.taper_fallback_weight "
                "instead.", DeprecationWarning, stacklevel=2,
            )
        if "manual_tps_support_radius_px" in transform_data:
            legacy_manual["taper_support_radius_px"] = transform_data.pop("manual_tps_support_radius_px")
            warnings.warn(
                "transform.manual_tps_support_radius_px is deprecated; set manual.taper_support_radius_px "
                "instead.", DeprecationWarning, stacklevel=2,
            )
        if legacy_manual:
            data["transform"] = transform_data
            existing_manual = data.get("manual") or {}
            if isinstance(existing_manual, BaseModel):
                existing_manual = existing_manual.model_dump()
            data["manual"] = {**legacy_manual, **existing_manual}
        return data
