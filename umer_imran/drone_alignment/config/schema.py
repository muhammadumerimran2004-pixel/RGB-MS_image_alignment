from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


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
    overlap_margin_px: int = Field(
        50,
        ge=0,
        description="Pixel margin beyond bounding box overlap to include during coarse crop."
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


class LoFTRConfig(BaseModel):
    """Optional learned correspondence candidate.

    LoFTR never publishes an alignment by itself.  Its matches enter the same
    transform, footprint and residual validation path as classical matches.
    """
    enabled: bool = Field(False, description="Attempt LoFTR after classical candidates fail.")
    representations: list[RepresentationType] = Field(
        default_factory=lambda: [RepresentationType.NORMALIZED, RepresentationType.GABOR_ENERGY],
        min_length=1,
        description="Cross-spectral representations tried in order.",
    )
    tile_grid: int = Field(7, ge=2, le=12)
    max_tiles: int = Field(
        12, ge=1, le=144,
        description="Maximum texture-ranked LoFTR tiles evaluated per candidate; bounds CPU/GPU runtime.",
    )
    tile_halo_px: int = Field(32, ge=0, le=256)
    min_confidence: float = Field(0.50, ge=0.0, le=1.0)
    max_matches_per_tile: int = Field(30, ge=1, le=200)
    max_residual_from_prior_px: float = Field(64.0, gt=0.0, le=512.0)
    device: str = Field("auto", pattern="^(auto|cpu|cuda)$")


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
    min_retained_valid_ratio: float = Field(0.80, gt=0.0, le=1.0)
    min_reference_overlap_ratio: float = Field(0.80, gt=0.0, le=1.0)
    min_inlier_coverage_ratio: float = Field(0.05, ge=0.0, le=1.0)


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


from pydantic import model_validator


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
    loftr: LoFTRConfig = Field(default_factory=LoFTRConfig)
    transform: TransformValidationConfig = Field(default_factory=TransformValidationConfig)
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
