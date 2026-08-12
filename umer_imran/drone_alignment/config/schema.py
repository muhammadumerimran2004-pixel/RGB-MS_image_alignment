from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ResolutionMode(str, Enum):
    MS_NATIVE = "ms"
    RGB_NATIVE = "rgb"


class AlignmentMode(str, Enum):
    MANUAL = "manual"
    AUTOMATED = "automated"


class DetectorType(str, Enum):
    ORB = "orb"
    SIFT = "sift"


class TransformType(str, Enum):
    AFFINE = "affine"
    HOMOGRAPHY = "homography"


class InterpolationType(str, Enum):
    BILINEAR = "bilinear"
    NEAREST = "nearest"


class RegistrationChannel(str, Enum):
    GREEN = "green"
    RED = "red"


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
    transform: TransformValidationConfig = Field(default_factory=TransformValidationConfig)
    warp: WarpConfig = Field(default_factory=WarpConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    
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
