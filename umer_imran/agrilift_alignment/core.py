from dataclasses import dataclass
from enum import Enum
import numpy as np

class Space(str, Enum):
    MS_SOURCE = "ms_source_pixel"
    RGB_REFERENCE = "rgb_reference_pixel"
    REGISTRATION = "registration_grid_pixel"
    RGB_OUTPUT = "rgb_output_pixel"

class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"

class AlignmentError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message); self.code = code

@dataclass(frozen=True)
class Transform:
    """Canonical forward transform: source_space pixels -> destination_space pixels."""
    forward: np.ndarray
    source_space: Space
    destination_space: Space
    method: str

    def __post_init__(self):
        if self.forward.shape not in ((2, 3), (3, 3)):
            raise ValueError("Transform must be affine 2x3 or homography 3x3.")

    @property
    def homogeneous(self):
        return self.forward if self.forward.shape == (3, 3) else np.vstack([self.forward, [0., 0., 1.]])

    def inverse(self) -> "Transform":
        return Transform(np.linalg.inv(self.homogeneous), self.destination_space, self.source_space, f"inverse({self.method})")

    def apply(self, points: np.ndarray) -> np.ndarray:
        p = np.c_[points, np.ones(len(points))]
        out = (self.homogeneous @ p.T).T
        return out[:, :2] / out[:, 2:3]

    def registration_to_native(self, sx: float, sy: float) -> "Transform":
        """Convert registration-grid forward matrix to native output pixel coordinates."""
        native_to_reg = np.array([[1/sx,0,0],[0,1/sy,0],[0,0,1.]], dtype=float)
        native = np.linalg.inv(native_to_reg) @ self.homogeneous @ native_to_reg
        return Transform(native, Space.MS_SOURCE, Space.RGB_OUTPUT, self.method)

@dataclass(frozen=True)
class Correspondences:
    ms: np.ndarray
    rgb: np.ndarray
    tiles: np.ndarray

@dataclass(frozen=True)
class Candidate:
    transform: Transform
    inlier_mask: np.ndarray
    correspondences: Correspondences
