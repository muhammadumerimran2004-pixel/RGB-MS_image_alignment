import cv2
import numpy as np
from .core import AlignmentError

def erode_valid_mask(mask: np.ndarray, radius_px: int = 8) -> np.ndarray:
    """Distance-based erosion; radius has an unambiguous pixel meaning."""
    if radius_px < 0: raise ValueError("radius_px must be non-negative")
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return distance >= radius_px

def normalize_valid(image: np.ndarray, mask: np.ndarray, low=2., high=98., clahe=True) -> np.ndarray:
    valid = mask & np.isfinite(image)
    values = image[valid]
    if values.size < 500: raise AlignmentError("ERR_INSUFFICIENT_VALID_DATA", "Too few valid pixels.")
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo: raise AlignmentError("ERR_INSUFFICIENT_TEXTURE", "No usable radiometric range.")
    out = np.zeros(image.shape, np.uint8)
    out[valid] = np.clip((image[valid]-lo)/(hi-lo)*255, 0, 255).astype(np.uint8)
    if clahe:
        out = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(out)
        out[~valid] = 0
    return out

def sobel_map(image: np.ndarray, eroded_mask: np.ndarray) -> np.ndarray:
    norm = normalize_valid(image, eroded_mask)
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    out = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out[~eroded_mask] = 0
    return out

def log_map(image: np.ndarray, eroded_mask: np.ndarray) -> np.ndarray:
    """Scale-tolerant structural map used when gradient magnitudes disagree across sensors."""
    norm = normalize_valid(image, eroded_mask)
    blur = cv2.GaussianBlur(norm, (0, 0), 1.4)
    lap = np.abs(cv2.Laplacian(blur, cv2.CV_32F, ksize=3))
    out = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out[~eroded_mask] = 0
    return out

def orientation_map(image: np.ndarray, eroded_mask: np.ndarray, bins: int = 8) -> np.ndarray:
    """Quantized gradient orientation map: structural direction is more cross-spectral-stable than brightness."""
    norm = normalize_valid(image, eroded_mask)
    gx=cv2.Sobel(norm,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(norm,cv2.CV_32F,0,1,ksize=3)
    mag,angle=cv2.cartToPolar(gx,gy,angleInDegrees=True)
    threshold=np.percentile(mag[eroded_mask],50) if eroded_mask.any() else np.inf
    codes=((angle % 180)/(180/bins)).astype(np.uint8)
    out=((codes+1)*(255//(bins+1))).astype(np.uint8)
    out[(mag<threshold)|(~eroded_mask)]=0
    return out

def gabor_orientation_map(image: np.ndarray, eroded_mask: np.ndarray, bins: int = 8) -> np.ndarray:
    """Dominant multi-scale Gabor orientation index for cross-spectral structure.

    This is deliberately named for what it is. It supplies the orientation-index
    representation used by RIFT-like matching, but is not presented as a complete
    RIFT implementation. Each valid pixel is labelled by its strongest response
    across spatial frequencies and directions, reducing dependence on brightness.
    """
    norm = normalize_valid(image, eroded_mask).astype(np.float32) / 255.0
    h, w = norm.shape
    best = np.zeros((h, w), np.float32)
    index = np.zeros((h, w), np.uint8)
    for scale_index, wavelength in enumerate((4.0, 8.0)):
        for direction in range(bins):
            theta = np.pi * direction / bins
            kernel = cv2.getGaborKernel((21, 21), 3.5, theta, wavelength, 0.6, 0, ktype=cv2.CV_32F)
            response = np.abs(cv2.filter2D(norm, cv2.CV_32F, kernel))
            stronger = response > best
            best[stronger] = response[stronger]
            index[stronger] = np.uint8(direction + scale_index * bins)
    threshold = np.percentile(best[eroded_mask], 45) if eroded_mask.any() else np.inf
    out = ((index.astype(np.uint16) + 1) * (255 // (2 * bins + 1))).astype(np.uint8)
    out[(best < threshold) | (~eroded_mask)] = 0
    return out

def gabor_energy_map(image: np.ndarray, eroded_mask: np.ndarray, bins: int = 8) -> np.ndarray:
    """Multi-scale, multi-direction structural energy without radiometric labels."""
    norm = normalize_valid(image, eroded_mask).astype(np.float32) / 255.0
    best=np.zeros(norm.shape,np.float32)
    for wavelength in (4.0, 8.0):
        for direction in range(bins):
            kernel=cv2.getGaborKernel((21,21),3.5,np.pi*direction/bins,wavelength,.6,0,ktype=cv2.CV_32F)
            best=np.maximum(best,np.abs(cv2.filter2D(norm,cv2.CV_32F,kernel)))
    out=cv2.normalize(best,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    out[~eroded_mask]=0
    return out
