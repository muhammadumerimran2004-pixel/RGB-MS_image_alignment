from pathlib import Path
import cv2
import numpy as np
from drone_alignment.alignment.feature_detector import normalize_to_uint8


def generate_alignment_preview(
    rgb_red: np.ndarray,
    ms_red_aligned: np.ndarray,
    output_path: Path,
    checker_size: int = 128,
    rgb_mask: np.ndarray | None = None,
    ms_mask: np.ndarray | None = None,
    max_dimension: int = 1600,
) -> Path:
    """
    Generates a 2x2 multi-panel QA image for visual inspection:
    1. RGB Red Channel
    2. Aligned MS Red Band
    3. Checkerboard Blend
    4. False-Color Overlay (Red=RGB Red, Green=MS Red, Blue=0)
    """
    output_path_obj = Path(output_path).resolve()
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if rgb_mask is None:
        rgb_mask = np.isfinite(rgb_red) & (rgb_red != 0)
    if ms_mask is None:
        ms_mask = np.isfinite(ms_red_aligned) & (ms_red_aligned != 0)
    h0, w0 = rgb_red.shape
    scale = min(1.0, max_dimension / max(h0, w0))
    if scale < 1.0:
        size = (max(1, round(w0 * scale)), max(1, round(h0 * scale)))
        rgb_red = cv2.resize(rgb_red, size, interpolation=cv2.INTER_AREA)
        ms_red_aligned = cv2.resize(ms_red_aligned, size, interpolation=cv2.INTER_AREA)
        rgb_mask = cv2.resize(rgb_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        ms_mask = cv2.resize(ms_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    r_norm = normalize_to_uint8(rgb_red, rgb_mask)
    ms_norm = normalize_to_uint8(ms_red_aligned, ms_mask)

    h, w = r_norm.shape

    # 1. Checkerboard Blend
    checker = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, checker_size):
        for x in range(0, w, checker_size):
            if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                checker[y:min(y + checker_size, h), x:min(x + checker_size, w)] = 1

    checkerboard_img = np.where(checker == 1, r_norm, ms_norm)

    # 2. False Color Overlay (misalignments show as magenta/cyan fringes)
    overlay_bgr = cv2.merge([
        np.zeros((h, w), dtype=np.uint8),  # B
        ms_norm,                           # G
        r_norm,                            # R
    ])

    # Convert single channel grayscale to 3-channel BGR for assembly
    r_bgr = cv2.cvtColor(r_norm, cv2.COLOR_GRAY2BGR)
    ms_bgr = cv2.cvtColor(ms_norm, cv2.COLOR_GRAY2BGR)
    chk_bgr = cv2.cvtColor(checkerboard_img, cv2.COLOR_GRAY2BGR)

    # Add text labels to each quadrant
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(w, h) / 1000.0)
    thickness = max(1, int(font_scale * 2))

    cv2.putText(r_bgr, "1. RGB Reference (Red Channel)", (20, 40), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(ms_bgr, "2. Aligned MS (Red Band)", (20, 40), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(chk_bgr, "3. Checkerboard Overlay", (20, 40), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(overlay_bgr, "4. False Color Overlay (R=RGB, G=MS)", (20, 40), font, font_scale, (0, 255, 255), thickness)

    # Assemble 2x2 grid
    top_row = np.hstack([r_bgr, ms_bgr])
    bottom_row = np.hstack([chk_bgr, overlay_bgr])
    quad_panel = np.vstack([top_row, bottom_row])

    cv2.imwrite(str(output_path_obj), quad_panel)
    return output_path_obj
