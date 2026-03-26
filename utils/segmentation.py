from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class HSVGreenMaskConfig:
    """
    HSV range for "green" as used in a simple HSV-based leaf segmentation.

    OpenCV uses H in [0, 179], S/V in [0, 255].
    """

    green_lower: Tuple[int, int, int] = (35, 40, 40)
    green_upper: Tuple[int, int, int] = (90, 255, 255)
    morph_kernel_size: int = 7
    morph_iterations: int = 1
    opening_iterations: int = 1


def _coverage_ratio(mask: np.ndarray) -> float:
    """Compute fraction of non-zero mask pixels."""
    if mask.ndim != 2:
        raise ValueError("Mask must be a single-channel (H, W) array.")
    return float(np.count_nonzero(mask)) / float(mask.size)


def segment_leaf_hsv(
    image_rgb: np.ndarray,
    config: HSVGreenMaskConfig = HSVGreenMaskConfig(),
    *,
    coverage_threshold: float = 0.01,
    raise_on_low_coverage: bool = True,
    mask_output_type: str = "uint8",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Segment a leaf using HSV green color masking.

    Returns
    -------
    segmented_rgb: np.ndarray
        RGB image with background removed (black background).
    mask: np.ndarray
        Binary mask (0 or 255 by default).
    coverage: float
        Fraction of pixels considered leaf by the mask.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3); got {image_rgb.shape}.")

    # HSV conversion is done in OpenCV using RGB->HSV.
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    lower = np.array(config.green_lower, dtype=np.uint8)
    upper = np.array(config.green_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)  # 0 or 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size)
    )

    # Morphological closing fills small holes; opening removes small noise.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.morph_iterations)
    if config.opening_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.opening_iterations)

    coverage = _coverage_ratio(mask)
    if raise_on_low_coverage and coverage < coverage_threshold:
        raise ValueError(
            f"Leaf segmentation coverage too low ({coverage:.4f}). "
            f"Try a different image or adjust HSV/mask thresholds."
        )

    segmented_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    if mask_output_type == "uint8":
        mask_out = mask.astype(np.uint8)
    elif mask_output_type == "bool":
        mask_out = (mask > 0)
    else:
        raise ValueError("mask_output_type must be one of: 'uint8', 'bool'.")

    return segmented_rgb, mask_out, coverage


def ensure_uint8_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """
    Ensure an RGB image is uint8 [0, 255] for OpenCV operations.

    Supports inputs in [0, 1] float range as well.
    """
    if image_rgb.dtype == np.uint8:
        return image_rgb

    img = image_rgb.astype(np.float32)
    maxv = float(np.max(img)) if img.size else 0.0
    if maxv <= 1.0:
        img = img * 255.0
    return np.clip(img, 0.0, 255.0).astype(np.uint8)

