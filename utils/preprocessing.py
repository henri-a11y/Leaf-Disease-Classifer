from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


TARGET_SIZE: Tuple[int, int] = (224, 224)


def resize_to_224(image: np.ndarray, size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Resize an RGB image to 224x224.

    Parameters
    ----------
    image: np.ndarray
        RGB image of shape (H, W, 3).
    size: Tuple[int, int]
        Target size, default (224, 224).
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3); got {image.shape}.")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_0_1(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1]."""
    image = image.astype(np.float32)
    return np.clip(image / 255.0, 0.0, 1.0)


def optional_gaussian_blur(
    image: np.ndarray,
    enabled: bool = False,
    ksize: Tuple[int, int] = (5, 5),
    sigma: float = 0.0,
) -> np.ndarray:
    """Optionally apply Gaussian blur (useful to reduce noise)."""
    if not enabled:
        return image
    return cv2.GaussianBlur(image, ksize=ksize, sigmaX=sigma)


def to_mobilenet_v2_range(image_0_1: np.ndarray) -> np.ndarray:
    """
    MobileNetV2 (tf.keras.applications) expects inputs scaled to [-1, 1].

    We first normalize to [0, 1] (required by the project spec) and then map to [-1, 1].
    """
    return image_0_1 * 2.0 - 1.0


def preprocess_for_mobilenet_v2(
    image_rgb: np.ndarray,
    *,
    blur: bool = False,
    blur_ksize: Tuple[int, int] = (5, 5),
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """
    End-to-end preprocessing for MobileNetV2.

    Steps:
    - Resize to 224x224
    - Optional Gaussian blur
    - Normalize to [0, 1]
    - Convert to [-1, 1] range for tf.keras MobileNetV2 weights
    """
    img = resize_to_224(image_rgb)
    img = optional_gaussian_blur(img, enabled=blur, ksize=blur_ksize, sigma=blur_sigma)
    img_0_1 = normalize_0_1(img)
    return to_mobilenet_v2_range(img_0_1).astype(np.float32)

