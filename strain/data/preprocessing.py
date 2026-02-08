"""Preprocessing utilities for cardiac MRI data.

Includes intensity normalization, spatial resampling, center/LV-centered
cropping, and a composable preprocessing pipeline builder.

All spatial functions accept both 3-D (..., H, W) and 4-D (T, D, H, W)
arrays; leading dimensions are preserved automatically.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.ndimage import center_of_mass, label, zoom


# ---------------------------------------------------------------------------
# Intensity normalization
# ---------------------------------------------------------------------------

def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Apply per-volume z-score normalization.

    Args:
        image: Input image array of any dimensionality.

    Returns:
        Normalized image with zero mean and unit variance.
    """
    mean = image.mean()
    std = image.std()
    if std < 1e-8:
        return image - mean
    return (image - mean) / std


# ---------------------------------------------------------------------------
# Spatial resampling
# ---------------------------------------------------------------------------

def resample_image(
    image: np.ndarray,
    current_spacing: tuple[float, ...],
    target_spacing: tuple[float, float] = (1.25, 1.25),
) -> np.ndarray:
    """Resample image to target in-plane spacing.

    Works for 2-D (H, W), 3-D (..., H, W), and 4-D (T, D, H, W) arrays.
    Only the last two spatial dimensions are resampled.

    Args:
        image: Input array with shape (..., H, W).
        current_spacing: Current pixel spacing (x, y) in mm.
        target_spacing: Desired pixel spacing (x, y) in mm.

    Returns:
        Resampled image array.
    """
    zoom_factors = [1.0] * (image.ndim - 2) + [
        current_spacing[1] / target_spacing[1],
        current_spacing[0] / target_spacing[0],
    ]
    return zoom(image, zoom_factors, order=3)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def center_crop(image: np.ndarray, crop_size: int = 128) -> np.ndarray:
    """Crop the image to a square centered on the image center.

    Args:
        image: Input array with shape (..., H, W).
        crop_size: Side length of the square crop.

    Returns:
        Cropped image array with shape (..., crop_size, crop_size).
    """
    h, w = image.shape[-2], image.shape[-1]
    start_h = max(0, (h - crop_size) // 2)
    start_w = max(0, (w - crop_size) // 2)
    return image[..., start_h : start_h + crop_size, start_w : start_w + crop_size]


def detect_lv_center(image: np.ndarray) -> tuple[int, int]:
    """Detect the left-ventricle center using image intensity moments.

    The heuristic assumes that the LV blood pool is the brightest large
    connected region near the image center.  For multi-dimensional input
    the detection runs on a single representative 2-D slice (mid slice
    for 3-D, mid time-frame mid slice for 4-D).

    Args:
        image: Input array with shape (H, W), (D, H, W), or (T, D, H, W).

    Returns:
        (row, col) center coordinates in the last two spatial dimensions.
    """
    # Reduce to a single 2D slice
    if image.ndim == 4:
        # (T, D, H, W) -> pick mid time frame, mid slice
        slice_2d = image[image.shape[0] // 2, image.shape[1] // 2]
    elif image.ndim == 3:
        # (D, H, W) -> pick mid slice
        slice_2d = image[image.shape[0] // 2]
    elif image.ndim == 2:
        slice_2d = image
    else:
        raise ValueError(f"Expected 2-4D array, got {image.ndim}D")

    slice_2d = slice_2d.astype(np.float64)

    # Threshold at the 75th percentile to isolate bright regions
    threshold = np.percentile(slice_2d, 75)
    binary = (slice_2d > threshold).astype(np.int32)

    # Label connected components and pick the one closest to center
    labeled, n_features = label(binary)
    if n_features == 0:
        # Fallback: image center
        return image.shape[-2] // 2, image.shape[-1] // 2

    img_center = np.array([slice_2d.shape[0] / 2.0, slice_2d.shape[1] / 2.0])
    best_label = 1
    best_dist = float("inf")

    centroids = center_of_mass(binary, labeled, range(1, n_features + 1))
    for i, centroid in enumerate(centroids, start=1):
        dist = np.linalg.norm(np.array(centroid) - img_center)
        if dist < best_dist:
            best_dist = dist
            best_label = i

    # Compute center of mass of the best component
    component_mask = (labeled == best_label).astype(np.float64)
    cy, cx = center_of_mass(component_mask)
    return int(round(cy)), int(round(cx))


def crop_around_lv(
    image: np.ndarray,
    center: tuple[int, int],
    crop_size: int = 128,
) -> np.ndarray:
    """Crop the image centered on a given (row, col) coordinate.

    If the crop would extend beyond image boundaries the center is shifted
    so the crop fits entirely within the image.

    Args:
        image: Input array with shape (..., H, W).
        center: (row, col) center of the crop.
        crop_size: Side length of the square crop.

    Returns:
        Cropped image with shape (..., crop_size, crop_size).
    """
    h, w = image.shape[-2], image.shape[-1]
    half = crop_size // 2

    # Clamp so the crop stays within bounds
    cr = min(max(center[0], half), h - half)
    cc = min(max(center[1], half), w - half)

    start_r = cr - half
    start_c = cc - half
    return image[..., start_r : start_r + crop_size, start_c : start_c + crop_size]


# ---------------------------------------------------------------------------
# Composable preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(config: dict[str, Any]) -> Callable[[np.ndarray, dict], np.ndarray]:
    """Build a composable preprocessing pipeline from a configuration dict.

    Supported config keys (all optional):

    - ``normalize`` (str): ``"zscore"`` for z-score normalization.
    - ``target_spacing`` (list[float]): target (x, y) spacing in mm.
    - ``crop_size`` (int): size of the square crop.
    - ``crop_mode`` (str): ``"center"`` or ``"lv"`` (LV-centered crop).

    The returned callable expects ``(image, metadata)`` where *metadata*
    must contain ``"spacing"`` when ``target_spacing`` is configured.

    Args:
        config: Dictionary of preprocessing parameters.

    Returns:
        A function ``preprocess(image, metadata) -> image``.
    """
    steps: list[Callable] = []

    # Normalization
    norm_mode = config.get("normalize", None)
    if norm_mode == "zscore":
        steps.append(lambda img, _meta: normalize_intensity(img.astype(np.float32)))

    # Resampling
    target_spacing = config.get("target_spacing", None)
    if target_spacing is not None:
        target_sp = tuple(target_spacing)

        def _resample(img: np.ndarray, meta: dict) -> np.ndarray:
            spacing = meta.get("spacing", (1.0, 1.0))
            # spacing from SimpleITK is (x, y [, z]) — we need in-plane (x, y)
            current_sp = (float(spacing[0]), float(spacing[1]))
            return resample_image(img, current_sp, target_sp)

        steps.append(_resample)

    # Cropping
    crop_size = config.get("crop_size", None)
    crop_mode = config.get("crop_mode", "center")
    if crop_size is not None:
        if crop_mode == "lv":
            def _crop_lv(img: np.ndarray, _meta: dict) -> np.ndarray:
                c = detect_lv_center(img)
                return crop_around_lv(img, c, crop_size)
            steps.append(_crop_lv)
        else:
            steps.append(lambda img, _meta: center_crop(img, crop_size))

    def pipeline(image: np.ndarray, metadata: dict | None = None) -> np.ndarray:
        if metadata is None:
            metadata = {}
        for step in steps:
            image = step(image, metadata)
        return image

    return pipeline
