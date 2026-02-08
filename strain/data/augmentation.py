"""Data augmentation transforms for cardiac MRI training.

All transforms that operate on image+mask pairs apply bilinear (order=3)
interpolation for the image and nearest-neighbor (order=0) for the mask
to preserve label integrity.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import map_coordinates, rotate, zoom


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def random_rotation(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    max_angle: float = 15.0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply random rotation to image and optional mask.

    Args:
        image: Input image (..., H, W).
        mask: Optional segmentation mask with same spatial dims.
        max_angle: Maximum rotation angle in degrees.

    Returns:
        Rotated image (and mask if provided).
    """
    angle = np.random.uniform(-max_angle, max_angle)
    rotated_image = rotate(image, angle, axes=(-2, -1), reshape=False, order=3)
    if mask is not None:
        rotated_mask = rotate(mask, angle, axes=(-2, -1), reshape=False, order=0)
        return rotated_image, rotated_mask
    return rotated_image


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def random_scale(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    scale_range: tuple[float, float] = (0.9, 1.1),
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply random scaling.

    Args:
        image: Input image (..., H, W).
        mask: Optional segmentation mask.
        scale_range: (min_scale, max_scale).

    Returns:
        Scaled image (and mask if provided).
    """
    scale = np.random.uniform(*scale_range)
    factors = [1.0] * (image.ndim - 2) + [scale, scale]
    scaled_image = zoom(image, factors, order=3)
    if mask is not None:
        scaled_mask = zoom(mask, factors, order=0)
        return scaled_image, scaled_mask
    return scaled_image


# ---------------------------------------------------------------------------
# Gamma correction
# ---------------------------------------------------------------------------

def random_gamma(
    image: np.ndarray,
    gamma_range: tuple[float, float] = (0.7, 1.5),
) -> np.ndarray:
    """Apply random gamma correction.

    Args:
        image: Input image (should be non-negative).
        gamma_range: (min_gamma, max_gamma).

    Returns:
        Gamma-corrected image.
    """
    gamma = np.random.uniform(*gamma_range)
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min < 1e-8:
        return image
    normalized = (image - img_min) / (img_max - img_min)
    return np.power(normalized, gamma) * (img_max - img_min) + img_min


# ---------------------------------------------------------------------------
# Elastic deformation
# ---------------------------------------------------------------------------

def random_elastic_deformation(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    alpha: float = 100.0,
    sigma: float = 10.0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply random elastic deformation using B-spline-like random displacement.

    A low-resolution random displacement field is generated, smoothed with a
    Gaussian filter (controlled by *sigma*), then scaled by *alpha*.  The
    deformation is applied via ``scipy.ndimage.map_coordinates``.

    Only the last two spatial dimensions (H, W) are deformed; any leading
    batch / depth / time dimensions are deformed identically.

    Args:
        image: Input image (..., H, W).
        mask: Optional segmentation mask with same spatial dims.
        alpha: Deformation intensity (higher = stronger).
        sigma: Gaussian smoothing sigma for the displacement field.

    Returns:
        Deformed image (and mask if provided).
    """
    from scipy.ndimage import gaussian_filter

    h, w = image.shape[-2], image.shape[-1]

    # Random displacement fields
    dx = gaussian_filter(np.random.randn(h, w), sigma, mode="constant") * alpha
    dy = gaussian_filter(np.random.randn(h, w), sigma, mode="constant") * alpha

    # Build coordinate grids
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords_y = np.clip(y + dy, 0, h - 1)
    coords_x = np.clip(x + dx, 0, w - 1)

    def _deform(arr: np.ndarray, order: int) -> np.ndarray:
        """Deform an array, handling arbitrary leading dimensions."""
        if arr.ndim == 2:
            return map_coordinates(arr, [coords_y, coords_x], order=order, mode="reflect")
        # Flatten leading dims, deform each 2D slice, reshape back
        leading_shape = arr.shape[:-2]
        flat = arr.reshape(-1, h, w)
        out = np.empty_like(flat)
        for i in range(flat.shape[0]):
            out[i] = map_coordinates(flat[i], [coords_y, coords_x], order=order, mode="reflect")
        return out.reshape(leading_shape + (h, w))

    deformed_image = _deform(image, order=3)
    if mask is not None:
        deformed_mask = _deform(mask, order=0)
        return deformed_image, deformed_mask
    return deformed_image


# ---------------------------------------------------------------------------
# Flip
# ---------------------------------------------------------------------------

def random_flip(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    axis: int = -1,
    prob: float = 0.5,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Randomly flip image (and mask) along a given axis.

    Args:
        image: Input image (..., H, W).
        mask: Optional segmentation mask.
        axis: Axis to flip. ``-1`` = horizontal (W), ``-2`` = vertical (H).
        prob: Probability of applying the flip.

    Returns:
        Flipped image (and mask if provided).
    """
    if np.random.random() < prob:
        image = np.flip(image, axis=axis).copy()
        if mask is not None:
            mask = np.flip(mask, axis=axis).copy()

    if mask is not None:
        return image, mask
    return image


# ---------------------------------------------------------------------------
# Additive Gaussian noise
# ---------------------------------------------------------------------------

def random_noise(
    image: np.ndarray,
    std: float = 0.02,
) -> np.ndarray:
    """Add zero-mean Gaussian noise to the image.

    Args:
        image: Input image array.
        std: Standard deviation of the noise.

    Returns:
        Noisy image (same shape as input).
    """
    noise = np.random.normal(0.0, std, size=image.shape).astype(image.dtype)
    return image + noise


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class Compose:
    """Chain multiple augmentation transforms with per-transform probability.

    Each entry in *transforms* is a callable that accepts ``(image, mask)``
    and returns ``(image, mask)``.  If a raw augmentation function returns
    only the image (no mask), the wrapper handles it transparently.

    Args:
        transforms: Sequence of ``(callable, probability)`` tuples.
            Each callable should accept ``(image, mask)`` keyword-style or
            positional and return the same.

    Example::

        aug = Compose([
            (lambda img, msk: random_rotation(img, msk, 15.0), 0.8),
            (lambda img, msk: random_elastic_deformation(img, msk), 0.3),
            (lambda img, msk: (random_noise(img, 0.02), msk), 0.5),
        ])
        image, mask = aug(image, mask)
    """

    def __init__(self, transforms: Sequence[tuple[callable, float]]):
        self.transforms = list(transforms)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        for fn, prob in self.transforms:
            if np.random.random() < prob:
                result = fn(image, mask)
                if isinstance(result, tuple):
                    image, mask = result
                else:
                    image = result
        if mask is not None:
            return image, mask
        return image

    def __repr__(self) -> str:
        lines = [f"Compose(["]
        for fn, p in self.transforms:
            lines.append(f"  ({fn}, p={p}),")
        lines.append("])")
        return "\n".join(lines)
