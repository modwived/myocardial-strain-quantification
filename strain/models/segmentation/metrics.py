"""Evaluation metrics for cardiac segmentation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff

# ACDC label convention
LABEL_MAP = {1: "RV", 2: "Myo", 3: "LV"}


def dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
) -> dict[str, float]:
    """Compute per-class Dice similarity coefficient.

    Args:
        pred: Predicted segmentation mask with integer labels. Shape (H, W)
            or (D, H, W).
        target: Ground-truth segmentation mask with integer labels. Same shape
            as *pred*.
        num_classes: Total number of classes including background.

    Returns:
        Dictionary mapping class names to Dice scores.
        Example: ``{"RV": 0.91, "Myo": 0.88, "LV": 0.95}``.
    """
    scores: dict[str, float] = {}
    for cls_id in range(1, num_classes):  # skip background
        pred_mask = (pred == cls_id).astype(np.float64)
        target_mask = (target == cls_id).astype(np.float64)

        intersection = (pred_mask * target_mask).sum()
        denominator = pred_mask.sum() + target_mask.sum()

        if denominator == 0:
            # Both masks are empty -- perfect agreement
            dice = 1.0
        else:
            dice = (2.0 * intersection) / denominator

        label_name = LABEL_MAP.get(cls_id, f"class_{cls_id}")
        scores[label_name] = float(dice)

    return scores


def _extract_surface_points(mask: np.ndarray) -> np.ndarray:
    """Extract surface (boundary) voxel coordinates from a binary mask.

    Surface points are defined as foreground voxels that are adjacent to at
    least one background voxel (i.e. they would be removed by binary erosion).

    Args:
        mask: Binary mask array.

    Returns:
        Array of shape (N, ndim) with coordinates of surface points, or an
        empty array with shape (0, ndim) if the mask is empty.
    """
    if mask.sum() == 0:
        return np.empty((0, mask.ndim), dtype=np.float64)

    eroded = binary_erosion(mask)
    surface = mask.astype(bool) & ~eroded
    coords = np.argwhere(surface)
    # If erosion removes everything (single-voxel structures), fall back
    if coords.shape[0] == 0:
        coords = np.argwhere(mask)
    return coords.astype(np.float64)


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    num_classes: int = 4,
) -> dict[str, float]:
    """Compute the 95th-percentile Hausdorff distance per class.

    The Hausdorff distance measures the maximum surface-to-surface distance
    between two segmentation masks. The 95th percentile variant is more robust
    to outliers.

    Args:
        pred: Predicted segmentation mask (integer labels).
        target: Ground-truth segmentation mask (integer labels).
        spacing: Voxel spacing in mm for each spatial dimension. If ``None``,
            unit spacing is assumed.
        num_classes: Total number of classes including background.

    Returns:
        Dictionary mapping class names to HD95 values in mm.
    """
    if spacing is None:
        spacing = tuple(1.0 for _ in range(pred.ndim))

    spacing_arr = np.array(spacing, dtype=np.float64)

    results: dict[str, float] = {}
    for cls_id in range(1, num_classes):
        pred_mask = (pred == cls_id).astype(np.uint8)
        target_mask = (target == cls_id).astype(np.uint8)

        label_name = LABEL_MAP.get(cls_id, f"class_{cls_id}")

        # Handle edge cases where one or both masks are empty
        if pred_mask.sum() == 0 and target_mask.sum() == 0:
            results[label_name] = 0.0
            continue
        if pred_mask.sum() == 0 or target_mask.sum() == 0:
            results[label_name] = float("inf")
            continue

        pred_surface = _extract_surface_points(pred_mask) * spacing_arr
        target_surface = _extract_surface_points(target_mask) * spacing_arr

        # Compute all pairwise surface-to-surface distances
        # For each point on surface A, find the min distance to surface B
        forward_distances = _surface_distances(pred_surface, target_surface)
        backward_distances = _surface_distances(target_surface, pred_surface)

        all_distances = np.concatenate([forward_distances, backward_distances])
        hd95 = float(np.percentile(all_distances, 95))
        results[label_name] = hd95

    return results


def _surface_distances(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """For each point in *points_a*, compute the minimum Euclidean distance
    to any point in *points_b*.

    Args:
        points_a: (N, D) array of coordinates.
        points_b: (M, D) array of coordinates.

    Returns:
        (N,) array of minimum distances.
    """
    # Use a chunked approach to avoid excessive memory for large surfaces
    chunk_size = 5000
    n = points_a.shape[0]
    min_dists = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # (chunk, 1, D) - (1, M, D) -> (chunk, M, D)
        diff = points_a[start:end, np.newaxis, :] - points_b[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))  # (chunk, M)
        min_dists[start:end] = dists.min(axis=1)
    return min_dists


def compute_volumes(
    segmentation: np.ndarray,
    spacing: tuple[float, ...],
) -> dict[str, float]:
    """Compute cardiac volumes and ejection fraction from a segmentation.

    Expects the segmentation to include at least the ED and ES frames (i.e. a
    time-resolved stack). If the input has no time dimension, it computes
    volumes for the single frame only (useful for per-frame analysis).

    Labels follow ACDC convention:
        0 = background, 1 = RV cavity, 2 = myocardium, 3 = LV cavity.

    Args:
        segmentation: Integer segmentation mask.
            - (T, D, H, W) for a 4D volume with T time-frames and D slices.
            - (T, H, W) for a 2D+time volume (single slice -- rarely used).
            - (D, H, W) for a single time-frame 3D volume.
        spacing: Voxel spacing. For 3D volumes: (sz, sy, sx) in mm.
            For 2D: (sy, sx). The product gives the voxel volume factor.

    Returns:
        Dictionary with volume measurements:
            - ``LV_EDV``: LV end-diastolic volume (mL)
            - ``LV_ESV``: LV end-systolic volume (mL)
            - ``LVEF``: LV ejection fraction (%)
            - ``RV_EDV``: RV end-diastolic volume (mL)
            - ``RV_ESV``: RV end-systolic volume (mL)
        If only a single frame is given, ``_EDV`` and ``_ESV`` keys are
        replaced by ``LV_vol`` and ``RV_vol``, and ``LVEF`` is not included.
    """
    # Determine voxel volume in mm^3 then convert to mL (1 mL = 1000 mm^3)
    voxel_vol_mm3 = float(np.prod(spacing))
    voxel_vol_ml = voxel_vol_mm3 / 1000.0

    is_time_resolved = segmentation.ndim >= 3 and _has_time_axis(segmentation)

    if not is_time_resolved:
        # Single frame
        lv_vol = float((segmentation == 3).sum()) * voxel_vol_ml
        rv_vol = float((segmentation == 1).sum()) * voxel_vol_ml
        return {"LV_vol": lv_vol, "RV_vol": rv_vol}

    # Time-resolved: compute per-frame LV and RV volumes and find ED/ES
    n_frames = segmentation.shape[0]
    lv_volumes = np.array(
        [float((segmentation[t] == 3).sum()) * voxel_vol_ml for t in range(n_frames)]
    )
    rv_volumes = np.array(
        [float((segmentation[t] == 1).sum()) * voxel_vol_ml for t in range(n_frames)]
    )

    # ED = frame with largest LV volume, ES = frame with smallest LV volume
    ed_idx = int(np.argmax(lv_volumes))
    es_idx = int(np.argmin(lv_volumes))

    lv_edv = lv_volumes[ed_idx]
    lv_esv = lv_volumes[es_idx]
    rv_edv = rv_volumes[ed_idx]
    rv_esv = rv_volumes[es_idx]

    lvef = ((lv_edv - lv_esv) / lv_edv * 100.0) if lv_edv > 0 else 0.0

    return {
        "LV_EDV": float(lv_edv),
        "LV_ESV": float(lv_esv),
        "LVEF": float(lvef),
        "RV_EDV": float(rv_edv),
        "RV_ESV": float(rv_esv),
    }


def _has_time_axis(segmentation: np.ndarray) -> bool:
    """Heuristic check for whether the first axis is a time axis.

    For a (T, D, H, W) or (T, H, W) array representing a time-resolved
    segmentation, T >= 2 is expected. We treat 4D arrays as always having a
    time axis. For 3D arrays we assume time axis if the number of frames
    (first dim) is >= 2, which is the standard convention for this project.
    """
    if segmentation.ndim == 4:
        return True
    if segmentation.ndim == 3 and segmentation.shape[0] >= 2:
        return True
    return False
