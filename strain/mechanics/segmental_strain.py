"""AHA 16-segment model for regional strain analysis.

The American Heart Association (AHA) 16-segment model divides the left
ventricle into standardised regions for reporting regional wall-motion and
strain.  The angular reference is the anterior RV insertion point (typically
at ~45 deg in standard short-axis orientation).

Segment numbering:
  - Basal   (6 segments): 1--6
  - Mid     (6 segments): 7--12
  - Apical  (4 segments): 13--16

Angular layout (AHA standard, starting at the anterior RV insertion point,
counter-clockwise):
  Basal / Mid:
    1/7  Anterior
    2/8  Anteroseptal
    3/9  Inferoseptal
    4/10 Inferior
    5/11 Inferolateral
    6/12 Anterolateral
  Apical:
    13  Anterior
    14  Septal
    15  Inferior
    16  Lateral
"""

from __future__ import annotations

import numpy as np

from strain.mechanics.coordinate_system import (
    compute_cardiac_axes,
    compute_rv_insertion_angle,
    get_lv_center,
)
from strain.mechanics.deformation import (
    compute_deformation_gradient,
    compute_green_lagrange_strain,
)


# ---------------------------------------------------------------------------
# AHA segment names
# ---------------------------------------------------------------------------

AHA_SEGMENT_NAMES: dict[int, str] = {
    1: "Basal anterior",
    2: "Basal anteroseptal",
    3: "Basal inferoseptal",
    4: "Basal inferior",
    5: "Basal inferolateral",
    6: "Basal anterolateral",
    7: "Mid anterior",
    8: "Mid anteroseptal",
    9: "Mid inferoseptal",
    10: "Mid inferior",
    11: "Mid inferolateral",
    12: "Mid anterolateral",
    13: "Apical anterior",
    14: "Apical septal",
    15: "Apical inferior",
    16: "Apical lateral",
}

# Default anterior RV insertion angle when no explicit point is provided.
# In standard short-axis CMR orientation this is approximately 45 deg.
_DEFAULT_RV_INSERTION_ANGLE_DEG: float = 45.0


# ---------------------------------------------------------------------------
# Segment assignment
# ---------------------------------------------------------------------------

def assign_aha_segments(
    myocardial_mask: np.ndarray,
    slice_level: str = "mid",
    *,
    lv_cavity_mask: np.ndarray | None = None,
    rv_insertion_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Assign AHA segment labels to myocardial pixels based on angular position.

    The angular reference is aligned to the anterior RV insertion point
    following the AHA standard.

    Args:
        myocardial_mask: (H, W) binary myocardial mask.
        slice_level: One of ``'basal'``, ``'mid'``, ``'apical'``.
        lv_cavity_mask: Optional (H, W) LV cavity mask for centre detection.
        rv_insertion_point: Optional (y, x) anterior RV insertion point.  If
            not supplied a default angle of 45 deg is assumed.

    Returns:
        (H, W) integer array with 1-based segment labels.  Background is 0.
    """
    segments = np.zeros_like(myocardial_mask, dtype=np.int32)
    if myocardial_mask.sum() == 0:
        return segments

    # -- LV centre ---------------------------------------------------------
    cy, cx = get_lv_center(myocardial_mask, lv_cavity_mask)

    # -- Reference angle (RV anterior insertion) ---------------------------
    if rv_insertion_point is not None:
        ref_angle = compute_rv_insertion_angle((cy, cx), rv_insertion_point)
    else:
        ref_angle = _DEFAULT_RV_INSERTION_ANGLE_DEG

    # -- Segment counts & label offsets ------------------------------------
    if slice_level == "basal":
        n_segments, offset = 6, 0
    elif slice_level == "mid":
        n_segments, offset = 6, 6
    elif slice_level == "apical":
        n_segments, offset = 4, 12
    else:
        raise ValueError(f"Unknown slice level: {slice_level}")

    segment_size = 360.0 / n_segments

    # -- Angular assignment ------------------------------------------------
    ys, xs = np.where(myocardial_mask > 0)
    angles = np.degrees(np.arctan2(ys - cy, xs - cx))  # [-180, 180)
    # Normalise relative to the RV insertion reference angle
    angles = (angles - ref_angle) % 360.0  # [0, 360)

    seg_indices = np.clip(
        (angles / segment_size).astype(np.int32), 0, n_segments - 1
    )
    segments[ys, xs] = seg_indices + 1 + offset

    return segments


# ---------------------------------------------------------------------------
# Slice-level classification
# ---------------------------------------------------------------------------

def classify_slice_level(
    slice_index: int,
    total_slices: int,
) -> str:
    """Map a short-axis slice index to basal / mid / apical.

    A simple thirds-based heuristic: the first third of slices (from base)
    are basal, the middle third are mid-ventricular, and the last third
    (toward the apex) are apical.

    Args:
        slice_index: 0-based index of the slice (0 = most basal).
        total_slices: Total number of short-axis slices.

    Returns:
        One of ``'basal'``, ``'mid'``, ``'apical'``.
    """
    if total_slices <= 0:
        raise ValueError("total_slices must be > 0")
    frac = slice_index / total_slices
    if frac < 1.0 / 3.0:
        return "basal"
    elif frac < 2.0 / 3.0:
        return "mid"
    else:
        return "apical"


# ---------------------------------------------------------------------------
# Per-segment strain computation
# ---------------------------------------------------------------------------

def compute_segmental_strain(
    displacement: np.ndarray,
    myocardial_mask: np.ndarray,
    segments: np.ndarray,
    *,
    lv_cavity_mask: np.ndarray | None = None,
    smooth_sigma: float | None = None,
) -> dict[int, dict[str, float]]:
    """Compute circumferential and radial strain per AHA segment.

    Args:
        displacement: (2, H, W) displacement field.
        myocardial_mask: (H, W) binary mask of myocardium.
        segments: (H, W) integer segment labels (1-based) as produced by
            :func:`assign_aha_segments`.
        lv_cavity_mask: Optional (H, W) LV cavity mask for centre detection.
        smooth_sigma: Optional Gaussian smoothing sigma.

    Returns:
        Dictionary mapping segment label to ``{'GCS': float, 'GRS': float}``
        (values in percent).  Only segments present in *segments* are included.
    """
    F = compute_deformation_gradient(displacement, smooth_sigma=smooth_sigma)
    E = compute_green_lagrange_strain(F)
    e_circ, e_rad = compute_cardiac_axes(
        myocardial_mask, lv_cavity_mask=lv_cavity_mask,
    )

    # Project strain tensor onto cardiac axes (full field)
    E_cc = np.zeros(myocardial_mask.shape, dtype=np.float64)
    E_rr = np.zeros(myocardial_mask.shape, dtype=np.float64)
    for i in range(2):
        for j in range(2):
            E_cc += e_circ[i] * E[i, j] * e_circ[j]
            E_rr += e_rad[i] * E[i, j] * e_rad[j]

    # Average per segment
    unique_labels = np.unique(segments)
    result: dict[int, dict[str, float]] = {}
    for seg in unique_labels:
        if seg == 0:
            continue  # background
        seg_mask = segments == seg
        n_pixels = seg_mask.sum()
        if n_pixels == 0:
            continue
        gcs = float(E_cc[seg_mask].mean()) * 100.0
        grs = float(E_rr[seg_mask].mean()) * 100.0
        result[int(seg)] = {"GCS": gcs, "GRS": grs}

    return result


def compute_all_segments_strain(
    displacements: np.ndarray,
    myocardial_masks: np.ndarray,
    slice_levels: list[str] | None = None,
    *,
    lv_cavity_masks: np.ndarray | None = None,
    rv_insertion_points: list[tuple[float, float] | None] | None = None,
    smooth_sigma: float | None = None,
) -> dict[int, dict[str, float]]:
    """Compute segmental strain across multiple short-axis slices.

    This is a convenience function that assigns AHA segments for each slice
    and aggregates the results into a single 16-segment dictionary.

    Args:
        displacements: (S, 2, H, W) displacement fields per slice.
        myocardial_masks: (S, H, W) myocardial masks per slice.
        slice_levels: List of length S with ``'basal'``/``'mid'``/``'apical'``
            per slice.  If *None*, :func:`classify_slice_level` is used.
        lv_cavity_masks: Optional (S, H, W) LV cavity masks per slice.
        rv_insertion_points: Optional list of (y, x) per slice.
        smooth_sigma: Optional Gaussian smoothing sigma.

    Returns:
        Dictionary mapping AHA segment label (1--16) to
        ``{'GCS': float, 'GRS': float}`` (in percent).
    """
    S = displacements.shape[0]

    if slice_levels is None:
        slice_levels = [classify_slice_level(s, S) for s in range(S)]

    all_seg: dict[int, dict[str, float]] = {}

    for s in range(S):
        lv_cav = lv_cavity_masks[s] if lv_cavity_masks is not None else None
        rv_pt = (
            rv_insertion_points[s]
            if rv_insertion_points is not None
            else None
        )

        segs = assign_aha_segments(
            myocardial_masks[s],
            slice_levels[s],
            lv_cavity_mask=lv_cav,
            rv_insertion_point=rv_pt,
        )

        seg_strain = compute_segmental_strain(
            displacements[s],
            myocardial_masks[s],
            segs,
            lv_cavity_mask=lv_cav,
            smooth_sigma=smooth_sigma,
        )

        # Merge (if a segment appears on multiple slices, average)
        for seg_id, vals in seg_strain.items():
            if seg_id not in all_seg:
                all_seg[seg_id] = {"GCS": vals["GCS"], "GRS": vals["GRS"], "_n": 1}
            else:
                all_seg[seg_id]["GCS"] += vals["GCS"]
                all_seg[seg_id]["GRS"] += vals["GRS"]
                all_seg[seg_id]["_n"] += 1

    # Average across slices
    for seg_id in all_seg:
        n = all_seg[seg_id].pop("_n")
        all_seg[seg_id]["GCS"] /= n
        all_seg[seg_id]["GRS"] /= n

    return all_seg
