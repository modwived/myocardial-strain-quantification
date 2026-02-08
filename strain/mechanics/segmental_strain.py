"""AHA 16-segment model for regional strain analysis."""

import numpy as np
from scipy.ndimage import center_of_mass


AHA_SEGMENT_NAMES = {
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


def assign_aha_segments(
    myocardial_mask: np.ndarray,
    slice_level: str = "mid",
) -> np.ndarray:
    """Assign AHA segment labels to myocardial pixels based on angular position.

    Args:
        myocardial_mask: (H, W) binary myocardial mask.
        slice_level: One of 'basal', 'mid', 'apical'.

    Returns:
        (H, W) integer array with segment labels (1-based).
    """
    segments = np.zeros_like(myocardial_mask, dtype=np.int32)
    cy, cx = center_of_mass(myocardial_mask)

    ys, xs = np.where(myocardial_mask > 0)
    angles = np.arctan2(ys - cy, xs - cx)  # range [-pi, pi]
    angles = np.degrees(angles) % 360  # range [0, 360)

    if slice_level == "basal":
        n_segments, offset = 6, 0
    elif slice_level == "mid":
        n_segments, offset = 6, 6
    elif slice_level == "apical":
        n_segments, offset = 4, 12
    else:
        raise ValueError(f"Unknown slice level: {slice_level}")

    segment_size = 360.0 / n_segments
    for i, (y, x) in enumerate(zip(ys, xs)):
        seg_idx = int(angles[i] / segment_size)
        seg_idx = min(seg_idx, n_segments - 1)
        segments[y, x] = seg_idx + 1 + offset

    return segments
