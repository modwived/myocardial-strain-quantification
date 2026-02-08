"""Define cardiac coordinate system (radial, circumferential, longitudinal).

The cardiac coordinate system is defined relative to the left-ventricular (LV)
geometry so that strain tensors can be projected onto clinically meaningful
directions:

* **Radial (e_rad)** -- points outward from the LV cavity centre.
* **Circumferential (e_circ)** -- tangent to the myocardial wall,
  perpendicular to the radial direction in the short-axis plane.
* **Longitudinal (e_long)** -- base-to-apex direction (requires information
  from multiple slices or a long-axis view).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass, label as ndlabel


# ---------------------------------------------------------------------------
# Helper: robust centroid from a (possibly multi-component) mask
# ---------------------------------------------------------------------------

def _robust_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return the centroid (cy, cx) of the *largest connected component* of
    *mask*.  This handles cases where the mask may contain small disconnected
    islands (e.g. from imperfect segmentation).

    If the mask is empty, raises ``ValueError``.
    """
    if mask.sum() == 0:
        raise ValueError("Cannot compute centroid of an empty mask.")

    labelled, n_components = ndlabel(mask > 0)
    if n_components <= 1:
        return center_of_mass(mask > 0)

    # Pick the largest connected component
    component_sizes = np.array(
        [(labelled == lbl).sum() for lbl in range(1, n_components + 1)]
    )
    largest = int(component_sizes.argmax()) + 1
    return center_of_mass(labelled == largest)


# ---------------------------------------------------------------------------
# Main function: radial & circumferential axes
# ---------------------------------------------------------------------------

def compute_cardiac_axes(
    myocardial_mask: np.ndarray,
    *,
    lv_cavity_mask: np.ndarray | None = None,
    rv_insertion_point: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute circumferential and radial unit vectors at each myocardial pixel.

    The circumferential direction is tangent to the myocardial contour
    (perpendicular to the vector pointing from centre to pixel).
    The radial direction points outward from the LV centre.

    Args:
        myocardial_mask: (H, W) binary mask of the myocardium (label 2).
        lv_cavity_mask: (H, W) binary mask of the LV cavity (label 3).
            If provided the cavity centroid is used as the LV centre;
            otherwise the centroid of the myocardial mask itself is used.
        rv_insertion_point: (y, x) coordinates of the anterior RV insertion
            point.  Not used for axis directions but stored/returned for
            AHA segmentation angular reference.

    Returns:
        Tuple of:
          - e_circ: (2, H, W) circumferential unit vectors.
          - e_rad:  (2, H, W) radial unit vectors.
    """
    H, W = myocardial_mask.shape
    e_circ = np.zeros((2, H, W), dtype=np.float64)
    e_rad = np.zeros((2, H, W), dtype=np.float64)

    # ---- Determine LV centre ------------------------------------------------
    if lv_cavity_mask is not None and lv_cavity_mask.sum() > 0:
        cy, cx = _robust_centroid(lv_cavity_mask)
    else:
        cy, cx = _robust_centroid(myocardial_mask)

    # ---- Vectorised computation of radial/circumferential directions ---------
    ys, xs = np.where(myocardial_mask > 0)
    if len(ys) == 0:
        return e_circ, e_rad

    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    norms = np.sqrt(dx ** 2 + dy ** 2)
    valid = norms > 1e-8

    # Radial (from centre outward): [dx, dy] normalised
    rad_x = np.where(valid, dx / norms, 0.0)
    rad_y = np.where(valid, dy / norms, 0.0)

    # Circumferential: 90-deg counter-clockwise rotation of the radial vector
    circ_x = -rad_y
    circ_y = rad_x

    e_rad[0, ys, xs] = rad_x
    e_rad[1, ys, xs] = rad_y
    e_circ[0, ys, xs] = circ_x
    e_circ[1, ys, xs] = circ_y

    return e_circ, e_rad


# ---------------------------------------------------------------------------
# Longitudinal axis (for long-axis views or multi-slice short-axis stacks)
# ---------------------------------------------------------------------------

def compute_longitudinal_axis(
    basal_centroid: tuple[float, float],
    apical_centroid: tuple[float, float],
) -> np.ndarray:
    """Compute the unit vector along the LV long axis (base -> apex).

    In a long-axis view the longitudinal direction runs from the mitral-valve
    plane (base) toward the apex.  This helper returns a normalised 2-D
    direction vector.

    Args:
        basal_centroid: (y, x) centroid of the basal myocardial mask.
        apical_centroid: (y, x) centroid of the apical myocardial mask.

    Returns:
        (2,) unit vector [vx, vy] pointing from base to apex.
    """
    dy = apical_centroid[0] - basal_centroid[0]
    dx = apical_centroid[1] - basal_centroid[1]
    norm = np.sqrt(dx ** 2 + dy ** 2)
    if norm < 1e-8:
        # Degenerate case -- return default downward direction
        return np.array([0.0, 1.0])
    return np.array([dx / norm, dy / norm])


def compute_longitudinal_axis_from_masks(
    basal_mask: np.ndarray,
    apical_mask: np.ndarray,
) -> np.ndarray:
    """Convenience wrapper: compute the longitudinal axis from binary masks.

    Args:
        basal_mask: (H, W) binary myocardial mask at the basal level.
        apical_mask: (H, W) binary myocardial mask at the apical level.

    Returns:
        (2,) unit vector [vx, vy] from base to apex.
    """
    bc = _robust_centroid(basal_mask)
    ac = _robust_centroid(apical_mask)
    return compute_longitudinal_axis(bc, ac)


# ---------------------------------------------------------------------------
# RV insertion point utilities
# ---------------------------------------------------------------------------

def compute_rv_insertion_angle(
    lv_center: tuple[float, float],
    rv_insertion_point: tuple[float, float],
) -> float:
    """Return the angular position (degrees, [0, 360)) of the RV anterior
    insertion point relative to the LV centre.

    This angle is used as the zero-degree reference for AHA segmental
    assignment.

    Args:
        lv_center: (cy, cx) LV centre coordinates.
        rv_insertion_point: (y, x) RV anterior insertion point.

    Returns:
        Angle in degrees, measured counter-clockwise from the positive-x axis.
    """
    dy = rv_insertion_point[0] - lv_center[0]
    dx = rv_insertion_point[1] - lv_center[1]
    angle = np.degrees(np.arctan2(dy, dx)) % 360.0
    return float(angle)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_orthogonality(
    e_circ: np.ndarray,
    e_rad: np.ndarray,
    myocardial_mask: np.ndarray,
    *,
    atol: float = 1e-6,
) -> bool:
    """Check that e_circ and e_rad are orthogonal at every myocardial pixel.

    Args:
        e_circ: (2, H, W) circumferential unit vectors.
        e_rad:  (2, H, W) radial unit vectors.
        myocardial_mask: (H, W) binary mask.
        atol: Absolute tolerance for the dot product.

    Returns:
        True if |e_circ . e_rad| < atol everywhere on the mask.
    """
    dot = e_circ[0] * e_rad[0] + e_circ[1] * e_rad[1]
    mask = myocardial_mask > 0
    if mask.sum() == 0:
        return True
    return bool(np.all(np.abs(dot[mask]) < atol))


def get_lv_center(
    myocardial_mask: np.ndarray,
    lv_cavity_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return the LV centre (cy, cx) used for coordinate-system construction.

    This is a convenience function that mirrors the centre-finding logic in
    :func:`compute_cardiac_axes`.
    """
    if lv_cavity_mask is not None and lv_cavity_mask.sum() > 0:
        return _robust_centroid(lv_cavity_mask)
    return _robust_centroid(myocardial_mask)
