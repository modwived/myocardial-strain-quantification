"""Compute global strain values (GCS, GRS) from strain tensors.

Global circumferential strain (GCS) and global radial strain (GRS) are
computed by projecting the Green-Lagrange strain tensor onto the cardiac
coordinate system and averaging over the myocardium.

This module also provides a time-series variant that processes an entire
cardiac cycle and returns strain-time curves together with peak-strain
detection.
"""

from __future__ import annotations

import numpy as np

from strain.mechanics.coordinate_system import compute_cardiac_axes
from strain.mechanics.deformation import (
    compute_deformation_gradient,
    compute_green_lagrange_strain,
)


# ---------------------------------------------------------------------------
# Single-frame global strain
# ---------------------------------------------------------------------------

def compute_global_strain(
    displacement: np.ndarray,
    myocardial_mask: np.ndarray,
    *,
    lv_cavity_mask: np.ndarray | None = None,
    smooth_sigma: float | None = None,
) -> dict[str, float]:
    """Compute global circumferential and radial strain for a single frame.

    Args:
        displacement: (2, H, W) displacement field.
        myocardial_mask: (H, W) binary mask of myocardium.
        lv_cavity_mask: Optional (H, W) LV cavity mask for improved centre
            detection.
        smooth_sigma: Optional Gaussian smoothing sigma applied to the
            displacement field before computing gradients.

    Returns:
        Dictionary with ``'GCS'`` and ``'GRS'`` values (in percent).
    """
    F = compute_deformation_gradient(displacement, smooth_sigma=smooth_sigma)
    E = compute_green_lagrange_strain(F)
    e_circ, e_rad = compute_cardiac_axes(
        myocardial_mask, lv_cavity_mask=lv_cavity_mask,
    )

    mask = myocardial_mask > 0
    if mask.sum() == 0:
        return {"GCS": 0.0, "GRS": 0.0}

    # Project strain tensor onto cardiac coordinate directions:
    #   E_cc = e_c^T E e_c   (circumferential)
    #   E_rr = e_r^T E e_r   (radial)
    E_cc = np.zeros_like(myocardial_mask, dtype=np.float64)
    E_rr = np.zeros_like(myocardial_mask, dtype=np.float64)

    for i in range(2):
        for j in range(2):
            E_cc += e_circ[i] * E[i, j] * e_circ[j]
            E_rr += e_rad[i] * E[i, j] * e_rad[j]

    gcs = float(E_cc[mask].mean()) * 100.0  # -> percentage
    grs = float(E_rr[mask].mean()) * 100.0

    return {"GCS": gcs, "GRS": grs}


# ---------------------------------------------------------------------------
# Full cardiac-cycle time-series
# ---------------------------------------------------------------------------

def compute_global_strain_timeseries(
    displacements: np.ndarray,
    myocardial_masks: np.ndarray,
    *,
    lv_cavity_masks: np.ndarray | None = None,
    smooth_sigma: float | None = None,
    time_points: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute global strain over an entire cardiac cycle.

    Args:
        displacements: (T, 2, H, W) cumulative displacement from ED at each
            time point.
        myocardial_masks: (T, H, W) myocardial mask at each time point.
        lv_cavity_masks: Optional (T, H, W) LV cavity masks for improved
            centre detection.
        smooth_sigma: Optional Gaussian smoothing sigma.
        time_points: (T,) time stamps in milliseconds.  If *None* a default
            ``0, 1, 2, ...`` index is used.

    Returns:
        Dictionary with keys:
          - ``'GCS'``: (T,) circumferential strain time series (%).
          - ``'GRS'``: (T,) radial strain time series (%).
          - ``'time'``: (T,) time points used.
    """
    T = displacements.shape[0]

    if time_points is None:
        time_points = np.arange(T, dtype=np.float64)
    else:
        time_points = np.asarray(time_points, dtype=np.float64)

    gcs_arr = np.zeros(T, dtype=np.float64)
    grs_arr = np.zeros(T, dtype=np.float64)

    for t in range(T):
        lv_cav = lv_cavity_masks[t] if lv_cavity_masks is not None else None
        result = compute_global_strain(
            displacements[t],
            myocardial_masks[t],
            lv_cavity_mask=lv_cav,
            smooth_sigma=smooth_sigma,
        )
        gcs_arr[t] = result["GCS"]
        grs_arr[t] = result["GRS"]

    return {"GCS": gcs_arr, "GRS": grs_arr, "time": time_points}


# ---------------------------------------------------------------------------
# Peak strain detection
# ---------------------------------------------------------------------------

def detect_peak_strain(
    strain_curve: np.ndarray,
    strain_type: str = "circumferential",
) -> tuple[int, float]:
    """Find the time index and value of peak strain.

    For circumferential and longitudinal strain the peak is the *most negative*
    value (maximum shortening).  For radial strain it is the *most positive*
    value (maximum thickening).

    Args:
        strain_curve: (T,) strain values over time.
        strain_type: One of ``'circumferential'``, ``'longitudinal'``,
            or ``'radial'``.

    Returns:
        Tuple of (peak_index, peak_value).
    """
    if len(strain_curve) == 0:
        return 0, 0.0

    if strain_type in ("circumferential", "longitudinal"):
        idx = int(np.argmin(strain_curve))
    elif strain_type == "radial":
        idx = int(np.argmax(strain_curve))
    else:
        raise ValueError(f"Unknown strain_type: {strain_type}")

    return idx, float(strain_curve[idx])
