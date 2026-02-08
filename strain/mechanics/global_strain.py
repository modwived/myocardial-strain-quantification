"""Compute global strain values (GCS, GRS) from strain tensors."""

import numpy as np

from strain.mechanics.coordinate_system import compute_cardiac_axes
from strain.mechanics.deformation import (
    compute_deformation_gradient,
    compute_green_lagrange_strain,
)


def compute_global_strain(
    displacement: np.ndarray,
    myocardial_mask: np.ndarray,
) -> dict[str, float]:
    """Compute global circumferential and radial strain.

    Args:
        displacement: (2, H, W) displacement field.
        myocardial_mask: (H, W) binary mask of myocardium.

    Returns:
        Dictionary with 'GCS' and 'GRS' values.
    """
    F = compute_deformation_gradient(displacement)
    E = compute_green_lagrange_strain(F)
    e_circ, e_rad = compute_cardiac_axes(myocardial_mask)

    # Project strain tensor onto cardiac coordinates
    # E_cc = e_c^T E e_c, E_rr = e_r^T E e_r
    mask = myocardial_mask > 0

    E_cc = np.zeros_like(myocardial_mask, dtype=np.float64)
    E_rr = np.zeros_like(myocardial_mask, dtype=np.float64)

    for i in range(2):
        for j in range(2):
            E_cc += e_circ[i] * E[i, j] * e_circ[j]
            E_rr += e_rad[i] * E[i, j] * e_rad[j]

    gcs = float(E_cc[mask].mean()) * 100  # percentage
    grs = float(E_rr[mask].mean()) * 100

    return {"GCS": gcs, "GRS": grs}
