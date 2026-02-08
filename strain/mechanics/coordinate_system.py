"""Define cardiac coordinate system (radial, circumferential) from myocardial geometry."""

import numpy as np
from scipy.ndimage import binary_erosion, center_of_mass


def compute_cardiac_axes(myocardial_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute circumferential and radial unit vectors at each myocardial pixel.

    The circumferential direction is tangent to the myocardial contour
    (perpendicular to the vector pointing from center to pixel).
    The radial direction points outward from the LV center.

    Args:
        myocardial_mask: (H, W) binary mask of the myocardium.

    Returns:
        Tuple of:
          - e_circ: (2, H, W) circumferential unit vectors
          - e_rad: (2, H, W) radial unit vectors
    """
    H, W = myocardial_mask.shape
    e_circ = np.zeros((2, H, W))
    e_rad = np.zeros((2, H, W))

    # Find center of the LV cavity (eroded myocardial mask)
    eroded = binary_erosion(myocardial_mask, iterations=5)
    if eroded.sum() == 0:
        # Fallback: use center of myocardial mask
        cy, cx = center_of_mass(myocardial_mask)
    else:
        cy, cx = center_of_mass(eroded)

    # For each myocardial pixel, compute radial and circumferential directions
    ys, xs = np.where(myocardial_mask > 0)
    for y, x in zip(ys, xs):
        # Radial: from center to pixel
        r = np.array([x - cx, y - cy])
        norm = np.linalg.norm(r)
        if norm < 1e-8:
            continue
        r = r / norm

        # Circumferential: perpendicular to radial (90° counterclockwise)
        c = np.array([-r[1], r[0]])

        e_rad[:, y, x] = r
        e_circ[:, y, x] = c

    return e_circ, e_rad
