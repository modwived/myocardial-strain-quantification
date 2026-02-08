"""Compute deformation gradient and strain tensors from displacement fields.

This module provides continuum-mechanics computations for myocardial strain
quantification.  It converts pixel-level displacement fields (from motion
tracking) into deformation gradient tensors, Green-Lagrange strain tensors,
and principal strains.

Key functions
-------------
- compute_deformation_gradient: F = I + grad(u)
- compute_green_lagrange_strain: E = 0.5 (F^T F - I)
- compute_jacobian: det(F) -- physical validity check
- compute_principal_strains: eigenvalue decomposition of E
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Displacement-field gradient (with optional smoothing & boundary handling)
# ---------------------------------------------------------------------------

def _gradient_with_boundary(field: np.ndarray, axis: int) -> np.ndarray:
    """Compute spatial gradient using central differences in the interior and
    one-sided (forward/backward) differences at the boundaries.

    Args:
        field: 2-D array (H, W).
        axis: 0 for row (y) direction, 1 for column (x) direction.

    Returns:
        Gradient array of the same shape.
    """
    grad = np.empty_like(field, dtype=np.float64)
    n = field.shape[axis]

    if n < 2:
        grad[:] = 0.0
        return grad

    # Interior: central differences
    slices_center = [slice(None)] * field.ndim
    slices_fwd = [slice(None)] * field.ndim
    slices_bwd = [slice(None)] * field.ndim
    slices_center[axis] = slice(1, n - 1)
    slices_fwd[axis] = slice(2, n)
    slices_bwd[axis] = slice(0, n - 2)
    grad[tuple(slices_center)] = (
        field[tuple(slices_fwd)] - field[tuple(slices_bwd)]
    ) / 2.0

    # Boundary: forward difference at start
    slices_start = [slice(None)] * field.ndim
    slices_start_next = [slice(None)] * field.ndim
    slices_start[axis] = 0
    slices_start_next[axis] = 1
    grad[tuple(slices_start)] = (
        field[tuple(slices_start_next)] - field[tuple(slices_start)]
    )

    # Boundary: backward difference at end
    slices_end = [slice(None)] * field.ndim
    slices_end_prev = [slice(None)] * field.ndim
    slices_end[axis] = n - 1
    slices_end_prev[axis] = n - 2
    grad[tuple(slices_end)] = (
        field[tuple(slices_end)] - field[tuple(slices_end_prev)]
    )

    return grad


def compute_deformation_gradient(
    displacement: np.ndarray,
    *,
    smooth_sigma: float | None = None,
) -> np.ndarray:
    """Compute the deformation gradient tensor F = I + grad(u).

    Args:
        displacement: (2, H, W) displacement field [dx, dy].
        smooth_sigma: If not None, apply Gaussian smoothing to the
            displacement field before computing the gradient to reduce noise.
            Typical values: 1.0 -- 2.0 pixels.

    Returns:
        (2, 2, H, W) deformation gradient tensor F.
    """
    disp = displacement.astype(np.float64, copy=True)

    if smooth_sigma is not None and smooth_sigma > 0:
        for c in range(disp.shape[0]):
            disp[c] = gaussian_filter(disp[c], sigma=smooth_sigma)

    H, W = disp.shape[1], disp.shape[2]
    F = np.zeros((2, 2, H, W), dtype=np.float64)

    # Identity
    F[0, 0] = 1.0
    F[1, 1] = 1.0

    # Gradient of displacement using boundary-aware differences
    # F[i, j] += du_i / dx_j
    # axis=1 -> d/dx (column direction), axis=0 -> d/dy (row direction)
    F[0, 0] += _gradient_with_boundary(disp[0], axis=1)  # du_x/dx
    F[0, 1] += _gradient_with_boundary(disp[0], axis=0)  # du_x/dy
    F[1, 0] += _gradient_with_boundary(disp[1], axis=1)  # du_y/dx
    F[1, 1] += _gradient_with_boundary(disp[1], axis=0)  # du_y/dy

    return F


# ---------------------------------------------------------------------------
# Green-Lagrange strain tensor
# ---------------------------------------------------------------------------

def compute_green_lagrange_strain(F: np.ndarray) -> np.ndarray:
    """Compute Green-Lagrange strain tensor E = 0.5 * (F^T F - I).

    Args:
        F: (2, 2, H, W) deformation gradient tensor.

    Returns:
        (2, 2, H, W) Green-Lagrange strain tensor E.
    """
    H, W = F.shape[2], F.shape[3]

    # C = F^T F  (right Cauchy-Green deformation tensor)
    C = np.zeros((2, 2, H, W), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                C[i, j] += F[k, i] * F[k, j]

    # E = 0.5 * (C - I)
    E = 0.5 * C
    E[0, 0] -= 0.5
    E[1, 1] -= 0.5

    return E


# ---------------------------------------------------------------------------
# Jacobian determinant (physical validity check)
# ---------------------------------------------------------------------------

def compute_jacobian(F: np.ndarray) -> np.ndarray:
    """Compute the Jacobian determinant det(F) at every pixel.

    For a physically valid deformation det(F) > 0 everywhere.  A value of 1.0
    means no local volume change; values < 1 indicate compression and > 1
    indicate expansion.

    Args:
        F: (2, 2, H, W) deformation gradient tensor.

    Returns:
        (H, W) array of determinant values.
    """
    return F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]


def check_jacobian(F: np.ndarray, *, tol: float = 0.0) -> tuple[bool, np.ndarray]:
    """Check that det(F) > tol everywhere (physical validity).

    Args:
        F: (2, 2, H, W) deformation gradient tensor.
        tol: Minimum acceptable Jacobian value (default 0).

    Returns:
        Tuple of (is_valid, J) where *is_valid* is True when every element
        satisfies J > tol and *J* is the (H, W) Jacobian field.
    """
    J = compute_jacobian(F)
    is_valid = bool(np.all(J > tol))
    return is_valid, J


# ---------------------------------------------------------------------------
# Principal strains (eigenvalue decomposition of E)
# ---------------------------------------------------------------------------

def compute_principal_strains(E: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute principal strains via eigenvalue decomposition of E.

    At each spatial point the 2x2 symmetric Green-Lagrange tensor is
    decomposed into its eigenvalues (principal strains) and eigenvectors
    (principal directions).

    Args:
        E: (2, 2, H, W) Green-Lagrange strain tensor.

    Returns:
        Tuple of:
          - e1: (H, W) first principal strain (larger eigenvalue).
          - e2: (H, W) second principal strain (smaller eigenvalue).
          - directions: (2, 2, H, W) eigenvectors at each pixel.
            directions[:, 0, ...] corresponds to e1, directions[:, 1, ...] to e2.
    """
    H, W = E.shape[2], E.shape[3]

    # Reshape E to (H*W, 2, 2) for vectorised eigendecomposition
    E_mat = np.empty((H * W, 2, 2), dtype=np.float64)
    E_mat[:, 0, 0] = E[0, 0].ravel()
    E_mat[:, 0, 1] = E[0, 1].ravel()
    E_mat[:, 1, 0] = E[1, 0].ravel()
    E_mat[:, 1, 1] = E[1, 1].ravel()

    # Eigendecomposition (symmetric real matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(E_mat)  # sorted ascending

    # eigh returns ascending order; we want descending (e1 >= e2)
    e1 = eigenvalues[:, 1].reshape(H, W)
    e2 = eigenvalues[:, 0].reshape(H, W)

    directions = np.empty((2, 2, H, W), dtype=np.float64)
    # eigenvectors[:, :, 1] -> largest eigenvalue direction
    directions[0, 0] = eigenvectors[:, 0, 1].reshape(H, W)
    directions[1, 0] = eigenvectors[:, 1, 1].reshape(H, W)
    # eigenvectors[:, :, 0] -> smallest eigenvalue direction
    directions[0, 1] = eigenvectors[:, 0, 0].reshape(H, W)
    directions[1, 1] = eigenvectors[:, 1, 0].reshape(H, W)

    return e1, e2, directions
