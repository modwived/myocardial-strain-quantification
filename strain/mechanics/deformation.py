"""Compute deformation gradient and strain tensors from displacement fields."""

import numpy as np


def compute_deformation_gradient(displacement: np.ndarray) -> np.ndarray:
    """Compute the deformation gradient tensor F = I + grad(u).

    Args:
        displacement: (2, H, W) displacement field [dx, dy].

    Returns:
        (2, 2, H, W) deformation gradient tensor F.
    """
    H, W = displacement.shape[1], displacement.shape[2]
    F = np.zeros((2, 2, H, W))

    # Identity
    F[0, 0] = 1.0
    F[1, 1] = 1.0

    # Gradient of displacement (central differences)
    # du_x/dx, du_x/dy
    F[0, 0] += np.gradient(displacement[0], axis=1)  # du_x/dx
    F[0, 1] += np.gradient(displacement[0], axis=0)  # du_x/dy
    # du_y/dx, du_y/dy
    F[1, 0] += np.gradient(displacement[1], axis=1)  # du_y/dx
    F[1, 1] += np.gradient(displacement[1], axis=0)  # du_y/dy

    return F


def compute_green_lagrange_strain(F: np.ndarray) -> np.ndarray:
    """Compute Green-Lagrange strain tensor E = 0.5 * (F^T F - I).

    Args:
        F: (2, 2, H, W) deformation gradient tensor.

    Returns:
        (2, 2, H, W) Green-Lagrange strain tensor E.
    """
    H, W = F.shape[2], F.shape[3]

    # C = F^T F (right Cauchy-Green tensor)
    C = np.zeros((2, 2, H, W))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                C[i, j] += F[k, i] * F[k, j]

    # E = 0.5 * (C - I)
    E = 0.5 * C
    E[0, 0] -= 0.5
    E[1, 1] -= 0.5

    return E
