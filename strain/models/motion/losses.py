"""Loss functions for unsupervised motion estimation.

Losses
------
* ``NCC`` -- local normalised cross-correlation (image similarity).
* ``DiffusionRegularizer`` -- first-order smoothness on displacement gradients.
* ``BendingEnergy`` -- second-order smoothness (penalises curvature).
* ``CyclicConsistencyLoss`` -- forward-backward displacement consistency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from strain.models.motion.warp import spatial_transform


# ---------------------------------------------------------------------------
# Image similarity
# ---------------------------------------------------------------------------

class NCC(nn.Module):
    """Local Normalized Cross-Correlation loss.

    Returns ``1 - NCC`` so that *lower is better* (minimisation objective).
    """

    def __init__(self, window_size: int = 9):
        super().__init__()
        self.window_size = window_size

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute NCC loss.

        Parameters
        ----------
        predicted : (B, 1, H, W)
            Warped source image.
        target : (B, 1, H, W)
            Target image.

        Returns
        -------
        Scalar loss (1 - NCC).
        """
        w = self.window_size
        pad = w // 2

        I = predicted
        J = target
        I2 = I * I
        J2 = J * J
        IJ = I * J

        pool = nn.AvgPool2d(w, stride=1, padding=pad)
        I_mean = pool(I)
        J_mean = pool(J)
        I2_mean = pool(I2)
        J2_mean = pool(J2)
        IJ_mean = pool(IJ)

        cross = IJ_mean - I_mean * J_mean
        I_var = I2_mean - I_mean * I_mean
        J_var = J2_mean - J_mean * J_mean

        cc = (cross * cross + 1e-5) / (I_var * J_var + 1e-5)
        return 1.0 - cc.mean()


# ---------------------------------------------------------------------------
# Regularizers
# ---------------------------------------------------------------------------

class DiffusionRegularizer(nn.Module):
    """Diffusion regularization on displacement field gradients (first order)."""

    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss.

        Parameters
        ----------
        displacement : (B, 2, H, W)

        Returns
        -------
        Scalar regularization loss (mean of squared spatial gradients).
        """
        dy = displacement[:, :, 1:, :] - displacement[:, :, :-1, :]
        dx = displacement[:, :, :, 1:] - displacement[:, :, :, :-1]
        return (dx.pow(2).mean() + dy.pow(2).mean()) / 2.0


class BendingEnergy(nn.Module):
    """Second-order bending energy regularizer.

    Penalises the second spatial derivatives of the displacement field,
    producing smoother (lower-curvature) deformations than the first-order
    diffusion regularizer.

    .. math::

        L_{bend} = \\frac{1}{N} \\sum \\left(
            \\frac{\\partial^2 u}{\\partial x^2}
          + \\frac{\\partial^2 u}{\\partial y^2}
          + 2 \\frac{\\partial^2 u}{\\partial x \\partial y}
        \\right)^2

    For efficiency the second derivatives are approximated with finite
    differences on the discrete grid.
    """

    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        """Compute bending energy.

        Parameters
        ----------
        displacement : (B, 2, H, W)

        Returns
        -------
        Scalar bending energy loss.
        """
        # Second derivatives via finite differences
        # d^2 u / dx^2
        dxx = (
            displacement[:, :, :, 2:]
            - 2 * displacement[:, :, :, 1:-1]
            + displacement[:, :, :, :-2]
        )
        # d^2 u / dy^2
        dyy = (
            displacement[:, :, 2:, :]
            - 2 * displacement[:, :, 1:-1, :]
            + displacement[:, :, :-2, :]
        )
        # d^2 u / dxdy  (cross derivative)
        dxy = (
            displacement[:, :, 1:, 1:]
            - displacement[:, :, 1:, :-1]
            - displacement[:, :, :-1, 1:]
            + displacement[:, :, :-1, :-1]
        )

        return dxx.pow(2).mean() + dyy.pow(2).mean() + 2.0 * dxy.pow(2).mean()


# ---------------------------------------------------------------------------
# Cyclic consistency
# ---------------------------------------------------------------------------

class CyclicConsistencyLoss(nn.Module):
    """Forward-backward cyclic consistency loss.

    Given a forward displacement field phi_AB (A -> B) and a backward field
    phi_BA (B -> A), the composed round-trip displacement should be zero:

        phi_AB(x) + phi_BA(x + phi_AB(x)) ≈ 0   for all x

    This loss penalises the L2 magnitude of that residual.
    """

    def forward(
        self,
        forward_disp: torch.Tensor,
        backward_disp: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cyclic consistency loss.

        Parameters
        ----------
        forward_disp : (B, 2, H, W)
            Displacement field from A to B.
        backward_disp : (B, 2, H, W)
            Displacement field from B to A.

        Returns
        -------
        Scalar consistency loss (mean L2 of residual round-trip displacement).
        """
        # Warp backward field by the forward field:  phi_BA sampled at (x + phi_AB(x))
        backward_warped = spatial_transform(backward_disp, forward_disp)
        residual = forward_disp + backward_warped  # should be ~ 0
        return residual.pow(2).mean()
